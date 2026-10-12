//! A local spend ledger: reserve before a call, settle from the response's own usage. **No network.**
//!
//! Anthropic has **no balance endpoint** for an individual account, and Gemini's project spend cap lags by about ten minutes; a prepaid
//! balance running out stops every key on the account at once. So the router keeps its own conservative ledger: before a hop it
//! **reserves** the worst case this call could cost; if the reserve does not fit in what is left, the call is refused **before anything is
//! sent** ([`ProviderError::BudgetExceeded`]); after the call it **settles** the reservation with the cost computed from the usage the
//! provider reported (cache fields included). A call that failed without being billed is **released** at no charge.
//!
//! Prices are **declared by whoever configures the provider** (they change, and a stale hard-coded price is worse than none); all
//! money is in micro-dollars so there is no floating point.

use crate::provider::ProviderError;
use std::sync::Mutex;

/// Token counts as a provider reports them. Cache fields are separate because they are priced differently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
}

/// Declared prices, micro-dollars per million tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pricing {
    pub input_micro_per_mtok: u64,
    pub output_micro_per_mtok: u64,
    pub cache_read_micro_per_mtok: u64,
    pub cache_write_micro_per_mtok: u64,
}

fn per_million(tokens: u64, micro_per_mtok: u64) -> u64 {
    // ceil, so a settled cost is never below the true one
    (u128::from(tokens) * u128::from(micro_per_mtok)).div_ceil(1_000_000) as u64
}

impl Pricing {
    /// The cost of a finished call from its reported usage.
    pub fn cost(&self, u: &Usage) -> u64 {
        per_million(u.input_tokens, self.input_micro_per_mtok)
            + per_million(u.output_tokens, self.output_micro_per_mtok)
            + per_million(u.cache_read_tokens, self.cache_read_micro_per_mtok)
            + per_million(u.cache_write_tokens, self.cache_write_micro_per_mtok)
    }

    /// The most a call could cost: every input token at the highest input rate (a cache write is the dearest) and the full output allowance.
    pub fn worst_case(&self, input_tokens: u64, max_output_tokens: u64) -> u64 {
        let in_rate = self
            .input_micro_per_mtok
            .max(self.cache_write_micro_per_mtok)
            .max(self.cache_read_micro_per_mtok);
        per_million(input_tokens, in_rate)
            + per_million(max_output_tokens, self.output_micro_per_mtok)
    }
}

/// Money set aside for one in-flight call.
#[derive(Debug)]
#[must_use = "settle or release a reservation, or the money stays reserved"]
pub struct Reservation(u64);

#[derive(Debug)]
struct State {
    remaining: u64,
    reserved: u64,
    spent: u64,
}

/// The ledger. Thread-safe; the sum of everything reserved plus everything spent never exceeds the budget it was created with, except
/// when a provider bills more than it was reserved (then `spent` shows the overrun and `remaining` floors at zero).
#[derive(Debug)]
pub struct Ledger(Mutex<State>);

impl Ledger {
    pub fn new(budget_micro_usd: u64) -> Self {
        Self(Mutex::new(State {
            remaining: budget_micro_usd,
            reserved: 0,
            spent: 0,
        }))
    }

    /// Set aside `amount`. Refused (nothing changes) if it does not fit in what is left after other reservations.
    pub fn reserve(&self, amount: u64) -> Result<Reservation, ProviderError> {
        let mut s = self.0.lock().map_err(|_| ProviderError::Other)?;
        if amount > s.remaining.saturating_sub(s.reserved) {
            return Err(ProviderError::BudgetExceeded);
        }
        s.reserved += amount;
        Ok(Reservation(amount))
    }

    /// The call finished and was billed `actual`: release the reservation and charge the real cost.
    pub fn settle(&self, r: Reservation, actual: u64) {
        if let Ok(mut s) = self.0.lock() {
            s.reserved = s.reserved.saturating_sub(r.0);
            s.spent += actual;
            s.remaining = s.remaining.saturating_sub(actual);
        }
    }

    /// The call failed without being billed: give the reservation back, charge nothing.
    pub fn release(&self, r: Reservation) {
        if let Ok(mut s) = self.0.lock() {
            s.reserved = s.reserved.saturating_sub(r.0);
        }
    }

    /// Budget left, not counting reservations in flight.
    pub fn remaining(&self) -> u64 {
        self.0.lock().map_or(0, |s| s.remaining)
    }
    pub fn spent(&self) -> u64 {
        self.0.lock().map_or(0, |s| s.spent)
    }
    /// Budget left after in-flight reservations (what the next `reserve` is checked against).
    pub fn available(&self) -> u64 {
        self.0
            .lock()
            .map_or(0, |s| s.remaining.saturating_sub(s.reserved))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const P: Pricing = Pricing {
        input_micro_per_mtok: 3_000_000, // $3 per million, as an example only
        output_micro_per_mtok: 15_000_000,
        cache_read_micro_per_mtok: 300_000,
        cache_write_micro_per_mtok: 3_750_000,
    };

    #[test]
    fn cost_counts_every_usage_field_and_rounds_up() {
        let u = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 1_000_000,
        };
        assert_eq!(P.cost(&u), 3_000_000 + 15_000_000 + 300_000 + 3_750_000);
        // one token is a fraction of a micro-dollar: rounded UP, never down to zero
        assert_eq!(
            P.cost(&Usage {
                input_tokens: 1,
                ..Usage::default()
            }),
            3
        );
        assert_eq!(
            Pricing::default().cost(&u),
            0,
            "an unpriced model costs nothing here (the caller must declare prices)"
        );
    }

    #[test]
    fn the_worst_case_prices_input_at_the_dearest_rate() {
        // 1M input tokens at the cache-write rate (3.75) plus 1M output tokens at 15
        assert_eq!(P.worst_case(1_000_000, 1_000_000), 3_750_000 + 15_000_000);
    }

    #[test]
    fn a_reservation_that_does_not_fit_is_refused_and_changes_nothing() {
        let l = Ledger::new(1_000);
        assert!(matches!(
            l.reserve(1_001),
            Err(ProviderError::BudgetExceeded)
        ));
        assert_eq!(l.available(), 1_000);
        let r = l.reserve(600).unwrap();
        assert!(
            matches!(l.reserve(401), Err(ProviderError::BudgetExceeded)),
            "the first reservation counts while it is in flight"
        );
        assert_eq!(l.available(), 400);
        l.release(r);
        assert_eq!(
            l.available(),
            1_000,
            "an unbilled failure gives the money back"
        );
    }

    #[test]
    fn settling_charges_the_real_cost_and_frees_the_rest() {
        let l = Ledger::new(1_000);
        let r = l.reserve(900).unwrap();
        l.settle(r, 250);
        assert_eq!((l.spent(), l.remaining(), l.available()), (250, 750, 750));
    }

    #[test]
    fn an_overrun_is_recorded_and_the_budget_floors_at_zero() {
        let l = Ledger::new(100);
        let r = l.reserve(50).unwrap();
        l.settle(r, 300);
        assert_eq!((l.spent(), l.remaining()), (300, 0));
        assert!(
            l.reserve(1).is_err(),
            "nothing more can be reserved once spent"
        );
    }

    #[test]
    fn concurrent_reservations_can_never_exceed_the_budget() {
        let l = Arc::new(Ledger::new(1_000));
        let handles: Vec<_> = (0..32)
            .map(|_| {
                let l = l.clone();
                std::thread::spawn(move || l.reserve(100).ok())
            })
            .collect();
        let granted = handles
            .into_iter()
            .filter_map(|h| h.join().unwrap())
            .count();
        assert_eq!(granted, 10, "exactly budget / amount reservations succeed");
    }
}
