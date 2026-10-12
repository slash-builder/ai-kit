//! A local spend ledger: reserve before a call, settle from the response's own usage. **No network.**
//!
//! Anthropic has **no balance endpoint** for an individual account, and Gemini's project spend cap lags by about ten minutes; a prepaid
//! balance running out stops every key on the account at once. So the router keeps its own conservative ledger: before a hop it
//! **reserves** the worst case this call could cost; if the reserve does not fit in what is left, the call is refused **before anything is
//! sent** ([`ProviderError::BudgetExceeded`]); after the call it **settles** the reservation with the cost computed from the usage the
//! provider reported (cache fields included). A call that failed without being billed is **released** at no charge.
//!
//! **A daily cap.** A total budget alone is a fine ceiling for a prepaid account but a poor guard against a runaway loop: a bug that
//! retries a paid call all night should be stopped by the first 24 hours, not by the whole balance. [`Ledger::with_daily_cap`] adds a
//! limit on what may be spent in **any rolling 24-hour window** (settled spend plus what is reserved right now). It is rolling, not "per
//! calendar day", so it needs no time zone and cannot be dodged by a call that straddles midnight. The kit never reads a clock: the
//! caller supplies a [`Clock`], so one clock serves the whole application and tests control time.
//!
//! Prices are **declared by whoever configures the provider** (they change, and a stale hard-coded price is worse than none); all
//! money is in micro-dollars so there is no floating point.

use crate::provider::ProviderError;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Where the ledger gets "now" (unix seconds). The kit never reads a clock itself.
pub trait Clock: Send + Sync {
    fn now(&self) -> u64;
}

/// The length of the rolling window the daily cap applies to.
pub const DAILY_WINDOW_S: u64 = 24 * 60 * 60;

struct Daily {
    cap: u64,
    clock: Arc<dyn Clock>,
    /// Settled spend inside the window, oldest first: (time, micro-dollars).
    spends: VecDeque<(u64, u64)>,
}

impl Daily {
    /// Forget spend that has aged out of the window, and return what is left in it.
    fn spent_in_window(&mut self) -> u64 {
        let now = self.clock.now();
        while let Some(&(t, _)) = self.spends.front() {
            if now.saturating_sub(t) >= DAILY_WINDOW_S {
                self.spends.pop_front();
            } else {
                break;
            }
        }
        self.spends.iter().map(|&(_, a)| a).sum()
    }
}

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

struct State {
    remaining: u64,
    reserved: u64,
    spent: u64,
    daily: Option<Daily>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State")
            .field("remaining", &self.remaining)
            .field("reserved", &self.reserved)
            .field("spent", &self.spent)
            .field("daily_cap", &self.daily.as_ref().map(|d| d.cap))
            .finish()
    }
}

impl State {
    /// What may still be reserved: the total budget left after reservations, and the daily cap left after the window's spend and the
    /// same reservations, whichever is smaller.
    fn available(&mut self) -> u64 {
        let total = self.remaining.saturating_sub(self.reserved);
        match self.daily.as_mut() {
            None => total,
            Some(d) => {
                let used = d.spent_in_window().saturating_add(self.reserved);
                total.min(d.cap.saturating_sub(used))
            }
        }
    }
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
            daily: None,
        }))
    }

    /// A total budget **and** a cap on spend in any rolling 24 hours. Both must have room for a reservation to succeed.
    pub fn with_daily_cap(
        budget_micro_usd: u64,
        daily_cap_micro_usd: u64,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self(Mutex::new(State {
            remaining: budget_micro_usd,
            reserved: 0,
            spent: 0,
            daily: Some(Daily {
                cap: daily_cap_micro_usd,
                clock,
                spends: VecDeque::new(),
            }),
        }))
    }

    /// Set aside `amount`. Refused (nothing changes) if it does not fit in what is left after other reservations.
    pub fn reserve(&self, amount: u64) -> Result<Reservation, ProviderError> {
        let mut s = self.0.lock().map_err(|_| ProviderError::Other)?;
        if amount > s.available() {
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
            if let Some(d) = s.daily.as_mut() {
                let now = d.clock.now();
                d.spends.push_back((now, actual));
            }
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
    /// Budget left after in-flight reservations (what the next `reserve` is checked against): the smaller of what the total budget and
    /// the daily cap each still allow.
    pub fn available(&self) -> u64 {
        self.0.lock().map_or(0, |mut s| s.available())
    }

    /// Spend settled in the last 24 hours (`0` for a ledger with no daily cap, which does not track it).
    pub fn spent_today(&self) -> u64 {
        self.0.lock().map_or(0, |mut s| {
            s.daily.as_mut().map_or(0, Daily::spent_in_window)
        })
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

    /// A clock the test moves by hand.
    struct Manual(std::sync::atomic::AtomicU64);
    impl Manual {
        fn at(t: u64) -> Arc<Self> {
            Arc::new(Self(std::sync::atomic::AtomicU64::new(t)))
        }
        fn set(&self, t: u64) {
            self.0.store(t, std::sync::atomic::Ordering::SeqCst);
        }
    }
    impl Clock for Manual {
        fn now(&self) -> u64 {
            self.0.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[test]
    fn the_daily_cap_stops_spending_even_when_the_total_budget_has_room() {
        let clock = Manual::at(1_000);
        let l = Ledger::with_daily_cap(10_000, 100, clock.clone());
        let r = l.reserve(60).unwrap();
        l.settle(r, 60);
        assert_eq!(l.spent_today(), 60);
        assert_eq!(
            l.available(),
            40,
            "the daily cap, not the total, is the smaller"
        );
        assert!(matches!(l.reserve(41), Err(ProviderError::BudgetExceeded)));
        assert!(l.reserve(40).is_ok());
    }

    #[test]
    fn the_window_rolls_and_old_spend_stops_counting() {
        let clock = Manual::at(1_000);
        let l = Ledger::with_daily_cap(10_000, 100, clock.clone());
        let r = l.reserve(100).unwrap();
        l.settle(r, 100);
        assert_eq!(l.available(), 0);
        clock.set(1_000 + DAILY_WINDOW_S - 1);
        assert_eq!(
            l.available(),
            0,
            "one second short of 24 hours: still counted"
        );
        clock.set(1_000 + DAILY_WINDOW_S);
        assert_eq!(l.available(), 100, "24 hours later it has aged out");
        assert_eq!(l.spent_today(), 0);
        assert_eq!(
            l.spent(),
            100,
            "the lifetime total is unaffected by the window"
        );
        assert_eq!(l.remaining(), 9_900);
    }

    #[test]
    fn spend_is_counted_when_it_is_settled_not_when_it_was_reserved() {
        let clock = Manual::at(0);
        let l = Ledger::with_daily_cap(10_000, 100, clock.clone());
        let r = l.reserve(100).unwrap();
        clock.set(DAILY_WINDOW_S + 500); // the call took a very long time
        l.settle(r, 100);
        assert_eq!(
            l.available(),
            0,
            "billed just now, so it counts against the window it was billed in"
        );
    }

    #[test]
    fn reservations_in_flight_count_against_the_daily_cap() {
        let clock = Manual::at(0);
        let l = Ledger::with_daily_cap(10_000, 100, clock);
        let r = l.reserve(70).unwrap();
        assert!(
            l.reserve(31).is_err(),
            "70 reserved leaves 30 of the daily cap"
        );
        l.release(r);
        assert!(
            l.reserve(100).is_ok(),
            "an unbilled failure frees the daily room too"
        );
    }

    #[test]
    fn the_total_budget_still_binds_when_the_daily_cap_is_larger() {
        let l = Ledger::with_daily_cap(50, 1_000_000, Manual::at(0));
        assert_eq!(l.available(), 50);
        assert!(l.reserve(51).is_err());
    }

    #[test]
    fn a_zero_daily_cap_allows_nothing_and_a_clock_going_backwards_does_not_panic() {
        let clock = Manual::at(5_000);
        let l = Ledger::with_daily_cap(1_000, 0, clock.clone());
        assert!(l.reserve(1).is_err());
        let l = Ledger::with_daily_cap(1_000, 10, clock.clone());
        let r = l.reserve(10).unwrap();
        l.settle(r, 10);
        clock.set(1); // earlier than the recorded spend
        assert_eq!(
            l.available(),
            0,
            "spend recorded in the 'future' still counts"
        );
    }

    #[test]
    fn a_ledger_without_a_daily_cap_behaves_as_before() {
        let l = Ledger::new(1_000);
        let r = l.reserve(500).unwrap();
        l.settle(r, 500);
        assert_eq!((l.available(), l.spent_today()), (500, 0));
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
