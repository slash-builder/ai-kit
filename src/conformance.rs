//! The provider conformance suite: one behavioural contract that **every** [`ProviderAdapter`] must meet.
//!
//! An adapter's own tests prove it parses its vendor's wire format. This suite proves something different and more important: that all
//! adapters behave **the same way** where the rest of the kit depends on it. The planner, the executor and the audit trail were written
//! against these rules, so an adapter that breaks one makes them wrong.
//!
//! The contract, scenario by scenario:
//!
//! | scenario | the adapter must |
//! |---|---|
//! | `Ok` | return a reply that stops normally, with text, the model actually served, and the bytes really sent and received; leave **no money reserved** |
//! | `Refusal` | return a *result* (`Refusal` or `ContentFilter`), never an error, and an executor must **not try a second provider** |
//! | `RateLimited` | fail with `RateLimited` carrying the server's `retry-after`; **release** the reservation |
//! | `Auth` | fail with `Auth`; release the reservation |
//! | `ServerError` | fail with `Unavailable` or `Overloaded`; release the reservation |
//! | `ContextOverflow` | fail with `ContextOverflow`, which must **not** count against the provider's health |
//! | `LeakyErrorBody` | when the upstream error body contains a secret, nothing the caller can see (error text, debug output, audit record) contains it |
//! | `BudgetTooSmall` | fail with `BudgetExceeded` **without sending anything** (the fixture's mock expects zero requests) |
//!
//! plus two rules that need no server: the adapter claims at least one kind and names a model, and its `Debug` output does not contain the
//! credential.
//!
//! **How to use it.** Implement [`Fixture`] for your adapter (build a mock upstream that behaves as each [`Scenario`] says, build the adapter
//! against it) and call [`check`]; it returns the list of violations, empty when the adapter conforms. See `providers/conformance_tests.rs`.
//! This module has no HTTP-mock dependency, so a downstream crate can use it.

use crate::budget::Ledger;
use crate::egress::{
    AgeBand, Label, Policy, ProviderMatch, Purpose, Rule, SourceKind, Subject, VecSink,
};
use crate::modality::{Output, Request, RequestKind};
use crate::planner::counts_against_health;
use crate::provider::{
    run_chain, ChainConfig, FinishReason, ProviderAdapter, ProviderError, ScriptedProvider,
};
use crate::types::InferenceParams;
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;

/// What the mock upstream should do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    Ok,
    Refusal,
    /// A rate limit with `retry-after: 7`.
    RateLimited,
    Auth,
    ServerError,
    ContextOverflow,
    /// An error whose body contains [`SECRET`].
    LeakyErrorBody,
    /// The adapter's budget cannot cover the call; the mock must receive no request.
    BudgetTooSmall,
}

impl Scenario {
    pub const ALL: [Scenario; 8] = [
        Scenario::Ok,
        Scenario::Refusal,
        Scenario::RateLimited,
        Scenario::Auth,
        Scenario::ServerError,
        Scenario::ContextOverflow,
        Scenario::LeakyErrorBody,
        Scenario::BudgetTooSmall,
    ];
}

/// The `retry-after` value the `RateLimited` scenario uses.
pub const RETRY_AFTER_S: u32 = 7;
/// The string a `LeakyErrorBody` upstream puts in its error body.
pub const SECRET: &str = "SECRET-UPSTREAM-BODY-TEXT";

/// An adapter under test, wired to a mock upstream.
pub struct Setup {
    pub adapter: Arc<dyn ProviderAdapter>,
    /// The adapter's ledger, if it has a budget (a remote provider must).
    pub ledger: Option<Arc<Ledger>>,
    /// Keeps the mock server alive for the check; dropping it lets the mock verify its own expectations.
    pub guard: Box<dyn Any + Send + Sync>,
    /// The credential the adapter was built with, if any (the suite checks it never shows in `Debug`).
    pub credential: Option<&'static str>,
}

#[async_trait]
pub trait Fixture: Send + Sync {
    fn name(&self) -> &'static str;
    /// Not every adapter has every scenario (a local server has no budget).
    fn supports(&self, _scenario: Scenario) -> bool {
        true
    }
    /// Build a mock upstream that behaves as `scenario` and an adapter pointed at it.
    async fn setup(&self, scenario: Scenario) -> Setup;
}

fn request() -> Request {
    Request::Text(InferenceParams {
        model_id: "conformance".into(),
        prompt: "hello".into(),
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 64,
        session_id: None,
        preferred_quantization: None,
        request_id: None,
    })
}

fn label() -> Label {
    Label::new(
        SourceKind::HouseholdFile,
        &["notes"],
        &[Subject::person("alice", AgeBand::Adult)],
    )
}

fn allow_all() -> Policy {
    Policy {
        version: 1,
        rules: vec![Rule::allow("conformance", ProviderMatch::Any, &[])],
    }
}

/// Run the whole contract against `fixture`. Returns the violations (`"scenario: what was wrong"`); empty means it conforms.
pub async fn check(fixture: &dyn Fixture) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    let mut bad =
        |scenario: &str, what: String| v.push(format!("{} / {scenario}: {what}", fixture.name()));

    for scenario in Scenario::ALL {
        if !fixture.supports(scenario) {
            continue;
        }
        let name = format!("{scenario:?}");
        let setup = fixture.setup(scenario).await;
        let a = &setup.adapter;
        let before = setup.ledger.as_ref().map(|l| (l.available(), l.spent()));

        // rules that need no server
        if scenario == Scenario::Ok {
            if a.info().kinds.is_empty() {
                bad(&name, "claims no request kind".into());
            }
            if a.model_id().is_empty() {
                bad(&name, "names no model".into());
            }
            if let Some(c) = setup.credential {
                if format!("{a:?}").contains(c) {
                    bad(&name, "its Debug output contains the credential".into());
                }
            }
        }

        let result = a.call(&request()).await;
        let released = |bad: &mut dyn FnMut(&str, String)| {
            if let (Some(l), Some((avail, spent))) = (&setup.ledger, before) {
                if (l.available(), l.spent()) != (avail, spent) {
                    bad(&name, format!("an unbilled failure must release the reservation (available {} -> {}, spent {} -> {})", avail, l.available(), spent, l.spent()));
                }
            }
        };

        match scenario {
            Scenario::Ok => match &result {
                Ok(r) => {
                    let text_ok = matches!(&r.output, Output::Text(t) if !t.completion.is_empty());
                    if !text_ok {
                        bad(&name, "a normal answer has no text".into());
                    }
                    if r.finish != FinishReason::Stop {
                        bad(
                            &name,
                            format!("a normal answer finished with {:?}", r.finish),
                        );
                    }
                    if r.model_served.is_empty() {
                        bad(&name, "does not record the model actually served".into());
                    }
                    if r.bytes_sent == 0 || r.bytes_received == 0 {
                        bad(
                            &name,
                            "does not measure the bytes it sent and received".into(),
                        );
                    }
                    if let Some(l) = &setup.ledger {
                        if l.available() != l.remaining() {
                            bad(
                                &name,
                                "money is still reserved after the call finished".into(),
                            );
                        }
                    }
                }
                Err(e) => bad(&name, format!("a normal answer failed with {e:?}")),
            },
            Scenario::Refusal => {
                match &result {
                    Ok(r)
                        if matches!(
                            r.finish,
                            FinishReason::Refusal | FinishReason::ContentFilter
                        ) => {}
                    Ok(r) => bad(&name, format!("a refusal finished with {:?}", r.finish)),
                    Err(e) => bad(
                        &name,
                        format!("a refusal must be a result, not the error {e:?}"),
                    ),
                }
                // an executor must not fall through to another provider
                let second = ScriptedProvider::new(
                    "second",
                    a.info().locality,
                    &[RequestKind::TextGeneration],
                );
                let chain: Vec<Arc<dyn ProviderAdapter>> = vec![a.clone(), second.clone()];
                let _ = run_chain(
                    &chain,
                    &allow_all(),
                    &label(),
                    &request(),
                    &ChainConfig::new(Purpose::Answer, 1),
                    &VecSink::default(),
                )
                .await;
                if second.calls() != 0 {
                    bad(
                        &name,
                        "a refusal led to a second provider being tried".into(),
                    );
                }
            }
            Scenario::RateLimited => {
                if result.as_ref().err()
                    != Some(&ProviderError::RateLimited {
                        retry_after_s: Some(RETRY_AFTER_S),
                    })
                {
                    bad(
                        &name,
                        format!(
                            "expected RateLimited with retry-after {RETRY_AFTER_S}, got {:?}",
                            result.as_ref().err()
                        ),
                    );
                }
                released(&mut bad);
            }
            Scenario::Auth => {
                if result.as_ref().err() != Some(&ProviderError::Auth) {
                    bad(
                        &name,
                        format!("expected Auth, got {:?}", result.as_ref().err()),
                    );
                }
                released(&mut bad);
            }
            Scenario::ServerError => {
                if !matches!(
                    result.as_ref().err(),
                    Some(ProviderError::Unavailable | ProviderError::Overloaded)
                ) {
                    bad(
                        &name,
                        format!(
                            "expected Unavailable or Overloaded, got {:?}",
                            result.as_ref().err()
                        ),
                    );
                }
                released(&mut bad);
            }
            Scenario::ContextOverflow => {
                match result.as_ref().err() {
                    Some(e) if *e == ProviderError::ContextOverflow => {
                        if counts_against_health(e) {
                            bad(
                                &name,
                                "a context overflow must not count against the provider's health"
                                    .into(),
                            );
                        }
                    }
                    other => bad(&name, format!("expected ContextOverflow, got {other:?}")),
                }
                released(&mut bad);
            }
            Scenario::LeakyErrorBody => {
                let sink = VecSink::default();
                let chain: Vec<Arc<dyn ProviderAdapter>> = vec![a.clone()];
                let outcome = run_chain(
                    &chain,
                    &allow_all(),
                    &label(),
                    &request(),
                    &ChainConfig::new(Purpose::Answer, 1),
                    &sink,
                )
                .await;
                match outcome {
                    Err(e) => {
                        let visible = format!(
                            "{e} {e:?} {:?} {}",
                            sink.0.lock().map(|r| r.clone()).unwrap_or_default(),
                            serde_json::to_string(&*sink.0.lock().unwrap()).unwrap_or_default()
                        );
                        if visible.contains(SECRET) {
                            bad(
                                &name,
                                "the upstream error body reached the caller or the audit record"
                                    .into(),
                            );
                        }
                    }
                    Ok(_) => bad(&name, "an error scenario succeeded".into()),
                }
                released(&mut bad);
            }
            Scenario::BudgetTooSmall => {
                if result.as_ref().err() != Some(&ProviderError::BudgetExceeded) {
                    bad(
                        &name,
                        format!(
                            "expected BudgetExceeded before sending, got {:?}",
                            result.as_ref().err()
                        ),
                    );
                }
                released(&mut bad);
                // the mock was told to expect zero requests: dropping the guard verifies it
            }
        }
        drop(setup);
    }
    v
}

/// Panic with the list of violations unless the adapter conforms. For use in tests.
pub async fn assert_conforms(fixture: &dyn Fixture) {
    let violations = check(fixture).await;
    assert!(
        violations.is_empty(),
        "{} does not conform to the provider contract:\n  - {}",
        fixture.name(),
        violations.join("\n  - ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egress::Locality;
    use crate::provider::ProviderReply;

    /// A well-behaved fake: a scripted provider that answers each scenario the way the contract wants.
    struct Behaves;

    #[derive(Debug)]
    struct Scripted(Arc<ScriptedProvider>);
    #[async_trait]
    impl ProviderAdapter for Scripted {
        fn info(&self) -> &crate::egress::ProviderInfo {
            self.0.info()
        }
        fn model_id(&self) -> &str {
            self.0.model_id()
        }
        async fn call(&self, req: &Request) -> Result<ProviderReply, ProviderError> {
            self.0.call(req).await
        }
    }

    #[async_trait]
    impl Fixture for Behaves {
        fn name(&self) -> &'static str {
            "scripted (conforming)"
        }
        fn supports(&self, s: Scenario) -> bool {
            s != Scenario::BudgetTooSmall
        }
        async fn setup(&self, s: Scenario) -> Setup {
            let p =
                ScriptedProvider::new("scripted", Locality::Device, &[RequestKind::TextGeneration]);
            let script = match s {
                Scenario::Ok => vec![Ok((FinishReason::Stop, "hello".into()))],
                Scenario::Refusal => vec![
                    Ok((FinishReason::Refusal, "no".into())),
                    Ok((FinishReason::Refusal, "no".into())),
                ],
                Scenario::RateLimited => vec![Err(ProviderError::RateLimited {
                    retry_after_s: Some(RETRY_AFTER_S),
                })],
                Scenario::Auth => vec![Err(ProviderError::Auth), Err(ProviderError::Auth)],
                Scenario::ServerError => vec![Err(ProviderError::Overloaded)],
                Scenario::ContextOverflow => vec![Err(ProviderError::ContextOverflow)],
                Scenario::LeakyErrorBody => {
                    vec![Err(ProviderError::Auth), Err(ProviderError::Auth)]
                }
                Scenario::BudgetTooSmall => vec![],
            };
            Setup {
                adapter: Arc::new(Scripted(p.then(script))),
                ledger: None,
                guard: Box::new(()),
                credential: None,
            }
        }
    }

    /// A fake that breaks three rules: wrong class for an auth failure, no retry-after, and a refusal returned as an error.
    struct Breaks;
    #[async_trait]
    impl Fixture for Breaks {
        fn name(&self) -> &'static str {
            "scripted (broken)"
        }
        fn supports(&self, s: Scenario) -> bool {
            s != Scenario::BudgetTooSmall
        }
        async fn setup(&self, s: Scenario) -> Setup {
            let p =
                ScriptedProvider::new("scripted", Locality::Device, &[RequestKind::TextGeneration]);
            let script = match s {
                Scenario::Ok => vec![Ok((FinishReason::Stop, "hello".into()))],
                Scenario::Refusal => vec![Err(ProviderError::Other)],
                Scenario::RateLimited => vec![Err(ProviderError::RateLimited {
                    retry_after_s: None,
                })],
                Scenario::Auth => vec![Err(ProviderError::Other), Err(ProviderError::Other)],
                Scenario::ServerError => vec![Err(ProviderError::Overloaded)],
                Scenario::ContextOverflow => vec![Err(ProviderError::ContextOverflow)],
                Scenario::LeakyErrorBody => {
                    vec![Err(ProviderError::Auth), Err(ProviderError::Auth)]
                }
                Scenario::BudgetTooSmall => vec![],
            };
            Setup {
                adapter: Arc::new(Scripted(p.then(script))),
                ledger: None,
                guard: Box::new(()),
                credential: None,
            }
        }
    }

    #[tokio::test]
    async fn a_conforming_adapter_has_no_violations() {
        let v = check(&Behaves).await;
        assert!(v.is_empty(), "{v:#?}");
    }

    #[tokio::test]
    async fn the_suite_detects_each_broken_rule_and_names_the_scenario() {
        let v = check(&Breaks).await;
        let joined = v.join("\n");
        assert!(
            joined.contains("Refusal") && joined.contains("must be a result"),
            "{joined}"
        );
        assert!(
            joined.contains("RateLimited") && joined.contains("retry-after"),
            "{joined}"
        );
        assert!(
            joined.contains("Auth") && joined.contains("expected Auth"),
            "{joined}"
        );
        assert!(v.len() >= 3, "{joined}");
    }

    #[test]
    fn every_scenario_is_listed_once() {
        let mut seen = Scenario::ALL.to_vec();
        seen.dedup();
        assert_eq!(seen.len(), 8);
    }
}
