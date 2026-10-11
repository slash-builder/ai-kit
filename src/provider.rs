//! Providers and the hop executor. **Prototype, no network.**
//!
//! A [`ProviderAdapter`] is one place a request can run (a local engine, a machine in the house, a SaaS API). The N-way router
//! plans an ordered chain of them; [`run_chain`] walks the chain and, **before every hop, asks the egress [`Policy`] again**
//! ("every hop is a fresh egress decision"). A provider the policy denies is never called, probed or sent content.
//!
//! What this module guarantees (each is a test):
//! - a denied hop makes **zero calls**;
//! - a **refusal** (`FinishReason::Refusal`) is a *result*, returned as is, and **never causes a second hop**;
//! - a provider **error** moves to the next hop, which gets its own fresh decision;
//! - the number of remote hops is capped ([`ChainConfig::max_remote_hops`]) because content already sent cannot be taken back;
//! - an audit record is produced for every remote or denied request and never contains content.
//!
//! Real adapters (Anthropic, Gemini, an OpenAI-compatible local server) implement [`ProviderAdapter`] in their own modules; the
//! [`ScriptedProvider`] here is the test double the conformance suite and the egress tests use.

use crate::egress::{
    Attempt, AuditSink, DenyReason, EgressRecord, Label, Locality, Outcome, Policy, ProviderInfo,
    Purpose, Verdict,
};
use crate::error::AiKitError;
use crate::modality::{Output, Request, RequestKind};
use crate::types::InferenceResponse;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Why a model stopped. A refusal is a **result**, not an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    /// The model declined. Billed, HTTP 200 on some APIs. Never retried on another provider.
    Refusal,
    ContentFilter,
}

/// A provider's answer plus what the audit record needs. Byte counts are the **serialized** sizes the adapter really sent and
/// received; an adapter that cannot measure them says 0 rather than guessing.
#[derive(Debug, Clone)]
pub struct ProviderReply {
    pub output: Output,
    pub finish: FinishReason,
    pub model_served: String,
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

/// What can go wrong at a provider. Classes only: never an upstream body, never a header.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProviderError {
    Auth,
    RateLimited {
        retry_after_s: Option<u32>,
    },
    Overloaded,
    Timeout,
    Network,
    Unavailable,
    ContextOverflow,
    /// A spend limit was reached (non-retryable on the same provider).
    BudgetExceeded,
    Other,
}

impl ProviderError {
    /// A short stable name for the audit record.
    pub fn class(&self) -> &'static str {
        match self {
            ProviderError::Auth => "auth",
            ProviderError::RateLimited { .. } => "rate_limited",
            ProviderError::Overloaded => "overloaded",
            ProviderError::Timeout => "timeout",
            ProviderError::Network => "network",
            ProviderError::Unavailable => "unavailable",
            ProviderError::ContextOverflow => "context_overflow",
            ProviderError::BudgetExceeded => "budget_exceeded",
            ProviderError::Other => "other",
        }
    }

    /// Could the same provider succeed if tried again later? (The executor never retries a hop; this is for adapters and callers.)
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            ProviderError::RateLimited { .. }
                | ProviderError::Overloaded
                | ProviderError::Timeout
                | ProviderError::Network
                | ProviderError::Unavailable
        )
    }
}

/// One place a request can run.
#[async_trait]
pub trait ProviderAdapter: Send + Sync {
    fn info(&self) -> &ProviderInfo;
    /// The model this adapter uses for requests of its kinds (recorded in the audit trail; the reply says what was actually served).
    fn model_id(&self) -> &str;
    async fn call(&self, req: &Request) -> Result<ProviderReply, ProviderError>;
}

/// How [`run_chain`] behaves.
#[derive(Debug, Clone)]
pub struct ChainConfig {
    pub purpose: Purpose,
    /// Unix seconds, passed in so the run is deterministic.
    pub now: u64,
    /// More than this many remote providers are never tried for one request (content already sent cannot be taken back).
    pub max_remote_hops: u32,
    /// Can an owner be asked? When false, "ask the user" is treated as deny.
    pub interactive: bool,
}

impl ChainConfig {
    pub fn new(purpose: Purpose, now: u64) -> Self {
        Self {
            purpose,
            now,
            max_remote_hops: 2,
            interactive: false,
        }
    }
}

/// Walk `chain` in order; every hop gets a fresh egress decision. Returns the first reply that ends the chain (a normal answer or a
/// refusal), or an error if nothing could serve it.
///
/// Errors: [`AiKitError::UnsupportedKind`] if no provider in the chain handles the kind; [`AiKitError::RequiresShareable`] if the only
/// providers that could were blocked by the egress policy; otherwise [`AiKitError::RemoteBackendError`] with the last error class.
pub async fn run_chain(
    chain: &[Arc<dyn ProviderAdapter>],
    policy: &Policy,
    label: &Label,
    req: &Request,
    cfg: &ChainConfig,
    sink: &dyn AuditSink,
) -> Result<(ProviderReply, EgressRecord), AiKitError> {
    let kind = req.kind();
    let mut attempts: Vec<Attempt> = Vec::new();
    let mut remote_hops = 0u32;
    let mut any_capable = false;
    let mut blocked_by_policy = false;
    let mut last_error: Option<ProviderError> = None;
    let mut result: Option<ProviderReply> = None;

    for provider in chain {
        let info = provider.info();
        if !info.kinds.contains(&kind) {
            continue;
        }
        any_capable = true;
        let n = attempts.len() as u32 + 1;
        let mk = |outcome: Outcome,
                  rules: Vec<(String, u32)>,
                  err: Option<&str>,
                  sent: u64,
                  recv: u64| Attempt {
            n,
            provider_id: info.provider_id.clone(),
            locality: info.locality,
            model_id: provider.model_id().to_string(),
            outcome,
            error_class: err.map(str::to_string),
            bytes_sent: sent,
            bytes_received: recv,
            rules,
        };

        // a fresh decision for THIS hop
        let rules = match policy.decide(label, info, cfg.purpose, kind, cfg.now) {
            Verdict::Allow { rules } => rules,
            Verdict::Deny { reason, rules } => {
                blocked_by_policy = true;
                let outcome = match reason {
                    DenyReason::RuleDenies => Outcome::DeniedPolicy,
                    DenyReason::Unlabelled | DenyReason::CredentialsNeverLeave => {
                        Outcome::BlockedLabel
                    }
                };
                attempts.push(mk(outcome, rules, None, 0, 0));
                continue;
            }
            Verdict::AskUser { .. } => {
                // no owner to ask in a non-interactive run; the interactive path is the caller's (it re-runs after consent)
                blocked_by_policy = true;
                attempts.push(mk(Outcome::BlockedLabel, Vec::new(), None, 0, 0));
                continue;
            }
        };
        if info.locality.leaves_home() {
            if remote_hops >= cfg.max_remote_hops {
                attempts.push(mk(
                    Outcome::BlockedLabel,
                    rules,
                    Some("remote_hop_limit"),
                    0,
                    0,
                ));
                continue;
            }
            remote_hops += 1;
        }

        match provider.call(req).await {
            Ok(reply) => {
                attempts.push(mk(
                    Outcome::Sent,
                    rules,
                    None,
                    reply.bytes_sent,
                    reply.bytes_received,
                ));
                // any successful reply is final, a refusal or content-filter included: it is never retried elsewhere
                result = Some(reply);
                break;
            }
            Err(e) => {
                let outcome = if e == ProviderError::Timeout {
                    Outcome::Timeout
                } else {
                    Outcome::ProviderError
                };
                // the request was sent to a remote provider even though it failed: count its size as sent, unknown if it reached
                attempts.push(mk(
                    outcome,
                    rules,
                    Some(e.class()),
                    req.payload_bytes() as u64,
                    0,
                ));
                last_error = Some(e);
            }
        }
    }

    let record = EgressRecord {
        record_id: uuid::Uuid::new_v4().to_string(),
        request_kind: kind,
        purpose: cfg.purpose,
        classes: label.classes.clone(),
        subjects: label.subjects.iter().map(|s| s.scope.clone()).collect(),
        policy_version: policy.version,
        attempts,
    };
    if record.is_worth_keeping() {
        sink.record(record.clone());
    }
    match result {
        Some(reply) => Ok((reply, record)),
        None if !any_capable => Err(AiKitError::UnsupportedKind {
            request_kind: kind.as_str().to_string(),
            reason: "no provider in the chain handles this kind".into(),
        }),
        None if last_error.is_none() && blocked_by_policy => Err(AiKitError::RequiresShareable {
            request_kind: kind.as_str().to_string(),
        }),
        None => Err(AiKitError::RemoteBackendError {
            backend: "chain".into(),
            reason: last_error.map_or("no hop could be tried".to_string(), |e| {
                e.class().to_string()
            }),
        }),
    }
}

/// A provider that answers from a script. For tests and the conformance suite.
pub struct ScriptedProvider {
    info: ProviderInfo,
    model: String,
    script: Mutex<VecDeque<Result<(FinishReason, String), ProviderError>>>,
    calls: AtomicU32,
}

impl ScriptedProvider {
    /// Answers every call with "ok" (`Stop`).
    pub fn new(provider_id: &str, locality: Locality, kinds: &[RequestKind]) -> Arc<Self> {
        Arc::new(Self {
            info: ProviderInfo {
                provider_id: provider_id.into(),
                locality,
                kinds: kinds.to_vec(),
            },
            model: format!("{provider_id}-model"),
            script: Mutex::new(VecDeque::new()),
            calls: AtomicU32::new(0),
        })
    }

    /// Queue the next answers; when the queue is empty it answers "ok".
    pub fn then(
        self: Arc<Self>,
        answers: Vec<Result<(FinishReason, String), ProviderError>>,
    ) -> Arc<Self> {
        if let Ok(mut q) = self.script.lock() {
            q.extend(answers);
        }
        self
    }

    /// How many times `call` was made (a denied hop must leave this at 0).
    pub fn calls(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ProviderAdapter for ScriptedProvider {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }
    fn model_id(&self) -> &str {
        &self.model
    }
    async fn call(&self, req: &Request) -> Result<ProviderReply, ProviderError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let next = self.script.lock().ok().and_then(|mut q| q.pop_front());
        let (finish, text) = next.unwrap_or_else(|| Ok((FinishReason::Stop, "ok".into())))?;
        Ok(ProviderReply {
            output: Output::Text(InferenceResponse {
                completion: text.clone(),
                total_tokens: 2,
                prompt_tokens: 1,
                completion_tokens: 1,
                inference_ms: 1,
                model_version: self.model.clone(),
                quantization: "n/a".into(),
                session_id: Default::default(),
                request_id: "scripted".into(),
            }),
            finish,
            model_served: self.model.clone(),
            bytes_sent: req.payload_bytes() as u64,
            bytes_received: text.len() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egress::{AgeBand, ProviderMatch, Rule, SourceKind, Subject, VecSink};
    use crate::types::InferenceParams;

    const TEXT: RequestKind = RequestKind::TextGeneration;

    fn req() -> Request {
        Request::Text(InferenceParams {
            model_id: "m".into(),
            prompt: "hello".into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 8,
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
    fn allow_all(version: u64) -> Policy {
        Policy {
            version,
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[])],
        }
    }
    fn chain(p: &[&Arc<ScriptedProvider>]) -> Vec<Arc<dyn ProviderAdapter>> {
        p.iter()
            .map(|x| (*x).clone() as Arc<dyn ProviderAdapter>)
            .collect()
    }
    fn cfg() -> ChainConfig {
        ChainConfig::new(Purpose::Answer, 100)
    }
    fn text_of(r: &ProviderReply) -> String {
        match &r.output {
            Output::Text(t) => t.completion.clone(),
            other => panic!("{other:?}"),
        }
    }

    #[tokio::test]
    async fn a_denied_remote_hop_makes_no_call_and_the_error_says_it_needs_consent() {
        let claude = ScriptedProvider::new("claude", Locality::Saas, &[TEXT]);
        let sink = VecSink::default();
        let e = run_chain(
            &chain(&[&claude]),
            &Policy::default(),
            &label(),
            &req(),
            &cfg(),
            &sink,
        )
        .await
        .unwrap_err();
        assert!(matches!(e, AiKitError::RequiresShareable { .. }), "{e:?}");
        assert_eq!(claude.calls(), 0, "a denied provider must never be called");
        let rec = sink.0.lock().unwrap();
        assert_eq!(rec.len(), 1, "a denial is audited");
        assert_eq!(rec[0].attempts[0].outcome, Outcome::BlockedLabel);
        assert_eq!(rec[0].attempts[0].bytes_sent, 0);
    }

    #[tokio::test]
    async fn local_first_then_remote_only_when_policy_allows_and_each_hop_is_decided_again() {
        let local = ScriptedProvider::new("ollama", Locality::Device, &[TEXT])
            .then(vec![Err(ProviderError::Unavailable)]);
        let claude = ScriptedProvider::new("claude", Locality::Saas, &[TEXT]);
        let sink = VecSink::default();
        // policy allows claude: local fails, claude answers
        let (reply, rec) = run_chain(
            &chain(&[&local, &claude]),
            &allow_all(7),
            &label(),
            &req(),
            &cfg(),
            &sink,
        )
        .await
        .unwrap();
        assert_eq!(text_of(&reply), "ok");
        assert_eq!((local.calls(), claude.calls()), (1, 1));
        assert_eq!(rec.attempts.len(), 2);
        assert_eq!(rec.attempts[1].rules, vec![("all".to_string(), 1)]);
        assert_eq!(rec.policy_version, 7);
        // same chain, policy denies claude: local fails, claude is NOT called, the error is the consent one
        let local2 = ScriptedProvider::new("ollama", Locality::Device, &[TEXT])
            .then(vec![Err(ProviderError::Unavailable)]);
        let claude2 = ScriptedProvider::new("claude", Locality::Saas, &[TEXT]);
        let e = run_chain(
            &chain(&[&local2, &claude2]),
            &Policy::default(),
            &label(),
            &req(),
            &cfg(),
            &sink,
        )
        .await
        .unwrap_err();
        assert_eq!(claude2.calls(), 0);
        assert!(
            matches!(e, AiKitError::RemoteBackendError { .. }),
            "the local error is the last real failure: {e:?}"
        );
    }

    #[tokio::test]
    async fn a_refusal_is_a_result_and_never_causes_a_second_hop() {
        let a = ScriptedProvider::new("claude", Locality::Saas, &[TEXT]).then(vec![Ok((
            FinishReason::Refusal,
            "I can't help with that".into(),
        ))]);
        let b = ScriptedProvider::new("gemini", Locality::Saas, &[TEXT]);
        let sink = VecSink::default();
        let (reply, rec) = run_chain(
            &chain(&[&a, &b]),
            &allow_all(1),
            &label(),
            &req(),
            &cfg(),
            &sink,
        )
        .await
        .unwrap();
        assert_eq!(reply.finish, FinishReason::Refusal);
        assert_eq!(
            (a.calls(), b.calls()),
            (1, 0),
            "a refusal must not be retried on another provider"
        );
        assert_eq!(rec.attempts.len(), 1);
    }

    #[tokio::test]
    async fn remote_hops_are_capped_because_sent_content_cannot_be_taken_back() {
        let mk = |id: &str| {
            ScriptedProvider::new(id, Locality::Saas, &[TEXT])
                .then(vec![Err(ProviderError::Overloaded)])
        };
        let (a, b, c) = (mk("a"), mk("b"), mk("c"));
        let sink = VecSink::default();
        let e = run_chain(
            &chain(&[&a, &b, &c]),
            &allow_all(1),
            &label(),
            &req(),
            &cfg(),
            &sink,
        )
        .await
        .unwrap_err();
        assert!(matches!(e, AiKitError::RemoteBackendError { .. }));
        assert_eq!(
            (a.calls(), b.calls(), c.calls()),
            (1, 1, 0),
            "default cap is 2 remote hops"
        );
    }

    #[tokio::test]
    async fn a_kind_no_provider_handles_is_unsupported_not_denied() {
        let embed_only = ScriptedProvider::new("voyage", Locality::Saas, &[RequestKind::Embedding]);
        let sink = VecSink::default();
        let e = run_chain(
            &chain(&[&embed_only]),
            &allow_all(1),
            &label(),
            &req(),
            &cfg(),
            &sink,
        )
        .await
        .unwrap_err();
        assert!(matches!(e, AiKitError::UnsupportedKind { .. }), "{e:?}");
        assert_eq!(embed_only.calls(), 0);
    }

    #[tokio::test]
    async fn the_audit_record_never_contains_the_prompt_or_the_answer() {
        let claude = ScriptedProvider::new("claude", Locality::Saas, &[TEXT])
            .then(vec![Ok((FinishReason::Stop, "SECRET-ANSWER".into()))]);
        let sink = VecSink::default();
        let secret_req = match req() {
            Request::Text(mut p) => {
                p.prompt = "SECRET-PROMPT".into();
                Request::Text(p)
            }
            r => r,
        };
        run_chain(
            &chain(&[&claude]),
            &allow_all(1),
            &label(),
            &secret_req,
            &cfg(),
            &sink,
        )
        .await
        .unwrap();
        let json = serde_json::to_string(&sink.0.lock().unwrap()[0]).unwrap();
        assert!(!json.contains("SECRET"), "{json}");
        assert!(json.contains("claude") && json.contains("bytes_sent"));
    }

    #[tokio::test]
    async fn property_no_remote_provider_is_called_for_content_the_policy_does_not_allow() {
        // exhaustive over a small universe: for every label shape and every policy shape, a remote provider that was called implies
        // the engine allowed it (the engine's invariants are tested in egress.rs; this proves the EXECUTOR obeys the verdict)
        let classes = [
            vec!["notes"],
            vec!["notes", "photos"],
            vec!["credentials"],
            vec![],
        ];
        let subjects = [
            vec![Subject::household()],
            vec![Subject::person("kid", AgeBand::Minor)],
            vec![
                Subject::household(),
                Subject::person("kid", AgeBand::Unknown),
            ],
            vec![],
        ];
        let policies = [
            Policy::default(),
            allow_all(1),
            Policy {
                version: 2,
                rules: vec![Rule::allow("a", ProviderMatch::Any, &[]).including_minors()],
            },
            Policy {
                version: 3,
                rules: vec![
                    Rule::allow("a", ProviderMatch::Any, &[]).including_minors(),
                    Rule::deny("d", ProviderMatch::Any, &["photos"]),
                ],
            },
        ];
        let mut called = 0;
        for c in &classes {
            for s in &subjects {
                for pol in &policies {
                    let l = Label::new(SourceKind::HouseholdFile, c, s);
                    let remote = ScriptedProvider::new("claude", Locality::Saas, &[TEXT]);
                    let sink = VecSink::default();
                    let r = run_chain(&chain(&[&remote]), pol, &l, &req(), &cfg(), &sink).await;
                    let verdict = pol.decide(&l, remote.info(), Purpose::Answer, TEXT, 100);
                    if remote.calls() > 0 {
                        called += 1;
                        assert!(
                            matches!(verdict, Verdict::Allow { .. }),
                            "called without an allow: {c:?} {s:?} v{}",
                            pol.version
                        );
                        assert!(!l.classes.contains(crate::egress::CREDENTIALS_CLASS));
                        assert!(r.is_ok());
                    } else {
                        assert!(r.is_err(), "no call and no error");
                    }
                }
            }
        }
        assert!(called > 0, "the universe must include allowed cases");
    }
}
