//! Hybrid local/cloud routing for Inference Kit — **POC, not ruled on**.
//!
//! [`HybridInferenceService`] wraps two [`InferenceService`]s — a local one and an
//! optional cloud one — and a [`Router`] that decides, per request, which one runs
//! it and whether a failed local answer is retried in the cloud.
//!
//! # Why this exists, and what it is not
//!
//! The locked invariants already say Inference Kit is *hybrid*: "local-first with
//! cloud fallback; never local-only, never cloud-first". Everything cloud-side is
//! deferred (v0.2), and the Claude proxy backend is explicitly **unruled**: it sends
//! household text to a third party. This module is the *routing mechanism* for that
//! future, built so the safe behaviour is the default and the risky behaviour must be
//! asked for. It does not change what ships: it is pure logic over two trait objects,
//! the cloud side stays optional, and nothing here talks to a network.
//!
//! # The rules, in the order they are applied
//!
//! 1. **Privacy gate (fails closed).** A request is [`Privacy::Private`] unless the
//!    caller says [`Privacy::Shareable`]. Private requests never reach the cloud:
//!    not as the first choice, not as an escalation, not as a fallback after a local
//!    error. If a private local answer fails verification it is returned with a note
//!    saying so; it is not silently upgraded.
//! 2. **Capability.** Raster image generation is not text-in/text-out, so it is
//!    [`Plan::Delegate`]d (the caller must use an image capability; Inference Kit has
//!    none). Vector graphics (SVG) *are* text and are routed like code.
//! 3. **Fit.** A prompt longer than the local model's context goes to the cloud when
//!    allowed (and runs locally, with a note, when not).
//! 4. **Policy.** Otherwise the [`RoutePolicy`] table says which tier goes first for
//!    the task class and whether to *cascade*: run locally, check the answer with the
//!    request's [`Verifier`], and only pay for the cloud when the check fails.
//!
//! # Honest limits
//!
//! - A verifier can only catch failures that are detectable. A JSON check cannot tell
//!   a wrong label from a right one, so a cascade is cheaper than always-cloud but
//!   cannot be as accurate as it. `docs/` of the study that produced the default
//!   policy table quantifies this.
//! - The classifier is a few keyword rules. It is deterministic and free, and wrong
//!   sometimes; callers who know the class should pass it.
//! - Cloud calls are counted in [`RouteStats`], not priced.

use crate::error::{AiKitError, Result};
use crate::service::InferenceService;
use crate::types::{CacheStats, InferenceParams, InferenceResponse, SessionId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// What kind of work a prompt is. Drives the policy table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    /// Pick one label from a small set.
    Classify,
    /// Pull structured facts out of given text.
    Extract,
    /// Pick which of several given documents answers a question.
    Retrieve,
    /// Turn a request into a device command.
    Command,
    /// Write code.
    Code,
    /// Write or transform prose.
    Text,
    /// Open-ended multi-step reasoning.
    Reason,
    /// Vector graphics as SVG source (text out, so it can run on either tier).
    Svg,
    /// Raster images (photo, painting). Not text; the text-only [`HybridInferenceService`] delegates them,
    /// [`crate::routed::MultiHybridService`] runs them on a backend that has the capability.
    ImageRaster,
    /// A question about a given image.
    ImageUnderstanding,
    /// Turn texts into vectors.
    Embedding,
    /// Transcribe audio.
    SpeechToText,
    /// Anything else.
    Other,
}

/// Whether a request's content may leave the device. Defaults to `Private`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Privacy {
    /// Never sent to the cloud.
    #[default]
    Private,
    /// The caller has decided this content may go to the cloud.
    Shareable,
}

/// Where a call ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Local,
    Cloud,
}

/// A cheap check on an answer. `check` returns `None` when there is nothing to check.
#[derive(Clone, Default)]
pub enum Verifier {
    /// No way to tell good from bad.
    #[default]
    None,
    /// The answer is not empty.
    NonEmpty,
    /// The answer contains a parseable JSON object or array.
    Json,
    /// Caller-supplied (run the tests, validate the schema, ...).
    Custom(Arc<dyn Fn(&str) -> bool + Send + Sync>),
}

impl Verifier {
    pub fn check(&self, output: &str) -> Option<bool> {
        match self {
            Verifier::None => None,
            Verifier::NonEmpty => Some(!output.trim().is_empty()),
            Verifier::Json => Some(contains_json(output)),
            Verifier::Custom(f) => Some(f(output)),
        }
    }
    pub fn is_present(&self) -> bool {
        !matches!(self, Verifier::None)
    }
}

impl std::fmt::Debug for Verifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verifier::None => write!(f, "None"),
            Verifier::NonEmpty => write!(f, "NonEmpty"),
            Verifier::Json => write!(f, "Json"),
            Verifier::Custom(_) => write!(f, "Custom(..)"),
        }
    }
}

/// True if `text` contains a JSON object or array that parses.
fn contains_json(text: &str) -> bool {
    text.char_indices()
        .filter(|(_, c)| *c == '{' || *c == '[')
        .any(|(i, _)| {
            serde_json::Deserializer::from_str(&text[i..])
                .into_iter::<serde_json::Value>()
                .next()
                .is_some_and(|r| r.is_ok())
        })
}

/// What to do for one task class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassPolicy {
    /// Tier tried first.
    pub first: Tier,
    /// If `first` is local and the request has a verifier: retry in the cloud when the check fails.
    pub cascade: bool,
}

/// Per-class routing table plus the local context limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutePolicy {
    pub classes: HashMap<TaskClass, ClassPolicy>,
    pub default: ClassPolicy,
    /// Prompts estimated above this many tokens do not fit the local model.
    pub local_context_tokens: u32,
}

impl RoutePolicy {
    /// The invariant's posture: everything local first, cascade where there is a verifier.
    pub fn local_first() -> Self {
        let d = ClassPolicy {
            first: Tier::Local,
            cascade: true,
        };
        Self {
            classes: HashMap::new(),
            default: d,
            local_context_tokens: 4096,
        }
    }

    pub fn from_yaml(s: &str) -> Result<Self> {
        Ok(serde_yaml::from_str(s)?)
    }

    fn for_class(&self, c: TaskClass) -> ClassPolicy {
        self.classes.get(&c).copied().unwrap_or(self.default)
    }
}

/// A request plus the hints that make routing possible.
#[derive(Debug, Clone)]
pub struct RouteRequest {
    /// Sampling and length. `model_id` is ignored: tiers use their configured model.
    pub params: InferenceParams,
    /// Task class, if the caller knows it; otherwise it is classified from the prompt.
    pub class: Option<TaskClass>,
    pub privacy: Privacy,
    pub verifier: Verifier,
}

impl RouteRequest {
    pub fn new(params: InferenceParams) -> Self {
        Self {
            params,
            class: None,
            privacy: Privacy::Private,
            verifier: Verifier::None,
        }
    }
}

/// What the router decided to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Plan {
    /// Run on `first`; if `escalate`, retry on the cloud when verification fails (or the local call errors).
    Run { first: Tier, escalate: bool },
    /// Not something this service can run; the caller must use `capability`.
    Delegate { capability: String },
    /// No configured tier can serve this request, or none this content is allowed to reach.
    Unavailable { capability: String, reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub class: TaskClass,
    pub plan: Plan,
    /// Why, in the order the rules fired.
    pub reasons: Vec<String>,
}

/// Rough token estimate (4 characters per token) used for the fit check.
pub fn estimate_tokens(text: &str) -> u32 {
    (text.chars().count() / 4) as u32 + 1
}

/// Guess the task class from the prompt. A handful of keyword rules: deterministic, free,
/// and sometimes wrong, so pass the class when you know it.
pub fn classify(prompt: &str) -> TaskClass {
    let p = prompt.to_lowercase();
    // Single words match whole words only ("icon" must not fire on "silicon");
    // anything with a space or punctuation is a plain substring.
    let toks: std::collections::HashSet<&str> = p
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let has = |ws: &[&str]| {
        ws.iter().any(|w| {
            if w.chars().all(|c| c.is_alphanumeric()) {
                toks.contains(w)
            } else {
                p.contains(w)
            }
        })
    };
    if has(&[
        "watercolor",
        "oil painting",
        "photo-style",
        "photograph",
        "painting of",
        "photorealistic",
        "digital art of",
        "watercolour",
    ]) {
        return TaskClass::ImageRaster;
    }
    if has(&[
        "svg",
        "icon",
        "icons",
        "logo",
        "logos",
        "diagram",
        "diagrams",
        "flowchart",
        "vector",
        "flat illustration",
    ]) {
        return TaskClass::Svg;
    }
    if has(&[
        "python function",
        "write a function",
        "```",
        "write a class",
        "def",
        "implement a ",
    ]) {
        return TaskClass::Code;
    }
    if has(&["note_id", "which one note", "household notes"]) {
        return TaskClass::Retrieve;
    }
    if has(&["home assistant", "allowed actions"]) {
        return TaskClass::Command;
    }
    if has(&["classify", "as exactly one of", "triage", "label\":"]) {
        return TaskClass::Classify;
    }
    if has(&["extract", "list every", "reply with only a json"]) {
        return TaskClass::Extract;
    }
    if has(&[
        "summarize",
        "rewrite",
        "translate",
        "tagline",
        "headline",
        "bullet point",
        "write a ",
    ]) {
        return TaskClass::Text;
    }
    if has(&["why ", "explain", "step by step", "root cause", "debug"]) {
        return TaskClass::Reason;
    }
    TaskClass::Other
}

/// Decides which tier runs a request. Pure: no I/O, so it is trivially testable.
#[derive(Debug, Clone)]
pub struct Router {
    pub policy: RoutePolicy,
}

impl Router {
    pub fn new(policy: RoutePolicy) -> Self {
        Self { policy }
    }

    pub fn decide(
        &self,
        class: TaskClass,
        privacy: Privacy,
        prompt_tokens: u32,
        cloud_available: bool,
        has_verifier: bool,
    ) -> Decision {
        let mut reasons = Vec::new();
        let done = |plan, reasons| Decision {
            class,
            plan,
            reasons,
        };

        if class == TaskClass::ImageRaster {
            reasons.push("raster images are not text-in/text-out".to_string());
            return done(
                Plan::Delegate {
                    capability: "image_generation".to_string(),
                },
                reasons,
            );
        }

        let cloud_allowed = cloud_available && privacy == Privacy::Shareable;
        if privacy == Privacy::Private {
            reasons.push("privacy=private: content never leaves the device".to_string());
        } else if !cloud_available {
            reasons.push("no cloud backend configured".to_string());
        }
        if !cloud_allowed {
            return done(
                Plan::Run {
                    first: Tier::Local,
                    escalate: false,
                },
                reasons,
            );
        }

        if prompt_tokens > self.policy.local_context_tokens {
            reasons.push(format!(
                "prompt ~{prompt_tokens} tokens exceeds local context {}",
                self.policy.local_context_tokens
            ));
            return done(
                Plan::Run {
                    first: Tier::Cloud,
                    escalate: false,
                },
                reasons,
            );
        }

        let cp = self.policy.for_class(class);
        reasons.push(format!(
            "policy for {class:?}: first={:?}, cascade={}",
            cp.first, cp.cascade
        ));
        let escalate = cp.first == Tier::Local && cp.cascade && has_verifier;
        if cp.first == Tier::Local && cp.cascade && !has_verifier {
            reasons.push(
                "no verifier, so a bad local answer cannot be detected and will not be retried"
                    .to_string(),
            );
        }
        done(
            Plan::Run {
                first: cp.first,
                escalate,
            },
            reasons,
        )
    }
}

impl Router {
    /// Capability-aware decision for any kind of request.
    ///
    /// `local_has` / `cloud_has` say whether each tier has a backend for `cap`. The rules:
    /// nobody has it -> `Unavailable`; only the cloud has it -> cloud if the content may go, otherwise
    /// `Unavailable` (we do not silently use a different kind of model, and we do not leak private content to get an answer);
    /// only local has it -> local; both -> the same text policy as [`Router::decide`].
    #[allow(clippy::too_many_arguments)]
    pub fn decide_multi(
        &self,
        cap: crate::modality::Capability,
        class: TaskClass,
        privacy: Privacy,
        prompt_tokens: u32,
        local_has: bool,
        cloud_has: bool,
        has_verifier: bool,
    ) -> Decision {
        let cap_s = cap.as_str().to_string();
        let mut reasons = Vec::new();
        let done = |plan, reasons| Decision {
            class,
            plan,
            reasons,
        };
        let cloud_ok = cloud_has && privacy == Privacy::Shareable;
        match (local_has, cloud_has) {
            (false, false) => {
                reasons.push("no configured backend provides this capability".to_string());
                done(
                    Plan::Unavailable {
                        capability: cap_s,
                        reason: "no configured backend provides it".to_string(),
                    },
                    reasons,
                )
            }
            (false, true) if !cloud_ok => {
                reasons.push("only the cloud tier provides this capability".to_string());
                reasons.push("privacy=private: content never leaves the device".to_string());
                done(
                    Plan::Unavailable {
                        capability: cap_s,
                        reason: "only the cloud tier provides it and the content is private"
                            .to_string(),
                    },
                    reasons,
                )
            }
            (false, true) => {
                reasons.push("only the cloud tier provides this capability".to_string());
                done(
                    Plan::Run {
                        first: Tier::Cloud,
                        escalate: false,
                    },
                    reasons,
                )
            }
            (true, false) => {
                reasons.push("only the local tier provides this capability".to_string());
                done(
                    Plan::Run {
                        first: Tier::Local,
                        escalate: false,
                    },
                    reasons,
                )
            }
            (true, true) => {
                if cap == crate::modality::Capability::TextGeneration {
                    // Both tiers have text: the original policy (privacy gate, fit, class table).
                    return self.decide(class, privacy, prompt_tokens, true, has_verifier);
                }
                if !cloud_ok {
                    reasons.push("privacy=private: content never leaves the device".to_string());
                    return done(
                        Plan::Run {
                            first: Tier::Local,
                            escalate: false,
                        },
                        reasons,
                    );
                }
                let cp = self.policy.for_class(class);
                reasons.push(format!(
                    "both tiers have {cap_s}; policy for {class:?}: first={:?}, cascade={}",
                    cp.first, cp.cascade
                ));
                done(
                    Plan::Run {
                        first: cp.first,
                        escalate: cp.first == Tier::Local && cp.cascade && has_verifier,
                    },
                    reasons,
                )
            }
        }
    }
}

/// One tier call within a routed request.
#[derive(Debug, Clone, Serialize)]
pub struct Attempt {
    pub tier: Tier,
    pub model_id: String,
    pub inference_ms: Option<u32>,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    /// `None`: nothing to verify. `Some(false)`: failed.
    pub verified: Option<bool>,
    pub error: Option<String>,
}

/// What happened, for logs and audit (invariant 5 wants an immutable trail; this is the data for it).
#[derive(Debug, Clone, Serialize)]
pub struct RouteTrace {
    pub decision: Decision,
    pub attempts: Vec<Attempt>,
    pub final_tier: Option<Tier>,
    pub notes: Vec<String>,
}

/// Running totals, to see what routing is doing over time.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RouteStats {
    pub requests: u64,
    pub local_calls: u64,
    pub cloud_calls: u64,
    pub escalations: u64,
    pub blocked_by_privacy: u64,
    pub cloud_prompt_tokens: u64,
    pub cloud_completion_tokens: u64,
    /// Bytes of request content sent to a cloud tier (text, images, audio), whatever the kind. Set by `MultiHybridService`.
    pub cloud_payload_bytes: u64,
}

/// A local service, an optional cloud one, and the router between them.
pub struct HybridInferenceService {
    local: Arc<dyn InferenceService>,
    cloud: Option<Arc<dyn InferenceService>>,
    local_model: String,
    cloud_model: String,
    router: Router,
    stats: Mutex<RouteStats>,
}

impl HybridInferenceService {
    pub fn new(
        local: Arc<dyn InferenceService>,
        local_model: impl Into<String>,
        cloud: Option<(Arc<dyn InferenceService>, String)>,
        policy: RoutePolicy,
    ) -> Self {
        let (cloud, cloud_model) = match cloud {
            Some((c, m)) => (Some(c), m),
            None => (None, String::new()),
        };
        Self {
            local,
            cloud,
            local_model: local_model.into(),
            cloud_model,
            router: Router::new(policy),
            stats: Mutex::new(RouteStats::default()),
        }
    }

    pub fn route_stats(&self) -> RouteStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// Decide only; run nothing. For `ai-kit-cli route --explain` and for callers that want to
    /// know whether to delegate before building a prompt.
    pub fn explain(&self, req: &RouteRequest) -> Decision {
        let class = req.class.unwrap_or_else(|| classify(&req.params.prompt));
        self.router.decide(
            class,
            req.privacy,
            estimate_tokens(&req.params.prompt),
            self.cloud.is_some(),
            req.verifier.is_present(),
        )
    }

    async fn call(&self, tier: Tier, req: &RouteRequest) -> (Attempt, Option<InferenceResponse>) {
        let (svc, model) = match tier {
            Tier::Local => (&self.local, &self.local_model),
            Tier::Cloud => match &self.cloud {
                Some(c) => (c, &self.cloud_model),
                None => {
                    return (
                        Attempt {
                            tier,
                            model_id: String::new(),
                            inference_ms: None,
                            prompt_tokens: None,
                            completion_tokens: None,
                            verified: None,
                            error: Some("no cloud backend configured".into()),
                        },
                        None,
                    )
                }
            },
        };
        let mut params = req.params.clone();
        params.model_id = model.clone();
        let result = svc.infer(params).await;
        if let Ok(mut s) = self.stats.lock() {
            match tier {
                Tier::Local => s.local_calls += 1,
                Tier::Cloud => s.cloud_calls += 1,
            }
        }
        match result {
            Ok(r) => {
                if tier == Tier::Cloud {
                    if let Ok(mut s) = self.stats.lock() {
                        s.cloud_prompt_tokens += u64::from(r.prompt_tokens);
                        s.cloud_completion_tokens += u64::from(r.completion_tokens);
                    }
                }
                let verified = req.verifier.check(&r.completion);
                (
                    Attempt {
                        tier,
                        model_id: model.clone(),
                        inference_ms: Some(r.inference_ms),
                        prompt_tokens: Some(r.prompt_tokens),
                        completion_tokens: Some(r.completion_tokens),
                        verified,
                        error: None,
                    },
                    Some(r),
                )
            }
            Err(e) => (
                Attempt {
                    tier,
                    model_id: model.clone(),
                    inference_ms: None,
                    prompt_tokens: None,
                    completion_tokens: None,
                    verified: None,
                    error: Some(e.to_string()),
                },
                None,
            ),
        }
    }

    /// Route and run one request.
    pub async fn infer_routed(&self, req: RouteRequest) -> Result<(InferenceResponse, RouteTrace)> {
        if let Ok(mut s) = self.stats.lock() {
            s.requests += 1;
        }
        let decision = self.explain(&req);
        let mut trace = RouteTrace {
            decision: decision.clone(),
            attempts: Vec::new(),
            final_tier: None,
            notes: Vec::new(),
        };
        let (first, escalate) = match decision.plan {
            Plan::Delegate { ref capability } => {
                return Err(AiKitError::InvalidParams {
                    reason: format!(
                        "this request needs the '{capability}' capability, which InferenceService (text in, text out) does not provide"
                    ),
                })
            }
            Plan::Unavailable {
                ref capability,
                ref reason,
            } => {
                return Err(crate::modality::unsupported(
                    capability.parse().unwrap_or(crate::modality::Capability::TextGeneration),
                    reason.clone(),
                ))
            }
            Plan::Run { first, escalate } => (first, escalate),
        };
        let private = req.privacy == Privacy::Private;
        let cloud_ok = self.cloud.is_some() && !private;

        let (a1, r1) = self.call(first, &req).await;
        let failed1 = r1.is_none() || a1.verified == Some(false);
        trace.attempts.push(a1);

        if let Some(r) = r1.as_ref() {
            if !failed1 || !(first == Tier::Local && escalate) {
                if failed1 && private && first == Tier::Local && req.verifier.is_present() {
                    self.note_blocked(&mut trace, "local answer failed verification; not retried in the cloud (privacy=private)");
                }
                trace.final_tier = Some(first);
                return Ok((r.clone(), trace));
            }
        }

        // Local errored, or local failed verification and the plan allows escalating.
        let may_escalate = first == Tier::Local && cloud_ok && (escalate || r1.is_none());
        if !may_escalate {
            return match r1 {
                Some(r) => {
                    trace.final_tier = Some(first);
                    Ok((r, trace))
                }
                None => {
                    if private && first == Tier::Local {
                        self.note_blocked(
                            &mut trace,
                            "local call failed; no cloud fallback (privacy=private)",
                        );
                    }
                    let err = trace
                        .attempts
                        .last()
                        .and_then(|a| a.error.clone())
                        .unwrap_or_else(|| "inference failed".into());
                    Err(AiKitError::Internal { reason: err })
                }
            };
        }

        if let Ok(mut s) = self.stats.lock() {
            s.escalations += 1;
        }
        trace.notes.push(if r1.is_none() {
            "local call failed; falling back to the cloud".into()
        } else {
            "local answer failed verification; escalating to the cloud".into()
        });
        let (a2, r2) = self.call(Tier::Cloud, &req).await;
        let err2 = a2.error.clone();
        trace.attempts.push(a2);
        match (r2, r1) {
            (Some(r), _) => {
                trace.final_tier = Some(Tier::Cloud);
                Ok((r, trace))
            }
            (None, Some(local)) => {
                trace
                    .notes
                    .push("cloud escalation failed; returning the unverified local answer".into());
                trace.final_tier = Some(Tier::Local);
                Ok((local, trace))
            }
            (None, None) => Err(AiKitError::RemoteBackendError {
                backend: "cloud".into(),
                reason: err2.unwrap_or_else(|| "both tiers failed".into()),
            }),
        }
    }

    fn note_blocked(&self, trace: &mut RouteTrace, note: &str) {
        trace.notes.push(note.into());
        if let Ok(mut s) = self.stats.lock() {
            s.blocked_by_privacy += 1;
        }
    }
}

/// The trait path has no hints, so it is local-first and fail-closed: classified from the
/// prompt, `Privacy::Private`, no verifier. In practice that means "local only"; callers who
/// want the cloud must use [`HybridInferenceService::infer_routed`] and say `Shareable`.
#[async_trait]
impl InferenceService for HybridInferenceService {
    async fn infer(&self, params: InferenceParams) -> Result<InferenceResponse> {
        self.infer_routed(RouteRequest::new(params))
            .await
            .map(|(r, _)| r)
    }
    async fn load_model(&self, model_id: &str) -> Result<()> {
        self.local.load_model(model_id).await
    }
    async fn unload_model(&self, model_id: &str) -> Result<()> {
        self.local.unload_model(model_id).await
    }
    async fn cache_stats(&self) -> Result<CacheStats> {
        self.local.cache_stats().await
    }
    async fn end_session(&self, session_id: &SessionId) -> Result<()> {
        self.local.end_session(session_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A scripted service: returns `answer`, or an error, and counts calls.
    struct Mock {
        answer: Result<String>,
        calls: AtomicU32,
    }
    impl Mock {
        fn ok(s: &str) -> Arc<Self> {
            Arc::new(Self {
                answer: Ok(s.to_string()),
                calls: AtomicU32::new(0),
            })
        }
        fn failing() -> Arc<Self> {
            Arc::new(Self {
                answer: Err(AiKitError::Internal {
                    reason: "boom".into(),
                }),
                calls: AtomicU32::new(0),
            })
        }
        fn calls(&self) -> u32 {
            self.calls.load(Ordering::SeqCst)
        }
    }
    #[async_trait]
    impl InferenceService for Mock {
        async fn infer(&self, p: InferenceParams) -> Result<InferenceResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match &self.answer {
                Ok(s) => Ok(InferenceResponse {
                    completion: s.clone(),
                    total_tokens: 20,
                    prompt_tokens: 10,
                    completion_tokens: 10,
                    inference_ms: 5,
                    model_version: p.model_id,
                    quantization: "q".into(),
                    session_id: SessionId::default(),
                    request_id: "r".into(),
                }),
                Err(_) => Err(AiKitError::Internal {
                    reason: "boom".into(),
                }),
            }
        }
        async fn load_model(&self, _: &str) -> Result<()> {
            Ok(())
        }
        async fn unload_model(&self, _: &str) -> Result<()> {
            Ok(())
        }
        async fn cache_stats(&self) -> Result<CacheStats> {
            Err(AiKitError::Internal {
                reason: "n/a".into(),
            })
        }
        async fn end_session(&self, _: &SessionId) -> Result<()> {
            Ok(())
        }
    }

    fn params(prompt: &str) -> InferenceParams {
        InferenceParams {
            model_id: String::new(),
            prompt: prompt.into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 32,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        }
    }
    fn req(prompt: &str, privacy: Privacy, v: Verifier) -> RouteRequest {
        RouteRequest {
            params: params(prompt),
            class: None,
            privacy,
            verifier: v,
        }
    }
    fn svc(
        local: &Arc<Mock>,
        cloud: Option<&Arc<Mock>>,
        policy: RoutePolicy,
    ) -> HybridInferenceService {
        HybridInferenceService::new(
            local.clone(),
            "local-m",
            cloud.map(|c| {
                (
                    c.clone() as Arc<dyn InferenceService>,
                    "cloud-m".to_string(),
                )
            }),
            policy,
        )
    }

    #[tokio::test]
    async fn private_never_reaches_the_cloud_even_when_verification_fails() {
        let (l, c) = (Mock::ok("not json"), Mock::ok("{\"a\":1}"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        let (r, t) = s
            .infer_routed(req("extract the facts", Privacy::Private, Verifier::Json))
            .await
            .unwrap();
        assert_eq!(r.completion, "not json");
        assert_eq!(c.calls(), 0, "private content reached the cloud");
        assert_eq!(t.final_tier, Some(Tier::Local));
        assert_eq!(s.route_stats().blocked_by_privacy, 1);
    }

    #[tokio::test]
    async fn private_local_error_is_not_rescued_by_the_cloud() {
        let (l, c) = (Mock::failing(), Mock::ok("x"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        assert!(s
            .infer_routed(req("hi", Privacy::Private, Verifier::None))
            .await
            .is_err());
        assert_eq!(c.calls(), 0);
    }

    #[tokio::test]
    async fn shareable_failed_verification_escalates() {
        let (l, c) = (Mock::ok("not json"), Mock::ok("{\"a\":1}"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        let (r, t) = s
            .infer_routed(req("extract the facts", Privacy::Shareable, Verifier::Json))
            .await
            .unwrap();
        assert_eq!(r.completion, "{\"a\":1}");
        assert_eq!((l.calls(), c.calls()), (1, 1));
        assert_eq!(t.final_tier, Some(Tier::Cloud));
        assert_eq!(t.attempts.len(), 2);
        assert_eq!(s.route_stats().escalations, 1);
    }

    #[tokio::test]
    async fn shareable_passing_verification_stays_local() {
        let (l, c) = (Mock::ok("{\"a\":1}"), Mock::ok("unused"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        let (_, t) = s
            .infer_routed(req("extract the facts", Privacy::Shareable, Verifier::Json))
            .await
            .unwrap();
        assert_eq!(c.calls(), 0);
        assert_eq!(t.final_tier, Some(Tier::Local));
    }

    #[tokio::test]
    async fn no_verifier_means_no_retry() {
        let (l, c) = (Mock::ok("garbage"), Mock::ok("good"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        let (r, _) = s
            .infer_routed(req("write a tagline", Privacy::Shareable, Verifier::None))
            .await
            .unwrap();
        assert_eq!(r.completion, "garbage");
        assert_eq!(c.calls(), 0);
    }

    #[tokio::test]
    async fn policy_can_send_a_class_straight_to_the_cloud() {
        let mut p = RoutePolicy::local_first();
        p.classes.insert(
            TaskClass::Code,
            ClassPolicy {
                first: Tier::Cloud,
                cascade: false,
            },
        );
        let (l, c) = (Mock::ok("local"), Mock::ok("cloud"));
        let s = svc(&l, Some(&c), p);
        let (r, t) = s
            .infer_routed(req(
                "Write a Python function f(x).",
                Privacy::Shareable,
                Verifier::None,
            ))
            .await
            .unwrap();
        assert_eq!(r.completion, "cloud");
        assert_eq!(l.calls(), 0);
        assert_eq!(t.decision.class, TaskClass::Code);
    }

    #[tokio::test]
    async fn without_a_cloud_backend_everything_is_local() {
        let l = Mock::ok("fine");
        let s = svc(&l, None, RoutePolicy::local_first());
        let (r, _) = s
            .infer_routed(req("x", Privacy::Shareable, Verifier::Json))
            .await
            .unwrap();
        assert_eq!(r.completion, "fine");
    }

    #[tokio::test]
    async fn oversize_prompt_goes_to_the_cloud_only_when_allowed() {
        let long = "word ".repeat(6000);
        let (l, c) = (Mock::ok("local"), Mock::ok("cloud"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        let (r, _) = s
            .infer_routed(req(&long, Privacy::Shareable, Verifier::None))
            .await
            .unwrap();
        assert_eq!(r.completion, "cloud");
        let (r, _) = s
            .infer_routed(req(&long, Privacy::Private, Verifier::None))
            .await
            .unwrap();
        assert_eq!(r.completion, "local");
    }

    #[tokio::test]
    async fn raster_images_are_delegated_not_attempted() {
        let (l, c) = (Mock::ok("x"), Mock::ok("y"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        let r = req(
            "a watercolor painting of a cottage",
            Privacy::Shareable,
            Verifier::None,
        );
        assert!(matches!(s.explain(&r).plan, Plan::Delegate { .. }));
        assert!(matches!(
            s.infer_routed(r).await,
            Err(AiKitError::InvalidParams { .. })
        ));
        assert_eq!((l.calls(), c.calls()), (0, 0));
    }

    #[tokio::test]
    async fn local_error_falls_back_to_the_cloud_when_shareable() {
        let (l, c) = (Mock::failing(), Mock::ok("cloud"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        let (r, t) = s
            .infer_routed(req("hi", Privacy::Shareable, Verifier::None))
            .await
            .unwrap();
        assert_eq!(r.completion, "cloud");
        assert_eq!(t.final_tier, Some(Tier::Cloud));
    }

    #[tokio::test]
    async fn the_plain_trait_path_is_local_only() {
        let (l, c) = (Mock::ok("local"), Mock::ok("cloud"));
        let s = svc(&l, Some(&c), RoutePolicy::local_first());
        let r = s.infer(params("anything at all")).await.unwrap();
        assert_eq!(r.completion, "local");
        assert_eq!(c.calls(), 0);
    }

    #[test]
    fn classifier_covers_the_study_prompts() {
        let cases = [
            (
                "Write a Python function f(x) that doubles.",
                TaskClass::Code,
            ),
            (
                "a watercolor painting of a misty forest cottage",
                TaskClass::ImageRaster,
            ),
            (
                "a flat, simple icon of a house with a heart in it",
                TaskClass::Svg,
            ),
            ("a diagram of three boxes joined by arrows", TaskClass::Svg),
            (
                "Which ONE note answers it? Reply {\"note_id\": \"n3\"}",
                TaskClass::Retrieve,
            ),
            (
                "Home Assistant devices: lamp (light). Turn the request into one command.",
                TaskClass::Command,
            ),
            (
                "Classify each as exactly one of: urgent, normal, ignore.",
                TaskClass::Classify,
            ),
            ("Summarize this in at most 35 words.", TaskClass::Text),
            (
                "Extract the action items. Reply with ONLY a JSON list.",
                TaskClass::Extract,
            ),
            ("hello there", TaskClass::Other),
        ];
        for (p, want) in cases {
            assert_eq!(classify(p), want, "{p}");
        }
    }

    #[test]
    fn json_verifier_finds_json_inside_prose_and_fences() {
        assert!(contains_json("sure: ```json\n{\"a\": [1,2]}\n``` done"));
        assert!(contains_json("[1, 2, 3]"));
        assert!(!contains_json("no braces here"));
        assert!(!contains_json("{broken"));
    }

    #[test]
    fn policy_round_trips_through_yaml() {
        let y = "classes:\n  code: {first: cloud, cascade: false}\ndefault: {first: local, cascade: true}\nlocal_context_tokens: 2048\n";
        let p = RoutePolicy::from_yaml(y).unwrap();
        assert_eq!(p.for_class(TaskClass::Code).first, Tier::Cloud);
        assert_eq!(p.for_class(TaskClass::Text).first, Tier::Local);
        assert_eq!(p.local_context_tokens, 2048);
    }
}
