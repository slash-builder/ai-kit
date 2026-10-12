//! The [`EgressRouter`] (not the older hybrid `Router`): one object a host application (Courier, the gateway glue, the CLI) assembles once and calls.
//!
//! It owns the pieces the egress-aware path needs, so a caller cannot get the wiring wrong by assembling them by hand:
//! the [`ProviderSet`] (adapters, model cards, health and scorecards), the current [`Policy`], the spend [`Ledger`], and the audit
//! sink. It adds no new decisions of its own; every decision is still made by the planner and the egress engine. What it adds is the
//! order of the checks that must come **before** any of them, and the guarantees that depend on that order:
//!
//! 1. **Granted kinds.** If the caller was granted only some request kinds (decided at the Message Kit boundary), anything else is
//!    refused with `CapabilityDenied` before it is planned, and no provider is looked at.
//! 2. **Validation.** An oversize or malformed request is rejected, never truncated and never sent (`Request::validate`).
//! 3. **One policy snapshot per call.** The policy is read once, at the start of the call, so a policy change made while a request is in
//!    flight cannot change that request's decisions, and the audit record's `policy_version` is the one actually used.
//! 4. **The ledger's real balance.** `budget_available` is read from the ledger at call time, not supplied by the caller.
//!
//! Nothing here reads a clock or the environment: the caller passes `now`, and credentials reach the adapters through the
//! [`CredentialSource`](crate::credentials::CredentialSource) the caller built them with.
//!
//! **Defaults fail closed.** A router built with no policy has none to allow anything (default deny); one built with no ledger has a
//! zero budget (no paid provider can be reserved); one built with no audit sink is an error, because silently dropping the record of what
//! left the house is not a default anyone should get by forgetting a line.

use crate::budget::Ledger;
use crate::egress::{AuditSink, EgressRecord, Label, Policy, Purpose};
use crate::error::{AiKitError, Result};
use crate::modality::{Request, RequestKind};
use crate::planner::{Entry, Needs, Plan, PlanRequest, Preference, ProviderSet};
use crate::provider::ProviderReply;
use std::sync::{Arc, RwLock};

/// One request to route, with everything the decision needs. Built per call; cheap to construct.
#[derive(Debug, Clone)]
pub struct Call<'a> {
    pub request: &'a Request,
    /// What the content is and whose it is, decided by the code that read it. **Unlabelled content is denied for every remote provider.**
    pub label: &'a Label,
    pub purpose: Purpose,
    /// Unix seconds. The kit never reads a clock.
    pub now: u64,
    /// Who is asking (an app id), for the `CapabilityDenied` error and the audit trail.
    pub caller: &'a str,
    /// The request kinds this caller is granted (`None` = no restriction, a trusted in-process caller).
    pub granted: Option<&'a [RequestKind]>,
    /// Nothing may leave this device.
    pub local_only: bool,
    /// Can an owner be asked? When false, "ask" is deny.
    pub interactive: bool,
    /// Regions content may be processed in (`None` = no residency requirement).
    pub allowed_regions: Option<&'a [String]>,
    pub preference: Preference,
    pub needs: Needs,
}

impl<'a> Call<'a> {
    pub fn new(request: &'a Request, label: &'a Label, purpose: Purpose, now: u64) -> Self {
        Self {
            request,
            label,
            purpose,
            now,
            caller: "",
            granted: None,
            local_only: false,
            interactive: false,
            allowed_regions: None,
            preference: Preference::LocalFirst,
            needs: Needs::default(),
        }
    }
    pub fn caller(mut self, app_id: &'a str) -> Self {
        self.caller = app_id;
        self
    }
    pub fn granted(mut self, kinds: &'a [RequestKind]) -> Self {
        self.granted = Some(kinds);
        self
    }
    pub fn local_only(mut self) -> Self {
        self.local_only = true;
        self
    }
    pub fn interactive(mut self) -> Self {
        self.interactive = true;
        self
    }
    pub fn allowed_regions(mut self, regions: &'a [String]) -> Self {
        self.allowed_regions = Some(regions);
        self
    }
    pub fn preference(mut self, p: Preference) -> Self {
        self.preference = p;
        self
    }
    pub fn needs(mut self, n: Needs) -> Self {
        self.needs = n;
        self
    }
}

/// What [`EgressRouter::run`] returns (not to be confused with the older `routed` module). `record` and `plan` are `None` only when the request was refused **before** planning (not granted, or
/// invalid): nothing was planned, nothing was sent, nothing was denied by policy, so there is nothing to audit.
#[derive(Debug)]
pub struct RouteOutcome {
    pub result: Result<ProviderReply>,
    pub record: Option<EgressRecord>,
    pub plan: Option<Plan>,
}

/// A conservative token estimate (3 bytes per token) for the planner's context-fit and budget filters. It over-counts on purpose: an
/// under-estimate could let a request through that the provider then refuses, or a worst-case cost that the ledger then cannot cover.
/// Audio and image bytes are not token-priced, so they count as zero here.
fn estimate_tokens(req: &Request) -> (u64, u64) {
    let input = match req {
        Request::Text(_) | Request::ImageGeneration(_) | Request::Embedding(_) => {
            (req.payload_bytes() as u64).div_ceil(3)
        }
        Request::ImageUnderstanding(p) => (p.prompt.len() as u64).div_ceil(3),
        Request::SpeechToText(_) => 0,
    };
    let output = match req {
        Request::Text(p) => u64::from(p.max_tokens),
        _ => 0,
    };
    (input, output)
}

pub struct EgressRouter {
    providers: ProviderSet,
    policy: RwLock<Arc<Policy>>,
    ledger: Arc<Ledger>,
    sink: Arc<dyn AuditSink>,
}

impl EgressRouter {
    pub fn builder() -> EgressRouterBuilder {
        EgressRouterBuilder::default()
    }

    /// Check the things that must hold before anything is planned.
    fn preflight(&self, call: &Call<'_>) -> Result<()> {
        let kind = call.request.kind();
        if let Some(granted) = call.granted {
            if !granted.contains(&kind) {
                return Err(AiKitError::CapabilityDenied {
                    app_id: call.caller.to_string(),
                    model_id: call.request.model_id().to_string(),
                    request_kind: kind.as_str().to_string(),
                });
            }
        }
        call.request.validate()
    }

    fn snapshot(&self) -> Arc<Policy> {
        // A poisoned lock still holds a valid Arc; take it rather than panic or silently use a different policy.
        match self.policy.read() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }

    fn plan_request<'a>(&self, call: &Call<'a>) -> PlanRequest<'a> {
        let (input, output) = estimate_tokens(call.request);
        let mut pr = PlanRequest::new(call.request.kind(), call.label, call.purpose, call.now);
        pr.input_tokens = input;
        pr.max_output_tokens = output;
        pr.needs = call.needs;
        pr.local_only = call.local_only;
        pr.interactive = call.interactive;
        pr.allowed_regions = call.allowed_regions.map(<[String]>::to_vec);
        pr.budget_available = Some(self.ledger.available());
        pr.preference = call.preference;
        pr
    }

    /// Which providers could serve this call, in what order, and why each other one cannot. **Pure: nothing is sent and no provider is
    /// contacted**, so it is safe to call for a "what could leave" preview. Fails the same pre-checks as [`EgressRouter::run`].
    pub fn plan(&self, call: &Call<'_>) -> Result<Plan> {
        self.preflight(call)?;
        let policy = self.snapshot();
        Ok(self.providers.plan(&self.plan_request(call), &policy))
    }

    /// Route the call. Hops that need the owner's consent are **not** run (see [`EgressRouter::plan`] to find them, ask, then add a rule and
    /// call again: the second call makes a fresh decision).
    pub async fn run(&self, call: &Call<'_>) -> RouteOutcome {
        if let Err(e) = self.preflight(call) {
            return RouteOutcome {
                result: Err(e),
                record: None,
                plan: None,
            };
        }
        let policy = self.snapshot();
        let pr = self.plan_request(call);
        let (result, record, plan) = self
            .providers
            .run(&pr, call.request, &policy, self.sink.as_ref())
            .await;
        RouteOutcome {
            result,
            record: Some(record),
            plan: Some(plan),
        }
    }

    /// Replace the policy. The version must be **strictly greater** than the one in force: a policy is never edited in place under a
    /// version that audit records already refer to. In-flight calls keep the snapshot they started with.
    pub fn set_policy(&self, policy: Policy) -> Result<()> {
        let mut guard = match self.policy.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if policy.version <= guard.version {
            return Err(AiKitError::InvalidParams {
                reason: format!(
                    "policy version {} does not exceed the version in force ({})",
                    policy.version, guard.version
                ),
            });
        }
        *guard = Arc::new(policy);
        Ok(())
    }

    pub fn policy_version(&self) -> u64 {
        self.snapshot().version
    }

    /// The ledger the router reads its budget from. Build the remote adapters with this same ledger, or their spending is invisible to it.
    pub fn ledger(&self) -> &Arc<Ledger> {
        &self.ledger
    }

    pub fn provider_count(&self) -> usize {
        self.providers.len()
    }
}

#[derive(Default)]
pub struct EgressRouterBuilder {
    providers: ProviderSet,
    policy: Policy,
    ledger: Option<Arc<Ledger>>,
    sink: Option<Arc<dyn AuditSink>>,
}

impl EgressRouterBuilder {
    pub fn provider(mut self, entry: Entry) -> Self {
        self.providers.add(entry);
        self
    }
    /// The starting policy. Without one, nothing may leave (default deny).
    pub fn policy(mut self, policy: Policy) -> Self {
        self.policy = policy;
        self
    }
    /// The ledger the remote adapters were built with. Without one the budget is zero and no paid provider can be reserved.
    pub fn ledger(mut self, ledger: Arc<Ledger>) -> Self {
        self.ledger = Some(ledger);
        self
    }
    /// Where audit records go. **Required.**
    pub fn audit_sink(mut self, sink: Arc<dyn AuditSink>) -> Self {
        self.sink = Some(sink);
        self
    }
    pub fn build(self) -> Result<EgressRouter> {
        let sink = self.sink.ok_or_else(|| AiKitError::InvalidParams {
            reason: "a router needs an audit sink: a record of what left the house is not optional"
                .into(),
        })?;
        Ok(EgressRouter {
            providers: self.providers,
            policy: RwLock::new(Arc::new(self.policy)),
            ledger: self.ledger.unwrap_or_else(|| Arc::new(Ledger::new(0))),
            sink,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Pricing;
    use crate::egress::{
        AgeBand, Locality, Outcome, ProviderMatch, Rule, SourceKind, Subject, VecSink,
    };
    use crate::model_card::{Cap, CardSource, ModelCard};
    use crate::planner::Filter;
    use crate::provider::{ProviderError, ScriptedProvider};
    use crate::types::InferenceParams;

    const TEXT: RequestKind = RequestKind::TextGeneration;

    fn entry(id: &str, loc: Locality) -> (Arc<ScriptedProvider>, Entry) {
        let p = ScriptedProvider::new(id, loc, &[TEXT]);
        let card = ModelCard {
            provider_id: id.into(),
            model_id: format!("{id}-model"),
            kinds: vec![TEXT],
            max_input_tokens: Some(100_000),
            max_output_tokens: None,
            retires_at: None,
            image_input: Cap::Unknown,
            structured_outputs: Cap::Unknown,
            tool_use: Cap::Unknown,
            source: CardSource::Declared,
        };
        (p.clone(), Entry::new(p, card))
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
    fn req(prompt: &str) -> Request {
        Request::Text(InferenceParams {
            model_id: "m".into(),
            prompt: prompt.into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 8,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        })
    }
    fn router(entries: Vec<Entry>, policy: Policy) -> (EgressRouter, Arc<VecSink>) {
        let sink = Arc::new(VecSink::default());
        let mut b = EgressRouter::builder()
            .policy(policy)
            .audit_sink(sink.clone());
        for e in entries {
            b = b.provider(e);
        }
        (b.build().unwrap(), sink)
    }

    #[tokio::test]
    async fn a_normal_call_runs_local_first_and_is_audited_under_the_policy_it_used() {
        let (dev, e1) = entry("ollama", Locality::Device);
        let (saas, e2) = entry("claude", Locality::Saas);
        let (r, sink) = router(vec![e2.with_prior(0.99), e1], allow_all(7));
        let (rq, l) = (req("hi"), label());
        let out = r.run(&Call::new(&rq, &l, Purpose::Answer, 1000)).await;
        assert!(out.result.is_ok());
        assert_eq!(
            (dev.calls(), saas.calls()),
            (1, 0),
            "local first, and the cloud model was never touched"
        );
        let rec = out.record.expect("a routed call has a record");
        assert_eq!(rec.policy_version, 7);
        assert_eq!(rec.attempts.len(), 1);
        assert_eq!(rec.attempts[0].outcome, Outcome::Sent);
        assert!(out.plan.is_some());
        // a purely local request needs no per-request record kept
        assert!(!rec.is_worth_keeping());
        assert!(sink.0.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn an_ungranted_kind_is_refused_before_any_provider_is_looked_at() {
        let (p, e) = entry("ollama", Locality::Device);
        let (r, sink) = router(vec![e], allow_all(1));
        let (rq, l) = (req("hi"), label());
        let only_embeddings = [RequestKind::Embedding];
        let out = r
            .run(
                &Call::new(&rq, &l, Purpose::Answer, 1000)
                    .caller("app-1")
                    .granted(&only_embeddings),
            )
            .await;
        match out.result {
            Err(AiKitError::CapabilityDenied {
                app_id,
                request_kind,
                ..
            }) => {
                assert_eq!(app_id, "app-1");
                assert_eq!(request_kind, TEXT.as_str());
            }
            other => panic!("expected CapabilityDenied, got {other:?}"),
        }
        assert_eq!(p.calls(), 0);
        assert!(out.record.is_none() && out.plan.is_none());
        assert!(sink.0.lock().unwrap().is_empty());
        // a granted kind goes through
        let granted = [TEXT];
        let ok = r
            .run(&Call::new(&rq, &l, Purpose::Answer, 1000).granted(&granted))
            .await;
        assert!(ok.result.is_ok());
    }

    #[tokio::test]
    async fn an_oversize_request_is_rejected_not_sent() {
        let (p, e) = entry("ollama", Locality::Device);
        let (r, _) = router(vec![e], allow_all(1));
        // (text prompts are bounded by the model's context, which the planner checks; image-generation prompts have a hard limit)
        let rq =
            Request::ImageGeneration(crate::modality::ImageGenParams::new("m", "x".repeat(5000)));
        let l = label();
        let out = r.run(&Call::new(&rq, &l, Purpose::Answer, 1000)).await;
        assert!(out.result.is_err());
        assert_eq!(p.calls(), 0);
        assert!(out.record.is_none());
    }

    #[tokio::test]
    async fn unlabelled_content_never_reaches_a_remote_provider_even_with_allow_all() {
        let (saas, e) = entry("claude", Locality::Saas);
        let (r, sink) = router(vec![e], allow_all(1));
        let rq = req("hi");
        let unlabelled = Label::unlabelled();
        let out = r
            .run(&Call::new(&rq, &unlabelled, Purpose::Answer, 1000))
            .await;
        assert!(matches!(
            out.result,
            Err(AiKitError::RequiresShareable { .. })
        ));
        assert_eq!(saas.calls(), 0);
        // a denial is worth keeping
        assert!(out.record.unwrap().is_worth_keeping());
        assert_eq!(sink.0.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn no_policy_means_nothing_leaves() {
        let (saas, e) = entry("claude", Locality::Saas);
        let (r, _) = router(vec![e], Policy::default());
        let (rq, l) = (req("hi"), label());
        let out = r.run(&Call::new(&rq, &l, Purpose::Answer, 1000)).await;
        assert!(out.result.is_err());
        assert_eq!(saas.calls(), 0);
    }

    #[test]
    fn a_policy_version_must_increase() {
        let (r, _) = router(vec![], allow_all(5));
        assert_eq!(r.policy_version(), 5);
        assert!(r.set_policy(allow_all(5)).is_err(), "same version");
        assert!(r.set_policy(allow_all(4)).is_err(), "older version");
        assert_eq!(r.policy_version(), 5, "a refused change changes nothing");
        r.set_policy(allow_all(6)).unwrap();
        assert_eq!(r.policy_version(), 6);
    }

    #[tokio::test]
    async fn a_policy_change_takes_effect_on_the_next_call() {
        let (saas, e) = entry("claude", Locality::Saas);
        let (r, _) = router(vec![e], Policy::default());
        let (rq, l) = (req("hi"), label());
        assert!(r
            .run(&Call::new(&rq, &l, Purpose::Answer, 1000))
            .await
            .result
            .is_err());
        r.set_policy(allow_all(1)).unwrap();
        let out = r.run(&Call::new(&rq, &l, Purpose::Answer, 1001)).await;
        assert!(out.result.is_ok());
        assert_eq!(out.record.unwrap().policy_version, 1);
        assert_eq!(saas.calls(), 1);
    }

    #[tokio::test]
    async fn the_budget_comes_from_the_ledger_not_from_the_caller() {
        let (saas, e) = entry("claude", Locality::Saas);
        let priced = e.with_pricing(Pricing {
            input_micro_per_mtok: 3_000_000,
            output_micro_per_mtok: 15_000_000,
            cache_read_micro_per_mtok: 0,
            cache_write_micro_per_mtok: 0,
        });
        // no ledger given: a zero budget, so a priced provider cannot be used
        let (r, _) = router(vec![priced], allow_all(1));
        assert_eq!(r.ledger().available(), 0);
        let (rq, l) = (req("hi"), label());
        let plan = r.plan(&Call::new(&rq, &l, Purpose::Answer, 1000)).unwrap();
        assert!(plan.hops.is_empty());
        assert!(plan.rejected.iter().any(|j| j.filter == Filter::Budget));
        assert_eq!(saas.calls(), 0);
        // and run says so, with the wire code for "nothing eligible", not "kind unsupported" and not "private content"
        let out = r.run(&Call::new(&rq, &l, Purpose::Answer, 1000)).await;
        match out.result {
            Err(e @ AiKitError::NoEligibleProvider { .. }) => {
                assert_eq!(
                    e.to_inference_error_code(),
                    crate::error::InferenceErrorCode::NoEligibleProvider
                );
                assert!(e.to_string().contains("Budget"), "{e}");
                assert!(
                    !e.to_string().contains("claude"),
                    "no provider names in the error: {e}"
                );
            }
            other => panic!("expected NoEligibleProvider, got {other:?}"),
        }
        assert_eq!(saas.calls(), 0);
    }

    #[test]
    fn plan_is_pure_and_does_not_touch_providers() {
        let (p, e) = entry("ollama", Locality::Device);
        let (r, sink) = router(vec![e], allow_all(1));
        let (rq, l) = (req("hi"), label());
        for _ in 0..3 {
            let plan = r.plan(&Call::new(&rq, &l, Purpose::Answer, 1000)).unwrap();
            assert_eq!(plan.hops.len(), 1);
        }
        assert_eq!(p.calls(), 0);
        assert!(sink.0.lock().unwrap().is_empty());
    }

    #[test]
    fn plan_fails_the_same_preflight_checks_as_run() {
        let (_, e) = entry("ollama", Locality::Device);
        let (r, _) = router(vec![e], allow_all(1));
        let (rq, l) = (req("hi"), label());
        let none: [RequestKind; 0] = [];
        assert!(matches!(
            r.plan(&Call::new(&rq, &l, Purpose::Answer, 1000).granted(&none)),
            Err(AiKitError::CapabilityDenied { .. })
        ));
    }

    #[test]
    fn a_router_without_an_audit_sink_cannot_be_built() {
        assert!(matches!(
            EgressRouter::builder().policy(allow_all(1)).build(),
            Err(AiKitError::InvalidParams { .. })
        ));
    }

    #[tokio::test]
    async fn a_failing_provider_is_learned_from_and_the_next_one_serves() {
        let (a, ea) = entry("ollama", Locality::Device);
        // `then` queues on the shared script; the entry holds the same adapter
        let _ = a.clone().then(vec![Err(ProviderError::Overloaded)]);
        let (b, eb) = entry("studio", Locality::Home);
        let (r, _) = router(vec![ea, eb], allow_all(1));
        let (rq, l) = (req("hi"), label());
        let out = r.run(&Call::new(&rq, &l, Purpose::Answer, 1000)).await;
        assert!(out.result.is_ok());
        assert_eq!((a.calls(), b.calls()), (1, 1));
        assert_eq!(out.record.unwrap().attempts.len(), 2);
    }
}
