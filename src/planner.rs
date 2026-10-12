//! The N-way planner: which providers may serve a request, in what order. **A pure function; no network, no clock, no randomness.**
//!
//! [`ProviderSet::plan`] filters first and ranks second ("constraints before objective"). Every candidate that is excluded gets a
//! [`Rejection`] naming the filter and the reason, so "why did it not use X?" always has an answer. The filters, in order:
//!
//! 1. **kind and required features** (an unknown capability counts as missing);
//! 2. **local only** (the caller said nothing may leave this device);
//! 3. **egress**: [`Policy::decide`] for this label, provider and purpose. A denied provider is rejected *here*, before anything about it
//!    (its health, its price, its score) is looked at, so it can never be probed or sent content. "Ask the user" is a rejection unless the
//!    caller is interactive, in which case the hop is kept, last, and flagged `needs_consent`;
//! 4. **context fit** (a model whose limit is unknown is assumed to hold only [`UNKNOWN_CONTEXT_ASSUMED`] tokens);
//! 5. **residency** (an unknown region is not an allowed region, for a provider outside the home);
//! 6. **health** (an open circuit breaker; a half-open one is kept as a last-resort trial);
//! 7. **budget** (the estimated worst-case cost against what is left).
//!
//! The survivors are then ordered by the [`Preference`]: by default **local first** (device, then household, then third party: the kit's
//! "never cloud-first" default), then measured quality (the lower bound of a Wilson interval, used only once a provider has enough
//! samples; otherwise its declared prior), then cheaper, then provider id. The last key makes the result **deterministic**, so the same
//! inputs always give the same plan, whatever order the providers were registered in.
//!
//! [`ProviderSet::run`] executes a plan through [`run_chain_recorded`] (a fresh egress decision at every hop), and learns from the outcome:
//! breakers and scorecards are updated from the audit record, including from requests that failed.

use crate::budget::Pricing;
use crate::egress::{AuditSink, EgressRecord, Label, Locality, Outcome, Policy, Purpose, Verdict};
use crate::error::AiKitError;
use crate::modality::{Request, RequestKind};
use crate::model_card::ModelCard;
use crate::provider::{
    run_chain_recorded, ChainConfig, ProviderAdapter, ProviderError, ProviderReply,
};
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};

/// What a model with an unknown context limit is assumed to hold, in tokens.
pub const UNKNOWN_CONTEXT_ASSUMED: u64 = 4096;

/// Samples a scorecard needs before it overrides the declared prior.
pub const MIN_SAMPLES: u32 = 20;

// ------------------------------------------------------------------------------------------------- health

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    /// Healthy.
    Closed,
    /// Too many recent failures: do not use until the cool-down passes.
    Open,
    /// The cool-down has passed: one trial is allowed (used last).
    HalfOpen,
}

/// A circuit breaker driven by the caller's clock (seconds).
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    threshold: u32,
    cooldown_s: u64,
    consecutive_failures: u32,
    opened_at: Option<u64>,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, cooldown_s: u64) -> Self {
        Self {
            threshold: threshold.max(1),
            cooldown_s,
            consecutive_failures: 0,
            opened_at: None,
        }
    }
    pub fn state(&self, now: u64) -> Health {
        match self.opened_at {
            None => Health::Closed,
            Some(t) if now < t.saturating_add(self.cooldown_s) => Health::Open,
            Some(_) => Health::HalfOpen,
        }
    }
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.opened_at = None;
    }
    pub fn record_failure(&mut self, now: u64) {
        self.consecutive_failures += 1;
        if self.consecutive_failures >= self.threshold {
            self.opened_at = Some(now); // also re-opens a half-open breaker whose trial failed
        }
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(3, 60)
    }
}

/// Does this error say something about the provider's health? (A too-long prompt or an exhausted budget is about the request, not the provider.)
pub fn counts_against_health(e: &ProviderError) -> bool {
    !matches!(
        e,
        ProviderError::ContextOverflow | ProviderError::BudgetExceeded
    )
}

// ------------------------------------------------------------------------------------------------- quality

/// Pass counts for one provider and model. `passes` counts requests that were answered (the planner has no verifier; a caller that has one
/// can record its own verdicts with [`Scorecard::record`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scorecard {
    pub n: u32,
    pub passes: u32,
}

impl Scorecard {
    pub fn record(&mut self, pass: bool) {
        self.n += 1;
        self.passes += u32::from(pass);
    }
    /// The lower bound of the 95% Wilson interval for the pass rate: what the data supports at least, not the optimistic average.
    pub fn lower_bound(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let (n, p, z) = (
            f64::from(self.n),
            f64::from(self.passes) / f64::from(self.n),
            1.96f64,
        );
        let centre = p + z * z / (2.0 * n);
        let margin = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt();
        ((centre - margin) / (1.0 + z * z / n)).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualitySource {
    /// From enough samples.
    Measured,
    /// The declared prior (not enough samples yet).
    Prior,
}

// ------------------------------------------------------------------------------------------------- request and result

/// Capabilities the request needs from the model. A capability the card does not report counts as missing.
#[derive(Debug, Clone, Copy, Default)]
pub struct Needs {
    pub image_input: bool,
    pub structured_outputs: bool,
    pub tool_use: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preference {
    /// Device, then household, then third party; within a tier by quality, then cost. The default ("never cloud-first").
    #[default]
    LocalFirst,
    /// Best measured quality first, wherever it runs. An explicit, per-class override; egress still applies to every hop.
    BestScore,
}

pub struct PlanRequest<'a> {
    pub kind: RequestKind,
    pub label: &'a Label,
    pub purpose: Purpose,
    /// Estimated input size in tokens.
    pub input_tokens: u64,
    pub max_output_tokens: u64,
    pub needs: Needs,
    /// The caller said nothing may leave this device.
    pub local_only: bool,
    /// Can an owner be asked? When false, "ask" is deny.
    pub interactive: bool,
    /// Regions content may be processed in (`None` = no residency requirement).
    pub allowed_regions: Option<Vec<String>>,
    /// What is left to spend, in micro-dollars (`None` = no budget constraint).
    pub budget_available: Option<u64>,
    pub preference: Preference,
    /// Unix seconds, passed in.
    pub now: u64,
}

impl<'a> PlanRequest<'a> {
    pub fn new(kind: RequestKind, label: &'a Label, purpose: Purpose, now: u64) -> Self {
        Self {
            kind,
            label,
            purpose,
            input_tokens: 0,
            max_output_tokens: 0,
            needs: Needs::default(),
            local_only: false,
            interactive: false,
            allowed_regions: None,
            budget_available: None,
            preference: Preference::LocalFirst,
            now,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Kind,
    Features,
    LocalOnly,
    Egress,
    Context,
    Residency,
    Health,
    Budget,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub provider_id: String,
    pub model_id: String,
    pub filter: Filter,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hop {
    pub provider_id: String,
    pub model_id: String,
    pub locality: Locality,
    pub quality: f64,
    pub quality_source: QualitySource,
    pub est_cost_micro: Option<u64>,
    /// The egress policy needs the owner's answer first; [`ProviderSet::run`] does not call these.
    pub needs_consent: bool,
    /// A half-open breaker: a single trial, ordered after healthy hops.
    pub probation: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub hops: Vec<Hop>,
    pub rejected: Vec<Rejection>,
}

impl Plan {
    /// The hops that can run without asking anyone.
    pub fn runnable(&self) -> impl Iterator<Item = &Hop> {
        self.hops.iter().filter(|h| !h.needs_consent)
    }
}

// ------------------------------------------------------------------------------------------------- the set

/// One registered provider and what the planner knows about it.
pub struct Entry {
    pub adapter: Arc<dyn ProviderAdapter>,
    pub card: ModelCard,
    /// Declared quality in `[0, 1]` for requests of this kind, used until the scorecard has [`MIN_SAMPLES`].
    pub prior_quality: f64,
    /// Prices for the cost estimate (`None` = not priced, treated as free, which is right for a local server).
    pub pricing: Option<Pricing>,
    /// Regions the provider processes in (empty = unknown).
    pub regions: Vec<String>,
    pub breaker: Mutex<CircuitBreaker>,
    pub scorecard: Mutex<Scorecard>,
}

impl Entry {
    pub fn new(adapter: Arc<dyn ProviderAdapter>, card: ModelCard) -> Self {
        Self {
            adapter,
            card,
            prior_quality: 0.5,
            pricing: None,
            regions: Vec::new(),
            breaker: Mutex::new(CircuitBreaker::default()),
            scorecard: Mutex::new(Scorecard::default()),
        }
    }
    pub fn with_prior(mut self, q: f64) -> Self {
        self.prior_quality = q.clamp(0.0, 1.0);
        self
    }
    pub fn with_pricing(mut self, p: Pricing) -> Self {
        self.pricing = Some(p);
        self
    }
    pub fn with_regions(mut self, r: &[&str]) -> Self {
        self.regions = r.iter().map(|s| (*s).to_string()).collect();
        self
    }
}

#[derive(Default)]
pub struct ProviderSet {
    entries: Vec<Entry>,
}

impl ProviderSet {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add(&mut self, e: Entry) {
        self.entries.push(e);
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn rank(l: Locality) -> u8 {
        match l {
            Locality::Device => 0,
            Locality::Home => 1,
            Locality::Saas => 2,
            Locality::Unspecified => 3,
        }
    }

    /// Decide which providers may serve `pr`, and in what order. Pure: the same inputs always give the same plan.
    pub fn plan(&self, pr: &PlanRequest<'_>, policy: &Policy) -> Plan {
        let mut rejected = Vec::new();
        let mut survivors: Vec<(&Entry, Hop)> = Vec::new();

        for e in &self.entries {
            let info = e.adapter.info();
            let (pid, mid) = (info.provider_id.clone(), e.card.model_id.clone());
            let mut reject = |filter: Filter, reason: String| {
                rejected.push(Rejection {
                    provider_id: pid.clone(),
                    model_id: mid.clone(),
                    filter,
                    reason,
                })
            };

            // 1. kind and features
            if !info.kinds.contains(&pr.kind) || !e.card.kinds.contains(&pr.kind) {
                reject(Filter::Kind, format!("does not serve {}", pr.kind.as_str()));
                continue;
            }
            let missing: Vec<&str> = [
                (pr.needs.image_input && !e.card.image_input.is_yes()).then_some("image input"),
                (pr.needs.structured_outputs && !e.card.structured_outputs.is_yes())
                    .then_some("structured output"),
                (pr.needs.tool_use && !e.card.tool_use.is_yes()).then_some("tool use"),
            ]
            .into_iter()
            .flatten()
            .collect();
            if !missing.is_empty() {
                reject(
                    Filter::Features,
                    format!("lacks (or does not report) {}", missing.join(", ")),
                );
                continue;
            }
            // 2. local only
            if pr.local_only && info.locality.leaves_home() {
                reject(
                    Filter::LocalOnly,
                    "the request must stay on this device".into(),
                );
                continue;
            }
            // 3. egress: BEFORE anything else about the provider is looked at
            let mut needs_consent = false;
            match policy.decide(pr.label, info, pr.purpose, pr.kind, pr.now) {
                Verdict::Allow { .. } => {}
                Verdict::Deny { reason, .. } => {
                    reject(Filter::Egress, format!("egress denied ({reason:?})"));
                    continue;
                }
                Verdict::AskUser { uncovered_pairs } => {
                    if pr.interactive {
                        needs_consent = true;
                    } else {
                        reject(Filter::Egress, format!("no rule covers {uncovered_pairs} (class, subject) pair(s) and nobody can be asked"));
                        continue;
                    }
                }
            }
            // 4. context fit
            let limit = e.card.max_input_tokens.unwrap_or(UNKNOWN_CONTEXT_ASSUMED);
            if pr.input_tokens > limit {
                let basis = if e.card.max_input_tokens.is_some() {
                    "its limit"
                } else {
                    "the assumed limit for an unknown model"
                };
                reject(
                    Filter::Context,
                    format!("{} input tokens exceed {basis} ({limit})", pr.input_tokens),
                );
                continue;
            }
            // 5. residency (only matters once content leaves the home)
            if let Some(allowed) = &pr.allowed_regions {
                if info.locality.leaves_home()
                    && !(!e.regions.is_empty() && e.regions.iter().all(|r| allowed.contains(r)))
                {
                    reject(
                        Filter::Residency,
                        "its region is unknown or not allowed".into(),
                    );
                    continue;
                }
            }
            // 6. health
            let health = e.breaker.lock().map_or(Health::Open, |b| b.state(pr.now));
            if health == Health::Open {
                reject(Filter::Health, "its circuit breaker is open".into());
                continue;
            }
            // 7. budget
            let est = e
                .pricing
                .map(|p| p.worst_case(pr.input_tokens, pr.max_output_tokens));
            if let (Some(cost), Some(left)) = (est, pr.budget_available) {
                if cost > left {
                    reject(
                        Filter::Budget,
                        format!("worst case {cost} exceeds the {left} left"),
                    );
                    continue;
                }
            }

            let card = e.scorecard.lock().map(|s| *s).unwrap_or_default();
            let (quality, quality_source) = if card.n >= MIN_SAMPLES {
                (card.lower_bound(), QualitySource::Measured)
            } else {
                (e.prior_quality, QualitySource::Prior)
            };
            survivors.push((
                e,
                Hop {
                    provider_id: pid,
                    model_id: mid,
                    locality: info.locality,
                    quality,
                    quality_source,
                    est_cost_micro: est,
                    needs_consent,
                    probation: health == Health::HalfOpen,
                },
            ));
        }

        survivors.sort_by(|(_, a), (_, b)| {
            // hops that need no consent and are healthy come first
            (a.needs_consent, a.probation)
                .cmp(&(b.needs_consent, b.probation))
                .then_with(|| match pr.preference {
                    Preference::LocalFirst => Self::rank(a.locality).cmp(&Self::rank(b.locality)),
                    Preference::BestScore => Ordering::Equal,
                })
                .then_with(|| b.quality.total_cmp(&a.quality))
                .then_with(|| {
                    a.est_cost_micro
                        .unwrap_or(0)
                        .cmp(&b.est_cost_micro.unwrap_or(0))
                })
                .then_with(|| a.provider_id.cmp(&b.provider_id))
                .then_with(|| a.model_id.cmp(&b.model_id))
        });
        rejected.sort_by(|a, b| (&a.provider_id, &a.model_id).cmp(&(&b.provider_id, &b.model_id)));
        Plan {
            hops: survivors.into_iter().map(|(_, h)| h).collect(),
            rejected,
        }
    }

    fn find(&self, h: &Hop) -> Option<&Entry> {
        self.entries.iter().find(|e| {
            e.adapter.info().provider_id == h.provider_id && e.card.model_id == h.model_id
        })
    }

    /// Plan, run the hops that need no consent through [`run_chain_recorded`], and learn from what happened.
    pub async fn run(
        &self,
        pr: &PlanRequest<'_>,
        req: &Request,
        policy: &Policy,
        sink: &dyn AuditSink,
    ) -> (Result<ProviderReply, AiKitError>, EgressRecord, Plan) {
        let plan = self.plan(pr, policy);
        let chain: Vec<Arc<dyn ProviderAdapter>> = plan
            .runnable()
            .filter_map(|h| self.find(h))
            .map(|e| e.adapter.clone())
            .collect();
        let mut cfg = ChainConfig::new(pr.purpose, pr.now);
        cfg.interactive = false;
        let (result, record) = run_chain_recorded(&chain, policy, pr.label, req, &cfg, sink).await;
        for a in &record.attempts {
            let Some(e) = self
                .entries
                .iter()
                .find(|e| e.adapter.info().provider_id == a.provider_id)
            else {
                continue;
            };
            match a.outcome {
                Outcome::Sent => {
                    if let Ok(mut b) = e.breaker.lock() {
                        b.record_success();
                    }
                    if let Ok(mut s) = e.scorecard.lock() {
                        s.record(true);
                    }
                }
                Outcome::ProviderError | Outcome::Timeout => {
                    // only errors that say something about the provider's health count against it
                    let about_provider = a
                        .error_class
                        .as_deref()
                        .is_none_or(|c| c != "context_overflow" && c != "budget_exceeded");
                    if about_provider {
                        if let Ok(mut b) = e.breaker.lock() {
                            b.record_failure(pr.now);
                        }
                        if let Ok(mut s) = e.scorecard.lock() {
                            s.record(false);
                        }
                    }
                }
                Outcome::DeniedPolicy | Outcome::BlockedLabel => {}
            }
        }
        (result, record, plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::egress::{AgeBand, ProviderMatch, Rule, SourceKind, Subject, VecSink};
    use crate::model_card::{Cap, CardSource};
    use crate::provider::{FinishReason, ScriptedProvider};
    use crate::types::InferenceParams;

    const TEXT: RequestKind = RequestKind::TextGeneration;

    fn card(provider: &str, model: &str, max_in: Option<u64>) -> ModelCard {
        ModelCard {
            provider_id: provider.into(),
            model_id: model.into(),
            kinds: vec![TEXT],
            max_input_tokens: max_in,
            max_output_tokens: None,
            retires_at: None,
            image_input: Cap::Unknown,
            structured_outputs: Cap::Unknown,
            tool_use: Cap::Unknown,
            source: CardSource::Declared,
        }
    }
    fn entry(id: &str, loc: Locality, max_in: Option<u64>) -> (Arc<ScriptedProvider>, Entry) {
        let p = ScriptedProvider::new(id, loc, &[TEXT]);
        let e = Entry::new(p.clone(), card(id, &format!("{id}-model"), max_in));
        (p, e)
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
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[])],
        }
    }
    fn pr<'a>(l: &'a Label) -> PlanRequest<'a> {
        let mut p = PlanRequest::new(TEXT, l, Purpose::Answer, 1000);
        p.input_tokens = 100;
        p.max_output_tokens = 100;
        p
    }
    fn ids(p: &Plan) -> Vec<&str> {
        p.hops.iter().map(|h| h.provider_id.as_str()).collect()
    }
    fn req() -> Request {
        Request::Text(InferenceParams {
            model_id: "m".into(),
            prompt: "hi".into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 8,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        })
    }

    #[test]
    fn local_first_is_the_default_order_whatever_the_quality() {
        let mut set = ProviderSet::new();
        let (_, saas) = entry("claude", Locality::Saas, Some(200_000));
        let (_, home) = entry("studio", Locality::Home, Some(32_000));
        let (_, dev) = entry("ollama", Locality::Device, Some(32_000));
        set.add(saas.with_prior(0.99));
        set.add(home.with_prior(0.5));
        set.add(dev.with_prior(0.1));
        let l = label();
        let plan = set.plan(&pr(&l), &allow_all());
        assert_eq!(ids(&plan), vec!["ollama", "studio", "claude"], "device, then household, then third party, even though the cloud model has the best prior");
        // the explicit override ranks by quality instead
        let mut p = pr(&l);
        p.preference = Preference::BestScore;
        assert_eq!(
            ids(&set.plan(&p, &allow_all())),
            vec!["claude", "studio", "ollama"]
        );
    }

    #[test]
    fn every_exclusion_has_a_named_filter_and_a_reason() {
        let mut set = ProviderSet::new();
        let embed_only = ScriptedProvider::new("voyage", Locality::Saas, &[RequestKind::Embedding]);
        set.add(Entry::new(
            embed_only,
            ModelCard {
                kinds: vec![RequestKind::Embedding],
                ..card("voyage", "v", Some(8000))
            },
        ));
        let (_, tiny) = entry("tiny", Locality::Device, Some(50));
        set.add(tiny);
        let (_, unknown) = entry("unknown", Locality::Device, None);
        set.add(unknown);
        let (_, fine) = entry("fine", Locality::Device, Some(10_000));
        set.add(fine);
        let l = label();
        let mut p = pr(&l);
        p.input_tokens = 5000;
        let plan = set.plan(&p, &allow_all());
        assert_eq!(ids(&plan), vec!["fine"]);
        let why: Vec<(&str, Filter)> = plan
            .rejected
            .iter()
            .map(|r| (r.provider_id.as_str(), r.filter))
            .collect();
        assert_eq!(
            why,
            vec![
                ("tiny", Filter::Context),
                ("unknown", Filter::Context),
                ("voyage", Filter::Kind)
            ]
        );
        assert!(plan.rejected.iter().all(|r| !r.reason.is_empty()));
        assert!(
            plan.rejected[1]
                .reason
                .contains("assumed limit for an unknown model"),
            "{}",
            plan.rejected[1].reason
        );
    }

    #[test]
    fn a_denied_provider_is_rejected_by_egress_and_never_reaches_the_scoring() {
        let mut set = ProviderSet::new();
        let (_, saas) = entry("claude", Locality::Saas, Some(200_000));
        set.add(saas.with_prior(1.0));
        let (_, dev) = entry("ollama", Locality::Device, Some(8000));
        set.add(dev);
        let l = Label::new(
            SourceKind::HouseholdFile,
            &["health"],
            &[Subject::person("kid", AgeBand::Minor)],
        );
        let plan = set.plan(&pr(&l), &Policy::default());
        assert_eq!(ids(&plan), vec!["ollama"]);
        assert_eq!(plan.rejected.len(), 1);
        assert_eq!(plan.rejected[0].filter, Filter::Egress);
        // with the credentials class it is denied outright, not merely unasked
        let creds = Label::new(
            SourceKind::HouseholdFile,
            &["credentials"],
            &[Subject::household()],
        );
        let plan = set.plan(&pr(&creds), &allow_all());
        assert_eq!(ids(&plan), vec!["ollama"]);
        assert!(
            plan.rejected[0].reason.contains("CredentialsNeverLeave"),
            "{}",
            plan.rejected[0].reason
        );
    }

    #[test]
    fn asking_the_user_is_a_rejection_unless_the_caller_can_ask_and_then_it_goes_last() {
        let mut set = ProviderSet::new();
        let (_, saas) = entry("claude", Locality::Saas, Some(200_000));
        set.add(saas);
        let (_, home) = entry("studio", Locality::Home, Some(8000));
        set.add(home);
        let l = label();
        let none = Policy::default(); // no rule: ask
        assert_eq!(ids(&set.plan(&pr(&l), &none)), vec!["studio"]);
        let mut p = pr(&l);
        p.interactive = true;
        let plan = set.plan(&p, &none);
        assert_eq!(ids(&plan), vec!["studio", "claude"]);
        assert!(plan.hops[1].needs_consent && !plan.hops[0].needs_consent);
        assert_eq!(
            plan.runnable().count(),
            1,
            "a hop that needs consent is not runnable"
        );
    }

    #[test]
    fn local_only_residency_budget_and_features_each_filter() {
        let mut set = ProviderSet::new();
        let (_, a) = entry("us-cloud", Locality::Saas, Some(100_000));
        set.add(a.with_regions(&["US"]).with_pricing(Pricing {
            input_micro_per_mtok: 1_000_000,
            output_micro_per_mtok: 1_000_000,
            ..Pricing::default()
        }));
        let (_, b) = entry("eu-cloud", Locality::Saas, Some(100_000));
        set.add(b.with_regions(&["EU"]));
        let (_, c) = entry("anywhere", Locality::Saas, Some(100_000));
        set.add(c);
        let (_, d) = entry("ollama", Locality::Device, Some(100_000));
        set.add(d);
        let l = label();
        let mut p = pr(&l);
        p.local_only = true;
        assert_eq!(ids(&set.plan(&p, &allow_all())), vec!["ollama"]);
        let mut p = pr(&l);
        p.allowed_regions = Some(vec!["EU".into()]);
        let plan = set.plan(&p, &allow_all());
        assert_eq!(
            ids(&plan),
            vec!["ollama", "eu-cloud"],
            "an unknown region is not an allowed region; the local one needs no residency"
        );
        let mut p = pr(&l);
        p.input_tokens = 1_000_000;
        p.max_output_tokens = 1_000_000;
        p.budget_available = Some(500);
        // (context rejects first for the 100k models; use a smaller input for the budget check)
        p.input_tokens = 50_000;
        p.max_output_tokens = 50_000;
        let plan = set.plan(&p, &allow_all());
        assert!(
            plan.rejected
                .iter()
                .any(|r| r.provider_id == "us-cloud" && r.filter == Filter::Budget),
            "{:?}",
            plan.rejected
        );
        assert!(
            ids(&plan).contains(&"anywhere"),
            "an unpriced provider has no estimate to exceed"
        );
        let mut p = pr(&l);
        p.needs.image_input = true;
        let plan = set.plan(&p, &allow_all());
        assert!(
            plan.hops.is_empty() && plan.rejected.iter().all(|r| r.filter == Filter::Features),
            "unknown capability counts as missing: {:?}",
            plan.rejected
        );
    }

    #[test]
    fn measured_quality_overrides_the_prior_only_with_enough_samples_and_uses_the_lower_bound() {
        let mut set = ProviderSet::new();
        let (_, a) = entry("steady", Locality::Device, Some(8000));
        let (_, b) = entry("lucky", Locality::Device, Some(8000));
        set.add(a.with_prior(0.5));
        set.add(b.with_prior(0.9));
        // 3 of 3 passes is not enough evidence: the prior still decides
        if let Ok(mut s) = set.entries[0].scorecard.lock() {
            *s = Scorecard { n: 3, passes: 3 };
        }
        let l = label();
        let plan = set.plan(&pr(&l), &allow_all());
        assert_eq!(
            (ids(&plan), plan.hops[0].quality_source),
            (vec!["lucky", "steady"], QualitySource::Prior)
        );
        // 95 of 100 is measured, and its lower bound (about 0.89) is below the other's prior 0.9: the lower bound is what ranks
        if let Ok(mut s) = set.entries[0].scorecard.lock() {
            *s = Scorecard { n: 100, passes: 95 };
        }
        let plan = set.plan(&pr(&l), &allow_all());
        let steady = plan
            .hops
            .iter()
            .find(|h| h.provider_id == "steady")
            .unwrap();
        assert_eq!(steady.quality_source, QualitySource::Measured);
        assert!(
            steady.quality < 0.95 && steady.quality > 0.85,
            "{}",
            steady.quality
        );
        assert_eq!(ids(&plan)[0], "lucky");
        assert!(
            Scorecard {
                n: 100,
                passes: 100
            }
            .lower_bound()
                < 1.0
        );
        assert!(Scorecard { n: 0, passes: 0 }.lower_bound() == 0.0);
    }

    #[test]
    fn the_circuit_breaker_opens_half_opens_and_closes() {
        let mut b = CircuitBreaker::new(2, 60);
        assert_eq!(b.state(0), Health::Closed);
        b.record_failure(10);
        assert_eq!(b.state(10), Health::Closed, "below the threshold");
        b.record_failure(20);
        assert_eq!(b.state(30), Health::Open);
        assert_eq!(b.state(79), Health::Open, "still cooling down");
        assert_eq!(
            b.state(80),
            Health::HalfOpen,
            "the open window is t < opened_at + cooldown"
        );
        assert_eq!(
            b.state(81),
            Health::HalfOpen,
            "the cool-down has passed: one trial"
        );
        b.record_failure(90);
        assert_eq!(b.state(100), Health::Open, "a failed trial re-opens it");
        b.record_success();
        assert_eq!(b.state(100), Health::Closed);
        assert!(
            !counts_against_health(&ProviderError::ContextOverflow)
                && !counts_against_health(&ProviderError::BudgetExceeded)
        );
        assert!(counts_against_health(&ProviderError::Overloaded));
    }

    #[test]
    fn an_open_breaker_excludes_and_a_half_open_one_goes_last_on_probation() {
        let mut set = ProviderSet::new();
        let (_, a) = entry("down", Locality::Device, Some(8000));
        let (_, b) = entry("trial", Locality::Device, Some(8000));
        let (_, c) = entry("ok", Locality::Saas, Some(8000));
        set.add(a);
        set.add(b);
        set.add(c);
        *set.entries[0].breaker.lock().unwrap() = {
            let mut x = CircuitBreaker::new(1, 100);
            x.record_failure(950);
            x
        };
        *set.entries[1].breaker.lock().unwrap() = {
            let mut x = CircuitBreaker::new(1, 40);
            x.record_failure(950); // open until 990; now is 1000: half-open
            x
        };
        let l = label();
        let plan = set.plan(&pr(&l), &allow_all());
        assert_eq!(
            ids(&plan),
            vec!["ok", "trial"],
            "the probation hop is after the healthy cloud one, despite being local"
        );
        assert!(plan.hops[1].probation);
        assert_eq!(
            plan.rejected
                .iter()
                .map(|r| (r.provider_id.as_str(), r.filter))
                .collect::<Vec<_>>(),
            vec![("down", Filter::Health)]
        );
    }

    #[test]
    fn the_plan_does_not_depend_on_the_order_providers_were_registered_in() {
        let mk = |order: &[usize]| {
            let mut set = ProviderSet::new();
            let specs = [
                ("b-cloud", Locality::Saas),
                ("a-cloud", Locality::Saas),
                ("z-local", Locality::Device),
                ("m-home", Locality::Home),
            ];
            for &i in order {
                let (_, e) = entry(specs[i].0, specs[i].1, Some(8000));
                set.add(e);
            }
            set
        };
        let l = label();
        let first = mk(&[0, 1, 2, 3]).plan(&pr(&l), &allow_all());
        assert_eq!(
            ids(&first),
            vec!["z-local", "m-home", "a-cloud", "b-cloud"],
            "ties are broken by provider id"
        );
        for order in [[3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1], [0, 3, 1, 2]] {
            assert_eq!(
                mk(&order).plan(&pr(&l), &allow_all()),
                first,
                "order {order:?}"
            );
        }
    }

    #[test]
    fn property_a_provider_the_policy_denies_is_never_in_a_plan() {
        // small universe: label shapes x policy shapes x provider localities
        let classes = [
            vec!["notes"],
            vec!["photos", "notes"],
            vec!["credentials"],
            vec![],
        ];
        let subjects = [
            vec![Subject::household()],
            vec![Subject::person("kid", AgeBand::Minor)],
            vec![],
        ];
        let policies = [
            Policy::default(),
            allow_all(),
            Policy {
                version: 2,
                rules: vec![
                    Rule::allow("a", ProviderMatch::Any, &[]).including_minors(),
                    Rule::deny("d", ProviderMatch::Id("claude".into()), &["photos"]),
                ],
            },
        ];
        let mut set = ProviderSet::new();
        for (id, loc) in [
            ("claude", Locality::Saas),
            ("gemini", Locality::Saas),
            ("studio", Locality::Home),
            ("ollama", Locality::Device),
        ] {
            let (_, e) = entry(id, loc, Some(8000));
            set.add(e);
        }
        let mut with_remote = 0;
        for c in &classes {
            for s in &subjects {
                for pol in &policies {
                    let l = Label::new(SourceKind::HouseholdFile, c, s);
                    let plan = set.plan(&pr(&l), pol);
                    for h in &plan.hops {
                        let info = set.find(h).unwrap().adapter.info();
                        assert!(
                            matches!(
                                pol.decide(&l, info, Purpose::Answer, TEXT, 1000),
                                Verdict::Allow { .. }
                            ),
                            "{} planned but not allowed: {c:?} {s:?}",
                            h.provider_id
                        );
                        if h.locality.leaves_home() {
                            with_remote += 1;
                        }
                    }
                    // and every provider is accounted for: planned or rejected, never silently dropped
                    assert_eq!(plan.hops.len() + plan.rejected.len(), 4);
                }
            }
        }
        assert!(
            with_remote > 0,
            "the universe must include plans with a remote hop"
        );
    }

    #[tokio::test]
    async fn running_a_plan_never_calls_a_rejected_provider_and_learns_from_failures() {
        let mut set = ProviderSet::new();
        let down = ScriptedProvider::new("local-down", Locality::Device, &[TEXT]).then(vec![
            Err(ProviderError::Unavailable),
            Err(ProviderError::Unavailable),
            Err(ProviderError::Unavailable),
        ]);
        set.add(
            Entry::new(
                down.clone(),
                card("local-down", "local-down-model", Some(8000)),
            )
            .with_prior(0.9),
        );
        let cloud = ScriptedProvider::new("cloud", Locality::Saas, &[TEXT]);
        set.add(Entry::new(
            cloud.clone(),
            card("cloud", "cloud-model", Some(8000)),
        ));
        let denied = ScriptedProvider::new("denied", Locality::Saas, &[TEXT]);
        set.add(Entry::new(
            denied.clone(),
            card("denied", "denied-model", Some(8000)),
        ));
        let policy = Policy {
            version: 1,
            rules: vec![
                Rule::allow("cloud-only", ProviderMatch::Id("cloud".into()), &[]),
                Rule::deny("no-denied", ProviderMatch::Id("denied".into()), &[]),
            ],
        };
        let l = label();
        let mut p = pr(&l);
        p.now = 1000;
        let sink = VecSink::default();
        let (result, record, plan) = set.run(&p, &req(), &policy, &sink).await;
        assert_eq!(ids(&plan), vec!["local-down", "cloud"]);
        let reply = result.unwrap();
        assert_eq!(reply.finish, FinishReason::Stop);
        assert_eq!(
            (down.calls(), cloud.calls(), denied.calls()),
            (1, 1, 0),
            "the denied provider is never called"
        );
        assert_eq!(record.attempts.len(), 2);
        // the local provider failed once: its scorecard shows it, and its breaker (threshold 3) is still closed
        assert_eq!(
            *set.entries[0].scorecard.lock().unwrap(),
            Scorecard { n: 1, passes: 0 }
        );
        assert_eq!(set.entries[1].scorecard.lock().unwrap().passes, 1);
        // two more failures open the breaker; the next plan excludes it
        for _ in 0..2 {
            let _ = set.run(&p, &req(), &policy, &sink).await;
        }
        let plan = set.plan(&p, &policy);
        assert_eq!(ids(&plan), vec!["cloud"]);
        assert!(plan
            .rejected
            .iter()
            .any(|r| r.provider_id == "local-down" && r.filter == Filter::Health));
    }

    #[tokio::test]
    async fn a_failed_request_still_teaches_the_planner() {
        let mut set = ProviderSet::new();
        let only = ScriptedProvider::new("flaky", Locality::Device, &[TEXT]).then(vec![
            Err(ProviderError::Timeout),
            Err(ProviderError::ContextOverflow),
        ]);
        set.add(Entry::new(only, card("flaky", "m", Some(8000))));
        let l = label();
        let p = pr(&l);
        let sink = VecSink::default();
        let (r1, _, _) = set.run(&p, &req(), &Policy::default(), &sink).await;
        assert!(r1.is_err());
        assert_eq!(
            set.entries[0].scorecard.lock().unwrap().n,
            1,
            "a timeout is held against the provider"
        );
        let (r2, _, _) = set.run(&p, &req(), &Policy::default(), &sink).await;
        assert!(r2.is_err());
        assert_eq!(
            set.entries[0].scorecard.lock().unwrap().n,
            1,
            "a context overflow is about the request, not the provider"
        );
    }
}
