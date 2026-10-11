//! The egress policy engine: labels, rules, the verdict, and the audit record. **Prototype, no network, no storage.**
//!
//! This is the library half of "what may leave the house" (DJ, 2026-10-11): household resources are managed by the house owners
//! and per-user resources by the user, for both the data that may be shared with a remote provider and the credentials used for it.
//! The kit does **not** store rules, assign labels or show UI (the gateway / Courier / Score Kit do). It receives a label (set where
//! the content entered, never by a model) and a [`Policy`] snapshot and answers one question: *may this content go to this provider
//! for this purpose?* The answer is evaluated **fresh at every hop** by [`crate::provider::run_chain`].
//!
//! # Rules of the engine (each is tested, see the property test at the bottom)
//!
//! - **Default deny.** A remote ([`Locality::Saas`]) provider receives content only if an allow rule covers **every**
//!   `(class, subject)` pair of the label. A device-local or home-local provider needs no rule (nothing leaves the home).
//! - **Deny overrides.** Any matching deny beats any allow, regardless of specificity.
//! - **Unlabelled is private.** A label with no classes or no subjects, or an unspecified provider locality, is treated as the
//!   most restrictive case: nothing remote.
//! - **`credentials` never leaves the home**, whatever the rules say.
//! - **Minors.** A subject whose age band is `minor` or `unknown` is covered only by an allow rule that sets `include_minors`.
//!   (A deny matches them regardless.)
//! - **Combining labels** (a multi-turn context, a tool result) is a union of classes and subjects, never a collapse to one value.
//! - **A non-interactive caller treats "ask the user" as deny.**
//!
//! Not in this prototype (see `wiki/coordination/2026-10-11-egress-and-multi-provider-design-memos.md`): transforms (redaction,
//! local-summary-first), scopes (`once`/`session`), derived-from hashes, retention classes, registry-computed sensitivity.

use crate::modality::RequestKind;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Mutex;

/// The data class that can never leave the home.
pub const CREDENTIALS_CLASS: &str = "credentials";

/// Where a provider runs. Zero value (`Unspecified`) is treated as the strictest, [`Locality::Saas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    #[default]
    Unspecified,
    /// This machine.
    Device,
    /// Another machine in the same household.
    Home,
    /// A third party.
    Saas,
}

impl Locality {
    /// Does content sent here leave the home? (`Unspecified` counts as yes.)
    pub fn leaves_home(self) -> bool {
        !matches!(self, Locality::Device | Locality::Home)
    }
}

/// Age band of a subject. `Unknown` (the zero value) counts as a minor.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum AgeBand {
    #[default]
    Unknown,
    Minor,
    Adult,
}

/// Whose data this is: the household, or one person.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubjectScope {
    Household,
    Person(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Subject {
    pub scope: SubjectScope,
    pub age: AgeBand,
}

impl Subject {
    pub fn household() -> Self {
        Self {
            scope: SubjectScope::Household,
            age: AgeBand::Adult,
        }
    }
    pub fn person(id: &str, age: AgeBand) -> Self {
        Self {
            scope: SubjectScope::Person(id.to_string()),
            age,
        }
    }
    fn is_adult(&self) -> bool {
        self.age == AgeBand::Adult
    }
}

/// Where the content came from (set by the adapter that read it, never by a model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    #[default]
    Unspecified,
    UserTyped,
    HouseholdService,
    HouseholdFile,
    Vault,
    ToolResult,
    Generated,
    /// Several sources were combined.
    Mixed,
}

/// What a piece of content is and whose it is. A set, never one value.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Label {
    pub source: SourceKind,
    /// Data-class nouns (`location`, `photos`, `health`, `credentials`, ...).
    pub classes: BTreeSet<String>,
    pub subjects: BTreeSet<Subject>,
}

impl Label {
    pub fn new(source: SourceKind, classes: &[&str], subjects: &[Subject]) -> Self {
        Self {
            source,
            classes: classes.iter().map(|c| (*c).to_string()).collect(),
            subjects: subjects.iter().cloned().collect(),
        }
    }

    /// A label that says nothing: treated as private everywhere.
    pub fn unlabelled() -> Self {
        Self::default()
    }

    /// Union of several labels (a multi-turn context, a tool result plus a prompt). Never lowers anything.
    pub fn combine<'a>(labels: impl IntoIterator<Item = &'a Label>) -> Label {
        let mut out = Label::default();
        let mut first = true;
        for l in labels {
            out.source = if first || out.source == l.source {
                l.source
            } else {
                SourceKind::Mixed
            };
            first = false;
            out.classes.extend(l.classes.iter().cloned());
            out.subjects.extend(l.subjects.iter().cloned());
        }
        out
    }

    /// Nothing said about the class or the owner: the most restrictive reading applies.
    pub fn is_unlabelled(&self) -> bool {
        self.classes.is_empty() || self.subjects.is_empty()
    }
}

/// Why content is being sent. Closed: a free-text purpose would match no rule. Zero value is deny.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    #[default]
    Unspecified,
    Answer,
    Summarize,
    GenerateImage,
    Transcribe,
    IndexSearch,
    CodeAssist,
    Plan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Allow,
    Deny,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderMatch {
    Any,
    Id(String),
}

/// Which (class, subject) pairs a rule is about. Empty `classes` = any class; `None` subjects = any subject.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Selector {
    pub classes: BTreeSet<String>,
    pub subjects: Option<BTreeSet<SubjectScope>>,
    /// An **allow** covers a minor or unknown-age subject only if this is set. (A deny matches regardless.)
    pub include_minors: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rule {
    pub rule_id: String,
    pub version: u32,
    pub selector: Selector,
    pub provider: ProviderMatch,
    /// Empty = any purpose.
    pub purposes: BTreeSet<Purpose>,
    /// Empty = any request kind.
    pub request_kinds: Vec<RequestKind>,
    pub effect: Effect,
    /// Unix seconds; the rule is inactive at or after this time.
    pub expires_at: Option<u64>,
    pub revoked: bool,
}

impl Rule {
    /// An allow rule with the given id, covering `classes` (empty = any) for this provider.
    pub fn allow(rule_id: &str, provider: ProviderMatch, classes: &[&str]) -> Self {
        Self::new(rule_id, Effect::Allow, provider, classes)
    }
    pub fn deny(rule_id: &str, provider: ProviderMatch, classes: &[&str]) -> Self {
        Self::new(rule_id, Effect::Deny, provider, classes)
    }
    fn new(rule_id: &str, effect: Effect, provider: ProviderMatch, classes: &[&str]) -> Self {
        Self {
            rule_id: rule_id.to_string(),
            version: 1,
            selector: Selector {
                classes: classes.iter().map(|c| (*c).to_string()).collect(),
                ..Selector::default()
            },
            provider,
            purposes: BTreeSet::new(),
            request_kinds: Vec::new(),
            effect,
            expires_at: None,
            revoked: false,
        }
    }
    pub fn including_minors(mut self) -> Self {
        self.selector.include_minors = true;
        self
    }
    pub fn for_subjects(mut self, subjects: &[SubjectScope]) -> Self {
        self.selector.subjects = Some(subjects.iter().cloned().collect());
        self
    }

    fn active(&self, now: u64) -> bool {
        !self.revoked && self.expires_at.is_none_or(|t| now < t)
    }

    fn matches_context(&self, provider_id: &str, purpose: Purpose, kind: RequestKind) -> bool {
        (match &self.provider {
            ProviderMatch::Any => true,
            ProviderMatch::Id(id) => id == provider_id,
        }) && (self.purposes.is_empty() || self.purposes.contains(&purpose))
            && (self.request_kinds.is_empty() || self.request_kinds.contains(&kind))
    }

    fn matches_pair(&self, class: &str, subject: &Subject) -> bool {
        let class_ok = self.selector.classes.is_empty() || self.selector.classes.contains(class);
        let subject_ok = self
            .selector
            .subjects
            .as_ref()
            .is_none_or(|s| s.contains(&subject.scope));
        // minors gate only allows; a deny always applies
        let minor_ok =
            self.effect == Effect::Deny || subject.is_adult() || self.selector.include_minors;
        class_ok && subject_ok && minor_ok
    }
}

/// A policy snapshot: the rules in force and their version. The caller assembles it (household rules plus the person's rules).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Policy {
    pub version: u64,
    pub rules: Vec<Rule>,
}

/// What the engine needs to know about a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderInfo {
    pub provider_id: String,
    pub locality: Locality,
    pub kinds: Vec<RequestKind>,
}

/// Why a request may not go to a provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenyReason {
    /// An explicit deny rule matched.
    RuleDenies,
    /// The label says nothing about class or owner (or the purpose is unspecified).
    Unlabelled,
    /// The content includes the `credentials` class.
    CredentialsNeverLeave,
}

/// The engine's answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Go ahead. `rules` are the (rule id, version) pairs that allowed it (empty for a provider that is not remote).
    Allow { rules: Vec<(String, u32)> },
    Deny {
        reason: DenyReason,
        rules: Vec<(String, u32)>,
    },
    /// No rule covers some pair; an interactive caller may ask the owner. A non-interactive caller treats this as deny.
    AskUser { uncovered_pairs: usize },
}

impl Policy {
    /// May `label` go to `provider` for `purpose`/`kind` at time `now` (unix seconds)?
    pub fn decide(
        &self,
        label: &Label,
        provider: &ProviderInfo,
        purpose: Purpose,
        kind: RequestKind,
        now: u64,
    ) -> Verdict {
        if !provider.locality.leaves_home() {
            return Verdict::Allow { rules: Vec::new() };
        }
        if label.classes.contains(CREDENTIALS_CLASS) {
            return Verdict::Deny {
                reason: DenyReason::CredentialsNeverLeave,
                rules: Vec::new(),
            };
        }
        if label.is_unlabelled() || purpose == Purpose::Unspecified {
            return Verdict::Deny {
                reason: DenyReason::Unlabelled,
                rules: Vec::new(),
            };
        }
        let live: Vec<&Rule> = self
            .rules
            .iter()
            .filter(|r| r.active(now) && r.matches_context(&provider.provider_id, purpose, kind))
            .collect();
        let mut used: Vec<(String, u32)> = Vec::new();
        let mut uncovered = 0usize;
        for class in &label.classes {
            for subject in &label.subjects {
                let matching: Vec<&&Rule> = live
                    .iter()
                    .filter(|r| r.matches_pair(class, subject))
                    .collect();
                if let Some(d) = matching.iter().find(|r| r.effect == Effect::Deny) {
                    return Verdict::Deny {
                        reason: DenyReason::RuleDenies,
                        rules: vec![(d.rule_id.clone(), d.version)],
                    };
                }
                match matching.iter().find(|r| r.effect == Effect::Allow) {
                    Some(a) => {
                        let id = (a.rule_id.clone(), a.version);
                        if !used.contains(&id) {
                            used.push(id);
                        }
                    }
                    None => uncovered += 1,
                }
            }
        }
        if uncovered > 0 {
            Verdict::AskUser {
                uncovered_pairs: uncovered,
            }
        } else {
            Verdict::Allow { rules: used }
        }
    }
}

/// What happened at one hop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Sent,
    DeniedPolicy,
    /// Blocked by the label (unlabelled, credentials) or because no rule covers the content.
    BlockedLabel,
    ProviderError,
    Timeout,
}

/// One hop. **No content, no prompt, no response text.**
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attempt {
    pub n: u32,
    pub provider_id: String,
    pub locality: Locality,
    pub model_id: String,
    pub outcome: Outcome,
    /// The provider error class, if any (never the upstream body).
    pub error_class: Option<String>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub rules: Vec<(String, u32)>,
}

/// The audit record for one request. Never contains content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EgressRecord {
    pub record_id: String,
    pub request_kind: RequestKind,
    pub purpose: Purpose,
    pub classes: BTreeSet<String>,
    pub subjects: BTreeSet<SubjectScope>,
    pub policy_version: u64,
    pub attempts: Vec<Attempt>,
}

impl EgressRecord {
    /// A request that never involved a remote provider and was never denied needs no per-request record.
    pub fn is_worth_keeping(&self) -> bool {
        self.attempts.iter().any(|a| {
            a.locality.leaves_home()
                || matches!(a.outcome, Outcome::DeniedPolicy | Outcome::BlockedLabel)
        })
    }
}

/// Where audit records go. The caller supplies it (append-only storage is not the kit's job).
pub trait AuditSink: Send + Sync {
    fn record(&self, record: EgressRecord);
}

/// An in-memory sink, for tests and the CLI.
#[derive(Default)]
pub struct VecSink(pub Mutex<Vec<EgressRecord>>);

impl AuditSink for VecSink {
    fn record(&self, record: EgressRecord) {
        if let Ok(mut v) = self.0.lock() {
            v.push(record);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn saas(id: &str) -> ProviderInfo {
        ProviderInfo {
            provider_id: id.into(),
            locality: Locality::Saas,
            kinds: vec![RequestKind::TextGeneration],
        }
    }
    fn alice() -> Subject {
        Subject::person("alice", AgeBand::Adult)
    }
    fn kid() -> Subject {
        Subject::person("kid", AgeBand::Minor)
    }
    fn lbl(classes: &[&str], subjects: &[Subject]) -> Label {
        Label::new(SourceKind::HouseholdFile, classes, subjects)
    }
    fn decide(p: &Policy, l: &Label, prov: &ProviderInfo) -> Verdict {
        p.decide(l, prov, Purpose::Answer, RequestKind::TextGeneration, 100)
    }
    fn allowed(v: &Verdict) -> bool {
        matches!(v, Verdict::Allow { .. })
    }

    #[test]
    fn default_is_deny_for_a_remote_provider_and_allow_for_a_local_one() {
        let l = lbl(&["notes"], &[alice()]);
        let none = Policy::default();
        assert!(matches!(
            decide(&none, &l, &saas("claude")),
            Verdict::AskUser { .. }
        ));
        for loc in [Locality::Device, Locality::Home] {
            let p = ProviderInfo {
                locality: loc,
                ..saas("ollama")
            };
            assert!(allowed(&decide(&none, &l, &p)), "{loc:?} needs no rule");
        }
        let unspecified = ProviderInfo {
            locality: Locality::Unspecified,
            ..saas("x")
        };
        assert!(
            !allowed(&decide(&none, &l, &unspecified)),
            "unknown locality counts as remote"
        );
    }

    #[test]
    fn an_allow_must_cover_every_class_and_subject_pair() {
        let policy = Policy {
            version: 1,
            rules: vec![Rule::allow(
                "r1",
                ProviderMatch::Id("claude".into()),
                &["notes"],
            )],
        };
        assert!(allowed(&decide(
            &policy,
            &lbl(&["notes"], &[alice()]),
            &saas("claude")
        )));
        // a second class is not covered
        assert!(matches!(
            decide(
                &policy,
                &lbl(&["notes", "photos"], &[alice()]),
                &saas("claude")
            ),
            Verdict::AskUser { uncovered_pairs: 1 }
        ));
        // another provider is not covered
        assert!(!allowed(&decide(
            &policy,
            &lbl(&["notes"], &[alice()]),
            &saas("gemini")
        )));
    }

    #[test]
    fn any_matching_deny_beats_any_allow() {
        let policy = Policy {
            version: 1,
            rules: vec![
                Rule::allow("all", ProviderMatch::Any, &[]),
                Rule::deny("no-health", ProviderMatch::Any, &["health"]),
            ],
        };
        assert!(allowed(&decide(
            &policy,
            &lbl(&["notes"], &[alice()]),
            &saas("claude")
        )));
        assert!(matches!(
            decide(
                &policy,
                &lbl(&["notes", "health"], &[alice()]),
                &saas("claude")
            ),
            Verdict::Deny {
                reason: DenyReason::RuleDenies,
                ..
            }
        ));
    }

    #[test]
    fn credentials_never_leave_even_with_an_allow_everything_rule() {
        let policy = Policy {
            version: 1,
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[]).including_minors()],
        };
        assert!(matches!(
            decide(&policy, &lbl(&["credentials"], &[alice()]), &saas("claude")),
            Verdict::Deny {
                reason: DenyReason::CredentialsNeverLeave,
                ..
            }
        ));
    }

    #[test]
    fn minors_and_unknown_ages_need_an_allow_that_includes_them() {
        let plain = Policy {
            version: 1,
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[])],
        };
        let with_minors = Policy {
            version: 1,
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[]).including_minors()],
        };
        let unknown = Subject::person("x", AgeBand::Unknown);
        for s in [kid(), unknown] {
            let l = lbl(&["notes"], &[s]);
            assert!(!allowed(&decide(&plain, &l, &saas("claude"))));
            assert!(allowed(&decide(&with_minors, &l, &saas("claude"))));
        }
    }

    #[test]
    fn an_unlabelled_request_or_an_unspecified_purpose_goes_nowhere_remote() {
        let policy = Policy {
            version: 1,
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[]).including_minors()],
        };
        assert!(matches!(
            decide(&policy, &Label::unlabelled(), &saas("claude")),
            Verdict::Deny {
                reason: DenyReason::Unlabelled,
                ..
            }
        ));
        assert!(!allowed(&policy.decide(
            &lbl(&["notes"], &[alice()]),
            &saas("claude"),
            Purpose::Unspecified,
            RequestKind::TextGeneration,
            100
        )));
    }

    #[test]
    fn expired_and_revoked_rules_do_not_count() {
        let mut expiring = Rule::allow("r", ProviderMatch::Any, &[]);
        expiring.expires_at = Some(100);
        let mut revoked = Rule::allow("r2", ProviderMatch::Any, &[]);
        revoked.revoked = true;
        let p = Policy {
            version: 1,
            rules: vec![expiring, revoked],
        };
        let l = lbl(&["notes"], &[alice()]);
        assert!(!allowed(&p.decide(
            &l,
            &saas("c"),
            Purpose::Answer,
            RequestKind::TextGeneration,
            100
        )));
        assert!(
            allowed(&p.decide(
                &l,
                &saas("c"),
                Purpose::Answer,
                RequestKind::TextGeneration,
                99
            )),
            "the expiring rule still covers it before its expiry"
        );
    }

    #[test]
    fn combining_labels_is_a_union_and_never_lowers() {
        let a = lbl(&["notes"], &[alice()]);
        let b = Label::new(SourceKind::ToolResult, &["health"], &[kid()]);
        let c = Label::combine([&a, &b]);
        assert_eq!(c.classes.len(), 2);
        assert_eq!(c.subjects.len(), 2);
        assert_eq!(c.source, SourceKind::Mixed);
        assert!(
            Label::combine([]).is_unlabelled(),
            "nothing combined says nothing"
        );
    }

    /// Exhaustive property test over a small universe: for every label (subsets of 3 classes x subsets of 3 subjects) and every
    /// combination of 5 candidate rules, whether a remote provider may receive the content satisfies five independent invariants.
    /// This is the "no remote hop for private content" test: it does not mirror the engine's code, it states the invariants.
    #[test]
    fn property_a_remote_provider_gets_content_only_when_every_invariant_holds() {
        let class_names = ["notes", "photos", CREDENTIALS_CLASS];
        let subjects = [Subject::household(), alice(), kid()];
        let rule_pool = [
            Rule::allow("a-notes", ProviderMatch::Any, &["notes"]),
            Rule::allow(
                "a-photos-minors",
                ProviderMatch::Id("claude".into()),
                &["photos"],
            )
            .including_minors(),
            Rule::allow("a-all", ProviderMatch::Any, &[]),
            Rule::deny("d-photos", ProviderMatch::Any, &["photos"]),
            Rule::allow("a-credentials", ProviderMatch::Any, &[CREDENTIALS_CLASS])
                .including_minors(),
        ];
        let mut checked = 0u32;
        let mut sent = 0u32;
        for cmask in 0u8..8 {
            for smask in 0u8..8 {
                let label = Label {
                    source: SourceKind::HouseholdFile,
                    classes: (0..3)
                        .filter(|i| cmask >> i & 1 == 1)
                        .map(|i| class_names[i].to_string())
                        .collect(),
                    subjects: (0..3)
                        .filter(|i| smask >> i & 1 == 1)
                        .map(|i| subjects[i].clone())
                        .collect(),
                };
                for rmask in 0u8..32 {
                    let rules: Vec<Rule> = (0..5)
                        .filter(|i| rmask >> i & 1 == 1)
                        .map(|i| rule_pool[i].clone())
                        .collect();
                    let policy = Policy {
                        version: 1,
                        rules: rules.clone(),
                    };
                    let v = decide(&policy, &label, &saas("claude"));
                    checked += 1;
                    if !allowed(&v) {
                        continue;
                    }
                    sent += 1;
                    // I1: something said what the content is and whose it is
                    assert!(!label.classes.is_empty() && !label.subjects.is_empty());
                    // I2: credentials never leave
                    assert!(!label.classes.contains(CREDENTIALS_CLASS));
                    // I3: no matching deny exists for any pair
                    for c in &label.classes {
                        for s in &label.subjects {
                            assert!(
                                !rules.iter().any(|r| r.effect == Effect::Deny
                                    && r.matches_context(
                                        "claude",
                                        Purpose::Answer,
                                        RequestKind::TextGeneration
                                    )
                                    && r.matches_pair(c, s)),
                                "a deny matched but content was allowed: {label:?} {rmask}"
                            );
                            // I4: every pair is covered by an allow that applies to its age band
                            let covering = rules.iter().any(|r| {
                                r.effect == Effect::Allow
                                    && r.matches_context(
                                        "claude",
                                        Purpose::Answer,
                                        RequestKind::TextGeneration,
                                    )
                                    && (r.selector.classes.is_empty()
                                        || r.selector.classes.contains(c))
                                    && (s.age == AgeBand::Adult || r.selector.include_minors)
                            });
                            assert!(covering, "uncovered pair {c}/{s:?} was sent: {rmask}");
                        }
                    }
                }
                // I5: with no rules at all, nothing is ever sent
                assert!(!allowed(&decide(
                    &Policy::default(),
                    &label,
                    &saas("claude")
                )));
            }
        }
        assert_eq!(checked, 8 * 8 * 32);
        assert!(
            sent > 0,
            "the universe must contain allowed cases or the test proves nothing"
        );
    }
}
