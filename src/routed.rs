//! Hybrid routing for **any kind of request**: [`MultiHybridService`].
//!
//! [`crate::HybridInferenceService`] is the text-only façade (two [`InferenceService`]s). This is the general
//! one: each tier is a list of [`RequestService`] backends, a request can be text, image generation, image
//! understanding, embeddings or speech, and the [`Router`] picks a tier from *which tiers have the request kind*
//! before it looks at privacy, fit, policy and verification (the same rules as the text path, in the same order).
//!
//! What this adds on top of the text path:
//!
//! - **RequestKind negotiation.** A request no tier can serve is [`AiKitError::UnsupportedKind`] with the
//!   reason, never a wrong-kind-of-model answer. A request only the *cloud* can serve, with private content, is
//!   also unavailable: the router will not leak private data to get an answer.
//! - **Re-typing.** Free-form text such as "paint a cottage at dawn" is classified, and if it is really an image
//!   request and an image backend exists, it runs as one. The caller does not have to know which request kind it needs.
//! - **Verification for every modality** ([`OutputCheck`]): a real PNG of the right size that is not flat, embeddings
//!   that are finite and the right shape, a non-empty transcript. A failed check can escalate to a cloud tier that
//!   has the request kind, exactly as for text.
//! - **Egress accounting.** [`RouteStats::cloud_payload_bytes`] counts what was sent off the device, whatever its type.

use crate::error::{AiKitError, Result};
use crate::hybrid::{
    classify, estimate_tokens, Attempt, Decision, Plan, Privacy, RoutePolicy, RouteStats,
    RouteTrace, Router, TaskClass, Tier,
};
use crate::modality::{
    unsupported, ImageGenParams, Output, OutputCheck, Request, RequestKind, RequestService,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// One backend in a tier, with the model it uses for each request kind.
#[derive(Clone)]
pub struct TierBackend {
    pub service: Arc<dyn RequestService>,
    pub model_ids: HashMap<RequestKind, String>,
}

impl TierBackend {
    pub fn new(service: Arc<dyn RequestService>, models: &[(RequestKind, &str)]) -> Self {
        Self {
            service,
            model_ids: models.iter().map(|(c, m)| (*c, m.to_string())).collect(),
        }
    }
}

#[derive(Clone, Default)]
struct TierSet {
    backends: Vec<TierBackend>,
}

impl TierSet {
    fn find(&self, cap: RequestKind) -> Option<&TierBackend> {
        self.backends
            .iter()
            .find(|b| b.service.kinds().contains(&cap) && b.model_ids.contains_key(&cap))
    }
    fn has(&self, cap: RequestKind) -> bool {
        self.find(cap).is_some()
    }
    fn kinds(&self) -> HashSet<RequestKind> {
        self.backends
            .iter()
            .flat_map(|b| {
                b.service
                    .kinds()
                    .into_iter()
                    .filter(|c| b.model_ids.contains_key(c))
            })
            .collect()
    }
}

/// A request plus the hints that make routing possible.
#[derive(Debug, Clone)]
pub struct RoutedRequest {
    pub request: Request,
    /// Task class, if the caller knows it; otherwise derived (from the prompt for text, from the request kind otherwise).
    pub class: Option<TaskClass>,
    /// `Private` (the default) never reaches a cloud tier.
    pub privacy: Privacy,
    pub check: OutputCheck,
}

impl RoutedRequest {
    pub fn new(request: Request) -> Self {
        Self {
            request,
            class: None,
            privacy: Privacy::Private,
            check: OutputCheck::None,
        }
    }
    pub fn shareable(mut self) -> Self {
        self.privacy = Privacy::Shareable;
        self
    }
    pub fn with_check(mut self, check: OutputCheck) -> Self {
        self.check = check;
        self
    }
}

pub struct MultiHybridService {
    local: TierSet,
    cloud: TierSet,
    router: Router,
    stats: Mutex<RouteStats>,
    retype_text: bool,
}

impl MultiHybridService {
    pub fn new(local: Vec<TierBackend>, cloud: Vec<TierBackend>, policy: RoutePolicy) -> Self {
        Self {
            local: TierSet { backends: local },
            cloud: TierSet { backends: cloud },
            router: Router::new(policy),
            stats: Mutex::new(RouteStats::default()),
            retype_text: true,
        }
    }

    /// Turn off treating image-like text prompts as image requests (they are then plain text requests).
    pub fn without_retyping(mut self) -> Self {
        self.retype_text = false;
        self
    }

    pub fn route_stats(&self) -> RouteStats {
        self.stats.lock().map(|s| s.clone()).unwrap_or_default()
    }

    /// What each tier can do, for display and for callers that want to know before asking.
    pub fn kinds(&self) -> (Vec<RequestKind>, Vec<RequestKind>) {
        let f = |t: &TierSet| {
            let s = t.kinds();
            RequestKind::ALL
                .into_iter()
                .filter(|c| s.contains(c))
                .collect()
        };
        (f(&self.local), f(&self.cloud))
    }

    /// Re-type free-form text that is really an image request, if an image backend exists to run it.
    fn effective(&self, r: &RoutedRequest) -> (Request, TaskClass) {
        let class = r.class.unwrap_or_else(|| match &r.request {
            Request::Text(p) => classify(&p.prompt),
            Request::ImageGeneration(_) => TaskClass::ImageRaster,
            Request::ImageUnderstanding(_) => TaskClass::ImageUnderstanding,
            Request::Embedding(_) => TaskClass::Embedding,
            Request::SpeechToText(_) => TaskClass::SpeechToText,
        });
        if self.retype_text && class == TaskClass::ImageRaster {
            if let Request::Text(p) = &r.request {
                return (
                    Request::ImageGeneration(ImageGenParams::new(String::new(), p.prompt.clone())),
                    class,
                );
            }
        }
        (r.request.clone(), class)
    }

    /// Decide only; run nothing.
    pub fn explain(&self, r: &RoutedRequest) -> Decision {
        let (req, class) = self.effective(r);
        let cap = req.kind();
        self.router.decide_multi(
            cap,
            class,
            r.privacy,
            estimate_tokens(req.prompt_text().unwrap_or("")) + (req.payload_bytes() / 4) as u32,
            self.local.has(cap),
            self.cloud.has(cap),
            r.check.is_present(),
        )
    }

    async fn call(
        &self,
        tier: Tier,
        req: &Request,
        check: &OutputCheck,
    ) -> (Attempt, Option<Output>, Option<AiKitError>) {
        let cap = req.kind();
        let set = if tier == Tier::Local {
            &self.local
        } else {
            &self.cloud
        };
        let Some(backend) = set.find(cap) else {
            return (
                Attempt {
                    tier,
                    model_id: String::new(),
                    inference_ms: None,
                    prompt_tokens: None,
                    completion_tokens: None,
                    verified: None,
                    error: Some(format!("no {tier:?} backend for {}", cap.as_str())),
                },
                None,
                Some(unsupported(
                    cap,
                    format!("no {tier:?} backend for {}", cap.as_str()),
                )),
            );
        };
        let model = backend.model_ids[&cap].clone();
        let result = backend
            .service
            .handle(req.clone().with_model_id(&model))
            .await;
        if let Ok(mut s) = self.stats.lock() {
            match tier {
                Tier::Local => s.local_calls += 1,
                Tier::Cloud => {
                    s.cloud_calls += 1;
                    s.cloud_payload_bytes += req.payload_bytes() as u64;
                }
            }
        }
        match result {
            Ok(out) => {
                let (pt, ct) = match &out {
                    Output::Text(t) => (Some(t.prompt_tokens), Some(t.completion_tokens)),
                    _ => (None, None),
                };
                if tier == Tier::Cloud {
                    if let (Some(p), Some(c), Ok(mut s)) = (pt, ct, self.stats.lock()) {
                        s.cloud_prompt_tokens += u64::from(p);
                        s.cloud_completion_tokens += u64::from(c);
                    }
                }
                let verified = check.check(req, &out);
                (
                    Attempt {
                        tier,
                        model_id: model,
                        inference_ms: Some(out.inference_ms()),
                        prompt_tokens: pt,
                        completion_tokens: ct,
                        verified,
                        error: None,
                    },
                    Some(out),
                    None,
                )
            }
            Err(e) => (
                Attempt {
                    tier,
                    model_id: model,
                    inference_ms: None,
                    prompt_tokens: None,
                    completion_tokens: None,
                    verified: None,
                    error: Some(e.to_string()),
                },
                None,
                Some(e),
            ),
        }
    }

    fn note_blocked(&self, trace: &mut RouteTrace, note: &str) {
        trace.notes.push(note.into());
        if let Ok(mut s) = self.stats.lock() {
            s.blocked_by_privacy += 1;
        }
    }

    /// Route and run one request of any kind.
    pub async fn handle_routed(&self, r: RoutedRequest) -> Result<(Output, RouteTrace)> {
        if let Ok(mut s) = self.stats.lock() {
            s.requests += 1;
        }
        let (req, _class) = self.effective(&r);
        req.validate_for_route()?;
        let decision = self.explain(&r);
        let mut trace = RouteTrace {
            decision: decision.clone(),
            attempts: Vec::new(),
            final_tier: None,
            notes: Vec::new(),
        };
        let cap = req.kind();
        if req.kind() != r.request.kind() {
            trace.notes.push(format!(
                "treated a {} request as {}",
                r.request.kind().as_str(),
                cap.as_str()
            ));
        }
        let (first, escalate) = match decision.plan {
            Plan::Run { first, escalate } => (first, escalate),
            Plan::Delegate { ref request_kind } => {
                return Err(unsupported(cap, format!("delegate to '{request_kind}'")))
            }
            Plan::Unavailable {
                ref reason,
                needs_shareable,
                ..
            } => {
                return Err(if needs_shareable {
                    AiKitError::RequiresShareable {
                        request_kind: cap.as_str().to_string(),
                    }
                } else {
                    unsupported(cap, reason.clone())
                })
            }
        };
        let private = r.privacy == Privacy::Private;
        let cloud_ok = self.cloud.has(cap) && !private;

        let (a1, o1, e1) = self.call(first, &req, &r.check).await;
        let failed1 = o1.is_none() || a1.verified == Some(false);
        trace.attempts.push(a1);

        if let Some(out) = o1.as_ref() {
            if !failed1 || !(first == Tier::Local && escalate) {
                if failed1 && private && first == Tier::Local && r.check.is_present() {
                    self.note_blocked(
                        &mut trace,
                        "local answer failed verification; not retried in the cloud (privacy=private)",
                    );
                }
                trace.final_tier = Some(first);
                return Ok((out.clone(), trace));
            }
        }

        let may_escalate = first == Tier::Local && cloud_ok && (escalate || o1.is_none());
        if !may_escalate {
            return match o1 {
                Some(out) => {
                    trace.final_tier = Some(first);
                    Ok((out, trace))
                }
                None => {
                    if private && first == Tier::Local {
                        self.note_blocked(
                            &mut trace,
                            "local call failed; no cloud fallback (privacy=private)",
                        );
                    }
                    // Keep the typed error (an `InferenceOom` must stay one, so callers and the wire contract can tell it apart).
                    Err(e1.unwrap_or_else(|| AiKitError::Internal {
                        reason: "request failed".into(),
                    }))
                }
            };
        }

        if let Ok(mut s) = self.stats.lock() {
            s.escalations += 1;
        }
        trace.notes.push(if o1.is_none() {
            "local call failed; falling back to the cloud".into()
        } else {
            "local answer failed verification; escalating to the cloud".into()
        });
        let (a2, o2, _e2) = self.call(Tier::Cloud, &req, &r.check).await;
        let err2 = a2.error.clone();
        trace.attempts.push(a2);
        match (o2, o1) {
            (Some(out), _) => {
                trace.final_tier = Some(Tier::Cloud);
                Ok((out, trace))
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
}

impl Request {
    /// Validation that runs before routing. Model ids are filled in per tier, so an empty one is fine here.
    fn validate_for_route(&self) -> Result<()> {
        let mut r = self.clone();
        if r.model_id().is_empty() {
            r = r.with_model_id("(chosen by the router)");
        }
        r.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modality::{
        EmbeddingParams, EmbeddingResponse, GeneratedImage, ImageUnderstandingParams,
    };
    use crate::types::{InferenceParams, InferenceResponse, SessionId};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A scripted backend: serves `caps`, counts calls, and produces outputs from `flat_images` etc.
    struct Mock {
        caps: Vec<RequestKind>,
        calls: AtomicU32,
        text: String,
        flat_image: bool,
        fail: bool,
    }
    impl Mock {
        fn new(caps: &[RequestKind]) -> Arc<Self> {
            Arc::new(Self {
                caps: caps.to_vec(),
                calls: AtomicU32::new(0),
                text: "ok".into(),
                flat_image: false,
                fail: false,
            })
        }
        fn flat(caps: &[RequestKind]) -> Arc<Self> {
            Arc::new(Self {
                caps: caps.to_vec(),
                calls: AtomicU32::new(0),
                text: "ok".into(),
                flat_image: true,
                fail: false,
            })
        }
        fn failing(caps: &[RequestKind]) -> Arc<Self> {
            Arc::new(Self {
                caps: caps.to_vec(),
                calls: AtomicU32::new(0),
                text: "ok".into(),
                flat_image: false,
                fail: true,
            })
        }
        fn calls(&self) -> u32 {
            self.calls.load(Ordering::SeqCst)
        }
    }
    fn png(w: u32, h: u32) -> Vec<u8> {
        let mut b = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13];
        b.extend_from_slice(b"IHDR");
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&h.to_be_bytes());
        b.extend_from_slice(&[8, 2, 0, 0, 0]);
        b
    }
    #[async_trait]
    impl RequestService for Mock {
        fn kinds(&self) -> Vec<RequestKind> {
            self.caps.clone()
        }
        async fn handle(&self, req: Request) -> Result<Output> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(AiKitError::InferenceOom {
                    model_id: "model".into(),
                    required_mb: 8800,
                    available_mb: 7600,
                });
            }
            let sid = SessionId::default();
            Ok(match req {
                Request::Text(p) => Output::Text(InferenceResponse {
                    completion: self.text.clone(),
                    total_tokens: 20,
                    prompt_tokens: 10,
                    completion_tokens: 10,
                    inference_ms: 5,
                    model_version: p.model_id,
                    quantization: "q".into(),
                    session_id: sid,
                    request_id: "r".into(),
                }),
                Request::ImageGeneration(p) => Output::Image(GeneratedImage {
                    png: png(p.width, p.height),
                    width: p.width,
                    height: p.height,
                    seed: 1,
                    steps: p.steps,
                    pixel_stddev: if self.flat_image { 0.0 } else { 50.0 },
                    inference_ms: 7,
                    model_version: p.model_id,
                    session_id: sid,
                    request_id: "r".into(),
                }),
                Request::Embedding(p) => Output::Embeddings(EmbeddingResponse {
                    vectors: p.inputs.iter().map(|_| vec![0.6, 0.8]).collect(),
                    dimensions: 2,
                    prompt_tokens: 3,
                    inference_ms: 2,
                    model_version: p.model_id,
                    session_id: sid,
                    request_id: "r".into(),
                }),
                other => return Err(unsupported(other.kind(), "mock")),
            })
        }
    }

    fn text_req(prompt: &str) -> Request {
        Request::Text(InferenceParams {
            model_id: String::new(),
            prompt: prompt.into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 16,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        })
    }
    fn img_req(prompt: &str) -> Request {
        Request::ImageGeneration(ImageGenParams::new(String::new(), prompt))
    }
    fn embed_req() -> Request {
        Request::Embedding(EmbeddingParams {
            model_id: String::new(),
            inputs: vec!["a".into(), "b".into()],
            normalize: true,
            session_id: None,
            request_id: None,
        })
    }
    fn tier(m: &Arc<Mock>, caps: &[RequestKind]) -> TierBackend {
        TierBackend::new(
            m.clone(),
            &caps.iter().map(|c| (*c, "model")).collect::<Vec<_>>(),
        )
    }
    use RequestKind::*;

    #[tokio::test]
    async fn every_kind_of_request_is_served_by_the_tier_that_has_it() {
        let local = Mock::new(&[TextGeneration, ImageGeneration, Embedding]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[TextGeneration, ImageGeneration, Embedding])],
            vec![],
            RoutePolicy::local_first(),
        );
        for r in [text_req("hello"), img_req("a cat"), embed_req()] {
            let want = r.kind();
            let (out, trace) = s.handle_routed(RoutedRequest::new(r)).await.unwrap();
            assert_eq!(trace.final_tier, Some(Tier::Local));
            let got = match out {
                Output::Text(_) => TextGeneration,
                Output::Image(_) => ImageGeneration,
                Output::Embeddings(_) => Embedding,
                Output::Transcript(_) => SpeechToText,
            };
            assert_eq!(got, want);
        }
        assert_eq!(local.calls(), 3);
    }

    #[tokio::test]
    async fn a_request_kind_nobody_has_is_unsupported_not_attempted() {
        let local = Mock::new(&[TextGeneration]);
        let cloud = Mock::new(&[TextGeneration]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[TextGeneration])],
            vec![tier(&cloud, &[TextGeneration])],
            RoutePolicy::local_first(),
        );
        for r in [img_req("a cat"), embed_req()] {
            let e = s
                .handle_routed(RoutedRequest::new(r).shareable())
                .await
                .unwrap_err();
            assert!(matches!(e, AiKitError::UnsupportedKind { .. }), "{e:?}");
        }
        assert_eq!((local.calls(), cloud.calls()), (0, 0));
    }

    #[tokio::test]
    async fn cloud_only_request_kind_with_private_content_is_unavailable_not_leaked() {
        let local = Mock::new(&[TextGeneration]);
        let cloud = Mock::new(&[TextGeneration, ImageUnderstanding]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[TextGeneration])],
            vec![tier(&cloud, &[TextGeneration, ImageUnderstanding])],
            RoutePolicy::local_first(),
        );
        let r = Request::ImageUnderstanding(ImageUnderstandingParams {
            model_id: String::new(),
            image: vec![1, 2, 3],
            mime: "image/png".into(),
            prompt: "what is this".into(),
            max_tokens: 16,
            session_id: None,
            request_id: None,
        });
        let e = s
            .handle_routed(RoutedRequest::new(r.clone()))
            .await
            .unwrap_err();
        assert!(matches!(e, AiKitError::RequiresShareable { .. }), "{e:?}");
        assert_eq!(
            e.to_inference_error_code(),
            crate::error::InferenceErrorCode::RequiresShareable
        );
        assert_eq!(cloud.calls(), 0, "private image reached the cloud");
        // Shareable: the cloud tier is allowed to take it. (The mock has no vision, so it answers Unsupported itself.)
        let _ = s.handle_routed(RoutedRequest::new(r).shareable()).await;
        assert_eq!(cloud.calls(), 1);
    }

    #[tokio::test]
    async fn free_text_that_is_really_an_image_request_runs_as_one() {
        let local = Mock::new(&[TextGeneration, ImageGeneration]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[TextGeneration, ImageGeneration])],
            vec![],
            RoutePolicy::local_first(),
        );
        let (out, trace) = s
            .handle_routed(RoutedRequest::new(text_req(
                "a watercolor painting of a misty forest cottage",
            )))
            .await
            .unwrap();
        assert!(matches!(out, Output::Image(_)));
        assert!(trace
            .notes
            .iter()
            .any(|n| n.contains("treated a text_generation request as image_generation")));
        // ...and not when retyping is off.
        let s2 = MultiHybridService::new(
            vec![tier(&local, &[TextGeneration, ImageGeneration])],
            vec![],
            RoutePolicy::local_first(),
        )
        .without_retyping();
        let (out, _) = s2
            .handle_routed(RoutedRequest::new(text_req(
                "a watercolor painting of a misty forest cottage",
            )))
            .await
            .unwrap();
        assert!(matches!(out, Output::Text(_)));
    }

    #[tokio::test]
    async fn a_flat_image_fails_the_sanity_check_and_private_content_stays_local() {
        let local = Mock::flat(&[ImageGeneration]);
        let cloud = Mock::new(&[ImageGeneration]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[ImageGeneration])],
            vec![tier(&cloud, &[ImageGeneration])],
            RoutePolicy::local_first(),
        );
        let (out, t) = s
            .handle_routed(RoutedRequest::new(img_req("a cat")).with_check(OutputCheck::Sane))
            .await
            .unwrap();
        assert!(matches!(out, Output::Image(_)));
        assert_eq!(cloud.calls(), 0);
        assert_eq!(t.attempts[0].verified, Some(false));
        assert_eq!(s.route_stats().blocked_by_privacy, 1);
    }

    #[tokio::test]
    async fn a_failed_image_check_escalates_when_the_content_is_shareable() {
        let local = Mock::flat(&[ImageGeneration]);
        let cloud = Mock::new(&[ImageGeneration]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[ImageGeneration])],
            vec![tier(&cloud, &[ImageGeneration])],
            RoutePolicy::local_first(),
        );
        let (_, t) = s
            .handle_routed(
                RoutedRequest::new(img_req("a cat"))
                    .shareable()
                    .with_check(OutputCheck::Sane),
            )
            .await
            .unwrap();
        assert_eq!(t.final_tier, Some(Tier::Cloud));
        assert_eq!((local.calls(), cloud.calls()), (1, 1));
        let st = s.route_stats();
        assert_eq!(
            (st.escalations, st.cloud_payload_bytes),
            (1, "a cat".len() as u64)
        );
    }

    #[tokio::test]
    async fn private_local_failure_is_never_rescued_by_the_cloud_for_any_kind() {
        let local = Mock::failing(&[Embedding]);
        let cloud = Mock::new(&[Embedding]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[Embedding])],
            vec![tier(&cloud, &[Embedding])],
            RoutePolicy::local_first(),
        );
        assert!(s
            .handle_routed(RoutedRequest::new(embed_req()))
            .await
            .is_err());
        assert_eq!(cloud.calls(), 0);
        // With permission, the same failure falls back.
        assert!(s
            .handle_routed(RoutedRequest::new(embed_req()).shareable())
            .await
            .is_ok());
        assert_eq!(cloud.calls(), 1);
    }

    #[tokio::test]
    async fn invalid_requests_are_rejected_before_any_backend_is_touched() {
        let local = Mock::new(&[ImageGeneration]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[ImageGeneration])],
            vec![],
            RoutePolicy::local_first(),
        );
        let mut bad = ImageGenParams::new(String::new(), "a cat");
        bad.width = 500;
        let e = s
            .handle_routed(RoutedRequest::new(Request::ImageGeneration(bad)))
            .await
            .unwrap_err();
        assert!(matches!(e, AiKitError::InvalidParams { .. }));
        assert_eq!(local.calls(), 0);
    }

    #[test]
    fn kinds_report_what_each_tier_really_has() {
        let local = Mock::new(&[TextGeneration, Embedding]);
        let cloud = Mock::new(&[TextGeneration]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[TextGeneration, Embedding])],
            vec![tier(&cloud, &[TextGeneration])],
            RoutePolicy::local_first(),
        );
        assert_eq!(
            s.kinds(),
            (vec![TextGeneration, Embedding], vec![TextGeneration])
        );
    }

    #[tokio::test]
    async fn a_local_out_of_memory_error_keeps_its_type_through_the_router() {
        let local = Mock::failing(&[ImageGeneration]);
        let s = MultiHybridService::new(
            vec![tier(&local, &[ImageGeneration])],
            vec![],
            RoutePolicy::local_first(),
        );
        let e = s
            .handle_routed(RoutedRequest::new(Request::ImageGeneration(
                crate::modality::ImageGenParams::new("m", "a cat"),
            )))
            .await
            .unwrap_err();
        assert!(matches!(e, AiKitError::InferenceOom { .. }), "{e:?}");
        assert_eq!(
            e.to_inference_error_code(),
            crate::error::InferenceErrorCode::Oom,
            "the wire code must say out of memory, not internal"
        );
    }
}
