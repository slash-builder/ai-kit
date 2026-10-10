//! The kit's request envelope: every kind of work Inference Kit can be asked for.
//!
//! [`InferenceService`] is deliberately text-in/text-out and its ABI is frozen, so it is left
//! alone. This module adds a **second, general contract** next to it:
//!
//! - [`Capability`]: what kind of work (text, image generation, image understanding,
//!   embeddings, speech-to-text).
//! - [`Request`] / [`Output`]: typed payloads for each, so a caller never smuggles an image
//!   through a text prompt, and an answer never needs guessing at.
//! - [`CapabilityService`]: "I can do these capabilities; give me a request". A backend
//!   implements it for whichever capabilities it really has.
//! - [`TextCapability`]: lifts any existing [`InferenceService`] into the new contract, so the
//!   text backends need no changes.
//!
//! # "Ready for any kind of request" means three things
//!
//! 1. Every kind of request is *representable* and validated here, whether or not a backend
//!    exists on this device.
//! 2. A request nobody can serve gets a **typed, honest answer**
//!    ([`AiKitError::UnsupportedCapability`], or a routing [`crate::Plan::Unavailable`]) that
//!    says which capability was missing and why. It is never a silent fallback to the wrong kind
//!    of model and never a confusing `InvalidParams`.
//! 3. The router and the hybrid service ([`crate::routed`]) pick a tier from *which tiers have the
//!    capability*, not from the assumption that everything is text.
//!
//! What has real backends today, and what is only a contract, is listed in the README. The
//! contract is complete for all five capabilities; the backends are not.

use crate::error::{AiKitError, Result};
use crate::service::InferenceService;
use crate::types::{InferenceParams, InferenceResponse, SessionId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::str::FromStr;
use std::sync::Arc;

/// A kind of work. New kinds are added here first, so every consumer sees them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Text in, text out (the original `InferenceService`).
    TextGeneration,
    /// Text prompt in, a raster image out.
    ImageGeneration,
    /// An image (and a question) in, text out.
    ImageUnderstanding,
    /// Texts in, one vector each out (semantic search).
    Embedding,
    /// Audio in, a transcript out.
    SpeechToText,
}

impl Capability {
    pub const ALL: [Capability; 5] = [
        Capability::TextGeneration,
        Capability::ImageGeneration,
        Capability::ImageUnderstanding,
        Capability::Embedding,
        Capability::SpeechToText,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Capability::TextGeneration => "text_generation",
            Capability::ImageGeneration => "image_generation",
            Capability::ImageUnderstanding => "image_understanding",
            Capability::Embedding => "embedding",
            Capability::SpeechToText => "speech_to_text",
        }
    }
}

impl FromStr for Capability {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Capability::ALL
            .into_iter()
            .find(|c| c.as_str() == s.trim().to_ascii_lowercase())
            .ok_or_else(|| {
                format!(
                    "unknown capability '{s}' (expected one of: {})",
                    Capability::ALL.map(|c| c.as_str()).join(", ")
                )
            })
    }
}

/// Text prompt in, a PNG out.
#[derive(Debug, Clone)]
pub struct ImageGenParams {
    pub model_id: String,
    pub prompt: String,
    /// What to steer away from. Only used when `guidance_scale > 1`.
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    /// Denoising steps. Turbo-class models want 1-4.
    pub steps: u32,
    /// Classifier-free guidance. `0.0` or `1.0` means off (what distilled turbo models want).
    pub guidance_scale: f32,
    /// `None` picks a fresh seed (reported back in the output); `Some` is reproducible.
    pub seed: Option<u64>,
    pub session_id: Option<SessionId>,
    pub request_id: Option<String>,
}

impl ImageGenParams {
    pub fn new(model_id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            prompt: prompt.into(),
            negative_prompt: None,
            width: 512,
            height: 512,
            steps: 2,
            guidance_scale: 0.0,
            seed: None,
            session_id: None,
            request_id: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |reason: String| Err(AiKitError::InvalidParams { reason });
        if self.model_id.is_empty() {
            return bad("model_id cannot be empty".into());
        }
        if self.prompt.trim().is_empty() {
            return bad("prompt cannot be empty".into());
        }
        if self.prompt.chars().count() > 4000 {
            return bad("prompt is longer than 4000 characters".into());
        }
        for (name, v) in [("width", self.width), ("height", self.height)] {
            if !(64..=2048).contains(&v) || v % 8 != 0 {
                return bad(format!(
                    "{name} must be a multiple of 8 between 64 and 2048, got {v}"
                ));
            }
        }
        if u64::from(self.width) * u64::from(self.height) > 2048 * 1536 {
            return bad("image is larger than 2048x1536 pixels".into());
        }
        if !(1..=100).contains(&self.steps) {
            return bad(format!("steps must be 1-100, got {}", self.steps));
        }
        if !(0.0..=20.0).contains(&self.guidance_scale) {
            return bad(format!(
                "guidance_scale must be in [0, 20], got {}",
                self.guidance_scale
            ));
        }
        Ok(())
    }
}

/// A finished image.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// PNG bytes.
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// The seed actually used (so a "random" image can be reproduced).
    pub seed: u64,
    pub steps: u32,
    /// Standard deviation of the pixel values (0-255 scale). A flat or black image, the usual sign of a
    /// numerical failure in half precision, has a value near zero; [`OutputCheck::Sane`] uses it.
    pub pixel_stddev: f32,
    pub inference_ms: u32,
    pub model_version: String,
    pub session_id: SessionId,
    pub request_id: String,
}

/// An image and a question about it.
#[derive(Debug, Clone)]
pub struct ImageUnderstandingParams {
    pub model_id: String,
    pub image: Vec<u8>,
    /// e.g. `image/png`, `image/jpeg`.
    pub mime: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub session_id: Option<SessionId>,
    pub request_id: Option<String>,
}

/// Texts to turn into vectors.
#[derive(Debug, Clone)]
pub struct EmbeddingParams {
    pub model_id: String,
    pub inputs: Vec<String>,
    /// Scale each vector to unit length (what cosine-similarity search wants).
    pub normalize: bool,
    pub session_id: Option<SessionId>,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// One vector per input, in order.
    pub vectors: Vec<Vec<f32>>,
    pub dimensions: usize,
    pub prompt_tokens: u32,
    pub inference_ms: u32,
    pub model_version: String,
    pub session_id: SessionId,
    pub request_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// RIFF/WAVE, PCM.
    Wav,
}

/// Audio to transcribe.
#[derive(Debug, Clone)]
pub struct SpeechParams {
    pub model_id: String,
    pub audio: Vec<u8>,
    pub format: AudioFormat,
    pub language: Option<String>,
    pub session_id: Option<SessionId>,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TranscriptResponse {
    pub text: String,
    pub language: Option<String>,
    pub inference_ms: u32,
    pub model_version: String,
    pub session_id: SessionId,
    pub request_id: String,
}

/// Any request the kit can be asked.
#[derive(Debug, Clone)]
pub enum Request {
    Text(InferenceParams),
    ImageGeneration(ImageGenParams),
    ImageUnderstanding(ImageUnderstandingParams),
    Embedding(EmbeddingParams),
    SpeechToText(SpeechParams),
}

impl Request {
    pub fn capability(&self) -> Capability {
        match self {
            Request::Text(_) => Capability::TextGeneration,
            Request::ImageGeneration(_) => Capability::ImageGeneration,
            Request::ImageUnderstanding(_) => Capability::ImageUnderstanding,
            Request::Embedding(_) => Capability::Embedding,
            Request::SpeechToText(_) => Capability::SpeechToText,
        }
    }

    pub fn model_id(&self) -> &str {
        match self {
            Request::Text(p) => &p.model_id,
            Request::ImageGeneration(p) => &p.model_id,
            Request::ImageUnderstanding(p) => &p.model_id,
            Request::Embedding(p) => &p.model_id,
            Request::SpeechToText(p) => &p.model_id,
        }
    }

    /// The same request aimed at a different model (a tier uses its own configured model).
    pub fn with_model_id(mut self, id: &str) -> Self {
        match &mut self {
            Request::Text(p) => p.model_id = id.to_string(),
            Request::ImageGeneration(p) => p.model_id = id.to_string(),
            Request::ImageUnderstanding(p) => p.model_id = id.to_string(),
            Request::Embedding(p) => p.model_id = id.to_string(),
            Request::SpeechToText(p) => p.model_id = id.to_string(),
        }
        self
    }

    /// The text a classifier can read, if the request has any.
    pub fn prompt_text(&self) -> Option<&str> {
        match self {
            Request::Text(p) => Some(&p.prompt),
            Request::ImageGeneration(p) => Some(&p.prompt),
            Request::ImageUnderstanding(p) => Some(&p.prompt),
            Request::Embedding(_) | Request::SpeechToText(_) => None,
        }
    }

    /// Rough size of what the request carries, in bytes (text plus any binary payload). Used to estimate
    /// how much data would leave the device if the request were sent to a cloud tier.
    pub fn payload_bytes(&self) -> usize {
        match self {
            Request::Text(p) => p.prompt.len(),
            Request::ImageGeneration(p) => {
                p.prompt.len() + p.negative_prompt.as_ref().map_or(0, String::len)
            }
            Request::ImageUnderstanding(p) => p.image.len() + p.prompt.len(),
            Request::Embedding(p) => p.inputs.iter().map(String::len).sum(),
            Request::SpeechToText(p) => p.audio.len(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |reason: &str| {
            Err(AiKitError::InvalidParams {
                reason: reason.to_string(),
            })
        };
        match self {
            Request::Text(p) => p.validate(),
            Request::ImageGeneration(p) => p.validate(),
            Request::ImageUnderstanding(p) => {
                if p.image.is_empty() {
                    return bad("image_understanding needs an image");
                }
                if !p.mime.starts_with("image/") {
                    return bad("mime must be an image/* type");
                }
                if p.prompt.trim().is_empty() {
                    return bad("image_understanding needs a question about the image");
                }
                Ok(())
            }
            Request::Embedding(p) => {
                if p.inputs.is_empty() {
                    return bad("embedding needs at least one input text");
                }
                if p.inputs.len() > 512 {
                    return bad("embedding takes at most 512 inputs per request");
                }
                if p.inputs.iter().any(|s| s.trim().is_empty()) {
                    return bad("embedding inputs cannot be empty");
                }
                Ok(())
            }
            Request::SpeechToText(p) => {
                if p.audio.is_empty() {
                    return bad("speech_to_text needs audio");
                }
                Ok(())
            }
        }
    }
}

/// Any answer the kit can give.
#[derive(Debug, Clone)]
pub enum Output {
    /// Text generation, and the answer to an image question.
    Text(InferenceResponse),
    Image(GeneratedImage),
    Embeddings(EmbeddingResponse),
    Transcript(TranscriptResponse),
}

impl Output {
    pub fn inference_ms(&self) -> u32 {
        match self {
            Output::Text(r) => r.inference_ms,
            Output::Image(r) => r.inference_ms,
            Output::Embeddings(r) => r.inference_ms,
            Output::Transcript(r) => r.inference_ms,
        }
    }
}

/// A check on a finished [`Output`], the multimodal counterpart of [`crate::Verifier`].
#[derive(Clone, Default)]
pub enum OutputCheck {
    /// Nothing to check.
    #[default]
    None,
    /// Cheap sanity for the modality: a real PNG of the requested size that is not flat; embeddings that are
    /// finite, non-zero and the right shape; a non-empty transcript or text answer.
    Sane,
    /// Caller-supplied.
    Custom(Arc<dyn Fn(&Output) -> bool + Send + Sync>),
}

impl std::fmt::Debug for OutputCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputCheck::None => write!(f, "None"),
            OutputCheck::Sane => write!(f, "Sane"),
            OutputCheck::Custom(_) => write!(f, "Custom(..)"),
        }
    }
}

impl OutputCheck {
    pub fn is_present(&self) -> bool {
        !matches!(self, OutputCheck::None)
    }

    /// `None` when there is nothing to check; otherwise whether `out` passes. `req` supplies what was asked for.
    pub fn check(&self, req: &Request, out: &Output) -> Option<bool> {
        match self {
            OutputCheck::None => None,
            OutputCheck::Custom(f) => Some(f(out)),
            OutputCheck::Sane => Some(match (req, out) {
                (Request::ImageGeneration(p), Output::Image(i)) => {
                    is_png(&i.png)
                        && png_dims(&i.png) == Some((p.width, p.height))
                        && i.pixel_stddev > 2.0
                }
                (Request::Embedding(p), Output::Embeddings(e)) => {
                    e.vectors.len() == p.inputs.len()
                        && e.vectors.iter().all(|v| {
                            v.len() == e.dimensions
                                && v.iter().all(|x| x.is_finite())
                                && v.iter().any(|x| *x != 0.0)
                        })
                }
                (_, Output::Transcript(t)) => !t.text.trim().is_empty(),
                (_, Output::Text(t)) => !t.completion.trim().is_empty(),
                _ => false,
            }),
        }
    }
}

fn is_png(b: &[u8]) -> bool {
    b.len() > 24 && b[..8] == [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
}

/// Width and height from a PNG's IHDR chunk, without decoding the image.
pub fn png_dims(b: &[u8]) -> Option<(u32, u32)> {
    if !is_png(b) || &b[12..16] != b"IHDR" {
        return None;
    }
    Some((
        u32::from_be_bytes(b[16..20].try_into().ok()?),
        u32::from_be_bytes(b[20..24].try_into().ok()?),
    ))
}

/// A backend that can do one or more [`Capability`]s.
///
/// Implement `handle` for the capabilities you really have and return
/// [`AiKitError::UnsupportedCapability`] (see [`unsupported`]) for the rest.
#[async_trait]
pub trait CapabilityService: Send + Sync {
    fn capabilities(&self) -> Vec<Capability>;
    async fn handle(&self, req: Request) -> Result<Output>;
}

/// The error for "this service does not do that".
pub fn unsupported(cap: Capability, reason: impl Into<String>) -> AiKitError {
    AiKitError::UnsupportedCapability {
        capability: cap.as_str().to_string(),
        reason: reason.into(),
    }
}

/// Lifts any text [`InferenceService`] into the general contract.
pub struct TextCapability {
    inner: Arc<dyn InferenceService>,
}

impl TextCapability {
    pub fn new(inner: Arc<dyn InferenceService>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl CapabilityService for TextCapability {
    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::TextGeneration]
    }
    async fn handle(&self, req: Request) -> Result<Output> {
        match req {
            Request::Text(p) => self.inner.infer(p).await.map(Output::Text),
            other => Err(unsupported(
                other.capability(),
                "this is a text-generation backend",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_round_trip_through_strings() {
        for c in Capability::ALL {
            assert_eq!(c.as_str().parse::<Capability>(), Ok(c));
        }
        assert!("telepathy".parse::<Capability>().is_err());
    }

    #[test]
    fn image_params_validate() {
        let ok = ImageGenParams::new("m", "a cat");
        assert!(ok.validate().is_ok());
        for (w, h) in [(0, 512), (500, 512), (512, 4096), (63, 64)] {
            let mut p = ok.clone();
            p.width = w;
            p.height = h;
            assert!(p.validate().is_err(), "{w}x{h} should be rejected");
        }
        let mut p = ok.clone();
        p.prompt = "  ".into();
        assert!(p.validate().is_err());
        p = ok.clone();
        p.steps = 0;
        assert!(p.validate().is_err());
        p = ok;
        p.guidance_scale = 99.0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn every_request_kind_reports_its_capability_and_validates() {
        let reqs = [
            Request::Text(InferenceParams {
                model_id: "m".into(),
                prompt: "hi".into(),
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 8,
                session_id: None,
                preferred_quantization: None,
                request_id: None,
            }),
            Request::ImageGeneration(ImageGenParams::new("m", "a cat")),
            Request::ImageUnderstanding(ImageUnderstandingParams {
                model_id: "m".into(),
                image: vec![1, 2, 3],
                mime: "image/png".into(),
                prompt: "what is this".into(),
                max_tokens: 32,
                session_id: None,
                request_id: None,
            }),
            Request::Embedding(EmbeddingParams {
                model_id: "m".into(),
                inputs: vec!["a".into(), "b".into()],
                normalize: true,
                session_id: None,
                request_id: None,
            }),
            Request::SpeechToText(SpeechParams {
                model_id: "m".into(),
                audio: vec![0; 10],
                format: AudioFormat::Wav,
                language: None,
                session_id: None,
                request_id: None,
            }),
        ];
        let caps: Vec<Capability> = reqs.iter().map(Request::capability).collect();
        assert_eq!(caps, Capability::ALL.to_vec());
        for r in &reqs {
            assert!(r.validate().is_ok(), "{:?}", r.capability());
            assert!(r.payload_bytes() > 0);
        }
    }

    #[test]
    fn invalid_requests_are_rejected_not_attempted() {
        let empty_embed = Request::Embedding(EmbeddingParams {
            model_id: "m".into(),
            inputs: vec![],
            normalize: true,
            session_id: None,
            request_id: None,
        });
        assert!(matches!(
            empty_embed.validate(),
            Err(AiKitError::InvalidParams { .. })
        ));
        let no_audio = Request::SpeechToText(SpeechParams {
            model_id: "m".into(),
            audio: vec![],
            format: AudioFormat::Wav,
            language: None,
            session_id: None,
            request_id: None,
        });
        assert!(no_audio.validate().is_err());
    }

    fn tiny_png(w: u32, h: u32) -> Vec<u8> {
        let mut b = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13];
        b.extend_from_slice(b"IHDR");
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&h.to_be_bytes());
        b.extend_from_slice(&[8, 2, 0, 0, 0]);
        b
    }

    fn image_out(png: Vec<u8>, w: u32, h: u32, stddev: f32) -> Output {
        Output::Image(GeneratedImage {
            png,
            width: w,
            height: h,
            seed: 1,
            steps: 2,
            pixel_stddev: stddev,
            inference_ms: 1,
            model_version: "v".into(),
            session_id: SessionId::default(),
            request_id: "r".into(),
        })
    }

    #[test]
    fn sane_check_catches_blank_wrong_size_and_garbage_images() {
        let req = Request::ImageGeneration(ImageGenParams::new("m", "a cat"));
        let c = OutputCheck::Sane;
        assert_eq!(
            c.check(&req, &image_out(tiny_png(512, 512), 512, 512, 40.0)),
            Some(true)
        );
        assert_eq!(
            c.check(&req, &image_out(tiny_png(512, 512), 512, 512, 0.0)),
            Some(false),
            "black image"
        );
        assert_eq!(
            c.check(&req, &image_out(tiny_png(256, 256), 256, 256, 40.0)),
            Some(false),
            "wrong size"
        );
        assert_eq!(
            c.check(&req, &image_out(vec![1, 2, 3], 512, 512, 40.0)),
            Some(false),
            "not a png"
        );
        assert_eq!(
            OutputCheck::None.check(&req, &image_out(vec![], 1, 1, 0.0)),
            None
        );
    }

    #[test]
    fn sane_check_validates_embeddings() {
        let req = Request::Embedding(EmbeddingParams {
            model_id: "m".into(),
            inputs: vec!["a".into(), "b".into()],
            normalize: true,
            session_id: None,
            request_id: None,
        });
        let mk = |vectors: Vec<Vec<f32>>| {
            Output::Embeddings(EmbeddingResponse {
                dimensions: 2,
                vectors,
                prompt_tokens: 2,
                inference_ms: 1,
                model_version: "v".into(),
                session_id: SessionId::default(),
                request_id: "r".into(),
            })
        };
        let c = OutputCheck::Sane;
        assert_eq!(
            c.check(&req, &mk(vec![vec![0.6, 0.8], vec![1.0, 0.0]])),
            Some(true)
        );
        assert_eq!(
            c.check(&req, &mk(vec![vec![0.6, 0.8]])),
            Some(false),
            "wrong count"
        );
        assert_eq!(
            c.check(&req, &mk(vec![vec![f32::NAN, 0.8], vec![1.0, 0.0]])),
            Some(false),
            "NaN"
        );
        assert_eq!(
            c.check(&req, &mk(vec![vec![0.0, 0.0], vec![1.0, 0.0]])),
            Some(false),
            "zero vector"
        );
    }

    #[test]
    fn png_dims_reads_the_header() {
        assert_eq!(png_dims(&tiny_png(640, 480)), Some((640, 480)));
        assert_eq!(png_dims(b"not a png at all, definitely not"), None);
    }
}
