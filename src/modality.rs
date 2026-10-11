//! The kit's request envelope: every kind of work Inference Kit can be asked for.
//!
//! [`InferenceService`] is deliberately text-in/text-out and its ABI is frozen, so it is left
//! alone. This module adds a **second, general contract** next to it:
//!
//! - [`RequestKind`]: what kind of work (text, image generation, image understanding,
//!   embeddings, speech-to-text).
//! - [`Request`] / [`Output`]: typed payloads for each, so a caller never smuggles an image
//!   through a text prompt, and an answer never needs guessing at.
//! - [`RequestService`]: "I can do these request kinds; give me a request". A backend
//!   implements it for whichever request kinds it really has.
//! - [`TextService`]: lifts any existing [`InferenceService`] into the new contract, so the
//!   text backends need no changes.
//!
//! # "Ready for any kind of request" means three things
//!
//! 1. Every kind of request is *representable* and validated here, whether or not a backend
//!    exists on this device.
//! 2. A request nobody can serve gets a **typed, honest answer**
//!    ([`AiKitError::UnsupportedKind`], or a routing [`crate::Plan::Unavailable`]) that
//!    says which request kind was missing and why. It is never a silent fallback to the wrong kind
//!    of model and never a confusing `InvalidParams`.
//! 3. The router and the hybrid service ([`crate::routed`]) pick a tier from *which tiers have the
//!    request kind*, not from the assumption that everything is text.
//!
//! What has real backends today, and what is only a contract, is listed in the README. The
//! contract is complete for all five request kinds; the backends are not.

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
#[non_exhaustive]
pub enum RequestKind {
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

impl RequestKind {
    pub const ALL: [RequestKind; 5] = [
        RequestKind::TextGeneration,
        RequestKind::ImageGeneration,
        RequestKind::ImageUnderstanding,
        RequestKind::Embedding,
        RequestKind::SpeechToText,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            RequestKind::TextGeneration => "text_generation",
            RequestKind::ImageGeneration => "image_generation",
            RequestKind::ImageUnderstanding => "image_understanding",
            RequestKind::Embedding => "embedding",
            RequestKind::SpeechToText => "speech_to_text",
        }
    }
}

impl FromStr for RequestKind {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        RequestKind::ALL
            .into_iter()
            .find(|c| c.as_str() == s.trim().to_ascii_lowercase())
            .ok_or_else(|| {
                format!(
                    "unknown request kind '{s}' (expected one of: {})",
                    RequestKind::ALL.map(|c| c.as_str()).join(", ")
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
        if self.prompt.chars().count() > limits::PROMPT_MAX_CHARS {
            return Err(too_large(
                "prompt_chars",
                limits::PROMPT_MAX_CHARS as u64,
                self.prompt.chars().count() as u64,
            ));
        }
        for (name, v) in [("width", self.width), ("height", self.height)] {
            if !(64..=2048).contains(&v) || v % 8 != 0 {
                return bad(format!(
                    "{name} must be a multiple of 8 between 64 and 2048, got {v}"
                ));
            }
        }
        let pixels = u64::from(self.width) * u64::from(self.height);
        if pixels > limits::IMAGE_GEN_MAX_PIXELS {
            return Err(too_large(
                "image_pixels",
                limits::IMAGE_GEN_MAX_PIXELS,
                pixels,
            ));
        }
        if self.steps == 0 {
            return bad("steps must be at least 1".into());
        }
        if self.steps > limits::IMAGE_GEN_MAX_STEPS {
            return Err(too_large(
                "steps",
                u64::from(limits::IMAGE_GEN_MAX_STEPS),
                u64::from(self.steps),
            ));
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
#[non_exhaustive]
pub enum Request {
    Text(InferenceParams),
    ImageGeneration(ImageGenParams),
    ImageUnderstanding(ImageUnderstandingParams),
    Embedding(EmbeddingParams),
    SpeechToText(SpeechParams),
}

impl Request {
    pub fn kind(&self) -> RequestKind {
        match self {
            Request::Text(_) => RequestKind::TextGeneration,
            Request::ImageGeneration(_) => RequestKind::ImageGeneration,
            Request::ImageUnderstanding(_) => RequestKind::ImageUnderstanding,
            Request::Embedding(_) => RequestKind::Embedding,
            Request::SpeechToText(_) => RequestKind::SpeechToText,
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
                if p.prompt.chars().count() > limits::PROMPT_MAX_CHARS {
                    return Err(too_large(
                        "prompt_chars",
                        limits::PROMPT_MAX_CHARS as u64,
                        p.prompt.chars().count() as u64,
                    ));
                }
                if p.image.len() > limits::IMAGE_INPUT_MAX_BYTES {
                    return Err(too_large(
                        "image_bytes",
                        limits::IMAGE_INPUT_MAX_BYTES as u64,
                        p.image.len() as u64,
                    ));
                }
                // judge the decoded size from the header, before any decoder sees the bytes
                if let Some((w, h)) = png_dims(&p.image) {
                    let px = u64::from(w) * u64::from(h);
                    if px > limits::IMAGE_INPUT_MAX_PIXELS {
                        return Err(too_large(
                            "image_pixels",
                            limits::IMAGE_INPUT_MAX_PIXELS,
                            px,
                        ));
                    }
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
                if let Some(n) = p
                    .inputs
                    .iter()
                    .map(|s| s.chars().count())
                    .find(|n| *n > limits::EMBED_MAX_CHARS_PER_INPUT)
                {
                    return Err(too_large(
                        "embedding_input_chars",
                        limits::EMBED_MAX_CHARS_PER_INPUT as u64,
                        n as u64,
                    ));
                }
                let total: usize = p.inputs.iter().map(String::len).sum();
                if total > limits::EMBED_MAX_TOTAL_BYTES {
                    return Err(too_large(
                        "embedding_total_bytes",
                        limits::EMBED_MAX_TOTAL_BYTES as u64,
                        total as u64,
                    ));
                }
                Ok(())
            }
            Request::SpeechToText(p) => {
                if p.audio.is_empty() {
                    return bad("speech_to_text needs audio");
                }
                if p.audio.len() > limits::AUDIO_MAX_BYTES {
                    return Err(too_large(
                        "audio_bytes",
                        limits::AUDIO_MAX_BYTES as u64,
                        p.audio.len() as u64,
                    ));
                }
                match (p.format, wav_duration_seconds(&p.audio)) {
                    (AudioFormat::Wav, None) => bad("audio is not a readable WAV file"),
                    (AudioFormat::Wav, Some(secs)) if secs > limits::AUDIO_MAX_SECONDS as f64 => {
                        Err(too_large(
                            "audio_seconds",
                            limits::AUDIO_MAX_SECONDS,
                            secs.ceil() as u64,
                        ))
                    }
                    _ => Ok(()),
                }
            }
        }
    }
}

/// Per-kind size limits, enforced by [`Request::validate`] **before** routing, so an oversized request is refused with
/// [`AiKitError::RequestTooLarge`] instead of reaching a backend (DJ ruling 2026-10-11: reject, do not crash).
///
/// These are the kit-wide ceilings; a particular model may allow less (`InferenceOom` still applies for what does not fit).
/// The figures are proposals from the security review, not benchmarks, and are expected to move.
pub mod limits {
    /// Largest generated image, in pixels (1024 x 1024).
    pub const IMAGE_GEN_MAX_PIXELS: u64 = 1024 * 1024;
    /// Most denoising steps a request may ask for.
    pub const IMAGE_GEN_MAX_STEPS: u32 = 50;
    /// Longest text prompt, in characters.
    pub const PROMPT_MAX_CHARS: usize = 4000;
    /// Largest input image for image understanding, in bytes.
    pub const IMAGE_INPUT_MAX_BYTES: usize = 10 << 20;
    /// Largest input image, in decoded pixels, checked from the header **before** decoding (decompression bombs).
    pub const IMAGE_INPUT_MAX_PIXELS: u64 = 4096 * 4096;
    /// Longest single text to embed, in characters.
    pub const EMBED_MAX_CHARS_PER_INPUT: usize = 8000;
    /// Most text one embedding request may carry, in bytes.
    pub const EMBED_MAX_TOTAL_BYTES: usize = 1 << 20;
    /// Largest audio input, in bytes.
    pub const AUDIO_MAX_BYTES: usize = 50 << 20;
    /// Longest audio input, in seconds (read from the WAV header).
    pub const AUDIO_MAX_SECONDS: u64 = 600;
}

fn too_large(limit_name: &str, limit: u64, actual: u64) -> AiKitError {
    AiKitError::RequestTooLarge {
        limit_name: limit_name.to_string(),
        limit,
        actual,
    }
}

/// Duration in seconds of a PCM WAV file, from its header (`None` if it is not a readable WAV).
pub fn wav_duration_seconds(b: &[u8]) -> Option<f64> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return None;
    }
    let (mut pos, mut byte_rate) = (12usize, None);
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().ok()?) as usize;
        let body = pos + 8;
        if id == b"fmt " && body + 12 <= b.len() {
            byte_rate = Some(u32::from_le_bytes(b[body + 8..body + 12].try_into().ok()?));
        } else if id == b"data" {
            // a streamed file may claim 0xFFFFFFFF bytes; never believe more than is actually there
            let data = size.min(b.len() - body);
            let rate = byte_rate.filter(|r| *r > 0)?;
            return Some(data as f64 / f64::from(rate));
        }
        pos = body.checked_add(size)?.checked_add(size & 1)?;
    }
    None
}

/// Any answer the kit can give.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

/// A backend that can do one or more [`RequestKind`]s.
///
/// Implement `handle` for the request kinds you really have and return
/// [`AiKitError::UnsupportedKind`] (see [`unsupported`]) for the rest.
#[async_trait]
pub trait RequestService: Send + Sync {
    fn kinds(&self) -> Vec<RequestKind>;
    async fn handle(&self, req: Request) -> Result<Output>;
}

/// The error for "this service does not do that".
pub fn unsupported(cap: RequestKind, reason: impl Into<String>) -> AiKitError {
    AiKitError::UnsupportedKind {
        request_kind: cap.as_str().to_string(),
        reason: reason.into(),
    }
}

/// Lifts any text [`InferenceService`] into the general contract.
pub struct TextService {
    inner: Arc<dyn InferenceService>,
}

impl TextService {
    pub fn new(inner: Arc<dyn InferenceService>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl RequestService for TextService {
    fn kinds(&self) -> Vec<RequestKind> {
        vec![RequestKind::TextGeneration]
    }
    async fn handle(&self, req: Request) -> Result<Output> {
        match req {
            Request::Text(p) => self.inner.infer(p).await.map(Output::Text),
            other => Err(unsupported(
                other.kind(),
                "this is a text-generation backend",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_kinds_round_trip_through_strings() {
        for c in RequestKind::ALL {
            assert_eq!(c.as_str().parse::<RequestKind>(), Ok(c));
        }
        assert!("telepathy".parse::<RequestKind>().is_err());
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
    fn every_request_kind_reports_its_kind_and_validates() {
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
                audio: wav(1, None),
                format: AudioFormat::Wav,
                language: None,
                session_id: None,
                request_id: None,
            }),
        ];
        let caps: Vec<RequestKind> = reqs.iter().map(Request::kind).collect();
        assert_eq!(caps, RequestKind::ALL.to_vec());
        for r in &reqs {
            assert!(r.validate().is_ok(), "{:?}", r.kind());
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

    fn wav(seconds: u32, claimed_data: Option<u32>) -> Vec<u8> {
        // 8 kHz mono 16-bit PCM: 16000 bytes per second
        let data = seconds * 16_000;
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&[1, 0, 1, 0]); // PCM, mono
        b.extend_from_slice(&8000u32.to_le_bytes());
        b.extend_from_slice(&16_000u32.to_le_bytes()); // byte rate
        b.extend_from_slice(&[2, 0, 16, 0]);
        b.extend_from_slice(b"data");
        b.extend_from_slice(&claimed_data.unwrap_or(data).to_le_bytes());
        b.extend(std::iter::repeat_n(0u8, data as usize));
        b
    }

    fn too_large_name(r: Result<()>) -> Option<String> {
        match r {
            Err(AiKitError::RequestTooLarge { limit_name, .. }) => Some(limit_name),
            _ => None,
        }
    }

    #[test]
    fn image_generation_over_the_pixel_or_step_limit_is_refused_and_at_the_limit_is_fine() {
        let mut p = ImageGenParams::new("m", "a cat");
        p.width = 1024;
        p.height = 1024;
        p.steps = 50;
        assert!(p.validate().is_ok(), "exactly at both limits is allowed");
        p.width = 2048; // 2048 x 1024 = 2x the pixel limit
        assert_eq!(
            too_large_name(p.validate()).as_deref(),
            Some("image_pixels")
        );
        p.width = 1024;
        p.steps = 51;
        assert_eq!(too_large_name(p.validate()).as_deref(), Some("steps"));
        p.steps = 0;
        assert!(
            matches!(p.validate(), Err(AiKitError::InvalidParams { .. })),
            "zero steps is a bad parameter, not an oversize"
        );
    }

    #[test]
    fn image_input_is_judged_by_bytes_and_by_its_header_before_decoding() {
        let mk = |image: Vec<u8>| {
            Request::ImageUnderstanding(ImageUnderstandingParams {
                model_id: "m".into(),
                image,
                mime: "image/png".into(),
                prompt: "what is this".into(),
                max_tokens: 16,
                session_id: None,
                request_id: None,
            })
        };
        assert!(
            mk(tiny_png(4096, 4096)).validate().is_ok(),
            "at the pixel limit"
        );
        // a 30-byte file whose header claims 100000 x 100000 pixels: a decompression bomb, refused without decoding
        assert_eq!(
            too_large_name(mk(tiny_png(100_000, 100_000)).validate()).as_deref(),
            Some("image_pixels")
        );
        let mut big = tiny_png(10, 10);
        big.resize(limits::IMAGE_INPUT_MAX_BYTES + 1, 0);
        assert_eq!(
            too_large_name(mk(big).validate()).as_deref(),
            Some("image_bytes")
        );
    }

    #[test]
    fn embedding_inputs_are_limited_per_text_and_in_total() {
        let mk = |inputs: Vec<String>| {
            Request::Embedding(EmbeddingParams {
                model_id: "m".into(),
                inputs,
                normalize: true,
                session_id: None,
                request_id: None,
            })
        };
        assert!(mk(vec!["x".repeat(limits::EMBED_MAX_CHARS_PER_INPUT)])
            .validate()
            .is_ok());
        assert_eq!(
            too_large_name(mk(vec!["x".repeat(limits::EMBED_MAX_CHARS_PER_INPUT + 1)]).validate())
                .as_deref(),
            Some("embedding_input_chars")
        );
        // 200 texts of 8000 bytes = 1.6 MB, each one fine on its own
        let many = vec!["x".repeat(limits::EMBED_MAX_CHARS_PER_INPUT); 200];
        assert_eq!(
            too_large_name(mk(many).validate()).as_deref(),
            Some("embedding_total_bytes")
        );
    }

    #[test]
    fn audio_is_limited_by_the_duration_in_its_wav_header_and_by_bytes() {
        let mk = |audio: Vec<u8>| {
            Request::SpeechToText(SpeechParams {
                model_id: "m".into(),
                audio,
                format: AudioFormat::Wav,
                language: None,
                session_id: None,
                request_id: None,
            })
        };
        assert!(mk(wav(3, None)).validate().is_ok());
        assert_eq!(
            wav_duration_seconds(&wav(3, None)).map(|d| d.round() as u32),
            Some(3)
        );
        // 601 s is 9.6 MB: under the byte limit, over the duration limit
        assert_eq!(
            too_large_name(mk(wav(601, None)).validate()).as_deref(),
            Some("audio_seconds")
        );
        // a header that claims 4 GB of data but carries 3 s is believed only for what is there
        assert!(mk(wav(3, Some(u32::MAX))).validate().is_ok());
        let mut huge = wav(1, None);
        huge.resize(limits::AUDIO_MAX_BYTES + 1, 0);
        assert_eq!(
            too_large_name(mk(huge).validate()).as_deref(),
            Some("audio_bytes")
        );
        assert!(matches!(
            mk(b"definitely not a wav file".to_vec()).validate(),
            Err(AiKitError::InvalidParams { .. })
        ));
    }
}
