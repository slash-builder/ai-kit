//! Core types for AI Kit.
//!
//! # Design Notes
//!
//! - `ModelMetadata`: OS-shipped model manifest (immutable, part of image signature).
//! - `InferenceParams`: Caller-provided parameters (temperature, top_p, max_tokens).
//! - `InferenceResponse`: Completed inference (single message or streaming chunks).
//! - `SessionId`: Per-session KV cache key (never reused cross-app by design).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for an inference model.
pub type ModelId = String;

/// BLAKE3 hash of model weights (content address).
pub type ModelHash = String;

/// Unique identifier for an app (from Hearth capability system).
pub type AppId = String;

/// Unique identifier for an inference session (for KV cache locality).
#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

impl SessionId {
    /// Generate a new random session ID.
    pub fn new() -> Self {
        SessionId(Uuid::new_v4().to_string())
    }

    /// Create a session ID from a string.
    pub fn from_string(s: String) -> Self {
        SessionId(s)
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantization variant of a model (e.g., "q4", "q8", "fp16").
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum Quantization {
    /// 4-bit quantization (typical for 7B–13B models on consumer hardware).
    Q4,
    /// 8-bit quantization (higher quality, more VRAM).
    Q8,
    /// Full float16 (highest quality, most VRAM).
    FP16,
    /// Custom quantization (unspecified format).
    Custom(String),
}

impl Quantization {
    pub fn as_str(&self) -> &str {
        match self {
            Quantization::Q4 => "q4",
            Quantization::Q8 => "q8",
            Quantization::FP16 => "fp16",
            Quantization::Custom(s) => s,
        }
    }
}

impl std::fmt::Display for Quantization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Model metadata (from OS catalog, immutable).
///
/// # Examples
///
/// ```yaml
/// id: "llama2-7b"
/// version: "2.0.0"
/// os_arch: ["x86_64", "aarch64"]
/// quantization_variants: ["q4", "q8", "fp16"]
/// default_quantization: "q4"
/// hash: "blake3-hash-of-weights"
/// size_bytes: 3_865_470_976
/// location: "file:///usr/share/models/llama2-7b-q4.bin"
/// metadata:
///   parameters: 7_000_000_000
///   context_window: 4096
///   vram_required_mb: 4096
///   inference_latency_ms: 45
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier (e.g., "llama2-7b").
    pub id: ModelId,

    /// Semantic version (e.g., "2.0.0").
    pub version: String,

    /// Supported architectures (e.g., ["x86_64", "aarch64"]).
    pub os_arch: Vec<String>,

    /// Available quantization variants (e.g., ["q4", "q8", "fp16"]).
    pub quantization_variants: Vec<String>,

    /// Default quantization if not specified (usually "q4").
    pub default_quantization: String,

    /// BLAKE3 hash of weights (content address).
    pub hash: ModelHash,

    /// Size in bytes.
    pub size_bytes: u64,

    /// File location (e.g., "file:///usr/share/models/llama2-7b-q4.bin").
    pub location: String,

    /// Extended metadata.
    pub metadata: ModelMetadataExtended,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadataExtended {
    /// Number of model parameters.
    pub parameters: u64,

    /// Maximum context window (tokens).
    pub context_window: u32,

    /// VRAM required in MB (for default quantization).
    pub vram_required_mb: u32,

    /// Typical inference latency in ms (1x GPU, batch=1).
    pub inference_latency_ms: u32,

    /// Framework (e.g., "candle", "tflite").
    #[serde(default)]
    pub framework: String,

    /// Custom metadata (key-value pairs).
    #[serde(default)]
    pub custom: HashMap<String, String>,
}

/// Parameters for an inference call.
///
/// # Design Notes
///
/// - `model_id`: Model to use (looked up in registry).
/// - `prompt`: Input text or tokens.
/// - `temperature`: Sampling randomness (0.0 = greedy, 1.0 = high randomness).
/// - `top_p`: Nucleus sampling threshold.
/// - `max_tokens`: Maximum completion length.
/// - `session_id`: For KV cache locality (never reused cross-app).
///
/// TODO #3 (decision memo): Lazy-load is default (minimize footprint);
/// pre-load is opt-in via config for Cumulus/always-on appliances.
/// Once DJ rules, add `prefer_preload: bool` field.
///
/// TODO #7 (decision memo): Quantization fallback strategy.
/// Add `acceptable_quantizations: Vec<Quantization>` field for app to specify fallback chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceParams {
    /// Model to use.
    pub model_id: ModelId,

    /// Input prompt.
    pub prompt: String,

    /// Sampling temperature (0.0–2.0).
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Nucleus sampling (0.0–1.0).
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Maximum completion tokens.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Session ID for KV cache locality.
    /// If not provided, a new session is created (per-inference isolation).
    pub session_id: Option<SessionId>,

    /// Optional: preferred quantization (otherwise use model default).
    pub preferred_quantization: Option<String>,

    /// Optional: request ID for correlation (generated if not provided).
    pub request_id: Option<String>,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.95
}

fn default_max_tokens() -> u32 {
    256
}

impl InferenceParams {
    /// Validate parameters.
    pub fn validate(&self) -> crate::Result<()> {
        use crate::error::AiKitError;

        if self.model_id.is_empty() {
            return Err(AiKitError::InvalidParams {
                reason: "model_id cannot be empty".to_string(),
            });
        }

        if self.prompt.is_empty() {
            return Err(AiKitError::InvalidParams {
                reason: "prompt cannot be empty".to_string(),
            });
        }

        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(AiKitError::InvalidParams {
                reason: format!(
                    "temperature must be in range [0.0, 2.0], got {}",
                    self.temperature
                ),
            });
        }

        if !(0.0..=1.0).contains(&self.top_p) {
            return Err(AiKitError::InvalidParams {
                reason: format!("top_p must be in range [0.0, 1.0], got {}", self.top_p),
            });
        }

        if self.max_tokens == 0 {
            return Err(AiKitError::InvalidParams {
                reason: "max_tokens must be > 0".to_string(),
            });
        }

        Ok(())
    }
}

/// Completed inference response.
///
/// # Design Notes
///
/// - Contains final completion + metadata.
/// - Never contains plaintext prompts or intermediate activations (audit trail policy).
/// - Structured errors map to ErrorCode enum (for Message Kit transport).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    /// Completed text.
    pub completion: String,

    /// Total tokens in response (prompt + completion).
    pub total_tokens: u32,

    /// Tokens in prompt.
    pub prompt_tokens: u32,

    /// Tokens in completion.
    pub completion_tokens: u32,

    /// Total inference time in milliseconds.
    pub inference_ms: u32,

    /// Model version used (e.g., "2.0.0").
    pub model_version: String,

    /// Quantization used.
    pub quantization: String,

    /// Session ID (for cache correlation).
    pub session_id: SessionId,

    /// Request ID (for correlation).
    pub request_id: String,
}

/// Statistics for inference cache.
///
/// TODO: Once Service Kit integration is ready, these metrics flow to observability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of models currently loaded.
    pub models_loaded: u32,

    /// Total size of loaded models in bytes.
    pub models_size_bytes: u64,

    /// Number of active sessions.
    pub sessions_active: u32,

    /// Total KV cache size in bytes.
    pub cache_size_bytes: u64,

    /// Cache hit rate (0.0–1.0).
    pub cache_hit_rate: f32,

    /// Timestamp (ISO 8601).
    pub timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_generation() {
        let s1 = SessionId::new();
        let s2 = SessionId::new();
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_quantization_display() {
        assert_eq!(Quantization::Q4.as_str(), "q4");
        assert_eq!(Quantization::Q4.to_string(), "q4");
    }

    #[test]
    fn test_params_validation() {
        let params = InferenceParams {
            model_id: "llama2-7b".to_string(),
            prompt: "Hello, world!".to_string(),
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 256,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        };

        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_params_validation_invalid_temperature() {
        let params = InferenceParams {
            model_id: "llama2-7b".to_string(),
            prompt: "Hello, world!".to_string(),
            temperature: 3.0, // Invalid
            top_p: 0.95,
            max_tokens: 256,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        };

        assert!(params.validate().is_err());
    }
}
