//! Error types and Result type for AI Kit.
//!
//! # Design Notes
//!
//! AI Kit errors map cleanly to Hearth ErrorCode enum for protocol-level errors.
//! See: https://github.com/slash-builder/message-kit/blob/main/proto/quickring/v1/message.proto

use thiserror::Error;

/// Result type for AI Kit operations.
pub type Result<T> = std::result::Result<T, AiKitError>;

/// AI Kit error types.
///
/// Maps to `quickring.v1.ErrorCode` enum in Message Kit protocol:
/// - `MODEL_NOT_FOUND = 20`
/// - `INFERENCE_TIMEOUT = 21`
/// - `CACHE_EVICTED = 22`
/// - `QUANTIZATION_UNSUPPORTED = 23`
/// - `INFERENCE_OOM = 24`
#[derive(Debug, Error)]
pub enum AiKitError {
    /// Model requested but not found in registry or OS image.
    #[error("Model not found: {model_id}")]
    ModelNotFound { model_id: String },

    /// Inference operation timed out.
    #[error("Inference timeout on model {model_id} after {timeout_ms}ms")]
    InferenceTimeout {
        model_id: String,
        timeout_ms: u32,
    },

    /// KV cache for session was evicted due to memory pressure or TTL.
    #[error("KV cache evicted for session {session_id}")]
    CacheEvicted { session_id: String },

    /// Requested quantization variant is not available locally.
    ///
    /// Per decision memo #7: App must handle fallback explicitly (Message Kit to Cumulus
    /// or accept lower quantization). Implicit downgrade violates app's accuracy SLA.
    #[error("Quantization {quantization} not available for model {model_id}; available: {available:?}")]
    QuantizationUnsupported {
        model_id: String,
        quantization: String,
        available: Vec<String>,
    },

    /// Inference out of memory (model too large for device, or cache exhausted).
    #[error("Out of memory inferencing on model {model_id} (required: {required_mb}MB, available: {available_mb}MB)")]
    InferenceOom {
        model_id: String,
        required_mb: u32,
        available_mb: u32,
    },

    /// Model integrity verification failed (BLAKE3 hash mismatch).
    #[error("Model integrity check failed for {model_id}: expected hash {expected}, got {actual}")]
    IntegrityCheckFailed {
        model_id: String,
        expected: String,
        actual: String,
    },

    /// Model could not be loaded into memory (corrupt file, missing permissions, etc).
    #[error("Failed to load model {model_id}: {reason}")]
    ModelLoadFailed { model_id: String, reason: String },

    /// Invalid inference parameters (e.g., negative temperature, max_tokens = 0).
    #[error("Invalid inference parameters: {reason}")]
    InvalidParams { reason: String },

    /// Session cache state is inconsistent (internal error).
    #[error("Session cache inconsistency for {session_id}: {reason}")]
    CacheInconsistency {
        session_id: String,
        reason: String,
    },

    /// Capability grant check failed or is missing.
    ///
    /// Note: Per decision memo #1, AI Kit trusts Hearth's capability check at call boundary.
    /// This error should only occur if Hearth fails to enforce the check (design violation).
    #[error("Capability check failed for app {app_id} on model {model_id}")]
    CapabilityDenied { app_id: String, model_id: String },

    /// Generic internal error.
    #[error("Internal error: {reason}")]
    Internal { reason: String },

    /// IO error (file access, etc).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// YAML parsing error.
    #[error("YAML parse error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    /// JSON parsing error.
    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl AiKitError {
    /// Map AI Kit error to approximate Hearth ErrorCode (for Message Kit transport).
    ///
    /// # Mapping
    ///
    /// - `ModelNotFound` → 20 (MODEL_NOT_FOUND)
    /// - `InferenceTimeout` → 21 (INFERENCE_TIMEOUT)
    /// - `CacheEvicted` → 22 (CACHE_EVICTED)
    /// - `QuantizationUnsupported` → 23 (QUANTIZATION_UNSUPPORTED)
    /// - `InferenceOom` → 24 (INFERENCE_OOM)
    /// - `CapabilityDenied` → 13 (PERMISSION_DENIED)
    /// - others → 90 (INTERNAL)
    ///
    /// TODO: Once messaging-architect confirms the error code allocations, pin these values.
    pub fn to_error_code(&self) -> i32 {
        match self {
            AiKitError::ModelNotFound { .. } => 20,
            AiKitError::InferenceTimeout { .. } => 21,
            AiKitError::CacheEvicted { .. } => 22,
            AiKitError::QuantizationUnsupported { .. } => 23,
            AiKitError::InferenceOom { .. } => 24,
            AiKitError::CapabilityDenied { .. } => 13,
            _ => 90,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_mapping() {
        let err = AiKitError::ModelNotFound {
            model_id: "llama2-7b".to_string(),
        };
        assert_eq!(err.to_error_code(), 20);

        let err = AiKitError::CapabilityDenied {
            app_id: "app123".to_string(),
            model_id: "llama2-7b".to_string(),
        };
        assert_eq!(err.to_error_code(), 13);
    }
}
