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
    InferenceTimeout { model_id: String, timeout_ms: u32 },

    /// KV cache for session was evicted due to memory pressure or TTL.
    #[error("KV cache evicted for session {session_id}")]
    CacheEvicted { session_id: String },

    /// Requested quantization variant is not available locally.
    ///
    /// Per decision memo #7: App must handle fallback explicitly (Message Kit to Cumulus
    /// or accept lower quantization). Implicit downgrade violates app's accuracy SLA.
    #[error(
        "Quantization {quantization} not available for model {model_id}; available: {available:?}"
    )]
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

    /// No configured backend provides the requested request kind (or none the caller may use).
    #[error("Request kind '{request_kind}' is not available: {reason}")]
    UnsupportedKind {
        request_kind: String,
        reason: String,
    },

    /// A request exceeded a per-kind size limit (see `modality::limits`). Rejected up front, before anything is loaded or sent.
    #[error("Request too large: {limit_name} is {actual}, the limit is {limit}")]
    RequestTooLarge {
        limit_name: String,
        limit: u64,
        actual: u64,
    },

    /// The only tier that could serve this request is the cloud, and the content is private.
    /// The caller must mark the content shareable (and be allowed to) for it to run.
    #[error(
        "Request kind '{request_kind}' can only run in the cloud tier and the content is private"
    )]
    RequiresShareable { request_kind: String },

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
    CacheInconsistency { session_id: String, reason: String },

    /// RequestKind grant check failed or is missing.
    ///
    /// Note: Per decision memo #1, AI Kit trusts Hearth's capability check at call boundary.
    /// This error should only occur if Hearth fails to enforce the check (design violation).
    #[error("Request kind '{request_kind}' is not granted (app '{app_id}', model '{model_id}')")]
    CapabilityDenied {
        app_id: String,
        model_id: String,
        /// The request kind that was refused (`RequestKind::as_str()`), for the audit record.
        request_kind: String,
    },

    /// A remote proxy backend (e.g. the `claude-proxy` feature's Anthropic
    /// binding) failed: network/transport failure, a non-2xx HTTP response,
    /// or a response body that didn't match the expected shape.
    ///
    /// Distinct from [`AiKitError::Internal`] on purpose: this is an
    /// *external* dependency failing, not an AI Kit bug, and callers
    /// (Message Kit transport, dashboards) may want to treat the two
    /// differently (e.g. retry/backoff heuristics for the former only).
    #[error("Remote backend '{backend}' request failed: {reason}")]
    RemoteBackendError { backend: String, reason: String },

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

/// Error codes for the inference wire contract (`hearth.inference.v1`), proposed by messaging-architect (2026-10-10).
///
/// These are carried in an `InferenceError` payload, never in the fabric `RError`/`ErrorCode`: that space is the
/// fabric's, and the numbers the earlier design used (20-24) are already `TOPIC_INVALID`..`SUBSCRIBE_RATE_LIMITED` there.
/// Values are provisional until the proto is ratified; do not hard-code them elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[non_exhaustive]
pub enum InferenceErrorCode {
    Unspecified = 0,
    ModelNotFound = 1,
    Timeout = 2,
    CacheEvicted = 3,
    QuantizationUnsupported = 4,
    Oom = 5,
    InvalidParams = 6,
    /// No tier on this device can do the requested kind of work.
    RequestKindUnsupported = 7,
    /// Only the cloud could do it and the content is private.
    RequiresShareable = 8,
    VerificationFailed = 9,
    RemoteBackendFailed = 10,
    Cancelled = 11,
    ContentUnavailable = 12,
    /// Integrity or load failure. The message must not include hashes.
    ModelUnavailable = 13,
    Internal = 15,
    /// A request exceeded a per-kind size limit (`{limit_name, limit}` belong in the details). Provisional.
    RequestTooLarge = 16,
}

impl AiKitError {
    /// The wire error code for this error (see [`InferenceErrorCode`]).
    ///
    /// `CapabilityDenied` is a Score Kit grant failure that should be enforced at the Message Kit boundary and never
    /// reach inference; if it does, it is reported as `Internal` so it raises an alert, not as a permissions answer.
    pub fn to_inference_error_code(&self) -> InferenceErrorCode {
        use InferenceErrorCode as C;
        match self {
            AiKitError::ModelNotFound { .. } => C::ModelNotFound,
            AiKitError::InferenceTimeout { .. } => C::Timeout,
            AiKitError::CacheEvicted { .. } => C::CacheEvicted,
            AiKitError::QuantizationUnsupported { .. } => C::QuantizationUnsupported,
            AiKitError::InferenceOom { .. } => C::Oom,
            AiKitError::InvalidParams { .. } => C::InvalidParams,
            AiKitError::UnsupportedKind { .. } => C::RequestKindUnsupported,
            AiKitError::RequiresShareable { .. } => C::RequiresShareable,
            AiKitError::RequestTooLarge { .. } => C::RequestTooLarge,
            AiKitError::RemoteBackendError { .. } => C::RemoteBackendFailed,
            AiKitError::ModelLoadFailed { .. } | AiKitError::IntegrityCheckFailed { .. } => {
                C::ModelUnavailable
            }
            _ => C::Internal,
        }
    }

    /// Map to a fabric `ErrorCode`.
    ///
    /// # Deprecated: this mapping is wrong
    /// The fabric `ErrorCode` numbers 20-24 are `TOPIC_INVALID`, `TOPIC_UNAUTHORIZED`, `SUBSCRIPTION_NOT_FOUND`,
    /// `PUBLISH_RATE_LIMITED` and `SUBSCRIBE_RATE_LIMITED`; 13 is `ENVELOPE_TOO_LARGE`; 90 is the ops-alert `INTERNAL`. So
    /// a remote peer could trigger an ops alert, and a grant failure looked like an oversized message. Kept only so existing
    /// callers compile; use [`AiKitError::to_inference_error_code`].
    #[deprecated(note = "collides with the fabric ErrorCode space; use to_inference_error_code")]
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
    fn wire_codes_cover_every_router_outcome() {
        use InferenceErrorCode as C;
        let cases = [
            (
                AiKitError::ModelNotFound {
                    model_id: "m".into(),
                },
                C::ModelNotFound,
            ),
            (
                AiKitError::InferenceOom {
                    model_id: "m".into(),
                    required_mb: 2,
                    available_mb: 1,
                },
                C::Oom,
            ),
            (
                AiKitError::InvalidParams { reason: "r".into() },
                C::InvalidParams,
            ),
            (
                AiKitError::UnsupportedKind {
                    request_kind: "image_generation".into(),
                    reason: "r".into(),
                },
                C::RequestKindUnsupported,
            ),
            (
                AiKitError::RequestTooLarge {
                    limit_name: "image_pixels".into(),
                    limit: 1,
                    actual: 2,
                },
                C::RequestTooLarge,
            ),
            (
                AiKitError::RequiresShareable {
                    request_kind: "image_understanding".into(),
                },
                C::RequiresShareable,
            ),
            (
                AiKitError::RemoteBackendError {
                    backend: "b".into(),
                    reason: "r".into(),
                },
                C::RemoteBackendFailed,
            ),
            (
                AiKitError::ModelLoadFailed {
                    model_id: "m".into(),
                    reason: "r".into(),
                },
                C::ModelUnavailable,
            ),
            (
                AiKitError::IntegrityCheckFailed {
                    model_id: "m".into(),
                    expected: "a".into(),
                    actual: "b".into(),
                },
                C::ModelUnavailable,
            ),
            (AiKitError::Internal { reason: "r".into() }, C::Internal),
        ];
        for (err, want) in cases {
            assert_eq!(err.to_inference_error_code(), want, "{err}");
        }
    }

    #[test]
    fn a_grant_failure_is_reported_as_internal_not_as_a_permissions_answer() {
        let e = AiKitError::CapabilityDenied {
            app_id: "a".into(),
            model_id: "m".into(),
            request_kind: "image_generation".into(),
        };
        assert_eq!(e.to_inference_error_code(), InferenceErrorCode::Internal);
    }

    #[test]
    #[allow(deprecated)]
    fn the_deprecated_fabric_mapping_is_still_the_old_one() {
        // Documented as wrong; pinned so nobody "fixes" it silently and breaks a caller that already depends on it.
        let e = AiKitError::CapabilityDenied {
            app_id: "a".into(),
            model_id: "m".into(),
            request_kind: "image_generation".into(),
        };
        assert_eq!(e.to_error_code(), 13);
    }
}
