//! Core inference service trait.
//!
//! # Design Notes
//!
//! The `InferenceService` trait is the primary public API. It is:
//! - Framework-agnostic (does not expose Candle, TFLite internals).
//! - Async-first (supports both in-process and Message Kit binding).
//! - Frozen ABI (conforms to Spec Kit, versioned with semantic versioning).
//!
//! # Capability Gating (Decision Memo #1)
//!
//! Per decision memo #1, AI Kit trusts Hearth's capability check at call boundary.
//! When an app calls `infer()`, the Hearth runtime has already verified the capability grant.
//! AI Kit does NOT call Score Kit directly; it trusts the transport layer.
//!
//! TODO: Once DJ rules, confirm whether AI Kit should add redundant local gating
//! (unlikely, but security audit may require it).

use crate::error::Result;
use crate::types::{CacheStats, InferenceParams, InferenceResponse, SessionId};
use async_trait::async_trait;

/// Primary inference service trait (frozen ABI).
///
/// All inference goes through this trait; no framework-specific APIs exposed to products.
/// Conforms to Spec Kit conformance gates.
///
/// # Examples
///
/// ```no_run
/// # use ai_kit::InferenceService;
/// # use ai_kit::types::{InferenceParams, SessionId};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let service: Box<dyn InferenceService> = todo!();
/// let params = InferenceParams {
///     model_id: "llama2-7b".to_string(),
///     prompt: "Summarize this article: [...]".to_string(),
///     temperature: 0.7,
///     top_p: 0.95,
///     max_tokens: 200,
///     session_id: None,
///     preferred_quantization: None,
///     request_id: None,
/// };
///
/// let response = service.infer(params).await?;
/// println!("Completion: {}", response.completion);
/// # Ok(())
/// # }
/// ```
///
/// # Thread Safety
///
/// Implementations must be Send + Sync for use in async contexts across multiple threads.
#[async_trait]
pub trait InferenceService: Send + Sync {
    /// Perform inference on a prompt.
    ///
    /// # Semantics
    ///
    /// - Validates `params` (model_id, prompt, temperature, etc).
    /// - Looks up model in registry (OS-shipped catalog).
    /// - Checks model availability (not unloaded, quantization available).
    /// - Performs inference in-process or delegates to Cumulus (remote fallback).
    /// - Returns final completion + metadata.
    ///
    /// # Errors
    ///
    /// - `ModelNotFound`: model_id not in registry.
    /// - `QuantizationUnsupported`: requested quantization unavailable (per decision memo #7).
    /// - `InferenceTimeout`: took longer than SLA (from Score Kit grant).
    /// - `InferenceOom`: model too large for device.
    /// - `InvalidParams`: validation failed.
    /// - `CapabilityDenied`: Hearth capability check failed (design violation if this occurs).
    ///
    /// # TODO: Open Questions
    ///
    /// - Decision memo #2 (remote fallback scope): Should fallback include Quickring cloud hub
    ///   or only Cumulus? MVP is Cumulus-only; cloud is v0.2+.
    /// - Decision memo #7 (quantization fallback): Auto-downgrade, fail, or fallback?
    ///   Current design: fail hard (force explicit fallback).
    async fn infer(&self, params: InferenceParams) -> Result<InferenceResponse>;

    /// Load a model into memory (eager).
    ///
    /// # Semantics
    ///
    /// - Finds model in OS catalog (/usr/share/models/...).
    /// - Verifies BLAKE3 hash.
    /// - Memory-maps weights (RO) from disk.
    /// - Indexes in model_cache for reuse across apps.
    ///
    /// # Errors
    ///
    /// - `ModelNotFound`: not in catalog.
    /// - `IntegrityCheckFailed`: BLAKE3 mismatch.
    /// - `ModelLoadFailed`: corrupt file, missing permissions, etc.
    /// - `InferenceOom`: model too large for device.
    ///
    /// # TODO: Decision Memo #3 (pre-load strategy)
    ///
    /// Default is lazy-load (minimize footprint). Pre-load is opt-in via config.
    /// Once DJ rules, may add `preload_all()` method for Cumulus always-on mode.
    async fn load_model(&self, model_id: &str) -> Result<()>;

    /// Unload a model from memory (eager).
    ///
    /// # Semantics
    ///
    /// - Removes from model_cache if no active inferences.
    /// - Returns error if model has active sessions (blocking unload).
    /// - Decrements ref_count on LoadedModel.
    ///
    /// # Errors
    ///
    /// - `ModelNotFound`: not loaded.
    /// - Other: internal inconsistency (rare).
    async fn unload_model(&self, model_id: &str) -> Result<()>;

    /// Query cache statistics (for observability).
    ///
    /// # Semantics
    ///
    /// - Returns snapshot of current cache state (models loaded, sessions active, etc).
    /// - No side effects.
    /// - Used by Service Kit + monitoring dashboards.
    ///
    /// # TODO: Service Kit Integration
    ///
    /// Once service-kit is ready, these metrics flow to structured logging + observability.
    async fn cache_stats(&self) -> Result<CacheStats>;

    /// Terminate a session (eagerly clean up KV cache).
    ///
    /// # Semantics
    ///
    /// - Removes KV cache for session_id.
    /// - No-op if session doesn't exist.
    /// - Called by app or by LRU eviction logic.
    ///
    /// # Errors
    ///
    /// - Rare; mostly a no-op.
    async fn end_session(&self, session_id: &SessionId) -> Result<()>;
}

/// Default stub implementation (for testing, before framework integration).
///
/// This is a no-op placeholder. Real implementations will bind to Candle, TFLite, etc.
#[derive(Debug, Clone)]
pub struct DefaultInferenceService;

#[async_trait]
impl InferenceService for DefaultInferenceService {
    async fn infer(&self, params: InferenceParams) -> Result<InferenceResponse> {
        params.validate()?;
        Err(crate::error::AiKitError::Internal {
            reason: "DefaultInferenceService is a stub; implement with Candle or TFLite bindings."
                .to_string(),
        })
    }

    async fn load_model(&self, _model_id: &str) -> Result<()> {
        Err(crate::error::AiKitError::Internal {
            reason: "DefaultInferenceService is a stub.".to_string(),
        })
    }

    async fn unload_model(&self, _model_id: &str) -> Result<()> {
        Err(crate::error::AiKitError::Internal {
            reason: "DefaultInferenceService is a stub.".to_string(),
        })
    }

    async fn cache_stats(&self) -> Result<CacheStats> {
        Err(crate::error::AiKitError::Internal {
            reason: "DefaultInferenceService is a stub.".to_string(),
        })
    }

    async fn end_session(&self, _session_id: &SessionId) -> Result<()> {
        Err(crate::error::AiKitError::Internal {
            reason: "DefaultInferenceService is a stub.".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_default_service_is_stub() {
        let service = DefaultInferenceService;
        let params = InferenceParams {
            model_id: "llama2-7b".to_string(),
            prompt: "Hello".to_string(),
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 256,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        };

        let result = service.infer(params).await;
        assert!(result.is_err());
    }
}
