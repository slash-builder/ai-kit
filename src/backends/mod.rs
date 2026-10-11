//! Real inference backends implementing [`crate::InferenceService`].
//!
//! - [`candle::CandleInferenceService`] (POC scope), feature-gated behind
//!   `candle` so the default build/CI never touches it. See `README.md`'s
//!   "Local Dev Inference (POC)" section and `AGENT.md`'s "Implemented (POC
//!   scope)" entry for what this does and doesn't prove. Runs on the CPU by
//!   default; NVIDIA GPU is the opt-in `cuda` feature (see `README.md`'s
//!   "GPU (CUDA) — POC").
//! - [`claude_proxy::ClaudeProxyInferenceService`] (experimental, v0.2+
//!   scope, feature-gated behind `claude-proxy`, off by default). See that
//!   module's doc comment for the decision-memo #2/#9 open-ruling caveat
//!   before using this beyond a dev POC.

#[cfg(feature = "text")]
pub mod candle;
#[cfg(feature = "candle-base")]
pub mod common;
#[cfg(feature = "image")]
pub mod diffusion;
#[cfg(feature = "embedding")]
pub mod embedding;
#[cfg(feature = "image")]
pub(crate) mod zimage;
#[cfg(feature = "image")]
pub(crate) mod zimage_text;

#[cfg(feature = "text")]
pub use candle::CandleInferenceService;
#[cfg(feature = "candle-base")]
pub use common::{describe_cuda_devices, DevicePreference};
#[cfg(feature = "image")]
pub use diffusion::CandleImageService;
#[cfg(feature = "embedding")]
pub use embedding::CandleEmbeddingService;

#[cfg(feature = "claude-proxy")]
pub mod claude_proxy;

#[cfg(feature = "claude-proxy")]
pub use claude_proxy::{ClaudeProxyConfig, ClaudeProxyInferenceService};
