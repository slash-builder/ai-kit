//! Real inference backends implementing [`crate::InferenceService`].
//!
//! - [`candle::CandleInferenceService`] (POC scope), feature-gated behind
//!   `candle` so the default build/CI never touches it. See `README.md`'s
//!   "Local Dev Inference (POC)" section and `AGENT.md`'s "Implemented (POC
//!   scope)" entry for what this does and doesn't prove.
//! - [`claude_proxy::ClaudeProxyInferenceService`] (experimental, v0.2+
//!   scope, feature-gated behind `claude-proxy`, off by default). See that
//!   module's doc comment for the decision-memo #2/#9 open-ruling caveat
//!   before using this beyond a dev POC.

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub use candle::CandleInferenceService;

#[cfg(feature = "claude-proxy")]
pub mod claude_proxy;

#[cfg(feature = "claude-proxy")]
pub use claude_proxy::{ClaudeProxyConfig, ClaudeProxyInferenceService};
