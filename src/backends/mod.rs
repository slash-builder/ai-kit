//! Real inference backends implementing [`crate::InferenceService`].
//!
//! Currently just [`candle::CandleInferenceService`] (POC scope), feature-gated
//! behind `candle` so the default build/CI never touches it. See
//! `README.md`'s "Local Dev Inference (POC)" section and `AGENT.md`'s
//! "Implemented (POC scope)" entry for what this does and doesn't prove.

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "candle")]
pub use candle::CandleInferenceService;
