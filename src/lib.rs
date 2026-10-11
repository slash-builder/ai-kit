//! AI Kit — Trait-based local inference orchestration for Hearth household clusters.
//!
//! # Overview
//!
//! AI Kit is a hybrid tier kit providing inference orchestration across Hearth devices.
//! Models are immutable OS resources (shipped signed, verified at boot). Inference is
//! household-scoped with capability gating via Score Kit and observability via Service Kit.
//!
//! # Core Design Principles
//!
//! 1. **Models are OS resources** — shipped signed in the image, immutable post-deploy.
//! 2. **No weights over network (MVP)** — local-only; cloud fallback is v0.2+ feature.
//! 3. **Trait-based API** — dual-layer (Rust trait + Message Kit binding).
//! 4. **KV cache isolation** — per-session, never reused cross-app.
//! 5. **Audit trail immutable** — kernel audit daemon or separate store.
//! 6. **Score Kit gates capability** — authorization at call boundary.
//! 7. **Versions lock at install** — no auto-upgrade; rebuild required.
//! 8. **Inference-only** — no training, fine-tuning, or backprop.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │  Household Hearth Cluster                               │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                           │
//! │  App (Pi5, embedded)                                    │
//! │  ├─ Trait call: ai_kit::infer("llama2-7b", prompt)  │
//! │  └─ In-process, RO mmap, 1–10ms                        │
//! │                                                           │
//! │  App (Cumulus, always-on)                              │
//! │  ├─ Trait call: same API                               │
//! │  └─ Or Message Kit RPC for load distribution           │
//! │                                                           │
//! │  ┌────────────────────────────────────────────┐        │
//! │  │  AI Kit Service (Trait + Message Kit)      │        │
//! │  ├────────────────────────────────────────────┤        │
//! │  │ • Model Registry (catalog.yaml, YAML)      │        │
//! │  │ • Inference Engine (trait-based)           │        │
//! │  │ • Model Cache (RO mmap, per-model)         │        │
//! │  │ • KV Cache (per-session, LRU)              │        │
//! │  │ • Message Kit Binding (async fallback)     │        │
//! │  └────────────────────────────────────────────┘        │
//! │         ↓           ↓            ↓          ↓           │
//! │         │           │            │          │           │
//! │      ┌──┴───┐  ┌────┴──┐  ┌─────┴──┐  ┌────┴────┐     │
//! │      │      │  │       │  │        │  │         │      │
//! │   Storage Kit Score Kit Service Kit Spec Kit Device Kit │
//! │   (models)  (caps, rate) (metrics) (conformance) (vRAM) │
//! │                                                           │
//! │  /usr/share/models/ (OS-shipped, RO, signed)           │
//! │  ├─ llama2-7b-q4/ (weights.bin, meta.json)             │
//! │  ├─ whisper-tiny/ (model.bin, meta.json)               │
//! │  └─ ...                                                  │
//! │                                                           │
//! │  Audit Trail (kernel audit daemon or Service Kit)      │
//! │  ├─ Model name, device, latency, app_id, timestamp     │
//! │  └─ Never: plaintext prompts, outputs                   │
//! │                                                           │
//! └─────────────────────────────────────────────────────────┘
//! ```

#[cfg(any(feature = "candle-base", feature = "claude-proxy"))]
pub mod backends;
pub mod budget;
pub mod cache;
pub mod credentials;
pub mod egress;
pub mod error;
pub mod hybrid;
pub mod modality;
pub mod model_card;
pub mod provider;
#[cfg(feature = "provider-anthropic")]
pub mod providers;
pub mod registry;
pub mod routed;
pub mod service;
pub mod types;

// Re-export key types at crate root
#[cfg(feature = "embedding")]
pub use backends::CandleEmbeddingService;
#[cfg(feature = "image")]
pub use backends::CandleImageService;
#[cfg(feature = "text")]
pub use backends::CandleInferenceService;
#[cfg(feature = "candle-base")]
pub use backends::{describe_cuda_devices, DevicePreference};
#[cfg(feature = "claude-proxy")]
pub use backends::{ClaudeProxyConfig, ClaudeProxyInferenceService};
pub use error::{AiKitError, Result};
pub use hybrid::{
    classify, ClassPolicy, Decision, HybridInferenceService, Plan, Privacy, RoutePolicy,
    RouteRequest, RouteStats, RouteTrace, Router, TaskClass, Tier, Verifier,
};
pub use modality::{
    AudioFormat, EmbeddingParams, EmbeddingResponse, GeneratedImage, ImageGenParams,
    ImageUnderstandingParams, Output, OutputCheck, Request, RequestKind, RequestService,
    SpeechParams, TextService, TranscriptResponse,
};
pub use routed::{MultiHybridService, RoutedRequest};
pub use service::InferenceService;
pub use types::{InferenceParams, InferenceResponse, ModelFile, ModelMetadata};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
