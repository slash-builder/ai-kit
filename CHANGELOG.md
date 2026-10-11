# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Candle 0.9.2 -> **0.11** (verified: KV isolation A/B/A on CPU and CUDA, real-weights multimodal tests).
- `AiKitError::to_error_code()` is **deprecated**: its numbers (20-24, 13, 90) collide with the fabric `ErrorCode` space (a remote peer could trigger an ops alert). Use `to_inference_error_code()`.
- The router keeps the original typed error when a call fails (an out-of-memory error used to become `Internal`).

### Added
- Any kind of request: a general `Request`/`Output` contract for five request kinds (`RequestKind`: text, image generation, image understanding, embeddings, speech-to-text), `RequestService`, `TextService`,
  `OutputCheck`, and `AiKitError::UnsupportedKind`. `MultiHybridService` routes any of them by which tier has the kind (then the usual privacy/fit/policy rules) and counts bytes sent to the cloud
  (`RouteStats::cloud_payload_bytes`). `Plan::Unavailable { needs_shareable }`; new `TaskClass` values; `RequestKind`, `Request`, `Output` and `TaskClass` are `#[non_exhaustive]`.
- `RequiresShareable` error (a private request only the cloud could serve), `InferenceErrorCode` and `AiKitError::to_inference_error_code()` for the `hearth.inference.v1` wire contract (proposed, not ratified).
- `RoutedRequest::with_granted` (the boundary's grant verdict): an ungranted kind is refused with `CapabilityDenied` (now carries `request_kind`) before anything runs. Re-typing free text into an image request is **opt-in**
  (`MultiHybridService::with_retyping`) and cannot widen a grant.
- Real backends: `CandleImageService` (`sd_turbo` for development; **`z_image` = Z-Image-Turbo, Apache-2.0**, quantized GGUF transformer dequantized one block at a time, layer-streamed Qwen3 text encoder, tiled VAE; fits a 12 GB card, ~4.5 GB peak at 512x512)
  and `CandleEmbeddingService` (BERT sentence encoder). Multi-file catalog entries (`files:`, each BLAKE3-verified), `metadata.custom.kind`, and `custom.vram_peak_mb` for models that stream their weights.
- `ModelRegistry::default_for(kind, allow_noncommercial)`: the catalog default per kind; a non-commercial model is never chosen unless opted in (`--allow-noncommercial`). `ai-kit-cli imagine` and `route --image-model default` use it.
- Cargo features `text`, `image`, `embedding` (on `candle-base`); `candle` stays the umbrella. `backends/common.rs` holds the shared device, VRAM and verification code.
- A GPU out-of-memory error from an image backend is a typed `InferenceOom`, not `Internal`.
- `ai-kit-cli imagine`, `ai-kit-cli embed`; `route` now takes batch lines of any kind.
- Hybrid local/cloud routing (POC, not ruled on): `HybridInferenceService`, `Router`, `RoutePolicy` (YAML), `Privacy` (default `Private`, fail closed), `Verifier`, `RouteTrace`/`RouteStats`,
  a keyword `classify()`, and `ai-kit-cli route` (`--explain`, `--batch`, `--cloud-cmd`, `--policy`). Private content never reaches the cloud; raster images are delegated, not attempted.
- `chat_template: llama3` and `chatml` for the Candle backend.
- NVIDIA GPU (CUDA) support for the Candle backend, behind the opt-in `cuda` feature (off by default; not part of CI).
  `DevicePreference` (`cpu` | `cuda[:N]` | `auto`), `CandleInferenceService::with_device`, `device_of`, `describe_cuda_devices`.
  An explicit CUDA device that is missing, or a model larger than the GPU's free memory, is a hard error
  (`ModelLoadFailed` / `InferenceOom` before the file is read) — never a silent CPU fallback.
  `ai-kit-cli` gains `--device`, `devices` and `bench`; `scripts/cuda-userspace-toolkit.sh` builds a toolkit without root.
  On an RTX 3060 with Llama-3.2-3B: ~89 tok/s decode and ~1,600 tok/s prompt processing versus 2.3 and 3 on the CPU.
- `ClaudeProxyInferenceService` (`src/backends/claude_proxy.rs`), feature-gated
  behind `claude-proxy` (off by default): an `InferenceService` backend that
  proxies `infer()` to Anthropic's Messages API. **Experimental, v0.2+ scope
  per decision memos #2/#9 — not yet ruled on for a real household product.**
  Inert without an explicit `api_key`. See README "Cloud Proxy Inference —
  Claude/Anthropic" and the module's own doc comment for the full caveat.
- Initial Rust crate scaffold (v0.1.0-dev).
- `InferenceService` trait (frozen ABI for trait-based API).
- Core types: `ModelMetadata`, `InferenceParams`, `InferenceResponse`, `SessionId`.
- Error types mapping to Hearth `ErrorCode` enum.
- `ModelRegistry` (OS-shipped catalog, immutable).
- `ModelCache` (weights shared, RO mmap).
- `KvCacheManager` (per-session, privacy-isolated).
- Comprehensive module docs with design rationale.
- Unit tests for types, errors, registry, cache.

### Changed
- Candle backend: stop on every end-of-reply token the tokenizer has (`</s>`, `<|eot_id|>`, `<|end_of_text|>`, `<|im_end|>`, ...). It used to look up only `</s>` and fall back to token id 2
  (the plain `#` token for Llama-3), so replies ran on past `<|eot_id|>` into invented further turns. `load_model` now fails if the tokenizer has none of them.
- `CandleInferenceService` builds `ModelWeights` once in `load_model` and reuses it (one request at a time per model) instead of
  rebuilding from raw bytes on every `infer()`; request isolation relies on Candle resetting its KV cache at position 0 and is covered by a test.
  `load_model` on an already-loaded model is now a no-op.

### TODO
- Candle integration (awaiting framework binding).
- Message Kit transport (awaiting proto finalization).
- Storage Kit integration (awaiting storage-kit v2).
- Full catalog.yaml loader (YAML parsing).
- LRU eviction logic with TTL.
- Service Kit observability bridge.
- Score Kit capability checks.

## [v0.1.0] — Not Yet Released

Target: Q4 2026 (MVP local-only inference).

### Planned
- Core trait + in-process sync API.
- Lazy-load by default (minimize footprint).
- Candle + optional TFLite bindings.
- Model integrity verification (BLAKE3).
- VRAM queries via Device Kit.
- Latency metrics via Service Kit.
- Structured error handling (Hearth ErrorCode).

### Dependencies
- Awaiting DJ ruling on 10 open design questions (decision memos #1–#10).
- Awaiting Storage Kit v2 (model artifact persistence).
- Awaiting Message Kit proto finalization (async fallback RPC).
- Awaiting Score Kit (capability grants).
- Awaiting Service Kit (observability).
- Awaiting Spec Kit (conformance validation).

### Invariants (Locked)
1. Models are OS resources (immutable).
2. No weights over network (MVP).
3. Trait-based API (frozen).
4. KV cache isolation (per-session).
5. Audit trail immutable (always-on).
6. Score Kit gates capability (single source of truth).
7. Versions lock at install (no auto-upgrade).
8. Inference-only (no training/fine-tuning).
