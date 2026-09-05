# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
