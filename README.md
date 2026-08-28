# AI Kit

**Trait-based local inference orchestration for Hearth household clusters.**

AI Kit provides a framework-agnostic inference API for Hearth devices. Models are immutable OS resources (shipped signed, verified at boot). Inference is household-scoped with capability gating via Score Kit and observability via Service Kit.

## Status

**MVP (v0.1)** — Design complete, implementation in progress.

- ✅ Unified design doc (`context/projects/ai-kit-unified-design.md`)
- ✅ Decision memos for DJ ruling (`context/ai-kit-decision-memos.md`)
- 🚧 Core trait definition (this repo)
- ⏳ Candle integration (blocked on framework binding)
- ⏳ Message Kit transport (blocked on proto finalization)
- ⏳ Storage Kit integration (blocked on storage-kit v2)

## Design Principles

1. **Models are OS resources** — shipped signed in the image, immutable post-deploy.
2. **No weights over network (MVP)** — local-only; cloud fallback is v0.2+ feature.
3. **Trait-based API** — dual-layer (Rust trait + Message Kit binding).
4. **KV cache isolation** — per-session, never reused cross-app.
5. **Audit trail immutable** — kernel audit daemon or separate store.
6. **Score Kit gates capability** — authorization at call boundary.
7. **Versions lock at install** — no auto-upgrade; rebuild required.
8. **Inference-only** — no training, fine-tuning, or backprop.

## Quick Start

```rust
use ai_kit::{InferenceService, InferenceParams};

let service: Box<dyn InferenceService> = todo!(); // Bind to Candle/TFLite

let params = InferenceParams {
    model_id: "llama2-7b".to_string(),
    prompt: "Summarize this: [...]".to_string(),
    temperature: 0.7,
    top_p: 0.95,
    max_tokens: 200,
    session_id: None,
    preferred_quantization: None,
    request_id: None,
};

let response = service.infer(params).await?;
println!("Completion: {}", response.completion);
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Household Hearth Cluster                               │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  App (Pi5, embedded)                                    │
│  ├─ Trait call: ai_kit::infer("llama2-7b", prompt)  │
│  └─ In-process, RO mmap, 1–10ms                        │
│                                                           │
│  ┌────────────────────────────────────────────┐        │
│  │  AI Kit Service (Trait + Message Kit)      │        │
│  ├────────────────────────────────────────────┤        │
│  │ • Model Registry (catalog.yaml, YAML)      │        │
│  │ • Inference Engine (trait-based)           │        │
│  │ • Model Cache (RO mmap, per-model)         │        │
│  │ • KV Cache (per-session, LRU)              │        │
│  │ • Message Kit Binding (async fallback)     │        │
│  └────────────────────────────────────────────┘        │
│         ↓           ↓            ↓          ↓           │
│         │           │            │          │           │
│      ┌──┴───┐  ┌────┴──┐  ┌─────┴──┐  ┌────┴────┐     │
│      │      │  │       │  │        │  │         │      │
│   Storage Kit Score Kit Service Kit Spec Kit Device Kit │
│   (models)  (caps, rate) (metrics) (conformance) (vRAM) │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## Modules

- **`types.rs`**: Core data types (ModelMetadata, InferenceParams, SessionId, etc).
- **`error.rs`**: Error types (map to Hearth ErrorCode).
- **`service.rs`**: `InferenceService` trait (frozen ABI).
- **`registry.rs`**: Model discovery from OS catalog.
- **`cache.rs`**: Model cache (RO mmap) + KV cache (per-session).

## Decision Memos & Open Questions

AI Kit v0.1 has 10 open design questions pending DJ's ruling:

1. **Capability gating**: Direct Score Kit call or trust Hearth boundary?
2. **Remote fallback scope**: Cumulus-only (MVP) or cloud hub?
3. **Pre-load strategy**: Lazy-load default or pre-load all?
4. **Audit backend**: Kernel daemon, Service Kit, or separate DB?
5. **Model source of truth**: /usr/share/models/ or Storage Kit?
6. **Fine-tuning lock**: Explicit gate or implicit (no API)?
7. **Quantization fallback**: Auto-downgrade, fail, or remote?
8. **TPM attestation**: Skip MVP or require?
9. **Hub relay auth**: Hearth token or separate credential?
10. **KV cache key**: app_id+session_id or session_id alone?

See `context/ai-kit-decision-memos.md` for details.

## Integration Roadmap

### Q4 2026 (MVP)
- Core trait + in-process sync
- Lazy-load by default
- Candle + TFLite bindings (optional)
- Score Kit capability grants
- Service Kit observability
- Spec Kit conformance gate

### Q1 2027 (v2)
- Pre-load optimization for Cumulus
- Message Kit async fallback (Pi5 → Cumulus over Hearth)
- Quantization selection hints

### Q2+ 2027 (v3+)
- Cloud hub relay (federation)
- TPM attestation (optional)
- Fine-grained privacy controls

## Development

```bash
cargo build
cargo test
cargo fmt
cargo clippy -- -D warnings
```

## Related Repos

- **Message Kit**: Protocol envelopes for async fallback (IPC over Hearth)
- **Storage Kit**: Model artifact persistence, BLAKE3 hashing, dedup
- **Score Kit**: Capability grants (rate limiting, SLA tracking)
- **Service Kit**: Observability bridge (metrics, logging)
- **Spec Kit**: Conformance validation (model shape validation in CI)
- **Device Kit**: vRAM queries, quantization hints per hardware profile
- **Substrate Kit**: Sidecar deployment (separate process for shared inference)

## License

Apache 2.0 (SlashBuilder open infrastructure standard).

## References

- **Unified design**: `context/projects/ai-kit-unified-design.md`
- **Decision memos**: `context/ai-kit-decision-memos.md`
- **Tensions & questions**: `context/ai-kit-tensions-and-questions.md`
- **Locked invariants**: `context/ai-kit-locked-invariants.md`
- **Kit model reference**: `context/kit-model.md` (in cross-repo-context)
- **Threat model** (security sketch): `context/projects/ai-kit-threat-model.md`
