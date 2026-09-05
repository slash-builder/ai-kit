# AI Kit

**Trait-based local inference orchestration for Hearth household clusters.**

AI Kit provides a framework-agnostic inference API for Hearth devices. Models are immutable OS resources (shipped signed, verified at boot). Inference is household-scoped with capability gating via Score Kit and observability via Service Kit.

## Status

**MVP (v0.1)** — Design complete, implementation in progress.

- ✅ Unified design doc (`context/projects/ai-kit-unified-design.md`)
- ✅ Decision memos for DJ ruling (`context/ai-kit-decision-memos.md`)
- ✅ Core trait definition (this repo)
- ✅ Model registry (real YAML catalog loading)
- ✅ Candle integration — **local dev POC** (see "Local Dev Inference (POC)" below); not the full MVP integration (no Storage Kit, Device Kit hints, or Score Kit wiring yet)
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

## Local Dev Inference (POC)

**This is a deliberate, documented exception to invariant #2** ("no weights
over network"). Production models ship signed in the OS image and are never
fetched at runtime. For local development — testing the trait, the registry,
the cache, and now a real inference backend without BenixOS, Storage Kit, or
an OS image — `--features candle` adds a `CandleInferenceService` backend
that downloads a small quantized model from Hugging Face onto your dev
machine. That download only ever happens via `make fetch-model`, is gitignored
(`dev-models/`), and is never part of any production build or boot path.

```bash
make fetch-model                 # downloads TinyLlama-1.1B-Chat (GGUF, ~700MB) into dev-models/
cargo build --features candle
cargo run --features candle --bin ai-kit-cli -- infer \
    --model tinyllama --prompt "What is the capital of France?"
```

Without `--features candle`, `ai-kit-cli infer` still builds and runs — it
exits with a message telling you to rebuild with the feature, rather than
failing to compile. `ai-kit-cli hash <path>` and `cache-stats` work either way.

**Known POC limitations** (see `AGENT.md` for the full list): each `infer()`
call runs a fresh generation — there's no cross-call transformer KV-cache
reuse at the Candle level (`KvCacheManager` still tracks session bookkeeping/
TTL, just not real tensor state); CPU-only, no CUDA/Metal; single small model,
not the full model catalog / quantization-fallback story described below.

## Cloud Proxy Inference — Claude/Anthropic (experimental, not yet ruled on)

**This is explicitly a `v0.2+` feature, off by default, and not yet blessed
for a real household product.** `ai-kit-unified-design.md`'s decision memos
#2 ("remote fallback scope — does it include Quickring cloud hub?") and #9
("hub relay authentication") are both still open — no DJ ruling. Sending
prompt text (not model weights) to a third-party cloud API is a real
data-egress decision that needs a security-engineer/DJ privacy ruling before
this goes beyond a dev POC, exactly the same caveat this repo already
applies to Candle's network-fetch-at-dev-time exception above.

`--features claude-proxy` adds `ClaudeProxyInferenceService`, a second
`InferenceService` implementation that proxies `infer()` to Anthropic's
Messages API (`POST /v1/messages`) instead of running local Candle
inference — same trait, same call site, no change to caller code. This
mirrors the pattern Apple's Foundation Models framework adopted at WWDC
2026 (a single `LanguageModelSession`-style call site bindable to on-device,
Private Cloud Compute, or a third-party provider via a provider/fallback
parameter) — cited here as precedent for the *shape*, not copied from. See
`src/backends/claude_proxy.rs`'s module doc for the full caveat, including a
note that a live web search during this backend's development could not
independently corroborate the specific Apple/Anthropic/Gemini framework
claim beyond WWDC 2026 coverage of Gemini-powered Siri.

Even with the feature compiled in, this backend is inert without an
explicit API key — `ClaudeProxyInferenceService::new` fails fast on an empty
key, so enabling the feature alone cannot send anything anywhere.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # your own key; never committed
cargo build --features claude-proxy
```

```rust
use ai_kit::{ClaudeProxyConfig, ClaudeProxyInferenceService, InferenceService};

let service = ClaudeProxyInferenceService::new(ClaudeProxyConfig {
    api_key: std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set"),
    base_url: None, // defaults to https://api.anthropic.com
})?;
```

**Known limitations** (deliberate, to fit the existing trait's single-shot
shape): no streaming, no tool use / function calling, no multi-turn
conversation — each `infer()` call is one independent request/response, and
`quantization` in the response is always the literal `"n/a (cloud proxy)"`
since there's no local weight file to quantize.

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
