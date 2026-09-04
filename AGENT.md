# AI Kit — Agent Briefing

**Status**: v0.1-dev (MVP design complete, implementation started 2026-08-27)

## Tl;dr

AI Kit is a trait-based inference orchestrator for Hearth household clusters. Models are immutable OS resources. Inference is household-scoped with capability gating via Score Kit. Design is locked; DJ ruling on 10 open questions is the only blocker to full implementation.

## Design Documentation

- **Unified design**: `../../../vault/context/projects/ai-kit-unified-design.md` (78 KB, comprehensive)
- **Decision memos**: `../../../vault/context/ai-kit-decision-memos.md` (1-page rulings for DJ)
- **Tensions & questions**: `../../../vault/context/ai-kit-tensions-and-questions.md` (conflict table, open Q's)
- **Locked invariants**: `../../../vault/context/ai-kit-locked-invariants.md` (8 inviolable constraints)
- **Threat model**: `../../../vault/context/projects/ai-kit-threat-model.md` (security analysis)

## 8 Locked Invariants (Inviolable)

Any PR that breaks these will be rejected:

1. **Models are OS resources** — shipped signed in image, immutable post-deploy.
2. **No weights over network (MVP)** — local-only inference; cloud fallback is v0.2+ gate.
3. **Trait-based API** — no framework-specific APIs exposed to products.
4. **KV cache isolation** — per-session, never reused cross-app (privacy).
5. **Audit trail immutable** — always-on logging, kernel audit daemon or separate store.
6. **Score Kit gates capability** — single source of truth for authorization.
7. **Versions lock at install** — no auto-upgrade; rebuild required for new version.
8. **Inference-only** — no training, fine-tuning, backprop; weights read-only.

## 10 Open Questions (Pending DJ Ruling)

Implementation can proceed on locked constraints; these are design decisions, not blockers:

| # | Question | Current Design | Notes |
|---|----------|---|---|
| 1 | Capability gating locus | Trust Hearth boundary | AI Kit does not call Score Kit directly; trusts transport layer. |
| 2 | Remote fallback scope | Cumulus-only MVP | Cloud hub is v0.2+; MVP is household-scoped. |
| 3 | Cumulus pre-load strategy | Lazy-load default | Pre-load is opt-in config; minimizes boot time. |
| 4 | Audit backend | Kernel audit daemon (primary) | Service Kit as secondary sink; immutable append-only. |
| 5 | Model source of truth | /usr/share/models/ primary | OS-shipped, signed; Storage Kit as cache layer. |
| 6 | Fine-tuning lock | Implicit (no API methods) | API design prevents fine-tuning; no Spec Kit gate. |
| 7 | Quantization fallback | Fail hard (force explicit) | No silent downgrade; app handles fallback. |
| 8 | TPM attestation | Skip MVP | Deferred to v0.2+ if FIPS required. |
| 9 | Hub relay auth | Hearth token primary | Separate credential deferred to v0.2+ federation. |
| 10 | KV cache key | app_id + session_id | Strict privacy isolation by design. |

**TODO**: Once DJ rules, update decision docs and implement accordingly. Work marked with `TODO: Decision memo #N` indicates pending decisions.

## Repo Home

**Name**: `ai-kit`  
**Path**: `slash-builder/ai-kit/`  
**Org**: `slash-builder` (OSS infrastructure)  
**Language**: Rust (edition 2021)  
**Status**: v0.1-dev (initial skeleton)

## Modules

### Core

- **`src/lib.rs`**: Top-level crate definition + module exports.
- **`src/types.rs`**: Core data structures (ModelMetadata, InferenceParams, SessionId, etc).
- **`src/error.rs`**: Error types + mapping to Hearth ErrorCode enum.
- **`src/service.rs`**: `InferenceService` trait (frozen ABI).

### Infrastructure

- **`src/registry.rs`**: Model discovery from OS catalog (/usr/share/models/catalog.yaml).
- **`src/cache.rs`**: Two-layer cache (model weights + per-session KV).

### Implemented (POC scope)

- **`src/bin/cli.rs`**: `ai-kit-cli` — `hash`/`infer`/`cache-stats` subcommands. See README "Local Dev Inference (POC)".
- **`src/backends/candle.rs`**: `CandleInferenceService`, feature-gated behind `candle`. CPU-only, single small GGUF model (TinyLlama by default), real generation honoring temperature/top_p/max_tokens. **POC limitations**: no cross-call transformer KV-cache reuse (each `infer()` is a fresh generation); model fetched via `scripts/fetch-model.sh` from Hugging Face at dev time (documented exception to invariant #2, never in a production build); not wired to Storage Kit, Device Kit, or Score Kit — this is the trait + a real backend, not the full MVP integration.

### Future (Stubs)

- **`src/proto/`**: Message Kit bindings (TODO, awaiting proto finalization).
- **`src/backends/tflite.rs`**: TensorFlow Lite integration (TODO, optional).

## Integration Dependencies

| Kit | Status | What | When |
|---|---|---|---|
| **Storage Kit** | ⏳ Awaiting v2 | Model artifact persistence (BLAKE3, PartitionStore) | Q3 2026 |
| **Message Kit** | ⏳ Awaiting proto | Async RPC envelope (fallback, federation) | Q3 2026 |
| **Score Kit** | ⏳ Ready | Capability grants (`MlCapabilityGrant` table) | Q4 2026 |
| **Service Kit** | ⏳ Ready | Observability (latency metrics, structured logging) | Q4 2026 |
| **Spec Kit** | ✅ Ready | Conformance validation (model shape in CI) | Q4 2026 |
| **Device Kit** | ⏳ Ready | vRAM queries, quantization hints per hardware | Q4 2026 |
| **Substrate Kit** | ✅ Ready | Sidecar deployment (separate process for shared inference) | v0.2 |

**Key dependency path**: Storage Kit → Refraction (bitchain storage backend). Once Storage Kit v2 ships, AI Kit can consume model artifacts.

## Implementation Roadmap

### Q4 2026 (MVP)
- [x] Core trait definition (`InferenceService`).
- [x] Types + error mapping.
- [x] Model registry + catalog loader.
- [x] Cache infrastructure (model + KV).
- [x] Candle bindings (in-process inference) — POC scope: CPU-only, single small model, feature-gated (`--features candle`), local dev model fetch via `make fetch-model`. Real MVP integration (Storage Kit, Device Kit hints, multi-model catalog) still open.
- [ ] Score Kit integration (capability checks).
- [ ] Service Kit integration (observability).
- [ ] Spec Kit conformance gate (CI validation).
- [ ] Integration tests (Pi5 + Cumulus reference hardware).

### Q1 2027 (v0.2)
- [ ] Message Kit async fallback (Cumulus load distribution).
- [ ] Pre-load optimization for Cumulus.
- [ ] Quantization selection hints from Device Kit.
- [ ] Substrate Kit sidecar mode.

### Q2+ 2027 (v0.3+)
- [ ] Cloud hub relay (federation over Message Kit).
- [ ] TPM attestation (if FIPS required).
- [ ] Fine-grained privacy controls (user consent UI).

## Common Tasks

### Adding a New Model Type

1. Update `catalog.yaml` schema in unified design (if needed).
2. Add `Quantization` variant in `src/types.rs`.
3. Add test case in `src/registry.rs`.
4. Update Spec Kit conformance spec.

### Adding Error Cases

1. Define in `src/error.rs` with `#[error(...)]` derive.
2. Map to ErrorCode in `to_error_code()` (message-kit protocol).
3. Update CONTRIBUTING.md if it's an integration point.

### Integrating a New Kit

1. Add dependency to `Cargo.toml` (commented out if not ready).
2. Add `TODO` note in AGENT.md with target date.
3. Create integration test skeleton.
4. Coordinate with the kit's PM.

## Pre-Commit Checklist

- [ ] All tests pass (`cargo test`).
- [ ] Code formatted (`cargo fmt --check`).
- [ ] Linter passes (`cargo clippy -- -D warnings`).
- [ ] No direct Score Kit calls (trust Hearth boundary — decision memo #1).
- [ ] No weights in KV cache (only mmap'd RO).
- [ ] No training/fine-tuning methods (inference-only).
- [ ] `TODO` comments for decision memo dependencies.
- [ ] Documentation updated (if public API changes).

## FAQ

**Q: Can I add a training/fine-tuning method to the trait?**  
A: No. Inference-only is invariant #8. This is not negotiable.

**Q: Should AI Kit call Score Kit directly?**  
A: No. Per decision memo #1, AI Kit trusts Hearth's capability check at call boundary. Redundant local gating violates separation of concerns.

**Q: What if Cumulus is saturated?**  
A: For v0.1 (local-only MVP), the app gets an error and must handle fallback explicitly. v0.2 adds Message Kit async fallback; v0.3+ adds cloud hub relay.

**Q: Where do model files live?**  
A: OS-shipped in `/usr/share/models/`, signed, verified at boot. Not fetched at runtime. Updates require OS image rebuild.

**Q: Can two apps share KV cache?**  
A: No. Invariant #4 forbids cross-app sharing. Cache key is (app_id, session_id). Privacy by design.

**Q: Why not just call Score Kit to verify capability?**  
A: Decision memo #1: Capabilities are gated at the Hearth transport boundary. Repeating the check in AI Kit adds latency and splits authorization logic. One place per concern.

## Contact

**PM** (once project is elevated): TBD  
**Domain Owner (DJ)**: Makes rulings on 10 open questions  
**Messaging Architect**: Owns Message Kit proto + Hearth wire contract  

---

**Last updated**: 2026-08-27  
**Skeleton created**: 2026-08-27  
**Status**: Ready for full implementation once DJ rules on 10 decisions.
