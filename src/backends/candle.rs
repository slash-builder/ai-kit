//! Candle-backed local inference — **POC scope, not the full MVP integration**.
//!
//! Implements [`InferenceService`] for real: loads a quantized GGUF model
//! (verified via BLAKE3 against the registry's expected hash), runs real
//! autoregressive generation on CPU via `candle-transformers`' quantized
//! Llama implementation, and fills every [`InferenceResponse`] field with
//! genuine values (real token counts, real wall-clock timing).
//!
//! # POC-only exception to invariant #2
//!
//! Model weights are fetched over the network by `scripts/fetch-model.sh`
//! for local developer-machine testing (see README "Local Dev Inference
//! (POC)"). This backend itself never fetches anything — it only reads
//! whatever `ModelMetadata::location` in the catalog points at, exactly like
//! a production backend reading `/usr/share/models/` would. The network
//! fetch is a dev-setup-time concern, not a runtime one.
//!
//! # Known POC limitations (see `AGENT.md` for the full list)
//!
//! - CPU-only (`Device::Cpu`), no CUDA/Metal.
//! - Single small model family verified (plain Llama architecture, e.g.
//!   TinyLlama) — not the full model-catalog / quantization-fallback story.
//! - Not wired to Storage Kit, Device Kit, or Score Kit.
//! - See [`LoadedCandleModel`] for the KV-cache-across-calls limitation,
//!   which is deliberate and documented there, not an oversight.

use crate::cache::{KvCacheManager, LoadedModel, ModelCache};
use crate::error::{AiKitError, Result};
use crate::registry::ModelRegistry;
use crate::service::InferenceService;
use crate::types::{CacheStats, InferenceParams, InferenceResponse, ModelMetadata, SessionId};

use async_trait::async_trait;
use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::collections::HashMap;
use std::io::Cursor;
use std::sync::{Arc, Mutex, RwLock};
use tokenizers::Tokenizer;

/// Everything needed to run inference for one loaded model.
///
/// # POC LIMITATION: no KV-cache reuse *across* `infer()` calls
///
/// candle-transformers' `LayerWeights` (inside `ModelWeights`) maintains a
/// real, incremental KV-cache internally across the forward-pass calls
/// *within* one generation loop — so autoregressive decoding inside a
/// single `infer()` call is properly incremental, not O(n²). But there is
/// no public API on `ModelWeights` to reset that internal state, so reusing
/// one instance across separate `infer()` calls would silently leak KV
/// state between unrelated requests. This backend avoids that by keeping
/// only the raw GGUF bytes (already in RAM, no disk I/O) and rebuilding a
/// fresh `ModelWeights` from them on every `infer()` call — correct, at the
/// cost of re-parsing tensors each call. `KvCacheManager` below still
/// tracks session bookkeeping (TTL, `end_session`), just not real tensor
/// state — consistent with `cache.rs`'s own documented, pre-existing scope.
struct LoadedCandleModel {
    gguf_bytes: Vec<u8>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    device: Device,
}

/// Real local inference backend using Candle. See module docs.
pub struct CandleInferenceService {
    registry: ModelRegistry,
    loaded: RwLock<HashMap<String, Arc<LoadedCandleModel>>>,
    model_cache: Mutex<ModelCache>,
    kv_cache: Mutex<KvCacheManager>,
}

impl CandleInferenceService {
    /// Load the model registry from `catalog_path` (see
    /// [`ModelRegistry::load_from_file`]). Call [`InferenceService::load_model`]
    /// before the first [`InferenceService::infer`] call for a given model.
    pub fn new(catalog_path: &str) -> Result<Self> {
        let registry = ModelRegistry::load_from_file(catalog_path)?;
        Ok(Self {
            registry,
            loaded: RwLock::new(HashMap::new()),
            model_cache: Mutex::new(ModelCache::new()),
            kv_cache: Mutex::new(KvCacheManager::new()),
        })
    }

    fn model_meta(&self, model_id: &str) -> Result<ModelMetadata> {
        self.registry
            .get(model_id)
            .cloned()
            .ok_or_else(|| AiKitError::ModelNotFound {
                model_id: model_id.to_string(),
            })
    }

    fn tokenizer_path_for(model: &ModelMetadata) -> Result<String> {
        model
            .metadata
            .custom
            .get("tokenizer_path")
            .cloned()
            .ok_or_else(|| AiKitError::ModelLoadFailed {
                model_id: model.id.clone(),
                reason: "catalog entry missing metadata.custom.tokenizer_path".to_string(),
            })
    }

    /// Parse a fresh `ModelWeights` from an in-memory GGUF byte buffer.
    /// Shared between `load_model` (fail-fast validation) and `infer` (see
    /// [`LoadedCandleModel`]'s doc comment for why this reparses per call).
    fn build_model_weights(
        gguf_bytes: &[u8],
        device: &Device,
        model_id: &str,
    ) -> Result<ModelWeights> {
        let mut cursor = Cursor::new(gguf_bytes);
        let content =
            gguf_file::Content::read(&mut cursor).map_err(|e| AiKitError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: format!("gguf parse failed: {e}"),
            })?;
        ModelWeights::from_gguf(content, &mut cursor, device).map_err(|e| {
            AiKitError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: format!("candle model build failed: {e}"),
            }
        })
    }
}

#[async_trait]
impl InferenceService for CandleInferenceService {
    async fn infer(&self, params: InferenceParams) -> Result<InferenceResponse> {
        params.validate()?;

        let loaded = {
            let guard = self.loaded.read().map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".to_string(),
            })?;
            guard.get(&params.model_id).cloned()
        }
        .ok_or_else(|| AiKitError::ModelNotFound {
            model_id: params.model_id.clone(),
        })?;

        let start = std::time::Instant::now();

        let mut weights =
            Self::build_model_weights(&loaded.gguf_bytes, &loaded.device, &params.model_id)?;

        // Chat-tuned GGUF models (this backend only fetches chat-tuned
        // checkpoints, e.g. TinyLlama-Chat) answer far more coherently when
        // the prompt is wrapped in the template they were fine-tuned on,
        // versus fed as a raw continuation. `chat_template` in the catalog's
        // `metadata.custom` opts a model into this; absent, the prompt is
        // used as-is (raw completion — the trait's baseline contract).
        let model = self.model_meta(&params.model_id)?;
        let formatted_prompt = match model
            .metadata
            .custom
            .get("chat_template")
            .map(String::as_str)
        {
            Some("zephyr") => format!(
                "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                params.prompt
            ),
            _ => params.prompt.clone(),
        };

        let encoding = loaded
            .tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(|e| AiKitError::Internal {
                reason: format!("tokenizer encode failed: {e}"),
            })?;
        let prompt_tokens = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            return Err(AiKitError::InvalidParams {
                reason: "prompt encoded to zero tokens".to_string(),
            });
        }

        // Deterministic seed: a POC demo should be reproducible run-to-run
        // for the same prompt/params, not flaky. Real product use would
        // want per-request entropy; out of scope here.
        let mut logits_processor = LogitsProcessor::new(
            0,
            Some(params.temperature as f64),
            Some(params.top_p as f64),
        );

        let mut generated: Vec<u32> = Vec::new();
        let mut index_pos = 0usize;

        let input = tensor_1x_n(&prompt_tokens, &loaded.device)?;
        let mut logits = weights
            .forward(&input, index_pos)
            .map_err(|e| AiKitError::Internal {
                reason: format!("forward failed: {e}"),
            })?;
        index_pos += prompt_tokens.len();

        for _ in 0..params.max_tokens {
            let logits_1d = logits.squeeze(0).map_err(|e| AiKitError::Internal {
                reason: e.to_string(),
            })?;
            let next_token =
                logits_processor
                    .sample(&logits_1d)
                    .map_err(|e| AiKitError::Internal {
                        reason: e.to_string(),
                    })?;

            if next_token == loaded.eos_token_id {
                break;
            }
            generated.push(next_token);

            let next_input = tensor_1x_n(&[next_token], &loaded.device)?;
            logits = weights
                .forward(&next_input, index_pos)
                .map_err(|e| AiKitError::Internal {
                    reason: format!("forward failed: {e}"),
                })?;
            index_pos += 1;
        }

        let completion =
            loaded
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| AiKitError::Internal {
                    reason: format!("tokenizer decode failed: {e}"),
                })?;

        let inference_ms = u32::try_from(start.elapsed().as_millis()).unwrap_or(u32::MAX);
        let session_id = params.session_id.clone().unwrap_or_default();

        // Bookkeeping only (session TTL/lookup) — not real per-app privacy
        // isolation. That's a known, pre-existing gap in cache.rs
        // (KvCacheManager is keyed by SessionId alone, not app_id+session_id),
        // out of scope for this POC. See cache.rs module doc + AGENT.md.
        {
            let mut kv = self.kv_cache.lock().map_err(|_| AiKitError::Internal {
                reason: "kv-cache lock poisoned".to_string(),
            })?;
            kv.get_or_create(
                session_id.clone(),
                model.hash.clone(),
                "poc-app".to_string(),
            );
        }

        Ok(InferenceResponse {
            completion,
            total_tokens: (prompt_tokens.len() + generated.len()) as u32,
            prompt_tokens: prompt_tokens.len() as u32,
            completion_tokens: generated.len() as u32,
            inference_ms,
            model_version: model.version.clone(),
            quantization: params
                .preferred_quantization
                .clone()
                .unwrap_or_else(|| model.default_quantization.clone()),
            session_id,
            request_id: params
                .request_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        })
    }

    async fn load_model(&self, model_id: &str) -> Result<()> {
        let model = self.model_meta(model_id)?;

        let path = model
            .location
            .strip_prefix("file://")
            .unwrap_or(model.location.as_str());
        let bytes = std::fs::read(path)?;

        // Verify integrity into the *same* buffer that gets parsed below —
        // no separate re-read between "verified" and "loaded" (no TOCTOU gap).
        let actual_hash = blake3::hash(&bytes).to_hex().to_string();
        if actual_hash != model.hash {
            return Err(AiKitError::IntegrityCheckFailed {
                model_id: model_id.to_string(),
                expected: model.hash.clone(),
                actual: actual_hash,
            });
        }

        // Fail fast: make sure the GGUF actually parses and the model
        // builds now, rather than only discovering a corrupt file on the
        // first `infer()` call.
        Self::build_model_weights(&bytes, &Device::Cpu, model_id)?;

        let tokenizer_path = Self::tokenizer_path_for(&model)?;
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| AiKitError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: format!("tokenizer load failed: {e}"),
            })?;
        let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(2);

        // One-time (per load_model call, not per-inference) clone: cache.rs's
        // LoadedModel owns its own Vec<u8> for bookkeeping/stats purposes, and
        // this backend separately needs to keep its own copy alive to rebuild
        // ModelWeights per `infer()` call (see LoadedCandleModel's doc
        // comment). Not sharing storage between the two is a deliberate
        // choice to avoid touching cache.rs's existing storage strategy.
        {
            let mut cache = self.model_cache.lock().map_err(|_| AiKitError::Internal {
                reason: "model-cache lock poisoned".to_string(),
            })?;
            cache.insert(LoadedModel::new(actual_hash, bytes.clone()));
        }

        let loaded = LoadedCandleModel {
            gguf_bytes: bytes,
            tokenizer,
            eos_token_id,
            device: Device::Cpu,
        };
        self.loaded
            .write()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".to_string(),
            })?
            .insert(model_id.to_string(), Arc::new(loaded));

        Ok(())
    }

    async fn unload_model(&self, model_id: &str) -> Result<()> {
        self.loaded
            .write()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".to_string(),
            })?
            .remove(model_id);
        if let Ok(model) = self.model_meta(model_id) {
            let mut cache = self.model_cache.lock().map_err(|_| AiKitError::Internal {
                reason: "model-cache lock poisoned".to_string(),
            })?;
            cache.remove(&model.hash);
        }
        Ok(())
    }

    async fn cache_stats(&self) -> Result<CacheStats> {
        let model_cache = self.model_cache.lock().map_err(|_| AiKitError::Internal {
            reason: "model-cache lock poisoned".to_string(),
        })?;
        let kv_cache = self.kv_cache.lock().map_err(|_| AiKitError::Internal {
            reason: "kv-cache lock poisoned".to_string(),
        })?;
        Ok(CacheStats {
            models_loaded: model_cache.len() as u32,
            models_size_bytes: model_cache.total_size_bytes(),
            sessions_active: kv_cache.len() as u32,
            cache_size_bytes: kv_cache.total_size_bytes(),
            cache_hit_rate: 0.0, // not tracked in this POC
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    async fn end_session(&self, session_id: &SessionId) -> Result<()> {
        self.kv_cache
            .lock()
            .map_err(|_| AiKitError::Internal {
                reason: "kv-cache lock poisoned".to_string(),
            })?
            .remove(session_id);
        Ok(())
    }
}

/// Build a `(1, tokens.len())` tensor, the shape `ModelWeights::forward` expects.
fn tensor_1x_n(tokens: &[u32], device: &Device) -> Result<Tensor> {
    Tensor::new(tokens, device)
        .and_then(|t| t.unsqueeze(0))
        .and_then(|t| t.to_dtype(DType::U32))
        .map_err(|e| AiKitError::Internal {
            reason: format!("tensor build failed: {e}"),
        })
}
