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
//! - CPU by default. NVIDIA GPU via the opt-in `cuda` feature and
//!   [`DevicePreference`] (see below); no Metal.
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
use std::str::FromStr;
use std::sync::{Arc, Mutex, RwLock};
use tokenizers::Tokenizer;

/// Where a model runs. The default for [`CandleInferenceService::new`] is
/// [`DevicePreference::Cpu`], so nothing changes unless a caller opts in.
///
/// # No silent fallback
///
/// `Cuda(n)` means exactly that device: if it is not there, or the model does
/// not fit in its free memory, `load_model` **fails** (see invariant "fail
/// hard on OOM / no degrade"). Only `Auto` chooses for you, and it picks CUDA
/// device 0 when one is usable and the CPU otherwise; the choice is visible
/// afterwards through [`CandleInferenceService::device_of`], never hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Run on the CPU (the default).
    #[default]
    Cpu,
    /// Run on this CUDA device ordinal. Requires the `cuda` feature at build time.
    Cuda(usize),
    /// CUDA device 0 if usable, otherwise CPU.
    Auto,
}

impl FromStr for DevicePreference {
    type Err = String;
    /// Accepts `cpu`, `auto`, `cuda` (= `cuda:0`) and `cuda:N`.
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda(0)),
            other => other
                .strip_prefix("cuda:")
                .and_then(|n| n.parse::<usize>().ok())
                .map(Self::Cuda)
                .ok_or_else(|| {
                    format!("unknown device '{s}' (expected cpu, auto, cuda or cuda:N)")
                }),
        }
    }
}

/// Headroom added to the weight size when checking that a model fits in free
/// GPU memory: KV cache, activations and the CUDA context itself.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const VRAM_OVERHEAD_FRACTION: f64 = 0.15;
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const VRAM_OVERHEAD_FLOOR_MB: u64 = 384;

/// Everything needed to run inference for one loaded model.
///
/// # Weights are built once and reused (changed from the CPU POC)
///
/// The first version of this backend kept only the raw GGUF bytes and rebuilt
/// a fresh `ModelWeights` on every `infer()`, on the belief that Candle offers
/// no way to reset its internal KV cache between requests. It does, implicitly:
/// in `candle-transformers` 0.9.2 `LayerWeights::forward_attn` discards the
/// stored cache and starts from the new `k`/`v` whenever `index_pos == 0`
/// (see the `if index_pos == 0` branch). Every `infer()` here starts its first
/// forward pass at position 0, so one request can never attend to another's
/// cache, and `tests/candle_cuda.rs` checks this by running A, B, A.
///
/// Rebuilding per call was tolerable on the CPU (a ~2 GB re-parse) but would
/// re-upload the whole model to VRAM on every request on a GPU, so the
/// weights now live in a `Mutex` and requests for one model are serialised.
/// A finished request's key/value tensors stay in memory (VRAM on a GPU) until
/// the next request overwrites them; that residue is the one thing a
/// security review should look at (see AGENT.md).
struct LoadedCandleModel {
    weights: Mutex<ModelWeights>,
    tokenizer: Tokenizer,
    /// Every token that ends a reply for this model (see `end_token_ids`).
    eos_token_ids: Vec<u32>,
    device: Device,
    device_label: String,
}

/// Real local inference backend using Candle. See module docs.
pub struct CandleInferenceService {
    registry: ModelRegistry,
    device_pref: DevicePreference,
    loaded: RwLock<HashMap<String, Arc<LoadedCandleModel>>>,
    model_cache: Mutex<ModelCache>,
    kv_cache: Mutex<KvCacheManager>,
}

impl CandleInferenceService {
    /// Load the model registry from `catalog_path` (see
    /// [`ModelRegistry::load_from_file`]). Call [`InferenceService::load_model`]
    /// before the first [`InferenceService::infer`] call for a given model.
    pub fn new(catalog_path: &str) -> Result<Self> {
        Self::with_device(catalog_path, DevicePreference::Cpu)
    }

    /// Like [`Self::new`], but choose where models run. See [`DevicePreference`]
    /// for the (hard-fail) semantics; the device is resolved per model at
    /// [`InferenceService::load_model`] time, not here, so constructing a
    /// service never touches a GPU.
    pub fn with_device(catalog_path: &str, device: DevicePreference) -> Result<Self> {
        let registry = ModelRegistry::load_from_file(catalog_path)?;
        Ok(Self {
            registry,
            device_pref: device,
            loaded: RwLock::new(HashMap::new()),
            model_cache: Mutex::new(ModelCache::new()),
            kv_cache: Mutex::new(KvCacheManager::new()),
        })
    }

    /// Human-readable device a loaded model is running on (e.g. `cpu`,
    /// `cuda:0 (NVIDIA GeForce RTX 3060)`), or `None` if it is not loaded.
    pub fn device_of(&self, model_id: &str) -> Option<String> {
        self.loaded
            .read()
            .ok()?
            .get(model_id)
            .map(|m| m.device_label.clone())
    }

    /// Turn the configured [`DevicePreference`] into a concrete device and a
    /// label for it. Fails hard (no silent CPU fallback) unless `Auto`.
    fn resolve_device(&self, model_id: &str) -> Result<(Device, String)> {
        let fail = |reason: String| AiKitError::ModelLoadFailed {
            model_id: model_id.to_string(),
            reason,
        };
        device_for(self.device_pref).map_err(fail)
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

        // One loaded model serves one request at a time. The first forward pass
        // below starts at `index_pos == 0`, which makes Candle drop whatever
        // KV cache the previous request left (see `LoadedCandleModel`).
        let mut weights = loaded.weights.lock().map_err(|_| AiKitError::Internal {
            reason: "model weights lock poisoned".to_string(),
        })?;

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
            Some("llama3") => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                params.prompt
            ),
            Some("chatml") => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
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

            if loaded.eos_token_ids.contains(&next_token) {
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

        // Idempotent: a second load of an already-loaded model is a no-op. (Models
        // are immutable after install, and a rebuild would briefly hold two copies,
        // which on a GPU is how a model that fits once fails to "reload".)
        if self
            .loaded
            .read()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".to_string(),
            })?
            .contains_key(model_id)
        {
            return Ok(());
        }

        if model.kind() != crate::modality::RequestKind::TextGeneration {
            return Err(AiKitError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: format!(
                    "'{model_id}' is a {} model, not a text-generation model",
                    model.kind().as_str()
                ),
            });
        }
        let (device, device_label) = self.resolve_device(model_id)?;
        // Decide whether it fits *before* reading a multi-GB file.
        check_vram(&device, &model)?;

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

        // Build the weights once, on the target device. This is both the
        // fail-fast "does it parse" check and the instance every `infer()` reuses.
        let weights = Self::build_model_weights(&bytes, &device, model_id)?;

        let tokenizer_path = Self::tokenizer_path_for(&model)?;
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| AiKitError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: format!("tokenizer load failed: {e}"),
            })?;
        let eos_token_ids = end_token_ids(&tokenizer);
        if eos_token_ids.is_empty() {
            return Err(AiKitError::ModelLoadFailed {
                model_id: model_id.to_string(),
                reason: "tokenizer has none of the known end-of-reply tokens, so generation could not stop"
                    .to_string(),
            });
        }

        // The raw bytes now only back `cache.rs`'s bookkeeping, so hand them over
        // instead of cloning (the weights above already own the tensors).
        {
            let mut cache = self.model_cache.lock().map_err(|_| AiKitError::Internal {
                reason: "model-cache lock poisoned".to_string(),
            })?;
            cache.insert(LoadedModel::new(actual_hash, bytes));
        }

        let loaded = LoadedCandleModel {
            weights: Mutex::new(weights),
            tokenizer,
            eos_token_ids,
            device,
            device_label,
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

/// Tokens that end a reply, for whichever chat format the model uses. The CPU POC looked up only
/// `</s>` and silently fell back to token id 2, which for Llama-3 is the plain `#` token: it never
/// stopped at `<|eot_id|>` and ran on, inventing further turns, until `max_tokens`.
fn end_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    [
        "</s>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|end|>",
    ]
    .iter()
    .filter_map(|t| tokenizer.token_to_id(t))
    .collect()
}

/// Turn a [`DevicePreference`] into a device and a label, failing hard (no silent CPU fallback) unless `Auto`.
pub(crate) fn device_for(pref: DevicePreference) -> std::result::Result<(Device, String), String> {
    match pref {
        DevicePreference::Cpu => Ok((Device::Cpu, "cpu".to_string())),
        DevicePreference::Cuda(n) => cuda_device(n),
        DevicePreference::Auto => Ok(cuda_device(0).unwrap_or((Device::Cpu, "cpu".to_string()))),
    }
}

/// BLAKE3 of a file, streamed (model files can be gigabytes).
pub(crate) fn hash_file(path: &str) -> Result<String> {
    let mut h = blake3::Hasher::new();
    h.update_reader(std::fs::File::open(path)?)?;
    Ok(h.finalize().to_hex().to_string())
}

/// Verify the main file and every extra file of a model against the catalog's hashes. Returns the main file's path.
pub(crate) fn verify_model_files(model: &ModelMetadata) -> Result<String> {
    let strip = |l: &str| l.strip_prefix("file://").unwrap_or(l).to_string();
    let main = strip(&model.location);
    let mut checks = vec![(main.clone(), model.hash.clone())];
    checks.extend(
        model
            .files
            .iter()
            .map(|f| (strip(&f.location), f.hash.clone())),
    );
    for (path, expected) in checks {
        let actual = hash_file(&path)?;
        if actual != expected {
            return Err(AiKitError::IntegrityCheckFailed {
                model_id: model.id.clone(),
                expected,
                actual,
            });
        }
    }
    Ok(main)
}

/// Open CUDA device `ordinal`, returning it with a label such as
/// `cuda:0 (NVIDIA GeForce RTX 3060)`. The error string says what to fix.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_device(ordinal: usize) -> std::result::Result<(Device, String), String> {
    let device = Device::new_cuda(ordinal)
        .map_err(|e| format!("CUDA device {ordinal} is not usable: {e}"))?;
    let name = device
        .as_cuda_device()
        .map_err(|e| e.to_string())?
        .cuda_stream()
        .context()
        .name()
        .unwrap_or_else(|_| "unknown GPU".to_string());
    Ok((device, format!("cuda:{ordinal} ({name})")))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn cuda_device(ordinal: usize) -> std::result::Result<(Device, String), String> {
    Err(format!(
        "CUDA device {ordinal} requested, but this ai-kit was built without the `cuda` \
         feature (rebuild with `--features cuda`)"
    ))
}

/// MB of weights + headroom the model needs on a GPU. The catalog's declared
/// `vram_required_mb` is honoured as a floor.
// Only the CUDA path calls this, but it is plain arithmetic, so it is compiled
// (and unit-tested) in every build rather than hidden behind the feature.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn vram_needed_mb(model: &ModelMetadata) -> u64 {
    let weights_mb = model.total_size_bytes().div_ceil(1 << 20);
    let overhead =
        ((weights_mb as f64 * VRAM_OVERHEAD_FRACTION) as u64).max(VRAM_OVERHEAD_FLOOR_MB);
    (weights_mb + overhead).max(u64::from(model.metadata.vram_required_mb))
}

/// Fail hard (`InferenceOom`, no spill to the CPU, no smaller quantisation) if a
/// model cannot fit in the free memory of the chosen GPU. No-op on the CPU.
#[cfg(feature = "cuda")]
pub(crate) fn check_vram(device: &Device, model: &ModelMetadata) -> Result<()> {
    if !device.is_cuda() {
        return Ok(());
    }
    let ctx = device
        .as_cuda_device()
        .map_err(|e| AiKitError::Internal {
            reason: e.to_string(),
        })?
        .cuda_stream()
        .context()
        .clone();
    let (free, _total) = ctx.mem_get_info().map_err(|e| AiKitError::Internal {
        reason: format!("cannot query free GPU memory: {e}"),
    })?;
    let free_mb = (free >> 20) as u64;
    let need_mb = vram_needed_mb(model);
    if need_mb > free_mb {
        return Err(AiKitError::InferenceOom {
            model_id: model.id.clone(),
            required_mb: u32::try_from(need_mb).unwrap_or(u32::MAX),
            available_mb: u32::try_from(free_mb).unwrap_or(u32::MAX),
        });
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn check_vram(_device: &Device, _model: &ModelMetadata) -> Result<()> {
    Ok(())
}

/// One line per usable CUDA device (`cuda:0 NVIDIA GeForce RTX 3060, 11.4/12.0 GB free`),
/// or a single line saying why there are none. For `ai-kit-cli devices`.
pub fn describe_cuda_devices() -> Vec<String> {
    #[cfg(feature = "cuda")]
    {
        let mut out = Vec::new();
        for ordinal in 0..16 {
            let Ok(device) = Device::new_cuda(ordinal) else {
                break;
            };
            let Ok(cuda) = device.as_cuda_device() else {
                break;
            };
            let ctx = cuda.cuda_stream().context().clone();
            let name = ctx.name().unwrap_or_else(|_| "unknown GPU".to_string());
            let mem = ctx
                .mem_get_info()
                .map(|(free, total)| {
                    format!("{:.1}/{:.1} GB free", free as f64 / 1e9, total as f64 / 1e9)
                })
                .unwrap_or_else(|e| format!("memory unknown ({e})"));
            out.push(format!("cuda:{ordinal} {name}, {mem}"));
        }
        if out.is_empty() {
            out.push("no usable CUDA device found (driver missing, or no GPU)".to_string());
        }
        out
    }
    #[cfg(not(feature = "cuda"))]
    {
        vec!["built without the `cuda` feature; only `cpu` is available".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelMetadataExtended;

    fn model(size_bytes: u64, vram_required_mb: u32) -> ModelMetadata {
        ModelMetadata {
            id: "m".to_string(),
            version: "1.0.0".to_string(),
            os_arch: vec!["x86_64".to_string()],
            quantization_variants: vec!["q4_k_m".to_string()],
            default_quantization: "q4_k_m".to_string(),
            hash: "0".to_string(),
            size_bytes,
            location: "file:///x".to_string(),
            metadata: ModelMetadataExtended {
                parameters: 1,
                context_window: 2048,
                vram_required_mb,
                inference_latency_ms: 0,
                framework: String::new(),
                custom: HashMap::new(),
            },
            files: Vec::new(),
        }
    }

    #[test]
    fn vram_estimate_adds_proportional_headroom() {
        // 4000 MB of weights: 15% = 600 MB of headroom, above the 384 MB floor.
        let m = model(4000 << 20, 1);
        assert_eq!(vram_needed_mb(&m), 4600);
    }

    #[test]
    fn vram_estimate_has_a_floor_for_small_models() {
        // 100 MB of weights: 15% = 15 MB, so the 384 MB floor applies.
        assert_eq!(vram_needed_mb(&model(100 << 20, 1)), 100 + 384);
    }

    #[test]
    fn vram_estimate_honours_the_catalog_declared_floor() {
        // The catalog says it needs 9000 MB, more than the file-size estimate.
        assert_eq!(vram_needed_mb(&model(1000 << 20, 9000)), 9000);
    }
}
