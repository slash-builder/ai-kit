//! Text embeddings on Candle (a BERT-family sentence encoder such as all-MiniLM-L6-v2). **POC.**
//!
//! [`CandleEmbeddingService`] implements [`RequestService`] for [`RequestKind::Embedding`]: the
//! "semantic search over Courier files" use the Inference Kit design names. It mean-pools the last hidden state under the
//! attention mask and, if asked, scales each vector to unit length (so a dot product is cosine similarity).
//!
//! Embeddings of household files are about as private as data gets, so the router's default (`Private`: never leaves the
//! device) is the right one and this backend has no cloud counterpart. The catalog entry has `model.safetensors` as the main file and
//! `config` and `tokenizer` as extra files; every file is BLAKE3-verified on load.

use crate::backends::candle::{check_vram, device_for, verify_model_files, DevicePreference};
use crate::error::{AiKitError, Result};
use crate::modality::{
    unsupported, EmbeddingParams, EmbeddingResponse, Output, Request, RequestKind, RequestService,
};
use crate::registry::ModelRegistry;
use crate::types::ModelMetadata;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

struct LoadedEmbeddingModel {
    bert: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    version: String,
}

pub struct CandleEmbeddingService {
    registry: ModelRegistry,
    device_pref: DevicePreference,
    loaded: RwLock<HashMap<String, Arc<Mutex<LoadedEmbeddingModel>>>>,
}

impl CandleEmbeddingService {
    pub fn new(catalog_path: &str, device: DevicePreference) -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::load_from_file(catalog_path)?,
            device_pref: device,
            loaded: RwLock::new(HashMap::new()),
        })
    }

    fn model_meta(&self, id: &str) -> Result<ModelMetadata> {
        let m = self
            .registry
            .get(id)
            .cloned()
            .ok_or_else(|| AiKitError::ModelNotFound {
                model_id: id.to_string(),
            })?;
        if m.kind() != RequestKind::Embedding {
            return Err(AiKitError::ModelLoadFailed {
                model_id: id.to_string(),
                reason: format!(
                    "'{id}' is a {} model, not an embedding model",
                    m.kind().as_str()
                ),
            });
        }
        Ok(m)
    }

    pub fn load_model(&self, id: &str) -> Result<()> {
        if self.loaded.read().is_ok_and(|l| l.contains_key(id)) {
            return Ok(());
        }
        let model = self.model_meta(id)?;
        let fail = |reason: String| AiKitError::ModelLoadFailed {
            model_id: id.to_string(),
            reason,
        };
        let (device, _) = device_for(self.device_pref).map_err(fail)?;
        check_vram(&device, &model)?;
        let weights = verify_model_files(&model)?;
        let need = |role: &str| {
            model
                .file_path(role)
                .ok_or_else(|| fail(format!("catalog entry has no '{role}' file")))
        };
        let (cfg_path, tok_path) = (need("config")?, need("tokenizer")?);
        let config: Config = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)
            .map_err(|e| fail(format!("config: {e}")))?;
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| fail(format!("tokenizer: {e}")))?;
        tokenizer.with_padding(Some(PaddingParams::default()));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_position_embeddings.min(256),
                ..Default::default()
            }))
            .map_err(|e| fail(format!("tokenizer: {e}")))?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device) }
            .map_err(|e| fail(format!("weights: {e}")))?;
        let bert = BertModel::load(vb, &config).map_err(|e| fail(format!("model: {e}")))?;
        self.loaded
            .write()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".into(),
            })?
            .insert(
                id.to_string(),
                Arc::new(Mutex::new(LoadedEmbeddingModel {
                    bert,
                    tokenizer,
                    device,
                    version: model.version.clone(),
                })),
            );
        Ok(())
    }

    fn embed(&self, p: &EmbeddingParams) -> Result<EmbeddingResponse> {
        Request::Embedding(p.clone()).validate()?;
        self.load_model(&p.model_id)?;
        let model = self
            .loaded
            .read()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".into(),
            })?
            .get(&p.model_id)
            .cloned()
            .ok_or_else(|| AiKitError::ModelNotFound {
                model_id: p.model_id.clone(),
            })?;
        let m = model.lock().map_err(|_| AiKitError::Internal {
            reason: "embedding model lock poisoned".into(),
        })?;
        let started = std::time::Instant::now();
        let internal = |what: &str, e: candle_core::Error| AiKitError::Internal {
            reason: format!("{what}: {e}"),
        };

        let enc = m
            .tokenizer
            .encode_batch(p.inputs.clone(), true)
            .map_err(|e| AiKitError::Internal {
                reason: format!("tokenize: {e}"),
            })?;
        let prompt_tokens: u32 = enc
            .iter()
            .map(|e| e.get_attention_mask().iter().sum::<u32>())
            .sum();
        let rows = |f: &dyn Fn(&tokenizers::Encoding) -> Vec<u32>| -> candle_core::Result<Tensor> {
            let v: Vec<Tensor> = enc
                .iter()
                .map(|e| Tensor::new(f(e).as_slice(), &m.device))
                .collect::<candle_core::Result<_>>()?;
            Tensor::stack(&v, 0)
        };
        let vectors = (|| -> candle_core::Result<Vec<Vec<f32>>> {
            let ids = rows(&|e| e.get_ids().to_vec())?;
            let mask = rows(&|e| e.get_attention_mask().to_vec())?;
            let types = ids.zeros_like()?;
            let hidden = m.bert.forward(&ids, &types, Some(&mask))?; // (batch, seq, dim)
            let maskf = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
            let summed = hidden.broadcast_mul(&maskf)?.sum(1)?;
            let mut pooled = summed.broadcast_div(&maskf.sum(1)?)?;
            if p.normalize {
                pooled = pooled.broadcast_div(&pooled.sqr()?.sum_keepdim(1)?.sqrt()?)?;
            }
            pooled.to_vec2::<f32>()
        })()
        .map_err(|e| internal("embed", e))?;

        let dimensions = vectors.first().map_or(0, Vec::len);
        Ok(EmbeddingResponse {
            vectors,
            dimensions,
            prompt_tokens,
            inference_ms: u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX),
            model_version: m.version.clone(),
            session_id: p.session_id.clone().unwrap_or_default(),
            request_id: p
                .request_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        })
    }
}

#[async_trait]
impl RequestService for CandleEmbeddingService {
    fn kinds(&self) -> Vec<RequestKind> {
        vec![RequestKind::Embedding]
    }

    async fn handle(&self, req: Request) -> Result<Output> {
        match req {
            Request::Embedding(p) => self.embed(&p).map(Output::Embeddings),
            other => Err(unsupported(other.kind(), "this is an embedding backend")),
        }
    }
}
