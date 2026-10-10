//! Text-to-image on Candle: the Stable Diffusion family with the distilled "turbo" sampler. **POC.**
//!
//! [`CandleImageService`] implements [`CapabilityService`] for [`Capability::ImageGeneration`]. It is pure Rust
//! (Candle's CLIP text encoder, UNet and VAE; a Euler-ancestral scheduler; the `png` crate to encode) and runs on the CPU or, with
//! the `cuda` feature, an NVIDIA GPU, under the same device policy as the text backend: an explicit device that is
//! missing is an error, and a model that does not fit in free VRAM is an `InferenceOom` *before* any file is read.
//!
//! # What it runs
//!
//! `architecture: sd_turbo` in the catalog entry: an SD-2.1-family UNet with an OpenCLIP-H text encoder, epsilon prediction and a
//! trailing-spaced Euler-ancestral sampler (what `stabilityai/sd-turbo` ships). 512x512 in 1-4 steps. The catalog entry has the
//! UNet as the main file and `vae`, `text_encoder` and `tokenizer` as extra files; **every file is BLAKE3-verified** on load,
//! like any other model (invariant: models are immutable and verified).
//!
//! # Licence, read before shipping
//!
//! The dev weights (`stabilityai/sd-turbo`) are under Stability AI's non-commercial research licence. They are a development
//! default so the *backend* can be tested end to end. A product needs a commercially licensed model; the studio's higher-quality
//! local model (Z-Image-Turbo, Apache-2.0) cannot be run by Candle 0.11 on a 12 GB card (no quantised loader, no offload), which is why
//! it is not the default here.
//!
//! # Known limits
//!
//! Reproducibility: the starting noise comes from `seed` on every device, but the Euler-ancestral sampler draws more noise
//! on later steps from Candle's own RNG, which can only be seeded on CUDA. With `steps = 1` (no extra noise) the same
//! seed gives the same image everywhere. No img2img, inpainting or LoRA. One request at a time per model.

use crate::backends::candle::{check_vram, device_for, verify_model_files, DevicePreference};
use crate::error::{AiKitError, Result};
use crate::modality::{
    unsupported, Capability, CapabilityService, GeneratedImage, ImageGenParams, Output, Request,
};
use crate::registry::ModelRegistry;
use crate::types::ModelMetadata;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::stable_diffusion::{
    self,
    clip::{ClipTextTransformer, Config as ClipConfig},
    euler_ancestral_discrete::EulerAncestralDiscreteSchedulerConfig,
    schedulers::{SchedulerConfig, TimestepSpacing},
    unet_2d::UNet2DConditionModel,
    vae::AutoEncoderKL,
    StableDiffusionConfig,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use tokenizers::Tokenizer;

/// SD-2.1 family VAE scaling factor.
const VAE_SCALE: f64 = 0.18215;

struct LoadedImageModel {
    clip: ClipTextTransformer,
    clip_cfg: ClipConfig,
    unet: UNet2DConditionModel,
    vae: AutoEncoderKL,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    version: String,
}

pub struct CandleImageService {
    registry: ModelRegistry,
    device_pref: DevicePreference,
    loaded: RwLock<HashMap<String, Arc<Mutex<LoadedImageModel>>>>,
}

impl CandleImageService {
    pub fn new(catalog_path: &str, device: DevicePreference) -> Result<Self> {
        Ok(Self {
            registry: ModelRegistry::load_from_file(catalog_path)?,
            device_pref: device,
            loaded: RwLock::new(HashMap::new()),
        })
    }

    /// Where a loaded model runs, or `None`.
    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.loaded.read().is_ok_and(|l| l.contains_key(model_id))
    }

    fn model_meta(&self, id: &str) -> Result<ModelMetadata> {
        let m = self
            .registry
            .get(id)
            .cloned()
            .ok_or_else(|| AiKitError::ModelNotFound {
                model_id: id.to_string(),
            })?;
        if m.capability() != Capability::ImageGeneration {
            return Err(AiKitError::ModelLoadFailed {
                model_id: id.to_string(),
                reason: format!(
                    "'{id}' is a {} model, not an image-generation model",
                    m.capability().as_str()
                ),
            });
        }
        Ok(m)
    }

    /// Load a model (verify files, build on the device). Called on first use; call it earlier to warm up.
    pub fn load_model(&self, id: &str) -> Result<()> {
        if self.is_loaded(id) {
            return Ok(());
        }
        let model = self.model_meta(id)?;
        let fail = |reason: String| AiKitError::ModelLoadFailed {
            model_id: id.to_string(),
            reason,
        };
        let arch = model
            .metadata
            .custom
            .get("architecture")
            .map_or("sd_turbo", String::as_str);
        if arch != "sd_turbo" {
            return Err(fail(format!(
                "unsupported image architecture '{arch}' (this backend runs: sd_turbo)"
            )));
        }
        let (device, _label) = device_for(self.device_pref).map_err(fail)?;
        check_vram(&device, &model)?;
        let unet_path = verify_model_files(&model)?;
        let need = |role: &str| {
            model
                .file_path(role)
                .ok_or_else(|| fail(format!("catalog entry has no '{role}' file")))
        };
        let (vae_path, clip_path, tok_path) =
            (need("vae")?, need("text_encoder")?, need("tokenizer")?);

        let dtype = if device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        let cfg = StableDiffusionConfig::v2_1(None, Some(512), Some(512));
        let clip_cfg = ClipConfig::v2_1();
        let wrap = |what: &str, e: candle_core::Error| fail(format!("{what}: {e}"));
        let clip = stable_diffusion::build_clip_transformer(&clip_cfg, &clip_path, &device, dtype)
            .map_err(|e| wrap("text encoder", e))?;
        let vae = cfg
            .build_vae(&vae_path, &device, dtype)
            .map_err(|e| wrap("vae", e))?;
        let unet = cfg
            .build_unet(&unet_path, &device, 4, false, dtype)
            .map_err(|e| wrap("unet", e))?;
        let tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| fail(format!("tokenizer: {e}")))?;
        self.loaded
            .write()
            .map_err(|_| AiKitError::Internal {
                reason: "loaded-model lock poisoned".into(),
            })?
            .insert(
                id.to_string(),
                Arc::new(Mutex::new(LoadedImageModel {
                    clip,
                    clip_cfg,
                    unet,
                    vae,
                    tokenizer,
                    device,
                    dtype,
                    version: model.version.clone(),
                })),
            );
        Ok(())
    }

    fn generate(&self, p: &ImageGenParams) -> Result<GeneratedImage> {
        p.validate()?;
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
            reason: "image model lock poisoned".into(),
        })?;
        let started = std::time::Instant::now();
        let seed = p.seed.unwrap_or_else(fresh_seed);
        let internal = |what: &str, e: candle_core::Error| AiKitError::Internal {
            reason: format!("{what}: {e}"),
        };

        let (h, w) = (p.height as usize, p.width as usize);
        let use_guide = p.guidance_scale > 1.0;
        let cond = encode_prompt(&m, &p.prompt).map_err(|e| internal("prompt", e))?;
        let text_embeddings = if use_guide {
            let uncond = encode_prompt(&m, p.negative_prompt.as_deref().unwrap_or(""))
                .map_err(|e| internal("negative prompt", e))?;
            Tensor::cat(&[&uncond, &cond], 0).map_err(|e| internal("embeddings", e))?
        } else {
            cond
        };
        let text_embeddings = text_embeddings
            .to_dtype(m.dtype)
            .map_err(|e| internal("embeddings", e))?;

        let mut scheduler = EulerAncestralDiscreteSchedulerConfig {
            timestep_spacing: TimestepSpacing::Trailing,
            ..Default::default()
        }
        .build(p.steps as usize)
        .map_err(|e| internal("scheduler", e))?;
        let noise = seeded_normal(seed, 4 * (h / 8) * (w / 8));
        let mut latents = Tensor::from_vec(noise, (1, 4, h / 8, w / 8), &m.device)
            .and_then(|t| t * scheduler.init_noise_sigma())
            .and_then(|t| t.to_dtype(m.dtype))
            .map_err(|e| internal("latents", e))?;

        let timesteps = scheduler.timesteps().to_vec();
        for &t in &timesteps {
            let step = (|| -> candle_core::Result<Tensor> {
                let input = if use_guide {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };
                let input = scheduler.scale_model_input(input, t)?;
                let pred = m.unet.forward(&input, t as f64, &text_embeddings)?;
                let pred = if use_guide {
                    let c = pred.chunk(2, 0)?;
                    (&c[0] + ((&c[1] - &c[0])? * f64::from(p.guidance_scale))?)?
                } else {
                    pred
                };
                scheduler.step(&pred, t, &latents)
            })();
            latents = step.map_err(|e| internal("denoising step", e))?;
        }

        let rgb = (|| -> candle_core::Result<(Vec<u8>, f32)> {
            let img = m.vae.decode(&(&latents / VAE_SCALE)?)?;
            let img = ((img.to_dtype(DType::F32)? / 2.)? + 0.5)?;
            let img = (img.clamp(0f32, 1f32)? * 255.)?
                .to_dtype(DType::U8)?
                .to_device(&Device::Cpu)?
                .i(0)?
                .permute((1, 2, 0))?
                .contiguous()?;
            let bytes = img.flatten_all()?.to_vec1::<u8>()?;
            let n = bytes.len() as f32;
            let mean = bytes.iter().map(|b| f32::from(*b)).sum::<f32>() / n;
            let var = bytes
                .iter()
                .map(|b| (f32::from(*b) - mean).powi(2))
                .sum::<f32>()
                / n;
            Ok((bytes, var.sqrt()))
        })()
        .map_err(|e| internal("decode", e))?;

        Ok(GeneratedImage {
            png: encode_png(&rgb.0, p.width, p.height)?,
            width: p.width,
            height: p.height,
            seed,
            steps: p.steps,
            pixel_stddev: rgb.1,
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
impl CapabilityService for CandleImageService {
    fn capabilities(&self) -> Vec<Capability> {
        vec![Capability::ImageGeneration]
    }

    async fn handle(&self, req: Request) -> Result<Output> {
        match req {
            Request::ImageGeneration(p) => self.generate(&p).map(Output::Image),
            other => Err(unsupported(
                other.capability(),
                "this is an image-generation backend",
            )),
        }
    }
}

/// CLIP-tokenise to the model's fixed length (padded with its pad token) and run the text encoder: `(1, 77, 1024)`.
fn encode_prompt(m: &LoadedImageModel, prompt: &str) -> candle_core::Result<Tensor> {
    let max_len = m.clip_cfg.max_position_embeddings;
    let pad = m.clip_cfg.pad_with.as_deref().unwrap_or("<|endoftext|>");
    let pad_id = m
        .tokenizer
        .token_to_id(pad)
        .ok_or_else(|| candle_core::Error::Msg(format!("tokenizer has no '{pad}' token")))?;
    let eos = m.tokenizer.token_to_id("<|endoftext|>");
    let mut ids: Vec<u32> = m
        .tokenizer
        .encode(prompt, true)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
        .get_ids()
        .to_vec();
    if ids.len() > max_len {
        ids.truncate(max_len);
        if let (Some(last), Some(e)) = (ids.last_mut(), eos) {
            *last = e; // keep the end-of-text token a CLIP model expects
        }
    }
    while ids.len() < max_len {
        ids.push(pad_id);
    }
    let tokens = Tensor::new(ids.as_slice(), &m.device)?.unsqueeze(0)?;
    candle_core::Module::forward(&m.clip, &tokens)
}

/// splitmix64 + Box-Muller: deterministic, identical on every device (Candle can only seed the CUDA RNG).
fn seeded_normal(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let unit = |x: u64| ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64; // (0, 1)
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (unit(next()), unit(next()));
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
        out.push((r * (2.0 * std::f64::consts::PI * u2).sin()) as f32);
    }
    out.truncate(n);
    out
}

fn fresh_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    t ^ u64::from(std::process::id()).rotate_left(32)
}

fn encode_png(rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rgb.len() / 2);
    {
        let mut enc = png::Encoder::new(&mut out, width, height);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().map_err(|e| AiKitError::Internal {
            reason: format!("png header: {e}"),
        })?;
        w.write_image_data(rgb).map_err(|e| AiKitError::Internal {
            reason: format!("png data: {e}"),
        })?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_noise_is_deterministic_and_roughly_standard_normal() {
        let a = seeded_normal(7, 20_000);
        assert_eq!(a, seeded_normal(7, 20_000));
        assert_ne!(a, seeded_normal(8, 20_000));
        let n = a.len() as f32;
        let mean = a.iter().sum::<f32>() / n;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
        assert_eq!(seeded_normal(1, 5).len(), 5, "odd lengths are handled");
    }

    #[test]
    fn png_encoding_round_trips_the_header() {
        let rgb = vec![128u8; 16 * 8 * 3];
        let png = encode_png(&rgb, 16, 8).unwrap();
        assert_eq!(crate::modality::png_dims(&png), Some((16, 8)));
    }
}
