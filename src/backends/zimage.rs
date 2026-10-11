//! Z-Image-Turbo (Apache-2.0) on Candle, fitted to a 12 GB card. **POC.**
//!
//! `architecture: z_image` in an image-generation catalog entry. This is the licence-clean, higher-quality counterpart to the
//! `sd_turbo` development model: a 6B single-stream diffusion transformer, a Qwen3-4B text encoder and a 16-channel VAE.
//! Candle 0.11 ships the blocks (`candle_transformers::models::z_image`) but its `ZImageTransformer2DModel` wants the whole
//! model resident in bf16 (about 12 GB for the transformer alone, 8 GB for the text encoder), which cannot run on a 12 GB card.
//! This module keeps Candle's blocks and replaces how they are loaded and driven.
//!
//! # How it fits
//!
//! 1. **Text encoding runs as its own phase.** The bf16 Qwen3 encoder is memory-mapped, uploaded, run once, and dropped before
//!    anything else is loaded. Peak is about 8 GB, and it never shares the card with the transformer.
//! 2. **The transformer stays quantized.** The Q4_K_M GGUF's K-quant tensors (Q4_K/Q5_K/Q6_K, about 4 GB for the 30 main layers)
//!    are uploaded as quantized tensors. Its few bf16 tensors (embedders, the four refiner blocks) stay as dense bf16.
//! 3. **One block is dequantized at a time.** For each block the quantized weights are dequantized on the GPU into a bf16
//!    `VarBuilder`, the block runs, and it is dropped, so the working set is one layer (about 0.4 GB), not the model.
//!    The GGUF's fused `attention.qkv` is split into `to_q`/`to_k`/`to_v` and a few keys are renamed to the diffusers names
//!    Candle's blocks expect (see [`block_tensors`]).
//! 4. Sampling is the diffusers `ZImagePipeline` recipe: shifted flow-matching sigmas (shift 3.0), Euler steps on the negated
//!    prediction, no guidance (this is a Turbo model; `guidance_scale` is ignored), 8 transformer evaluations by default.
//!
//! # Honest limits
//!
//! - Weights are loaded per request and freed after it (idle VRAM is zero, first-image latency is the load time). A residency
//!   policy is future work.
//! - bf16 text encoder only (8 GB of disk). A quantized encoder is planned but needs its own converter.
//! - Dimensions must be multiples of 16. No img2img, LoRA or batching.
//! - Candle's attention here is the plain matmul-softmax form, so 1024x1024 uses a lot of activation memory.

use crate::error::{AiKitError, Result};
use crate::modality::ImageGenParams;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::RmsNorm;
use candle_transformers::models::z_image::transformer::{
    patchify, unpatchify, Config as DitConfig, FinalLayer, RopeEmbedder, TimestepEmbedder,
    ZImageTransformerBlock, ADALN_EMBED_DIM, SEQ_MULTI_OF,
};
use candle_transformers::models::z_image::{AutoEncoderKL, VaeConfig};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// diffusers `FlowMatchEulerDiscreteScheduler` shift for Z-Image-Turbo (`use_dynamic_shifting: false`).
const SHIFT: f64 = 3.0;

/// Where the files of a `z_image` catalog entry are (all verified before this is built).
pub(crate) struct ZImageFiles {
    pub gguf: String,
    pub text_encoder_shards: Vec<String>,
    pub vae: String,
    pub tokenizer: String,
}

/// Test instrumentation: the latents after each step, and optionally reference latents to start each step from.
#[derive(Default)]
struct Probe {
    teacher: Option<Vec<Tensor>>,
    trace: Vec<Tensor>,
}

/// What one generation asks for.
#[derive(Clone, Copy)]
struct Job {
    h: usize,
    w: usize,
    steps: usize,
    seed: u64,
}

pub(crate) struct ZImagePipeline {
    files: ZImageFiles,
    device: Device,
    dtype: DType,
}

fn internal(what: &str, e: impl std::fmt::Display) -> AiKitError {
    AiKitError::Internal {
        reason: format!("z_image {what}: {e}"),
    }
}

impl ZImagePipeline {
    pub(crate) fn new(files: ZImageFiles, device: Device) -> Self {
        // bf16 is what the model is trained and distributed in; every CUDA card the kit targets (sm_80+) has it.
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        Self {
            files,
            device,
            dtype,
        }
    }

    /// Check the request against what this model can do (before any weight is read).
    pub(crate) fn validate(p: &ImageGenParams) -> Result<()> {
        if !p.width.is_multiple_of(16) || !p.height.is_multiple_of(16) {
            return Err(AiKitError::InvalidParams {
                reason: format!(
                    "Z-Image needs width and height that are multiples of 16 (got {}x{})",
                    p.width, p.height
                ),
            });
        }
        Ok(())
    }

    /// Generate an image; returns RGB bytes (row-major, 3 per pixel). `noise` overrides the seeded start (used by the parity test).
    pub(crate) fn generate(
        &self,
        p: &ImageGenParams,
        seed: u64,
        noise: Option<Tensor>,
    ) -> Result<Vec<u8>> {
        Self::validate(p)?;
        let (h, w) = (p.height as usize, p.width as usize);
        let cap_feats = self.encode_prompt(&p.prompt)?; // phase 1: text encoder in, out
        let job = Job {
            h,
            w,
            steps: p.steps as usize,
            seed,
        };
        let latents = self.denoise(&cap_feats, job, noise, None)?; // phase 2: transformer
        drop(cap_feats);
        self.decode(&latents) // phase 3: VAE
    }

    // ------------------------------------------------------------------ phase 1: text

    /// The Qwen3 chat template diffusers applies (thinking enabled adds nothing after the assistant tag).
    fn chat_prompt(prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    }

    /// Token ids of the chat-templated prompt (diffusers keeps only the valid tokens, truncated to 512).
    fn prompt_ids(&self, prompt: &str) -> Result<Vec<u32>> {
        let tok =
            Tokenizer::from_file(&self.files.tokenizer).map_err(|e| internal("tokenizer", e))?;
        let enc = tok
            .encode(Self::chat_prompt(prompt), false)
            .map_err(|e| internal("tokenize", e))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(512);
        Ok(ids)
    }

    /// `(1, tokens, 2560)` hidden states of the second-to-last layer, unpadded. Streams the encoder one layer at a time,
    /// so this phase needs well under 1 GB of GPU memory instead of the encoder's 8 GB.
    pub(crate) fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let ids = self.prompt_ids(prompt)?;
        let paths: Vec<&str> = self
            .files
            .text_encoder_shards
            .iter()
            .map(String::as_str)
            .collect();
        crate::backends::zimage_text::encode_streaming(&ids, &paths, &self.device, self.dtype)
            .and_then(|t| t.contiguous())
            .map_err(|e| internal("text encoder", e))
    }

    /// Reference path: Candle's own encoder with every layer resident (8 GB). Used only to check the streaming one.
    #[cfg(test)]
    fn encode_prompt_resident(&self, prompt: &str) -> Result<Tensor> {
        use candle_transformers::models::z_image::{TextEncoderConfig, ZImageTextEncoder};
        let ids = self.prompt_ids(prompt)?;
        let input = Tensor::new(ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| internal("ids", e))?;
        let paths: Vec<&str> = self
            .files
            .text_encoder_shards
            .iter()
            .map(String::as_str)
            .collect();
        // SAFETY: verified before; not modified while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, self.dtype, &self.device) }
            .map_err(|e| internal("text encoder weights", e))?;
        let te = ZImageTextEncoder::new(&TextEncoderConfig::z_image(), vb)
            .map_err(|e| internal("text encoder", e))?;
        te.forward(&input)
            .and_then(|t| t.contiguous())
            .map_err(|e| internal("text encoder forward", e))
    }

    // ------------------------------------------------------------------ phase 2: denoise

    /// The sigma schedule diffusers builds: `linspace(1, 1/n, n)` shifted by `SHIFT`, then a terminal 0.
    pub(crate) fn sigmas(steps: usize) -> Vec<f64> {
        let n = steps as f64;
        let mut s: Vec<f64> = (0..steps)
            .map(|i| {
                let x = if steps == 1 {
                    1.0
                } else {
                    1.0 + (1.0 / n - 1.0) * i as f64 / (n - 1.0)
                };
                SHIFT * x / (1.0 + (SHIFT - 1.0) * x)
            })
            .collect();
        s.push(0.0);
        s
    }

    fn denoise(
        &self,
        cap_feats: &Tensor,
        job: Job,
        noise: Option<Tensor>,
        mut probe: Option<&mut Probe>,
    ) -> Result<Tensor> {
        let Job { h, w, steps, seed } = job;
        let cfg = DitConfig::z_image_turbo();
        let model = StreamingDit::load(&self.files.gguf, &cfg, &self.device, self.dtype)?;
        let (lh, lw) = (h / 8, w / 8);
        let mut latents = match noise {
            Some(n) => n
                .to_device(&self.device)
                .and_then(|t| t.to_dtype(DType::F32)),
            None => {
                let v = crate::backends::diffusion::seeded_normal(seed, 16 * lh * lw);
                Tensor::from_vec(v, (1, 16, lh, lw), &self.device)
            }
        }
        .map_err(|e| internal("noise", e))?;
        let sigmas = Self::sigmas(steps);
        let cap_feats = cap_feats
            .to_dtype(self.dtype)
            .map_err(|e| internal("cap", e))?;
        for i in 0..steps {
            // Test hook: start this step from the reference's latents instead of our own, so error cannot compound.
            if let Some(Probe {
                teacher: Some(teacher),
                ..
            }) = probe.as_deref_mut()
            {
                if i > 0 {
                    latents = teacher[i - 1]
                        .to_device(&self.device)
                        .map_err(|e| internal("teacher", e))?;
                }
            }
            let t_norm = 1.0 - sigmas[i]; // = (1000 - timestep) / 1000
            let t = Tensor::from_vec(vec![t_norm as f32], (1,), &self.device)
                .map_err(|e| internal("t", e))?;
            let x = latents
                .unsqueeze(2)
                .and_then(|x| x.to_dtype(self.dtype))
                .map_err(|e| internal("latents", e))?;
            let out = model
                .forward(&x, &t, &cap_feats)
                .map_err(|e| internal(&format!("step {i}"), e))?;
            // diffusers: noise_pred = -model_out (squeezed over the frame axis); Euler: x += (sigma_next - sigma) * noise_pred
            let pred = out
                .squeeze(2)
                .and_then(|o| o.to_dtype(DType::F32))
                .and_then(|o| o.neg())
                .map_err(|e| internal("pred", e))?;
            let dt = sigmas[i + 1] - sigmas[i];
            latents = (&latents + (pred * dt).map_err(|e| internal("euler", e))?)
                .map_err(|e| internal("euler", e))?;
            if let Some(pr) = probe.as_deref_mut() {
                pr.trace.push(latents.clone());
            }
        }
        Ok(latents)
    }

    // ------------------------------------------------------------------ phase 3: VAE

    fn decode(&self, latents: &Tensor) -> Result<Vec<u8>> {
        // SAFETY: verified immediately before; not modified while mapped.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[self.files.vae.as_str()],
                self.dtype,
                &self.device,
            )
        }
        .map_err(|e| internal("vae weights", e))?;
        let vae = AutoEncoderKL::new(&VaeConfig::z_image(), vb).map_err(|e| internal("vae", e))?;
        let (lh, lw) = (
            latents.dim(2).map_err(|e| internal("latents", e))?,
            latents.dim(3).map_err(|e| internal("latents", e))?,
        );
        let latents = latents
            .to_dtype(self.dtype)
            .map_err(|e| internal("vae in", e))?;
        // CHW f32 in [-1, 1]
        let chw: Vec<f32> = if lh * lw > TILE_ABOVE_LATENT_AREA {
            self.decode_tiled(&vae, &latents, TILE_LATENT, TILE_OVERLAP_LATENT)?
        } else {
            vae.decode(&latents)
                .and_then(|i| {
                    i.to_dtype(DType::F32)?
                        .to_device(&Device::Cpu)?
                        .flatten_all()?
                        .to_vec1::<f32>()
                })
                .map_err(|e| internal("vae decode", e))?
        };
        let (h, w) = (lh * 8, lw * 8);
        let plane = h * w;
        let mut out = vec![0u8; 3 * plane];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = (chw[c * plane + y * w + x].clamp(-1.0, 1.0) + 1.0) * 127.5;
                    out[(y * w + x) * 3 + c] = v as u8; // truncates, as Candle's F32->U8 cast does
                }
            }
        }
        Ok(out)
    }

    /// Decode in overlapping tiles blended with linear ramps, so memory does not grow with image size (a 1024x1024 decode in one
    /// piece ran the 12 GB card out of memory). Returns CHW f32.
    fn decode_tiled(
        &self,
        vae: &AutoEncoderKL,
        latents: &Tensor,
        tile: usize,
        overlap: usize,
    ) -> Result<Vec<f32>> {
        let (lh, lw) = (
            latents.dim(2).map_err(|e| internal("tile", e))?,
            latents.dim(3).map_err(|e| internal("tile", e))?,
        );
        let (h, w) = (lh * 8, lw * 8);
        let stride = tile - overlap;
        let starts = |n: usize| -> Vec<usize> {
            let mut s = vec![0];
            while s.last().copied().unwrap_or(0) + tile < n {
                let next = (s.last().copied().unwrap_or(0) + stride).min(n - tile);
                s.push(next);
            }
            s
        };
        let mut acc = vec![0f32; 3 * h * w];
        let mut wsum = vec![0f32; h * w];
        for &ty in &starts(lh) {
            for &tx in &starts(lw) {
                let (th, tw) = (tile.min(lh), tile.min(lw));
                let piece = latents
                    .narrow(2, ty, th)
                    .and_then(|t| t.narrow(3, tx, tw))
                    .and_then(|t| t.contiguous())
                    .map_err(|e| internal("tile", e))?;
                let img = vae
                    .decode(&piece)
                    .and_then(|i| {
                        i.to_dtype(DType::F32)?
                            .to_device(&Device::Cpu)?
                            .flatten_all()?
                            .to_vec1::<f32>()
                    })
                    .map_err(|e| internal("vae decode (tile)", e))?;
                let (ph, pw) = (th * 8, tw * 8);
                let ramp = |i: usize, n: usize, at_start: bool, at_end: bool| -> f32 {
                    let o = (overlap * 8) as f32;
                    let from_start = if at_start {
                        f32::MAX
                    } else {
                        (i + 1) as f32 / o
                    };
                    let from_end = if at_end { f32::MAX } else { (n - i) as f32 / o };
                    from_start.min(from_end).min(1.0)
                };
                for y in 0..ph {
                    let wy = ramp(y, ph, ty == 0, ty + th >= lh);
                    for x in 0..pw {
                        let wt = wy * ramp(x, pw, tx == 0, tx + tw >= lw);
                        let (gy, gx) = (ty * 8 + y, tx * 8 + x);
                        wsum[gy * w + gx] += wt;
                        for c in 0..3 {
                            acc[c * h * w + gy * w + gx] += wt * img[c * ph * pw + y * pw + x];
                        }
                    }
                }
            }
        }
        for c in 0..3 {
            for i in 0..h * w {
                acc[c * h * w + i] /= wsum[i].max(1e-6);
            }
        }
        Ok(acc)
    }
}

/// Latent area (h*w) above which the VAE decodes in tiles. 96x96 latent = 768x768 px, which fits in one piece.
const TILE_ABOVE_LATENT_AREA: usize = 96 * 96;
const TILE_LATENT: usize = 64;
const TILE_OVERLAP_LATENT: usize = 16;

// ---------------------------------------------------------------------- the streaming transformer

/// A GGUF tensor: K-quantized on the device (dequantized on use), or dense bf16/f32 in host RAM (uploaded on use).
enum Weight {
    Quant(Arc<QTensor>),
    Dense(Tensor),
}

/// All GGUF tensors on the device, with a way to turn any of them into a dense tensor on demand.
struct Weights {
    map: HashMap<String, Weight>,
    device: Device,
    dtype: DType,
}

impl Weights {
    fn open(gguf: &str, device: &Device, dtype: DType) -> Result<Self> {
        let mut file = std::fs::File::open(gguf)?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| internal("gguf header", e))?;
        let mut map = HashMap::new();
        for (name, info) in &content.tensor_infos {
            let w = match info.ggml_dtype {
                GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => Weight::Quant(Arc::new(
                    content
                        .tensor(&mut file, name, device)
                        .map_err(|e| internal(&format!("upload {name}"), e))?,
                )),
                _ => {
                    let qt = content
                        .tensor(&mut file, name, &Device::Cpu)
                        .map_err(|e| internal(&format!("read {name}"), e))?;
                    Weight::Dense(
                        qt.dequantize(&Device::Cpu)
                            .and_then(|t| t.to_dtype(dtype))
                            .map_err(|e| internal(&format!("convert {name}"), e))?,
                    )
                }
            };
            map.insert(name.clone(), w);
        }
        Ok(Self {
            map,
            device: device.clone(),
            dtype,
        })
    }

    fn get(&self, name: &str) -> Result<Tensor> {
        let w = self
            .map
            .get(name)
            .ok_or_else(|| internal(name, "missing from the GGUF"))?;
        self.dequant(w).map_err(|e| internal(name, e))
    }

    fn dequant(&self, w: &Weight) -> candle_core::Result<Tensor> {
        match w {
            Weight::Quant(q) => q.dequantize(&self.device)?.to_dtype(self.dtype),
            // dense tensors live in host RAM (the refiner blocks are 1.6 GB of bf16) and are uploaded when used
            Weight::Dense(t) => t.to_device(&self.device),
        }
    }

    /// The small non-block tensors, under the names Candle's model uses.
    fn vb_top(&self) -> Result<VarBuilder<'static>> {
        let mut map = HashMap::new();
        for (name, w) in &self.map {
            let new = if let Some(r) = name.strip_prefix("x_embedder.") {
                format!("all_x_embedder.2-1.{r}")
            } else if let Some(r) = name.strip_prefix("final_layer.") {
                format!("all_final_layer.2-1.{r}")
            } else if name.starts_with("cap_embedder.") || name.starts_with("t_embedder.") {
                name.clone()
            } else {
                continue;
            };
            map.insert(new, self.dequant(w).map_err(|e| internal(name, e))?);
        }
        Ok(VarBuilder::from_tensors(map, self.dtype, &self.device))
    }
}

/// The Z-Image transformer with its K-quant weights left quantized on the GPU.
struct StreamingDit {
    cfg: DitConfig,
    w: Weights,
    t_embedder: TimestepEmbedder,
    cap_norm: RmsNorm,
    cap_linear: candle_nn::Linear,
    x_embedder: candle_nn::Linear,
    final_layer: FinalLayer,
    rope: RopeEmbedder,
    x_pad_token: Tensor,
    cap_pad_token: Tensor,
}

impl StreamingDit {
    fn load(gguf: &str, cfg: &DitConfig, device: &Device, dtype: DType) -> Result<Self> {
        let w = Weights::open(gguf, device, dtype)?;
        let top = w.vb_top()?;
        let adaln = cfg.dim.min(ADALN_EMBED_DIM);
        let patch_dim = cfg.all_f_patch_size[0] * cfg.all_patch_size[0].pow(2) * cfg.in_channels;
        let e = |what: &str| {
            let what = what.to_string();
            move |e: candle_core::Error| internal(&what, e)
        };
        Ok(Self {
            t_embedder: TimestepEmbedder::new(adaln, 1024, top.pp("t_embedder"))
                .map_err(e("t_embedder"))?,
            cap_norm: RmsNorm::new(
                cfg.cap_feat_dim,
                cfg.norm_eps,
                top.pp("cap_embedder").pp("0"),
            )
            .map_err(e("cap norm"))?,
            cap_linear: candle_nn::linear(
                cfg.cap_feat_dim,
                cfg.dim,
                top.pp("cap_embedder").pp("1"),
            )
            .map_err(e("cap linear"))?,
            x_embedder: candle_nn::linear(patch_dim, cfg.dim, top.pp("all_x_embedder").pp("2-1"))
                .map_err(e("x embedder"))?,
            final_layer: FinalLayer::new(cfg.dim, patch_dim, top.pp("all_final_layer").pp("2-1"))
                .map_err(e("final layer"))?,
            rope: RopeEmbedder::new(
                cfg.rope_theta,
                cfg.axes_dims.clone(),
                cfg.axes_lens.clone(),
                device,
                dtype,
            )
            .map_err(e("rope"))?,
            x_pad_token: w.get("x_pad_token")?,
            cap_pad_token: w.get("cap_pad_token")?,
            cfg: cfg.clone(),
            w,
        })
    }

    /// Build one transformer block, dequantizing its weights now. Dropping the block frees them.
    fn block(&self, prefix: &str, modulation: bool) -> Result<ZImageTransformerBlock> {
        let tensors = block_tensors(prefix, &self.w.map, |w| self.w.dequant(w))
            .map_err(|e| internal(prefix, e))?;
        let vb = VarBuilder::from_tensors(tensors, self.w.dtype, &self.w.device);
        ZImageTransformerBlock::new(&self.cfg, modulation, vb).map_err(|e| internal(prefix, e))
    }

    /// The diffusers `ZImageTransformer2DModel.forward` for one image and one caption, with blocks built one at a time.
    ///
    /// Candle's own `forward` differs from diffusers (and so from how the model was trained and is distributed) in how it
    /// treats padding: it masks the caption's pad tokens out of attention and never uses `cap_pad_token`/`x_pad_token`.
    /// diffusers pads the caption (and the image tokens, when their count is not a multiple of 32) up to a multiple of 32,
    /// replaces the pad rows with the learned pad tokens, gives them position (0,0,0), and **attends to them**. The parity
    /// test measured 17-28% error per step with Candle's masking and checks this version against the reference.
    fn forward(&self, x: &Tensor, t: &Tensor, cap_feats: &Tensor) -> Result<Tensor> {
        let f = |what: &'static str| move |e: candle_core::Error| internal(what, e);
        let device = x.device();
        let (_b, _c, fr, h, w) = x.dims5().map_err(f("dims"))?;
        let (ps, fps) = (self.cfg.all_patch_size[0], self.cfg.all_f_patch_size[0]);
        let pad_to = |n: usize| (SEQ_MULTI_OF - n % SEQ_MULTI_OF) % SEQ_MULTI_OF;

        let t_scaled = (t * self.cfg.t_scale).map_err(f("t"))?;
        let adaln = self.t_embedder.forward(&t_scaled).map_err(f("t embed"))?;

        // caption: embed, pad with cap_pad_token, positions (1+i,0,0) then (0,0,0) for the pad rows
        let cap_len = cap_feats.dim(1).map_err(f("text len"))?;
        let cap_pad = pad_to(cap_len);
        let cap_emb = self
            .cap_norm
            .forward(cap_feats)
            .and_then(|c| c.apply(&self.cap_linear))
            .map_err(f("cap embed"))?;
        let mut cap = self.pad_rows(cap_emb, &self.cap_pad_token, cap_pad)?;
        let cap_pos =
            grid_with_pad((cap_len, 1, 1), (1, 0, 0), cap_pad, device).map_err(f("grid"))?;
        let cap_total = cap_len + cap_pad;

        // image: patchify, embed, pad with x_pad_token; positions start after the PADDED caption
        let (patches, orig) = patchify(x, ps, fps).map_err(f("patchify"))?;
        let xs = patches.apply(&self.x_embedder).map_err(f("x embed"))?;
        let img_len = xs.dim(1).map_err(f("img len"))?;
        let img_pad = pad_to(img_len);
        let mut xs = self.pad_rows(xs, &self.x_pad_token, img_pad)?;
        let x_pos = grid_with_pad(
            (fr / fps, h / ps, w / ps),
            (cap_total + 1, 0, 0),
            img_pad,
            device,
        )
        .map_err(f("grid"))?;

        let (x_cos, x_sin) = self.rope.forward(&x_pos).map_err(f("rope"))?;
        let (cap_cos, cap_sin) = self.rope.forward(&cap_pos).map_err(f("rope"))?;

        // One image, one caption: nothing is batch-padded, so no attention mask anywhere (pad tokens are attended to).
        for i in 0..self.cfg.n_refiner_layers {
            let blk = self.block(&format!("noise_refiner.{i}"), true)?;
            xs = blk
                .forward(&xs, None, &x_cos, &x_sin, Some(&adaln))
                .map_err(f("noise refiner"))?;
        }
        for i in 0..self.cfg.n_refiner_layers {
            let blk = self.block(&format!("context_refiner.{i}"), false)?;
            cap = blk
                .forward(&cap, None, &cap_cos, &cap_sin, None)
                .map_err(f("context refiner"))?;
        }

        let mut unified = Tensor::cat(&[&xs, &cap], 1).map_err(f("cat"))?; // image tokens first, then caption
        let pos = Tensor::cat(&[&x_pos, &cap_pos], 0).map_err(f("cat"))?;
        let (u_cos, u_sin) = self.rope.forward(&pos).map_err(f("rope"))?;
        for i in 0..self.cfg.n_layers {
            let blk = self.block(&format!("layers.{i}"), true)?;
            unified = blk
                .forward(&unified, None, &u_cos, &u_sin, Some(&adaln))
                .map_err(f("layer"))?;
        }

        let out = unified.narrow(1, 0, img_len).map_err(f("narrow"))?;
        let out = self.final_layer.forward(&out, &adaln).map_err(f("final"))?;
        unpatchify(&out, orig, ps, fps, self.cfg.in_channels).map_err(f("unpatchify"))
    }

    /// Append `n` copies of the learned pad token row to `(1, len, dim)`.
    fn pad_rows(&self, xs: Tensor, token: &Tensor, n: usize) -> Result<Tensor> {
        if n == 0 {
            return Ok(xs);
        }
        let dim = xs.dim(2).map_err(|e| internal("pad", e))?;
        let pad = token
            .to_dtype(xs.dtype())
            .and_then(|t| {
                t.reshape((1, 1, dim))?
                    .broadcast_as((1, n, dim))?
                    .contiguous()
            })
            .map_err(|e| internal("pad token", e))?;
        Tensor::cat(&[&xs, &pad], 1).map_err(|e| internal("pad", e))
    }
}

/// Position ids `(n + pad, 3)`: a `size` grid starting at `start`, then `pad` rows of (0,0,0), as diffusers' `_pad_with_ids`.
fn grid_with_pad(
    size: (usize, usize, usize),
    start: (usize, usize, usize),
    pad: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let (f, h, w) = size;
    let mut ids = Vec::with_capacity((f * h * w + pad) * 3);
    for fi in 0..f {
        for hi in 0..h {
            for wi in 0..w {
                ids.extend([
                    (start.0 + fi) as u32,
                    (start.1 + hi) as u32,
                    (start.2 + wi) as u32,
                ]);
            }
        }
    }
    ids.extend(std::iter::repeat_n(0u32, pad * 3));
    Tensor::from_vec(ids, (f * h * w + pad, 3), device)
}

/// The tensors of one block under `prefix`, renamed from the GGUF's names to the diffusers names Candle's blocks load:
/// the fused `attention.qkv` is split into `to_q`/`to_k`/`to_v` (rows q, k, v), `attention.out` becomes `to_out.0`, and
/// `q_norm`/`k_norm` become `norm_q`/`norm_k`.
pub(crate) fn block_tensors<W>(
    prefix: &str,
    weights: &HashMap<String, W>,
    mut load: impl FnMut(&W) -> candle_core::Result<Tensor>,
) -> candle_core::Result<HashMap<String, Tensor>> {
    let dot = format!("{prefix}.");
    let mut out = HashMap::new();
    for (name, w) in weights {
        let Some(rest) = name.strip_prefix(&dot) else {
            continue;
        };
        let t = load(w)?;
        match rest {
            "attention.qkv.weight" => {
                let rows = t.dim(0)? / 3;
                for (i, n) in ["to_q", "to_k", "to_v"].iter().enumerate() {
                    out.insert(
                        format!("attention.{n}.weight"),
                        t.narrow(0, i * rows, rows)?.contiguous()?,
                    );
                }
            }
            "attention.out.weight" => {
                out.insert("attention.to_out.0.weight".into(), t);
            }
            "attention.q_norm.weight" => {
                out.insert("attention.norm_q.weight".into(), t);
            }
            "attention.k_norm.weight" => {
                out.insert("attention.norm_k.weight".into(), t);
            }
            other => {
                out.insert(other.to_string(), t);
            }
        }
    }
    if out.is_empty() {
        candle_core::bail!("no tensors with prefix '{prefix}' in the GGUF");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_match_the_diffusers_schedule() {
        // linspace(1, 1/9, 9) shifted by 3: first 1.0 -> 1.0, last (1/9) -> 3*(1/9)/(1+2/9) = 0.2727...; terminal 0.
        let s = ZImagePipeline::sigmas(9);
        assert_eq!(s.len(), 10);
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[8] - 0.272_727_272_7).abs() < 1e-9, "{}", s[8]);
        assert_eq!(s[9], 0.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]), "strictly decreasing");
    }

    #[test]
    fn dimensions_that_are_not_multiples_of_16_are_rejected_before_any_weight_is_read() {
        let mut p = ImageGenParams::new("z-image-turbo", "x");
        p.width = 520;
        assert!(matches!(
            ZImagePipeline::validate(&p),
            Err(AiKitError::InvalidParams { .. })
        ));
        p.width = 512;
        assert!(ZImagePipeline::validate(&p).is_ok());
    }

    #[test]
    fn grid_positions_follow_diffusers_with_pad_rows_at_the_origin() {
        let g = grid_with_pad((1, 2, 2), (5, 0, 0), 2, &Device::Cpu).unwrap();
        assert_eq!(g.dims(), &[6, 3]);
        let v = g.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(&v[..12], &[5, 0, 0, 5, 0, 1, 5, 1, 0, 5, 1, 1]);
        assert_eq!(
            &v[12..],
            &[0, 0, 0, 0, 0, 0],
            "pad rows sit at (0,0,0), not continuing the grid"
        );
    }

    #[test]
    fn chat_prompt_is_the_qwen3_template_with_thinking_on() {
        assert_eq!(
            ZImagePipeline::chat_prompt("a cat"),
            "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qkv_is_split_into_q_k_v_in_order_and_keys_are_renamed() {
        let dev = Device::Cpu;
        // qkv rows: 0..3 = q, 3..6 = k, 6..9 = v
        let qkv =
            Tensor::from_vec((0..9).map(|r| r as f32).collect::<Vec<_>>(), (9, 1), &dev).unwrap();
        let mut w: HashMap<String, Tensor> = HashMap::new();
        w.insert("layers.0.attention.qkv.weight".into(), qkv);
        w.insert(
            "layers.0.attention.out.weight".into(),
            Tensor::zeros((1, 1), DType::F32, &dev).unwrap(),
        );
        w.insert(
            "layers.0.attention.q_norm.weight".into(),
            Tensor::zeros(1, DType::F32, &dev).unwrap(),
        );
        w.insert(
            "layers.0.attention.k_norm.weight".into(),
            Tensor::zeros(1, DType::F32, &dev).unwrap(),
        );
        w.insert(
            "layers.0.ffn_norm1.weight".into(),
            Tensor::zeros(1, DType::F32, &dev).unwrap(),
        );
        w.insert(
            "layers.1.ffn_norm1.weight".into(),
            Tensor::zeros(1, DType::F32, &dev).unwrap(),
        );
        let m = block_tensors("layers.0", &w, |t| Ok(t.clone())).unwrap();
        let col = |k: &str| m[k].flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(col("attention.to_q.weight"), [0., 1., 2.]);
        assert_eq!(col("attention.to_k.weight"), [3., 4., 5.]);
        assert_eq!(col("attention.to_v.weight"), [6., 7., 8.]);
        for k in [
            "attention.to_out.0.weight",
            "attention.norm_q.weight",
            "attention.norm_k.weight",
            "ffn_norm1.weight",
        ] {
            assert!(m.contains_key(k), "missing {k}");
        }
        assert_eq!(
            m.len(),
            7,
            "only layers.0 tensors, none from layers.1: {:?}",
            m.keys()
        );
        assert!(
            block_tensors("layers.9", &w, |t| Ok(t.clone())).is_err(),
            "a missing block is an error, not an empty block"
        );
    }

    // ------------------------------------------------------------------ parity with the diffusers reference (needs weights)

    fn rel_err(a: &Tensor, b: &Tensor) -> f64 {
        let (a, b) = (
            a.to_dtype(DType::F32).unwrap(),
            b.to_dtype(DType::F32).unwrap(),
        );
        let d = (a - &b)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64;
        let n = b
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64;
        (d / n).sqrt()
    }

    fn env(k: &str) -> String {
        std::env::var(k)
            .unwrap_or_else(|_| panic!("set {k} (see zimage_ref_dump.py in the studio vault)"))
    }

    /// Compare against `zimage_ref_dump.py` output: prompt embeddings, the latents after every step from the same noise, and the image.
    /// `AI_KIT_ZIMAGE_NATIVE` = the Tongyi-MAI/Z-Image-Turbo snapshot dir, `AI_KIT_ZIMAGE_GGUF` = z-image-turbo-Q4_K_M.gguf,
    /// `AI_KIT_ZIMAGE_REF` = the dump dir (prompt "a cozy cottage at dusk, watercolor", 512x512, 9 steps, seed 0).
    #[test]
    #[ignore = "needs the Z-Image weights and a diffusers reference dump"]
    fn matches_the_diffusers_reference() {
        let native = env("AI_KIT_ZIMAGE_NATIVE");
        let device = if std::env::var("AI_KIT_DEVICE").is_ok_and(|d| d.starts_with("cuda")) {
            crate::backends::common::device_for(crate::backends::common::DevicePreference::Cuda(0))
                .unwrap()
                .0
        } else {
            Device::Cpu
        };
        let te = ["00001", "00002", "00003"]
            .map(|n| format!("{native}/text_encoder/model-{n}-of-00003.safetensors"));
        let pipe = ZImagePipeline::new(
            ZImageFiles {
                gguf: env("AI_KIT_ZIMAGE_GGUF"),
                text_encoder_shards: te.to_vec(),
                vae: format!("{native}/vae/diffusion_pytorch_model.safetensors"),
                tokenizer: format!("{native}/tokenizer/tokenizer.json"),
            },
            device.clone(),
        );
        let refdir = env("AI_KIT_ZIMAGE_REF");
        let r = candle_core::safetensors::load(format!("{refdir}/ref.safetensors"), &Device::Cpu)
            .unwrap();

        let t0 = std::time::Instant::now();
        let emb = pipe
            .encode_prompt("a cozy cottage at dusk, watercolor")
            .unwrap();
        println!("text encoder: {:?}", t0.elapsed());
        let emb_err = rel_err(
            &emb.squeeze(0).unwrap().to_device(&Device::Cpu).unwrap(),
            &r["prompt_embeds"],
        );
        println!(
            "prompt embeds: shape {:?} rel err vs reference {emb_err:.4}",
            emb.dims()
        );
        assert_eq!(
            emb.dim(1).unwrap(),
            r["prompt_embeds"].dim(0).unwrap(),
            "token count differs: the chat template or tokenizer is off"
        );
        // The transformer is checked with the reference's OWN prompt embeddings so text-encoder differences do not mix in.
        let ref_emb = r["prompt_embeds"]
            .unsqueeze(0)
            .unwrap()
            .to_device(&device)
            .unwrap();

        let job = Job {
            h: 512,
            w: 512,
            steps: 9,
            seed: 0,
        };
        let noise = r["noise"].clone();
        let cpu = Device::Cpu;
        let reference_latents = |i: usize| r[&format!("latents_after_step_{i}")].clone();

        // (a) one step at a time from the reference's own latents: measures the numerics of a single step, errors cannot compound.
        let teacher: Vec<Tensor> = (0..8).map(reference_latents).collect();
        let mut forced = Probe {
            teacher: Some(teacher),
            trace: Vec::new(),
        };
        pipe.denoise(&ref_emb, job, Some(noise.clone()), Some(&mut forced))
            .unwrap();
        let mut worst_step = 0f64;
        for (i, ours) in forced.trace.iter().enumerate() {
            let before = if i == 0 {
                noise.clone()
            } else {
                reference_latents(i - 1)
            };
            let d_ours = (ours.to_device(&cpu).unwrap() - &before).unwrap();
            let d_ref = (reference_latents(i) - &before).unwrap();
            let e = rel_err(&d_ours, &d_ref);
            println!("  step {i} (from the reference's latents): update rel err {e:.4}");
            worst_step = worst_step.max(e);
        }

        // (b) free-running from the same noise: only has to stay a sane image (chaotic divergence from bf16 is expected).
        let t1 = std::time::Instant::now();
        let mut free = Probe::default();
        let lat = pipe
            .denoise(&ref_emb, job, Some(noise.clone()), Some(&mut free))
            .unwrap();
        println!("denoise (9 steps): {:?}", t1.elapsed());
        for (i, l) in free.trace.iter().enumerate() {
            println!(
                "  free-running latents after step {i}: rel err vs reference {:.4}",
                rel_err(&l.to_device(&cpu).unwrap(), &reference_latents(i))
            );
        }
        let worst = worst_step;
        // (c) the whole pipeline with OUR text encoder: this is what users get.
        let mut ours = Probe::default();
        let lat_ours = pipe
            .denoise(&emb, job, Some(noise), Some(&mut ours))
            .unwrap();
        println!(
            "end to end (our text encoder), latents rel err vs reference {:.4}",
            rel_err(&lat_ours.to_device(&cpu).unwrap(), &reference_latents(8))
        );
        let rgb_ours = pipe.decode(&lat_ours).unwrap();
        std::fs::write(format!("{refdir}/rust_e2e.png"), image_from_rgb(&rgb_ours)).unwrap();
        let rgb = pipe.decode(&lat).unwrap();
        let png = image_from_rgb(&rgb);
        let reference = png_decode(&format!("{refdir}/ref.png"));
        let mse = rgb
            .iter()
            .zip(&reference)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>()
            / rgb.len() as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10();
        println!("image vs reference: PSNR {psnr:.1} dB");
        std::fs::write(format!("{refdir}/rust.png"), png).unwrap();
        // Thresholds from the measurements in wiki/experiments/z-image-turbo-rust-2026-10-10.md: bf16 rounding alone gives 3% per step
        // against diffusers (the f32 control), Candle's own masking of the caption pad tokens gave up to 28%, this port peaks at 8%.
        assert!(emb_err < 0.03, "text embeddings drift: {emb_err}");
        assert!(
            worst < 0.12,
            "a single step differs from the reference by {worst} (bf16 rounding is ~0.04)"
        );
        // Free-running images diverge chaotically from bf16 rounding (~20 dB against the reference), so only require the same scene.
        assert!(psnr > 15.0, "image is not the reference scene: {psnr} dB");
    }

    /// GPU K-quant dequantization must equal the CPU reference, per quant type.
    #[test]
    #[ignore = "needs the Z-Image GGUF and a CUDA device"]
    fn gpu_dequantization_matches_the_cpu_reference() {
        let gguf = env("AI_KIT_ZIMAGE_GGUF");
        let gpu =
            crate::backends::common::device_for(crate::backends::common::DevicePreference::Cuda(0))
                .unwrap()
                .0;
        let mut file = std::fs::File::open(&gguf).unwrap();
        let content = gguf_file::Content::read(&mut file).unwrap();
        let mut seen = std::collections::HashSet::new();
        for (name, info) in &content.tensor_infos {
            let kind = format!("{:?}", info.ggml_dtype);
            if !seen.insert(kind.clone()) || !name.starts_with("layers.") && kind != "BF16" {
                continue;
            }
            let on_cpu = content
                .tensor(&mut file, name, &Device::Cpu)
                .unwrap()
                .dequantize(&Device::Cpu)
                .unwrap();
            let on_gpu = content
                .tensor(&mut file, name, &gpu)
                .unwrap()
                .dequantize(&gpu)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let e = rel_err(&on_gpu, &on_cpu);
            println!("{kind} {name}: GPU vs CPU dequant rel err {e:.2e}");
            assert!(e < 1e-3, "{kind} dequantizes differently on the GPU: {e}");
        }
    }

    /// Control for the parity test: is the per-step difference from diffusers more than bf16 rounding? Runs step 0 in f32 and bf16
    /// from the same noise and embeddings and compares all three pairs. If f32-vs-bf16 is as large as bf16-vs-diffusers, the port
    /// is as close to the exact answer as the reference itself is.
    #[test]
    #[ignore = "needs the Z-Image weights, a diffusers reference dump and a CUDA device"]
    fn step0_numerics_control() {
        let native = env("AI_KIT_ZIMAGE_NATIVE");
        let device =
            crate::backends::common::device_for(crate::backends::common::DevicePreference::Cuda(0))
                .unwrap()
                .0;
        let te = ["00001", "00002", "00003"]
            .map(|n| format!("{native}/text_encoder/model-{n}-of-00003.safetensors"));
        let mut pipe = ZImagePipeline::new(
            ZImageFiles {
                gguf: env("AI_KIT_ZIMAGE_GGUF"),
                text_encoder_shards: te.to_vec(),
                vae: String::new(),
                tokenizer: String::new(),
            },
            device.clone(),
        );
        let r = candle_core::safetensors::load(
            format!("{}/ref.safetensors", env("AI_KIT_ZIMAGE_REF")),
            &Device::Cpu,
        )
        .unwrap();
        let emb = r["prompt_embeds"]
            .unsqueeze(0)
            .unwrap()
            .to_device(&device)
            .unwrap();
        let noise = r["noise"].clone();
        let job = Job {
            h: 512,
            w: 512,
            steps: 9,
            seed: 0,
        };
        let step0 = |pipe: &ZImagePipeline| {
            let mut pr = Probe::default();
            pipe.denoise(
                &emb,
                Job { steps: 9, ..job },
                Some(noise.clone()),
                Some(&mut pr),
            )
            .unwrap();
            (pr.trace[0].to_device(&Device::Cpu).unwrap() - &noise).unwrap()
        };
        let bf16 = step0(&pipe);
        pipe.dtype = DType::F32;
        let f32_ = step0(&pipe);
        let py = (&r["latents_after_step_0"] - &noise).unwrap();
        println!(
            "step 0 update: bf16 vs f32 {:.4}, bf16 vs diffusers {:.4}, f32 vs diffusers {:.4}",
            rel_err(&bf16, &f32_),
            rel_err(&bf16, &py),
            rel_err(&f32_, &py)
        );
    }

    /// The layer-streaming encoder must equal Candle's resident one (same maths, same weights).
    #[test]
    #[ignore = "needs the Z-Image weights and a CUDA device with ~9 GB free"]
    fn streaming_matches_candles_full_encoder() {
        let native = env("AI_KIT_ZIMAGE_NATIVE");
        let device =
            crate::backends::common::device_for(crate::backends::common::DevicePreference::Cuda(0))
                .unwrap()
                .0;
        let te = ["00001", "00002", "00003"]
            .map(|n| format!("{native}/text_encoder/model-{n}-of-00003.safetensors"));
        let pipe = ZImagePipeline::new(
            ZImageFiles {
                gguf: String::new(),
                text_encoder_shards: te.to_vec(),
                vae: String::new(),
                tokenizer: format!("{native}/tokenizer/tokenizer.json"),
            },
            device,
        );
        for prompt in ["a cozy cottage at dusk, watercolor", "a very long description of a market square with many vendors, banners, lanterns, fruit stalls and a fountain in the middle, golden hour, photographed on film"] {
            let a = pipe.encode_prompt(prompt).unwrap().to_device(&Device::Cpu).unwrap();
            let b = pipe.encode_prompt_resident(prompt).unwrap().to_device(&Device::Cpu).unwrap();
            let e = rel_err(&a, &b);
            println!("{} tokens: streaming vs resident rel err {e:.2e}", a.dim(1).unwrap());
            assert!(e < 8e-3, "streaming encoder differs from Candle's (bf16 rounding is ~3e-3): {e}");
        }
    }

    /// Tiled VAE decoding must agree with the single-pass decode (no seams, no blending bug).
    #[test]
    #[ignore = "needs the Z-Image weights, a diffusers reference dump and a CUDA device"]
    fn tiled_vae_decode_matches_the_single_pass() {
        let native = env("AI_KIT_ZIMAGE_NATIVE");
        let device =
            crate::backends::common::device_for(crate::backends::common::DevicePreference::Cuda(0))
                .unwrap()
                .0;
        let pipe = ZImagePipeline::new(
            ZImageFiles {
                gguf: String::new(),
                text_encoder_shards: vec![],
                vae: format!("{native}/vae/diffusion_pytorch_model.safetensors"),
                tokenizer: String::new(),
            },
            device.clone(),
        );
        let r = candle_core::safetensors::load(
            format!("{}/ref.safetensors", env("AI_KIT_ZIMAGE_REF")),
            &Device::Cpu,
        )
        .unwrap();
        let lat = r["latents_after_step_8"]
            .to_device(&device)
            .unwrap()
            .to_dtype(pipe.dtype)
            .unwrap(); // 1x16x64x64
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[pipe.files.vae.as_str()], pipe.dtype, &device)
        }
        .unwrap();
        let vae = AutoEncoderKL::new(&VaeConfig::z_image(), vb).unwrap();
        let whole = vae
            .decode(&lat)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // 64x64 latent in 32-wide tiles with 8 overlap: 3x3 tiles, interior blending on every edge
        let tiled = pipe.decode_tiled(&vae, &lat, 32, 8).unwrap();
        assert_eq!(whole.len(), tiled.len());
        let mad = whole
            .iter()
            .zip(&tiled)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / whole.len() as f32
            * 127.5;
        let max = whole
            .iter()
            .zip(&tiled)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            * 127.5;
        println!("tiled vs single-pass decode: mean abs diff {mad:.2} / 255, max {max:.1}");
        assert!(mad < 2.0, "tiled decode differs from single pass: {mad}");
    }

    fn image_from_rgb(rgb: &[u8]) -> Vec<u8> {
        crate::backends::diffusion::encode_png_for_tests(rgb, 512, 512)
    }

    fn png_decode(path: &str) -> Vec<u8> {
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
        let mut reader = dec.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf.truncate(info.buffer_size());
        buf
    }
}
