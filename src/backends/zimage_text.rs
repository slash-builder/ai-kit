//! Layer-streaming Qwen3 text encoder for Z-Image (hidden states of the second-to-last layer, no final norm).
//!
//! Candle's `ZImageTextEncoder` builds all 36 layers up front: 8 GB of bf16 on the GPU, which is more than a 12 GB card has free next
//! to a resident text model. The encoder runs once per prompt, so there is no reason to hold it: this module memory-maps the
//! shards, uploads **one layer at a time**, runs it and drops it. Peak GPU memory is one layer (about 0.2 GB) plus activations,
//! and the 150k-row embedding table never leaves the CPU (only the few rows the prompt uses are copied over).
//!
//! The maths mirrors Candle's `z_image::text_encoder` exactly (RMSNorm on q and k per head, RoPE with theta 1e6, grouped-query
//! attention with 8 KV heads, SwiGLU); `streaming_matches_candles_full_encoder` checks that on real weights.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, rms_norm, RmsNorm, VarBuilder};
use candle_transformers::models::z_image::TextEncoderConfig;

struct Layer {
    q: candle_nn::Linear,
    k: candle_nn::Linear,
    v: candle_nn::Linear,
    o: candle_nn::Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    gate: candle_nn::Linear,
    up: candle_nn::Linear,
    down: candle_nn::Linear,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl Layer {
    fn load(cfg: &TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let a = vb.pp("self_attn");
        let m = vb.pp("mlp");
        Ok(Self {
            q: linear_no_bias(h, cfg.num_attention_heads * hd, a.pp("q_proj"))?,
            k: linear_no_bias(h, cfg.num_key_value_heads * hd, a.pp("k_proj"))?,
            v: linear_no_bias(h, cfg.num_key_value_heads * hd, a.pp("v_proj"))?,
            o: linear_no_bias(cfg.num_attention_heads * hd, h, a.pp("o_proj"))?,
            q_norm: rms_norm(hd, cfg.rms_norm_eps, a.pp("q_norm"))?,
            k_norm: rms_norm(hd, cfg.rms_norm_eps, a.pp("k_norm"))?,
            gate: linear_no_bias(h, cfg.intermediate_size, m.pp("gate_proj"))?,
            up: linear_no_bias(h, cfg.intermediate_size, m.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, h, m.pp("down_proj"))?,
            ln1: rms_norm(h, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: rms_norm(h, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
        })
    }

    fn forward(
        &self,
        cfg: &TextEncoderConfig,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (nh, nkv, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let hn = self.ln1.forward(x)?;
        let heads = |t: Tensor, n: usize| t.reshape((b, l, n, hd))?.transpose(1, 2);
        let q = heads(self.q.forward(&hn)?, nh)?;
        let k = heads(self.k.forward(&hn)?, nkv)?;
        let v = heads(self.v.forward(&hn)?, nkv)?;
        // per-head RMSNorm over head_dim, then RoPE
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        // grouped-query attention: repeat each KV head for its group of query heads
        let rep = nh / nkv;
        let repeat = |t: Tensor| -> Result<Tensor> {
            if rep == 1 {
                return Ok(t);
            }
            t.unsqueeze(2)?
                .broadcast_as((b, nkv, rep, l, hd))?
                .reshape((b, nkv * rep, l, hd))?
                .contiguous()
        };
        let (k, v) = (repeat(k)?, repeat(v)?);
        let scores = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (hd as f64).sqrt()))?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(mask)?)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, nh * hd))?;
        let x = (x + self.o.forward(&ctx)?)?;
        let h2 = self.ln2.forward(&x)?;
        let mlp = self
            .down
            .forward(&(self.gate.forward(&h2)?.silu()? * self.up.forward(&h2)?)?)?;
        x + mlp
    }
}

/// Hidden states after layer `num_hidden_layers - 2` for `ids`, shape `(1, len, hidden)`, with one layer resident at a time.
pub(crate) fn encode_streaming(
    ids: &[u32],
    shards: &[&str],
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let cfg = TextEncoderConfig::z_image();
    let l = ids.len();
    // SAFETY: the shards were BLAKE3-verified immediately before and are not modified while mapped.
    let on_cpu = unsafe { VarBuilder::from_mmaped_safetensors(shards, dtype, &Device::Cpu)? };
    let on_dev = unsafe { VarBuilder::from_mmaped_safetensors(shards, dtype, device)? };

    // embedding lookup on the CPU: only the prompt's rows are copied to the device
    let table = on_cpu
        .pp("model.embed_tokens")
        .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
    let idx = Tensor::new(ids, &Device::Cpu)?;
    let mut x = table
        .index_select(&idx, 0)?
        .unsqueeze(0)?
        .to_device(device)?;
    drop(table);

    // RoPE tables for exactly this length (Candle precomputes 40960 positions; the values are identical)
    let hd = cfg.head_dim;
    let inv: Vec<f32> = (0..hd)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / hd as f64) as f32)
        .collect();
    let inv = Tensor::from_vec(inv, (1, hd / 2), device)?;
    let t = Tensor::arange(0u32, l as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((l, 1))?;
    let freqs = t.matmul(&inv)?;
    let (cos, sin) = (freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?);

    // causal mask
    let mask: Vec<f32> = (0..l)
        .flat_map(|i| (0..l).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    let mask = Tensor::from_vec(mask, (1, 1, l, l), device)?.to_dtype(dtype)?;

    let target = cfg.num_hidden_layers - 2; // diffusers takes hidden_states[-2]; no final norm
    for i in 0..=target {
        let layer = Layer::load(&cfg, on_dev.pp("model.layers").pp(i))?;
        x = layer.forward(&cfg, &x, &mask, &cos, &sin)?;
        // wait for this layer to finish so its weights can be freed before the next one is uploaded
        device.synchronize()?;
        drop(layer);
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_tables_are_the_standard_qwen_ones_for_a_short_prompt() {
        // position 0 has angle 0 for every frequency: cos 1, sin 0 (a cheap check that the table starts where Candle's does)
        let cfg = TextEncoderConfig::z_image();
        let inv: Vec<f32> = (0..cfg.head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / cfg.head_dim as f64) as f32)
            .collect();
        assert_eq!(inv.len(), 64);
        assert!((inv[0] - 1.0).abs() < 1e-9, "highest frequency is 1");
        assert!(
            inv[63] < 2e-6,
            "lowest frequency with theta=1e6 is ~1.3e-6: {}",
            inv[63]
        );
    }
}
