//! Helpers shared by every Candle-backed service (text, image, embedding, and future kinds): where a model runs, whether
//! it fits, and whether its files are the ones the catalog promised.
//!
//! These used to live in `candle.rs` and were reached into by the other backends. Moving them here means a new backend
//! (for example Z-Image) depends on this module, not on the text backend.

use crate::error::{AiKitError, Result};
use crate::types::ModelMetadata;
use candle_core::Device;
use std::str::FromStr;

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

/// Turn a [`DevicePreference`] into a device and a label, failing hard (no silent CPU fallback) unless `Auto`.
pub(crate) fn device_for(pref: DevicePreference) -> std::result::Result<(Device, String), String> {
    match pref {
        DevicePreference::Cpu => Ok((Device::Cpu, "cpu".to_string())),
        DevicePreference::Cuda(n) => cuda_device(n),
        DevicePreference::Auto => Ok(cuda_device(0).unwrap_or((Device::Cpu, "cpu".to_string()))),
    }
}

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
/// BLAKE3 of a file, streamed (model files can be gigabytes).
pub(crate) fn hash_file(path: &str) -> Result<String> {
    let mut h = blake3::Hasher::new();
    h.update_reader(std::fs::File::open(path)?)?;
    Ok(h.finalize().to_hex().to_string())
}

#[cfg_attr(not(any(feature = "image", feature = "embedding")), allow(dead_code))]
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
    // A model that streams its weights (Z-Image) never holds them all at once, so the sum of its file sizes overstates what it
    // needs. Its catalog entry declares the measured peak in `custom.vram_peak_mb`, which replaces the file-size estimate.
    if let Some(peak) = model
        .metadata
        .custom
        .get("vram_peak_mb")
        .and_then(|v| v.parse::<u64>().ok())
    {
        return peak.max(u64::from(model.metadata.vram_required_mb));
    }
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

#[cfg_attr(not(feature = "image"), allow(dead_code))]
/// True if an error message is a GPU out-of-memory (Candle surfaces it as text, e.g. `CUDA_ERROR_OUT_OF_MEMORY`).
pub(crate) fn looks_like_oom(text: &str) -> bool {
    let t = text.to_ascii_uppercase();
    t.contains("OUT_OF_MEMORY") || t.contains("OUT OF MEMORY")
}

/// Free GPU memory in MB, if `device` is a CUDA device that can be queried.
#[cfg(feature = "cuda")]
#[cfg_attr(not(feature = "image"), allow(dead_code))]
fn free_vram_mb(device: &Device) -> Option<u64> {
    let ctx = device
        .as_cuda_device()
        .ok()?
        .cuda_stream()
        .context()
        .clone();
    ctx.mem_get_info().ok().map(|(free, _)| (free >> 20) as u64)
}
#[cfg(not(feature = "cuda"))]
#[cfg_attr(not(feature = "image"), allow(dead_code))]
fn free_vram_mb(_device: &Device) -> Option<u64> {
    None
}

#[cfg_attr(not(feature = "image"), allow(dead_code))]
/// Turn an `Internal` error that is really a GPU out-of-memory into a typed `InferenceOom`, so callers and the wire contract can tell
/// "does not fit" from "broken". `required_mb` is 0 because the real need is not known at that point; `available_mb` is the free
/// memory right now when it can be read. Any other error is returned unchanged.
pub(crate) fn promote_oom(e: AiKitError, model_id: &str, device: Option<&Device>) -> AiKitError {
    match e {
        AiKitError::Internal { ref reason } if looks_like_oom(reason) => AiKitError::InferenceOom {
            model_id: model_id.to_string(),
            required_mb: 0,
            available_mb: device
                .and_then(free_vram_mb)
                .map_or(0, |m| u32::try_from(m).unwrap_or(u32::MAX)),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelMetadataExtended;
    use std::collections::HashMap;

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
    fn a_gpu_out_of_memory_message_becomes_a_typed_error_and_nothing_else_does() {
        let oom = AiKitError::Internal {
            reason: "z_image vae decode: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
                .into(),
        };
        assert!(matches!(
            promote_oom(oom, "m", None),
            AiKitError::InferenceOom { .. }
        ));
        let other = AiKitError::Internal {
            reason: "shape mismatch".into(),
        };
        assert!(matches!(
            promote_oom(other, "m", None),
            AiKitError::Internal { .. }
        ));
        let typed = AiKitError::ModelNotFound {
            model_id: "m".into(),
        };
        assert!(matches!(
            promote_oom(typed, "m", None),
            AiKitError::ModelNotFound { .. }
        ));
    }

    #[test]
    fn a_streaming_model_declares_its_measured_peak_instead_of_its_file_sizes() {
        let mut m = model(12_000 << 20, 1);
        m.metadata
            .custom
            .insert("vram_peak_mb".to_string(), "8500".to_string());
        assert_eq!(vram_needed_mb(&m), 8500);
        m.metadata.vram_required_mb = 9000; // the declared floor still wins if it is higher
        assert_eq!(vram_needed_mb(&m), 9000);
    }

    #[test]
    fn vram_estimate_honours_the_catalog_declared_floor() {
        // The catalog says it needs 9000 MB, more than the file-size estimate.
        assert_eq!(vram_needed_mb(&model(1000 << 20, 9000)), 9000);
    }
}
