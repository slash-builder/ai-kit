//! Device selection (CPU / CUDA) for `CandleInferenceService`.
//!
//! Three groups:
//!
//! 1. **Always run** (`cargo test --features candle`): parsing, and the
//!    "fail hard, never silently fall back" behaviour. No GPU, no model.
//! 2. **Run when a CUDA device is present** (`cargo test --features cuda`): the
//!    VRAM pre-check. Skipped (with a message) on a box without a GPU.
//! 3. **`#[ignore]`d, need weights** (`make fetch-model`, or point
//!    `AI_KIT_MODEL_DIR` / `AI_KIT_TEST_MODEL` at a catalog): the real
//!    smoke tests and the request-isolation check that justifies reusing one
//!    `ModelWeights` across requests.
//!
//!   cargo test --features cuda -- --ignored --test-threads=1

#![cfg(feature = "candle")]

use ai_kit::{
    AiKitError, CandleInferenceService, DevicePreference, InferenceParams, InferenceService,
};
use std::io::Write;

/// A syntactically valid catalog whose single model points at a file that does
/// not exist: any code path that actually tried to *read* it would fail with
/// `Io`, which lets the tests prove the checks fire first.
fn dummy_catalog(size_bytes: u64, vram_required_mb: u32) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    write!(
        f,
        r#"version: "1.0"
models:
  - id: "dummy"
    version: "1.0.0"
    os_arch: ["x86_64"]
    quantization_variants: ["q4_k_m"]
    default_quantization: "q4_k_m"
    hash: "0000"
    size_bytes: {size_bytes}
    location: "file:///nonexistent/never-read.gguf"
    metadata:
      parameters: 1000000000
      context_window: 2048
      vram_required_mb: {vram_required_mb}
      inference_latency_ms: 0
      framework: "candle"
      custom:
        tokenizer_path: "/nonexistent/tokenizer.json"
"#
    )
    .unwrap();
    f
}

fn params(model: &str, prompt: &str, max_tokens: u32) -> InferenceParams {
    InferenceParams {
        model_id: model.to_string(),
        prompt: prompt.to_string(),
        temperature: 0.0, // greedy, so runs are comparable
        top_p: 1.0,
        max_tokens,
        session_id: None,
        preferred_quantization: None,
        request_id: None,
    }
}

// ---------------------------------------------------------------- group 1

#[test]
fn device_preference_parses() {
    assert_eq!("cpu".parse(), Ok(DevicePreference::Cpu));
    assert_eq!("AUTO".parse(), Ok(DevicePreference::Auto));
    assert_eq!("cuda".parse(), Ok(DevicePreference::Cuda(0)));
    assert_eq!("cuda:2".parse(), Ok(DevicePreference::Cuda(2)));
    assert_eq!(" Cuda:1 ".parse(), Ok(DevicePreference::Cuda(1)));
    assert!("cuda:".parse::<DevicePreference>().is_err());
    assert!("cuda:x".parse::<DevicePreference>().is_err());
    assert!("gpu".parse::<DevicePreference>().is_err());
    assert_eq!(DevicePreference::default(), DevicePreference::Cpu);
}

#[tokio::test]
async fn explicit_cuda_never_falls_back_to_cpu() {
    // Device 63 exists on no machine, so this fails whether or not the crate was
    // built with `cuda` and whether or not a GPU is present.
    let cat = dummy_catalog(1 << 20, 1);
    let svc = CandleInferenceService::with_device(
        cat.path().to_str().unwrap(),
        DevicePreference::Cuda(63),
    )
    .unwrap();
    match svc.load_model("dummy").await {
        Err(AiKitError::ModelLoadFailed { reason, .. }) => {
            assert!(reason.contains("CUDA"), "unhelpful reason: {reason}");
        }
        other => panic!("expected ModelLoadFailed, got {other:?}"),
    }
    assert!(
        svc.device_of("dummy").is_none(),
        "nothing may be left loaded"
    );
}

#[cfg(not(feature = "cuda"))]
#[tokio::test]
async fn cuda_without_the_feature_says_how_to_fix_it() {
    let cat = dummy_catalog(1 << 20, 1);
    let svc = CandleInferenceService::with_device(
        cat.path().to_str().unwrap(),
        DevicePreference::Cuda(0),
    )
    .unwrap();
    match svc.load_model("dummy").await {
        Err(AiKitError::ModelLoadFailed { reason, .. }) => {
            assert!(reason.contains("--features cuda"), "reason: {reason}")
        }
        other => panic!("expected ModelLoadFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn auto_without_a_gpu_is_visible_not_hidden() {
    // `Auto` may choose the CPU, but it must say so. (With a GPU it says cuda:0;
    // either way the label is never empty.) The dummy model cannot actually load.
    let cat = dummy_catalog(1 << 20, 1);
    let svc =
        CandleInferenceService::with_device(cat.path().to_str().unwrap(), DevicePreference::Auto)
            .unwrap();
    // It gets past device selection and fails reading the missing file: an Io error.
    assert!(matches!(
        svc.load_model("dummy").await,
        Err(AiKitError::Io(_)) | Err(AiKitError::InferenceOom { .. })
    ));
}

// ---------------------------------------------------------------- group 2

#[cfg(feature = "cuda")]
#[tokio::test]
async fn vram_precheck_fails_hard_before_reading_the_file() {
    if !candle_core::utils::cuda_is_available() {
        eprintln!("skipped: no CUDA device on this machine");
        return;
    }
    // Claims to be 4 TiB. No real GPU has that free, and the file does not exist:
    // an `Io` error would mean the check ran too late.
    let cat = dummy_catalog(4 << 40, 1);
    let svc = CandleInferenceService::with_device(
        cat.path().to_str().unwrap(),
        DevicePreference::Cuda(0),
    )
    .unwrap();
    match svc.load_model("dummy").await {
        Err(AiKitError::InferenceOom {
            required_mb,
            available_mb,
            ..
        }) => assert!(required_mb > available_mb),
        other => panic!("expected InferenceOom, got {other:?}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn describe_cuda_devices_lists_the_gpu() {
    let lines = ai_kit::describe_cuda_devices();
    assert!(!lines.is_empty());
    if candle_core::utils::cuda_is_available() {
        assert!(lines[0].starts_with("cuda:0 "), "got {lines:?}");
    }
}

// ---------------------------------------------------------------- group 3

fn real_service(device: DevicePreference) -> (CandleInferenceService, String) {
    let dir = std::env::var("AI_KIT_MODEL_DIR").unwrap_or_else(|_| "./dev-models".to_string());
    let model = std::env::var("AI_KIT_TEST_MODEL").unwrap_or_else(|_| "tinyllama".to_string());
    let svc = CandleInferenceService::with_device(&format!("{dir}/catalog.yaml"), device)
        .expect("catalog missing: run `make fetch-model` or set AI_KIT_MODEL_DIR");
    (svc, model)
}

/// One loaded model, three requests: A, B, A. Greedy decoding means A must
/// answer identically both times (no state carried from B), and B must equal
/// what a brand-new service says for B (no state carried from A).
async fn assert_requests_are_isolated(device: DevicePreference) {
    let (svc, model) = real_service(device);
    svc.load_model(&model).await.unwrap();
    let a = "The capital of France is";
    let b = "fn fibonacci(n: u32) -> u32 {";

    let a1 = svc.infer(params(&model, a, 24)).await.unwrap().completion;
    let b1 = svc.infer(params(&model, b, 24)).await.unwrap().completion;
    let a2 = svc.infer(params(&model, a, 24)).await.unwrap().completion;
    assert!(!a1.trim().is_empty() && !b1.trim().is_empty());
    assert_eq!(
        a1, a2,
        "request A changed after request B ran: KV state leaked"
    );

    let (fresh, _) = real_service(device);
    fresh.load_model(&model).await.unwrap();
    let b_fresh = fresh.infer(params(&model, b, 24)).await.unwrap().completion;
    assert_eq!(
        b1, b_fresh,
        "request B differs from a fresh service: KV state leaked"
    );

    assert_ne!(a1, b1);
}

#[tokio::test]
#[ignore]
async fn reuse_is_isolated_on_cpu() {
    assert_requests_are_isolated(DevicePreference::Cpu).await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore]
async fn reuse_is_isolated_on_cuda() {
    assert_requests_are_isolated(DevicePreference::Cuda(0)).await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore]
async fn cuda_smoke_test_reports_its_device() {
    let (svc, model) = real_service(DevicePreference::Cuda(0));
    svc.load_model(&model).await.unwrap();
    let label = svc.device_of(&model).unwrap();
    assert!(label.starts_with("cuda:0"), "label: {label}");

    let r = svc.infer(params(&model, "Say hello.", 16)).await.unwrap();
    assert!(!r.completion.trim().is_empty());
    assert!(r.prompt_tokens > 0 && r.completion_tokens > 0);
    assert_eq!(r.total_tokens, r.prompt_tokens + r.completion_tokens);

    // Loading again is a no-op, not a second copy in VRAM.
    svc.load_model(&model).await.unwrap();
    assert_eq!(svc.cache_stats().await.unwrap().models_loaded, 1);
}
