//! Image-generation and embedding backends, and the general router over them.
//!
//! Two groups:
//!
//! 1. **Always run** (`cargo test --features candle`): the honest-failure paths. No weights needed. A model of the
//!    wrong kind, a file that does not match its catalog hash, bad parameters and the wrong request kind must each give a
//!    specific, typed error and never touch a backend.
//! 2. **`#[ignore]`d, need weights**: real images and real embeddings. Point `AI_KIT_MM_CATALOG` at a catalog with an
//!    `sd-turbo` (image) and a `minilm` (embedding) entry (see `make_multimodal_catalog.py` in the studio vault):
//!
//!      AI_KIT_MM_CATALOG=/tmp/ak-models-mm/catalog.yaml cargo test --release --features cuda --test multimodal -- --ignored --test-threads=1

#![cfg(feature = "candle")]

use ai_kit::routed::TierBackend;
use ai_kit::{
    AiKitError, CandleEmbeddingService, CandleImageService, CandleInferenceService,
    DevicePreference, EmbeddingParams, ImageGenParams, InferenceParams, InferenceService,
    MultiHybridService, Output, OutputCheck, Request, RequestKind, RequestService, RoutePolicy,
    RoutedRequest,
};
use std::io::Write;
use std::sync::Arc;

/// A catalog with one model of each kind whose files are tiny junk, so loading gets as far as the checks under test.
fn junk_catalog(dir: &tempfile::TempDir, hash_ok: bool) -> String {
    let mut files = Vec::new();
    for name in [
        "unet",
        "vae",
        "text_encoder",
        "tokenizer",
        "config",
        "weights",
    ] {
        let p = dir.path().join(format!("{name}.bin"));
        std::fs::File::create(&p)
            .unwrap()
            .write_all(name.as_bytes())
            .unwrap();
        files.push((
            name,
            p.to_string_lossy().to_string(),
            blake3::hash(name.as_bytes()).to_hex().to_string(),
        ));
    }
    let h = |i: usize| {
        if hash_ok {
            files[i].2.clone()
        } else {
            "0".repeat(64)
        }
    };
    let l = |i: usize| files[i].1.clone();
    let entry = |id: &str, cap: &str, main: usize, extra: &[(&str, usize)]| {
        let extras: String = extra
            .iter()
            .map(|(role, i)| format!("      - {{role: {role}, location: \"file://{}\", hash: \"{}\", size_bytes: 6}}\n", l(*i), h(*i)))
            .collect();
        format!(
            "  - id: {id}\n    version: \"1.0.0\"\n    os_arch: [x86_64]\n    quantization_variants: [default]\n    default_quantization: default\n    hash: \"{}\"\n    size_bytes: 4\n    location: \"file://{}\"\n    files:\n{extras}    metadata:\n      parameters: 1\n      context_window: 77\n      vram_required_mb: 1\n      inference_latency_ms: 0\n      framework: candle\n      custom: {{kind: {cap}, architecture: sd_turbo}}\n",
            h(main),
            l(main)
        )
    };
    let yaml = format!(
        "version: \"1.0\"\nmodels:\n{}{}{}",
        entry(
            "img",
            "image_generation",
            0,
            &[("vae", 1), ("text_encoder", 2), ("tokenizer", 3)]
        ),
        entry("emb", "embedding", 5, &[("config", 4), ("tokenizer", 3)]),
        entry("txt", "text_generation", 0, &[]),
    );
    let path = dir.path().join("catalog.yaml");
    std::fs::write(&path, yaml).unwrap();
    path.to_string_lossy().to_string()
}

fn image_req(model: &str, w: u32) -> Request {
    let mut p = ImageGenParams::new(model, "a cat");
    p.width = w;
    Request::ImageGeneration(p)
}

// ------------------------------------------------------------ group 1

#[tokio::test]
async fn image_backend_refuses_a_model_of_the_wrong_kind_and_an_unknown_one() {
    let dir = tempfile::tempdir().unwrap();
    let svc = CandleImageService::new(&junk_catalog(&dir, true), DevicePreference::Cpu).unwrap();
    match svc.handle(image_req("txt", 512)).await {
        Err(AiKitError::ModelLoadFailed { reason, .. }) => {
            assert!(reason.contains("not an image-generation model"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        svc.handle(image_req("nope", 512)).await,
        Err(AiKitError::ModelNotFound { .. })
    ));
}

#[tokio::test]
async fn bad_parameters_are_rejected_before_any_model_is_loaded() {
    let dir = tempfile::tempdir().unwrap();
    let svc = CandleImageService::new(&junk_catalog(&dir, true), DevicePreference::Cpu).unwrap();
    assert!(matches!(
        svc.handle(image_req("img", 500)).await,
        Err(AiKitError::InvalidParams { .. })
    ));
    assert!(!svc.is_loaded("img"));
}

#[tokio::test]
async fn a_file_that_does_not_match_its_catalog_hash_is_never_loaded() {
    let dir = tempfile::tempdir().unwrap();
    let cat = junk_catalog(&dir, false);
    let img = CandleImageService::new(&cat, DevicePreference::Cpu).unwrap();
    assert!(matches!(
        img.handle(image_req("img", 512)).await,
        Err(AiKitError::IntegrityCheckFailed { .. })
    ));
    let emb = CandleEmbeddingService::new(&cat, DevicePreference::Cpu).unwrap();
    let r = Request::Embedding(EmbeddingParams {
        model_id: "emb".into(),
        inputs: vec!["hello".into()],
        normalize: true,
        session_id: None,
        request_id: None,
    });
    assert!(matches!(
        emb.handle(r).await,
        Err(AiKitError::IntegrityCheckFailed { .. })
    ));
}

#[tokio::test]
async fn each_backend_says_so_when_asked_for_something_it_does_not_do() {
    let dir = tempfile::tempdir().unwrap();
    let cat = junk_catalog(&dir, true);
    let img = CandleImageService::new(&cat, DevicePreference::Cpu).unwrap();
    let emb = CandleEmbeddingService::new(&cat, DevicePreference::Cpu).unwrap();
    assert_eq!(img.kinds(), vec![RequestKind::ImageGeneration]);
    assert_eq!(emb.kinds(), vec![RequestKind::Embedding]);
    let text = Request::Text(InferenceParams {
        model_id: "txt".into(),
        prompt: "hi".into(),
        temperature: 0.0,
        top_p: 1.0,
        max_tokens: 4,
        session_id: None,
        preferred_quantization: None,
        request_id: None,
    });
    for r in [img.handle(text.clone()).await, emb.handle(text).await] {
        assert!(
            matches!(r, Err(AiKitError::UnsupportedKind { .. })),
            "{r:?}"
        );
    }
    assert!(matches!(
        emb.handle(image_req("img", 512)).await,
        Err(AiKitError::UnsupportedKind { .. })
    ));
}

#[tokio::test]
async fn the_text_backend_refuses_an_image_model() {
    let dir = tempfile::tempdir().unwrap();
    let svc = CandleInferenceService::new(&junk_catalog(&dir, true)).unwrap();
    match svc.load_model("img").await {
        Err(AiKitError::ModelLoadFailed { reason, .. }) => assert!(
            reason.contains("image_generation model, not a text-generation"),
            "{reason}"
        ),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn an_explicit_cuda_device_that_is_missing_fails_for_every_backend() {
    let dir = tempfile::tempdir().unwrap();
    let cat = junk_catalog(&dir, true);
    let img = CandleImageService::new(&cat, DevicePreference::Cuda(63)).unwrap();
    match img.handle(image_req("img", 512)).await {
        Err(AiKitError::ModelLoadFailed { reason, .. }) => {
            assert!(reason.contains("CUDA"), "{reason}")
        }
        other => panic!("{other:?}"),
    }
}

// ------------------------------------------------------------ group 2

fn real_catalog() -> String {
    std::env::var("AI_KIT_MM_CATALOG")
        .expect("set AI_KIT_MM_CATALOG to a catalog with sd-turbo and minilm")
}
fn device() -> DevicePreference {
    std::env::var("AI_KIT_DEVICE")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(DevicePreference::Cpu)
}

#[tokio::test]
#[ignore]
async fn a_real_image_is_a_real_png_of_the_requested_size_and_is_reproducible() {
    let svc = CandleImageService::new(&real_catalog(), device()).unwrap();
    let mk = |seed: u64| {
        let mut p = ImageGenParams::new("sd-turbo", "a red apple on a wooden table, photo");
        p.steps = 1; // one step adds no sampler noise, so a seed fully determines the image on every device
        p.seed = Some(seed);
        Request::ImageGeneration(p)
    };
    let get = |o: Output| match o {
        Output::Image(i) => i,
        other => panic!("{other:?}"),
    };
    let a = get(svc.handle(mk(11)).await.unwrap());
    assert_eq!(ai_kit::modality::png_dims(&a.png), Some((512, 512)));
    assert!(a.pixel_stddev > 10.0, "flat image: {}", a.pixel_stddev);
    assert!(OutputCheck::Sane
        .check(&mk(11), &Output::Image(a.clone()))
        .unwrap());
    let b = get(svc.handle(mk(11)).await.unwrap());
    assert_eq!(a.png, b.png, "same seed, one step: not reproducible");
    let c = get(svc.handle(mk(12)).await.unwrap());
    assert_ne!(a.png, c.png, "different seeds gave the same image");

    // On CUDA the sampler's per-step noise is seeded too, so even a multi-step image is reproducible there.
    if matches!(device(), DevicePreference::Cuda(_)) {
        let multi = |seed: u64| {
            let mut p = ImageGenParams::new("sd-turbo", "a red apple on a wooden table, photo");
            p.steps = 3;
            p.seed = Some(seed);
            Request::ImageGeneration(p)
        };
        let x = get(svc.handle(multi(5)).await.unwrap());
        let y = get(svc.handle(multi(5)).await.unwrap());
        assert_eq!(
            x.png, y.png,
            "3 steps, same seed, on CUDA: not reproducible"
        );
    }
}

#[tokio::test]
#[ignore]
async fn real_embeddings_rank_the_right_note_first_and_are_unit_length() {
    let svc = CandleEmbeddingService::new(&real_catalog(), device()).unwrap();
    let notes = [
        "Guest wifi network is LakeHouse-Guest",
        "Trash pickup is Tuesday morning",
        "Biscuit the dog gets food at 7am and the vet is Dr. Okafor on Mill Road",
        "The router admin page is at 192.168.1.1",
    ];
    let mut inputs: Vec<String> = notes.iter().map(|s| s.to_string()).collect();
    inputs.push("who is the vet for the dog?".into());
    let out = svc
        .handle(Request::Embedding(EmbeddingParams {
            model_id: "minilm".into(),
            inputs,
            normalize: true,
            session_id: None,
            request_id: None,
        }))
        .await
        .unwrap();
    let Output::Embeddings(e) = out else { panic!() };
    assert_eq!((e.vectors.len(), e.dimensions), (5, 384));
    for v in &e.vectors {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "not unit length: {norm}");
    }
    let q = &e.vectors[4];
    let score = |i: usize| e.vectors[i].iter().zip(q).map(|(a, b)| a * b).sum::<f32>();
    let best = (0..4)
        .max_by(|a, b| score(*a).total_cmp(&score(*b)))
        .unwrap();
    assert_eq!(best, 2, "the vet note should win");
}

#[tokio::test]
#[ignore]
async fn one_service_routes_every_kind_of_request_and_is_honest_about_the_rest() {
    let cat = real_catalog();
    let local = vec![
        TierBackend::new(
            Arc::new(CandleImageService::new(&cat, device()).unwrap()),
            &[(RequestKind::ImageGeneration, "sd-turbo")],
        ),
        TierBackend::new(
            Arc::new(CandleEmbeddingService::new(&cat, device()).unwrap()),
            &[(RequestKind::Embedding, "minilm")],
        ),
    ];
    let svc = MultiHybridService::new(local, vec![], RoutePolicy::local_first());
    let (out, _) = svc
        .handle_routed(RoutedRequest::new(image_req("", 512)).with_check(OutputCheck::Sane))
        .await
        .unwrap();
    assert!(matches!(out, Output::Image(_)));
    let (out, trace) = svc
        .handle_routed(RoutedRequest::new(Request::Text(InferenceParams {
            model_id: String::new(),
            prompt: "paint me a watercolor of a lighthouse".into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 8,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        })))
        .await
        .unwrap();
    assert!(
        matches!(out, Output::Image(_)),
        "free text about a painting should run as an image request"
    );
    assert!(trace
        .notes
        .iter()
        .any(|n| n.contains("treated a text_generation request as image_generation")));
    let e = svc
        .handle_routed(RoutedRequest::new(Request::Text(InferenceParams {
            model_id: String::new(),
            prompt: "What is 2+2?".into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 8,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        })))
        .await
        .unwrap_err();
    assert!(
        matches!(e, AiKitError::UnsupportedKind { .. }),
        "no text backend is configured: {e:?}"
    );
}

#[tokio::test]
#[ignore]
async fn z_image_turbo_makes_a_real_reproducible_image_through_the_service() {
    // Needs a catalog with `z-image-turbo` (architecture: z_image) and a CUDA device with ~9 GB free.
    let svc = CandleImageService::new(&real_catalog(), device()).unwrap();
    let mk = |seed: u64| {
        let mut p = ImageGenParams::new(
            "z-image-turbo",
            "a lighthouse on a cliff at night, watercolor",
        );
        p.steps = 8;
        p.seed = Some(seed);
        Request::ImageGeneration(p)
    };
    let get = |o: Output| match o {
        Output::Image(i) => i,
        other => panic!("{other:?}"),
    };
    let a = get(svc.handle(mk(21)).await.unwrap());
    assert_eq!(ai_kit::modality::png_dims(&a.png), Some((512, 512)));
    assert!(a.pixel_stddev > 20.0, "flat image: {}", a.pixel_stddev);
    assert!(OutputCheck::Sane
        .check(&mk(21), &Output::Image(a.clone()))
        .unwrap());
    let b = get(svc.handle(mk(21)).await.unwrap());
    assert_eq!(
        a.png, b.png,
        "same seed: not reproducible (Z-Image uses no sampler noise, so it must be)"
    );
    let c = get(svc.handle(mk(22)).await.unwrap());
    assert_ne!(a.png, c.png, "different seeds gave the same image");
}
