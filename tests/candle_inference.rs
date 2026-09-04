//! Real-inference smoke test for `CandleInferenceService`.
//!
//! Requires `make fetch-model` to have downloaded weights into `dev-models/`
//! first (see README "Local Dev Inference (POC)"). Deliberately `#[ignore]`d
//! so it never runs in default `cargo test`/CI (no network, no ~700MB
//! download, no ~30s CPU generation in a CI job) — run it manually:
//!
//!   cargo test --features candle -- --ignored real_inference_smoke_test

#![cfg(feature = "candle")]

use ai_kit::{CandleInferenceService, InferenceParams, InferenceService};

#[tokio::test]
#[ignore]
async fn real_inference_smoke_test() {
    let catalog_path = std::env::var("AI_KIT_MODEL_DIR")
        .unwrap_or_else(|_| "./dev-models".to_string())
        + "/catalog.yaml";

    let service = CandleInferenceService::new(&catalog_path)
        .expect("failed to load dev-models/catalog.yaml — run `make fetch-model` first");

    service
        .load_model("tinyllama")
        .await
        .expect("failed to load tinyllama — run `make fetch-model` first");

    let params = InferenceParams {
        model_id: "tinyllama".to_string(),
        prompt: "Say the word 'hello' and nothing else.".to_string(),
        temperature: 0.0, // greedy — deterministic-ish for a smoke test
        top_p: 1.0,
        max_tokens: 20,
        session_id: None,
        preferred_quantization: None,
        request_id: None,
    };

    let response = service
        .infer(params)
        .await
        .expect("real inference call failed");

    assert!(
        !response.completion.trim().is_empty(),
        "completion should not be empty"
    );
    assert!(
        response.inference_ms > 0,
        "inference_ms should be real elapsed time"
    );
    assert!(
        response.prompt_tokens > 0,
        "prompt_tokens should be real token count"
    );
    assert!(
        response.completion_tokens > 0,
        "completion_tokens should be real token count"
    );
    assert_eq!(
        response.total_tokens,
        response.prompt_tokens + response.completion_tokens
    );

    let stats = service.cache_stats().await.expect("cache_stats failed");
    assert_eq!(stats.models_loaded, 1);
}
