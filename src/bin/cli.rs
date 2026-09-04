//! `ai-kit-cli` — minimal CLI for local dev/testing.
//!
//! Hand-rolled flag parsing on purpose: a handful of flags doesn't justify
//! a new dependency (no `clap` in Cargo.toml's dependency list).
//!
//! Subcommands:
//!   hash <path>                    Print the BLAKE3 hex digest of a file.
//!                                   Always available (no `candle` feature needed) —
//!                                   this is what `scripts/fetch-model.sh` shells out to.
//!   infer --model <id> --prompt <text> [--temperature F] [--top-p F] [--max-tokens N]
//!                                   Real local inference. Requires `--features candle`
//!                                   and a fetched model (`make fetch-model`).
//!   cache-stats                    Print cache_stats() for a freshly constructed service.

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let Some(command) = args.get(1) else {
        eprint_usage();
        return ExitCode::from(2);
    };

    match command.as_str() {
        "hash" => cmd_hash(&args[2..]),
        "infer" => cmd_infer(&args[2..]),
        "cache-stats" => cmd_cache_stats(&args[2..]),
        other => {
            eprintln!("error: unknown subcommand '{other}'");
            eprint_usage();
            ExitCode::from(2)
        }
    }
}

fn eprint_usage() {
    eprintln!("Usage:");
    eprintln!("  ai-kit-cli hash <path>");
    eprintln!(
        "  ai-kit-cli infer --model <id> --prompt <text> [--temperature F] [--top-p F] [--max-tokens N] [--session-id S]"
    );
    eprintln!("  ai-kit-cli cache-stats [--catalog <path>]");
}

fn cmd_hash(args: &[String]) -> ExitCode {
    let Some(path) = args.first() else {
        eprintln!("error: hash requires a <path> argument");
        return ExitCode::from(2);
    };
    match std::fs::read(path) {
        Ok(bytes) => {
            println!("{}", blake3::hash(&bytes).to_hex());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: failed to read {path}: {e}");
            ExitCode::FAILURE
        }
    }
}

// Only used by the candle-gated subcommands below; without the feature,
// there's nothing that needs flag parsing beyond `hash <path>`.
#[cfg(feature = "candle")]
fn flag_value(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

#[cfg(feature = "candle")]
fn default_catalog_path() -> String {
    env::var("AI_KIT_MODEL_DIR").unwrap_or_else(|_| "./dev-models".to_string()) + "/catalog.yaml"
}

#[cfg(feature = "candle")]
fn cmd_infer(args: &[String]) -> ExitCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(cmd_infer_async(args))
}

#[cfg(feature = "candle")]
async fn cmd_infer_async(args: &[String]) -> ExitCode {
    use ai_kit::{CandleInferenceService, InferenceParams, InferenceService};

    let Some(model_id) = flag_value(args, "--model") else {
        eprintln!("error: infer requires --model <id>");
        return ExitCode::from(2);
    };
    let Some(prompt) = flag_value(args, "--prompt") else {
        eprintln!("error: infer requires --prompt <text>");
        return ExitCode::from(2);
    };
    let temperature = flag_value(args, "--temperature")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.7);
    let top_p = flag_value(args, "--top-p")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.95);
    let max_tokens = flag_value(args, "--max-tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let session_id = flag_value(args, "--session-id").map(ai_kit::types::SessionId::from_string);
    let catalog_path = flag_value(args, "--catalog").unwrap_or_else(default_catalog_path);

    let service = match CandleInferenceService::new(&catalog_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to load catalog {catalog_path}: {e}");
            eprintln!("hint: run `make fetch-model` first.");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = service.load_model(&model_id).await {
        eprintln!("error: failed to load model '{model_id}': {e}");
        return ExitCode::FAILURE;
    }

    let params = InferenceParams {
        model_id,
        prompt,
        temperature,
        top_p,
        max_tokens,
        session_id,
        preferred_quantization: None,
        request_id: None,
    };

    match service.infer(params).await {
        Ok(response) => {
            println!("{}", response.completion);
            eprintln!(
                "\n[{} prompt + {} completion = {} tokens, {}ms, model {} ({})]",
                response.prompt_tokens,
                response.completion_tokens,
                response.total_tokens,
                response.inference_ms,
                response.model_version,
                response.quantization,
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: inference failed: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "candle"))]
fn cmd_infer(_args: &[String]) -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature — `infer` is unavailable.");
    eprintln!("Rebuild with: cargo build --features candle");
    eprintln!("(see README.md \"Local Dev Inference (POC)\")");
    ExitCode::from(2)
}

#[cfg(feature = "candle")]
fn cmd_cache_stats(args: &[String]) -> ExitCode {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(cmd_cache_stats_async(args))
}

#[cfg(feature = "candle")]
async fn cmd_cache_stats_async(args: &[String]) -> ExitCode {
    use ai_kit::{CandleInferenceService, InferenceService};

    let catalog_path = flag_value(args, "--catalog").unwrap_or_else(default_catalog_path);
    let service = match CandleInferenceService::new(&catalog_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: failed to load catalog {catalog_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    match service.cache_stats().await {
        Ok(stats) => {
            println!("{stats:#?}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(feature = "candle"))]
fn cmd_cache_stats(_args: &[String]) -> ExitCode {
    eprintln!("ai-kit-cli was built without the `candle` feature — `cache-stats` is unavailable.");
    eprintln!("Rebuild with: cargo build --features candle");
    ExitCode::from(2)
}
