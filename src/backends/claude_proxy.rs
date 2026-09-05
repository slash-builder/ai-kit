//! Claude (Anthropic) cloud proxy backend — **experimental, not yet ruled
//! on, off by default**.
//!
//! Implements [`InferenceService`] for real by proxying `infer()` to
//! Anthropic's public Messages API (`POST /v1/messages`) over HTTPS. No
//! model weights transit the network — only prompt/completion text — but
//! that text *is* real user data leaving the household boundary to a
//! third-party cloud service, which is exactly the scope
//! `ai-kit-unified-design.md`'s decision memos #2 ("remote fallback scope
//! ... does it include Quickring cloud hub?") and #9 ("hub relay
//! authentication") leave open. **Neither memo has a DJ ruling yet.**
//!
//! # Why build this now, if it's unruled
//!
//! Apple's Foundation Models framework (as of its WWDC 2026 update) adopts
//! the same shape this backend follows: a single `LanguageModelSession`-style
//! call site that can be bound to the on-device model, Apple's own Private
//! Cloud Compute, or a third-party cloud provider, via a
//! provider/fallback parameter — with no change to call-site logic. That is
//! genuinely validating precedent *for the pattern* (trait-based backend
//! swap, uniform call site) and is why this backend is implemented as "just
//! another `InferenceService` impl" rather than a special case. It is cited
//! here as precedent for the *shape*, not copied from — Apple's framework is
//! Swift/on-device-first and shares no code or wire format with this crate.
//!
//! Caveat on that citation: a live web search run as part of building this
//! backend could not independently corroborate the specific claim that
//! Apple's Foundation Models framework exposes a documented
//! provider/fallback parameter naming Anthropic Claude or Google Gemini as
//! bindable third-party providers (searches surfaced WWDC 2026 coverage of
//! Apple licensing Google's Gemini model specifically to power Siri, a
//! narrower and different integration than a general developer-facing
//! multi-provider framework parameter). Treat the precedent above as the
//! task's premise, not as independently verified by this implementation —
//! flagged here rather than silently presented as confirmed fact.
//!
//! # Not yet ruled on — do not ship beyond a dev POC without sign-off
//!
//! Sending household prompt text to a third-party API is a real data-egress
//! decision. This is explicitly a `v0.2+`/experimental feature, exactly like
//! `candle.rs`'s own POC exception to invariant #2 is scoped and flagged.
//! **A real household-product ship needs a security-engineer/DJ privacy
//! ruling first** — this backend does not imply that ruling has happened.
//! Concretely:
//!
//! - Off by default: gated behind the `claude-proxy` feature, so default
//!   `cargo build`/`cargo test`/`cargo clippy` (what Jenkins runs) never
//!   touch it.
//! - Inert even when compiled in: [`ClaudeProxyInferenceService::new`]
//!   fails fast on an empty `api_key`, so simply enabling the feature does
//!   not cause any prompt to leave the device.
//! - No audit-trail wiring here (invariant #5 territory) — a real ship needs
//!   that decided and implemented at the integration layer, not assumed.
//!
//! # Known limitations (single-shot only, by design)
//!
//! This backend implements exactly the trait's existing shape — a single
//! prompt in, a single completion out. It deliberately does **not** support:
//! streaming responses, tool use / function calling, or multi-turn
//! conversation (each `infer()` call is a fresh, independent Messages API
//! request with a single `user` message — no conversation history is
//! threaded across calls). Extending the trait to cover any of those is a
//! separate, larger design conversation, out of scope here.

use crate::error::{AiKitError, Result};
use crate::service::InferenceService;
use crate::types::{CacheStats, InferenceParams, InferenceResponse};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Default Anthropic API base URL, used when [`ClaudeProxyConfig::base_url`] is `None`.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// Anthropic Messages API version header value this backend was built against.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Configuration for [`ClaudeProxyInferenceService`].
#[derive(Clone)]
pub struct ClaudeProxyConfig {
    /// Anthropic API key. Must be non-empty — see
    /// [`ClaudeProxyInferenceService::new`].
    pub api_key: String,

    /// Override for the Anthropic API base URL. `None` uses
    /// [`DEFAULT_BASE_URL`]. Exists so tests (and any future self-hosted
    /// proxy in front of Anthropic) can point this at something other than
    /// the real Anthropic API.
    pub base_url: Option<String>,
}

impl std::fmt::Debug for ClaudeProxyConfig {
    /// Manual impl: never print `api_key` in full — this is a secret that
    /// ends up in logs/panics otherwise.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaudeProxyConfig")
            .field("api_key", &"<redacted>")
            .field("base_url", &self.base_url)
            .finish()
    }
}

/// Cloud proxy backend binding [`InferenceService`] to Anthropic's Messages
/// API. See module docs for scope, precedent, and the open-ruling caveat.
pub struct ClaudeProxyInferenceService {
    api_key: String,
    base_url: String,
    client: reqwest::Client,
}

impl ClaudeProxyInferenceService {
    /// Construct a new proxy backend.
    ///
    /// Fails fast with [`AiKitError::InvalidParams`] if `api_key` is empty —
    /// this is the "inert even with the feature compiled in" gate described
    /// in the module doc: enabling `--features claude-proxy` alone cannot
    /// cause any prompt to leave the device, an explicit key is also
    /// required.
    pub fn new(config: ClaudeProxyConfig) -> Result<Self> {
        if config.api_key.trim().is_empty() {
            return Err(AiKitError::InvalidParams {
                reason: "ClaudeProxyConfig.api_key must not be empty (this backend is inert \
                         without an explicit key, even when the claude-proxy feature is compiled in)"
                    .to_string(),
            });
        }

        let client =
            reqwest::Client::builder()
                .build()
                .map_err(|e| AiKitError::RemoteBackendError {
                    backend: "claude".to_string(),
                    reason: format!("failed to build HTTP client: {e}"),
                })?;

        Ok(Self {
            api_key: config.api_key,
            base_url: config
                .base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            client,
        })
    }
}

/// Request body for `POST /v1/messages`.
#[derive(Debug, Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    messages: Vec<MessageParam<'a>>,
}

#[derive(Debug, Serialize)]
struct MessageParam<'a> {
    role: &'a str,
    content: &'a str,
}

/// Successful response body shape from `POST /v1/messages`.
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<ContentBlock>,
    usage: Option<Usage>,
    /// Anthropic's own echo of the model that served the request. Not used
    /// for [`InferenceResponse::model_version`] — see `infer()`'s doc
    /// comment on that choice — kept here (and via `#[derive(Debug)]`, read)
    /// for future diagnostic use rather than discarded at parse time.
    #[serde(default)]
    #[allow(dead_code)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}

/// Anthropic's error response shape: `{"type":"error","error":{"type":"...","message":"..."}}`.
#[derive(Debug, Deserialize)]
struct AnthropicErrorBody {
    error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

#[async_trait]
impl InferenceService for ClaudeProxyInferenceService {
    async fn infer(&self, params: InferenceParams) -> Result<InferenceResponse> {
        params.validate()?;

        // `preferred_quantization` doesn't apply to a stateless remote proxy
        // — there is no local weight file to quantize. Per the task's
        // instruction, `Some(_)` here is silently ignored rather than
        // erroring (it simply doesn't mean anything in this backend), and
        // `InferenceResponse::quantization` below always reports the literal
        // "n/a (cloud proxy)" rather than pretending to honor it.

        let body = MessagesRequest {
            model: &params.model_id,
            max_tokens: params.max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            messages: vec![MessageParam {
                role: "user",
                content: &params.prompt,
            }],
        };

        let url = format!("{}/v1/messages", self.base_url);

        let start = std::time::Instant::now();
        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AiKitError::RemoteBackendError {
                backend: "claude".to_string(),
                reason: format!("HTTP request to {url} failed: {e}"),
            })?;
        let inference_ms = u32::try_from(start.elapsed().as_millis()).unwrap_or(u32::MAX);

        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| AiKitError::RemoteBackendError {
                backend: "claude".to_string(),
                reason: format!("failed to read response body: {e}"),
            })?;

        if !status.is_success() {
            let reason = match serde_json::from_slice::<AnthropicErrorBody>(&bytes) {
                Ok(err_body) => format!(
                    "HTTP {status}: {} — {}",
                    err_body.error.error_type, err_body.error.message
                ),
                Err(_) => format!("HTTP {status}: {}", String::from_utf8_lossy(&bytes)),
            };
            return Err(AiKitError::RemoteBackendError {
                backend: "claude".to_string(),
                reason,
            });
        }

        let parsed: MessagesResponse =
            serde_json::from_slice(&bytes).map_err(|e| AiKitError::RemoteBackendError {
                backend: "claude".to_string(),
                reason: format!("response did not match expected Messages API shape: {e}"),
            })?;

        // Realistic to have more than one text block; concatenate them all.
        let completion: String = parsed
            .content
            .iter()
            .filter(|block| block.block_type == "text")
            .filter_map(|block| block.text.as_deref())
            .collect::<Vec<_>>()
            .join("");

        let usage = parsed.usage.ok_or_else(|| AiKitError::RemoteBackendError {
            backend: "claude".to_string(),
            reason: "response missing usage object".to_string(),
        })?;

        let session_id = params.session_id.clone().unwrap_or_default();

        Ok(InferenceResponse {
            completion,
            total_tokens: usage.input_tokens + usage.output_tokens,
            prompt_tokens: usage.input_tokens,
            completion_tokens: usage.output_tokens,
            inference_ms,
            // Echo the caller's requested model_id rather than the
            // response's own `model` field: `params.model_id` is guaranteed
            // non-empty by `validate()` above, so this is never ambiguous,
            // and it reflects what the *caller* asked for (consistent with
            // how `CandleInferenceService` reports `model.version` from its
            // own registry lookup, not from anything the backend invents).
            model_version: params.model_id.clone(),
            // There is no quantization concept for a cloud proxy — don't
            // fake one.
            quantization: "n/a (cloud proxy)".to_string(),
            session_id,
            request_id: params
                .request_id
                .clone()
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        })
    }

    async fn load_model(&self, model_id: &str) -> Result<()> {
        // Nothing to load — this backend is a stateless remote proxy. Only
        // validate that a model_id was actually given; deliberately does
        // *not* maintain a local allowlist of known Claude model names
        // (e.g. "claude-opus-4", "claude-sonnet-4-5") since that list would
        // just go stale as Anthropic ships new models. Anthropic's own API
        // is the source of truth for whether a model_id is valid — an
        // invalid one surfaces as a `RemoteBackendError` from `infer()`.
        if model_id.trim().is_empty() {
            return Err(AiKitError::InvalidParams {
                reason: "model_id cannot be empty".to_string(),
            });
        }
        Ok(())
    }

    async fn unload_model(&self, _model_id: &str) -> Result<()> {
        // No-op: nothing was loaded.
        Ok(())
    }

    async fn cache_stats(&self) -> Result<CacheStats> {
        // Cache stats are meaningless for a stateless remote proxy — there
        // is nothing loaded and nothing cached locally. Returns a
        // zeroed-out snapshot rather than an error so callers that poll
        // `cache_stats()` across a mixed set of backends don't need special
        // handling for this one.
        Ok(CacheStats {
            models_loaded: 0,
            models_size_bytes: 0,
            sessions_active: 0,
            cache_size_bytes: 0,
            cache_hit_rate: 0.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    async fn end_session(&self, _session_id: &crate::types::SessionId) -> Result<()> {
        // No-op: no local session state is kept by this backend.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::InferenceParams;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn params(prompt: &str) -> InferenceParams {
        InferenceParams {
            model_id: "claude-sonnet-4-5".to_string(),
            prompt: prompt.to_string(),
            temperature: 0.7,
            top_p: 0.95,
            max_tokens: 100,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        }
    }

    fn service_for(mock: &MockServer, api_key: &str) -> ClaudeProxyInferenceService {
        ClaudeProxyInferenceService::new(ClaudeProxyConfig {
            api_key: api_key.to_string(),
            base_url: Some(mock.uri()),
        })
        .expect("valid config should construct")
    }

    #[test]
    fn empty_api_key_is_rejected_at_construction() {
        let result = ClaudeProxyInferenceService::new(ClaudeProxyConfig {
            api_key: String::new(),
            base_url: None,
        });
        assert!(matches!(result, Err(AiKitError::InvalidParams { .. })));
    }

    #[test]
    fn whitespace_only_api_key_is_rejected_at_construction() {
        let result = ClaudeProxyInferenceService::new(ClaudeProxyConfig {
            api_key: "   ".to_string(),
            base_url: None,
        });
        assert!(matches!(result, Err(AiKitError::InvalidParams { .. })));
    }

    #[test]
    fn default_base_url_is_used_when_none() {
        let service = ClaudeProxyInferenceService::new(ClaudeProxyConfig {
            api_key: "sk-test".to_string(),
            base_url: None,
        })
        .unwrap();
        assert_eq!(service.base_url, DEFAULT_BASE_URL);
    }

    #[tokio::test]
    async fn infer_success_round_trip() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-test-key"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "msg_01abc",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [
                    {"type": "text", "text": "Paris "},
                    {"type": "text", "text": "is the capital of France."}
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 8}
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let service = service_for(&mock_server, "sk-test-key");
        let response = service
            .infer(params("What is the capital of France?"))
            .await
            .expect("infer should succeed");

        assert_eq!(response.completion, "Paris is the capital of France.");
        assert_eq!(response.prompt_tokens, 12);
        assert_eq!(response.completion_tokens, 8);
        assert_eq!(response.total_tokens, 20);
        assert_eq!(response.model_version, "claude-sonnet-4-5");
        assert_eq!(response.quantization, "n/a (cloud proxy)");
        assert!(!response.request_id.is_empty());
    }

    #[tokio::test]
    async fn infer_sends_exact_request_body_shape() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(wiremock::matchers::body_json(json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.95,
                "messages": [{"role": "user", "content": "Hello there"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [{"type": "text", "text": "Hi!"}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
                "model": "claude-sonnet-4-5"
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let service = service_for(&mock_server, "sk-test-key");
        let response = service
            .infer(params("Hello there"))
            .await
            .expect("infer should succeed with matching body");
        assert_eq!(response.completion, "Hi!");
    }

    #[tokio::test]
    async fn infer_maps_non_2xx_to_remote_backend_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key"
                }
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let service = service_for(&mock_server, "sk-bad-key");
        let err = service
            .infer(params("Hello"))
            .await
            .expect_err("non-2xx should error");

        match err {
            AiKitError::RemoteBackendError { backend, reason } => {
                assert_eq!(backend, "claude");
                assert!(reason.contains("authentication_error"));
                assert!(reason.contains("invalid x-api-key"));
            }
            other => panic!("expected RemoteBackendError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn infer_maps_malformed_response_body_to_remote_backend_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json at all"))
            .expect(1)
            .mount(&mock_server)
            .await;

        let service = service_for(&mock_server, "sk-test-key");
        let err = service
            .infer(params("Hello"))
            .await
            .expect_err("malformed body should error");

        assert!(matches!(err, AiKitError::RemoteBackendError { .. }));
    }

    #[tokio::test]
    async fn infer_maps_missing_usage_to_remote_backend_error() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [{"type": "text", "text": "Hi!"}]
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let service = service_for(&mock_server, "sk-test-key");
        let err = service
            .infer(params("Hello"))
            .await
            .expect_err("missing usage should error");

        match err {
            AiKitError::RemoteBackendError { reason, .. } => {
                assert!(reason.contains("usage"));
            }
            other => panic!("expected RemoteBackendError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn load_model_validates_non_empty_id_only() {
        let mock_server = MockServer::start().await;
        let service = service_for(&mock_server, "sk-test-key");

        assert!(service.load_model("claude-sonnet-4-5").await.is_ok());
        assert!(matches!(
            service.load_model("").await,
            Err(AiKitError::InvalidParams { .. })
        ));
    }

    #[tokio::test]
    async fn unload_model_and_end_session_are_no_ops() {
        let mock_server = MockServer::start().await;
        let service = service_for(&mock_server, "sk-test-key");

        assert!(service.unload_model("anything").await.is_ok());
        assert!(service
            .end_session(&crate::types::SessionId::new())
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn cache_stats_is_a_zeroed_stub() {
        let mock_server = MockServer::start().await;
        let service = service_for(&mock_server, "sk-test-key");

        let stats = service.cache_stats().await.unwrap();
        assert_eq!(stats.models_loaded, 0);
        assert_eq!(stats.models_size_bytes, 0);
        assert_eq!(stats.sessions_active, 0);
        assert_eq!(stats.cache_size_bytes, 0);
        assert_eq!(stats.cache_hit_rate, 0.0);
    }

    #[tokio::test]
    async fn preferred_quantization_is_ignored_not_errored() {
        let mock_server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .mount(&mock_server)
            .await;

        let service = service_for(&mock_server, "sk-test-key");
        let mut p = params("Hello");
        p.preferred_quantization = Some("q4".to_string());

        let response = service.infer(p).await.expect("should not error");
        assert_eq!(response.quantization, "n/a (cloud proxy)");
    }
}
