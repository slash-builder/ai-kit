//! The Claude Messages API as a [`ProviderAdapter`]. **First slice: text requests, non-streaming. Tested against mocks only; no key exists yet.**
//!
//! What it does (each point is a wiremock test):
//! - the key is read from the [`CredentialSource`] **on every call** (rotation needs no restart); a missing credential means the adapter is never built;
//! - before sending it **reserves the worst-case cost in the [`Ledger`]**; if that does not fit, nothing is sent (`BudgetExceeded`);
//! - a **refusal** (`stop_reason: "refusal"`, an HTTP 200 that is billed) is a *result*, [`FinishReason::Refusal`];
//! - **two different spend-limit errors** map to `BudgetExceeded`, not `RateLimited`: a 400 "reached your specified API usage limits", and a 429
//!   whose `error.details.error_code` is `enforced_spend_limit_reached` (no `retry-after`);
//! - every other error maps to a class (`Auth`, `RateLimited{retry-after}`, `Overloaded`, `Unavailable`, `Timeout`, `ContextOverflow`, `Network`) and
//!   **never carries the upstream body or any header**;
//! - **no embeddings**: Anthropic has no embeddings API, so this adapter's `kinds` is text generation only.
//!
//! Not in this slice: streaming (SSE, including the mid-stream `event: error`), tool definitions, structured output, images, batch, prompt-cache
//! control. Model discovery ([`AnthropicAdapter::discover_models`]) is written against the vendor's documented fields **as summarised by a
//! research memo, not against a live response**: re-check the shape with a real key before relying on any capability flag.

use crate::budget::{Ledger, Pricing, Usage};
use crate::credentials::{CredentialError, CredentialSource};
use crate::egress::{Locality, ProviderInfo};
use crate::modality::{Output, Request, RequestKind};
use crate::model_card::{Cap, CardSource, ModelCard};
use crate::provider::{FinishReason, ProviderAdapter, ProviderError, ProviderReply};
use crate::types::InferenceResponse;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

const API_VERSION: &str = "2023-06-01";

/// Settings for one Anthropic provider (one credential, one workspace).
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Provider id as the policy and the audit record see it, for example `"anthropic"`.
    pub provider_id: String,
    pub base_url: String,
    /// The credential's NAME (resolved through the credential source), never the key.
    pub credential_name: String,
    pub model: String,
    /// Declared prices; Anthropic's prices change, so they are configured, never hard-coded.
    pub pricing: Pricing,
    pub default_max_tokens: u32,
    pub timeout: Duration,
}

impl AnthropicConfig {
    pub fn new(credential_name: &str, model: &str, pricing: Pricing) -> Self {
        Self {
            provider_id: "anthropic".into(),
            base_url: "https://api.anthropic.com".into(),
            credential_name: credential_name.into(),
            model: model.into(),
            pricing,
            default_max_tokens: 1024,
            timeout: Duration::from_secs(120),
        }
    }
}

pub struct AnthropicAdapter {
    info: ProviderInfo,
    cfg: AnthropicConfig,
    creds: Arc<dyn CredentialSource>,
    ledger: Arc<Ledger>,
    http: reqwest::Client,
}

impl fmt::Debug for AnthropicAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // the credential source is deliberately not printed
        f.debug_struct("AnthropicAdapter")
            .field("provider_id", &self.cfg.provider_id)
            .field("model", &self.cfg.model)
            .finish()
    }
}

impl AnthropicAdapter {
    /// Build the adapter. **Fails closed**: if the credential cannot be loaded now, no adapter exists.
    pub fn new(
        cfg: AnthropicConfig,
        creds: Arc<dyn CredentialSource>,
        ledger: Arc<Ledger>,
    ) -> Result<Self, CredentialError> {
        creds.get(&cfg.credential_name)?;
        let http = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|_| CredentialError::Unreadable("http client".into()))?;
        Ok(Self {
            info: ProviderInfo {
                provider_id: cfg.provider_id.clone(),
                locality: Locality::Saas,
                kinds: vec![RequestKind::TextGeneration],
            },
            cfg,
            creds,
            ledger,
            http,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.base_url.trim_end_matches('/'), path)
    }

    /// `GET /v1/models` as model cards. A flag the response does not carry is `Unknown`, which the router treats as no. `today` is `YYYY-MM-DD`.
    pub async fn discover_models(&self, today: &str) -> Result<Vec<ModelCard>, ProviderError> {
        let key = self
            .creds
            .get(&self.cfg.credential_name)
            .map_err(|_| ProviderError::Auth)?;
        let resp = self
            .http
            .get(self.url("/v1/models"))
            .header("x-api-key", key.expose())
            .header("anthropic-version", API_VERSION)
            .send()
            .await
            .map_err(transport_error)?;
        let status = resp.status().as_u16();
        let retry_after = retry_after(resp.headers());
        let text = resp.text().await.map_err(transport_error)?;
        let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if status != 200 {
            return Err(classify(status, retry_after, &body));
        }
        Ok(parse_models(&self.cfg.provider_id, &body, today))
    }
}

/// A capability flag may be a plain boolean or an object with `supported`; anything else (or absent) is unknown.
fn cap(v: Option<&Value>) -> Cap {
    match v {
        Some(Value::Bool(b)) => Cap::from_flag(Some(*b)),
        Some(Value::Object(o)) => Cap::from_flag(o.get("supported").and_then(Value::as_bool)),
        _ => Cap::Unknown,
    }
}

pub(crate) fn parse_models(provider_id: &str, body: &Value, today: &str) -> Vec<ModelCard> {
    body.get("data")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?.to_string();
                    let caps = m.get("capabilities");
                    Some(ModelCard {
                        provider_id: provider_id.to_string(),
                        model_id: id,
                        kinds: vec![RequestKind::TextGeneration],
                        max_input_tokens: m.get("max_input_tokens").and_then(Value::as_u64),
                        max_output_tokens: m.get("max_tokens").and_then(Value::as_u64),
                        retires_at: m
                            .get("retires_at")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        image_input: cap(caps.and_then(|c| c.get("image_input"))),
                        structured_outputs: cap(caps.and_then(|c| c.get("structured_outputs"))),
                        tool_use: cap(caps.and_then(|c| c.get("tool_use"))),
                        source: CardSource::Probed {
                            on: today.to_string(),
                        },
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn transport_error(e: reqwest::Error) -> ProviderError {
    if e.is_timeout() {
        ProviderError::Timeout
    } else {
        ProviderError::Network
    }
}

fn retry_after(h: &reqwest::header::HeaderMap) -> Option<u32> {
    h.get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

/// Map an error response to a class. Reads the body only to tell the two spend-limit errors and a context overflow apart; **the body itself
/// is never kept**.
pub(crate) fn classify(status: u16, retry_after: Option<u32>, body: &Value) -> ProviderError {
    let err = body.get("error");
    let message = err
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let code = err
        .and_then(|e| e.get("details"))
        .and_then(|d| d.get("error_code"))
        .and_then(Value::as_str)
        .unwrap_or("");
    match status {
        400 if message.contains("usage limit") => ProviderError::BudgetExceeded,
        400 if message.contains("prompt is too long") || message.contains("context") => {
            ProviderError::ContextOverflow
        }
        401 | 403 => ProviderError::Auth,
        402 => ProviderError::BudgetExceeded,
        413 => ProviderError::ContextOverflow,
        429 if code == "enforced_spend_limit_reached" => ProviderError::BudgetExceeded,
        429 => ProviderError::RateLimited {
            retry_after_s: retry_after,
        },
        504 => ProviderError::Timeout,
        529 => ProviderError::Overloaded,
        500..=599 => ProviderError::Unavailable,
        _ => ProviderError::Other,
    }
}

fn finish_of(stop_reason: &str) -> FinishReason {
    match stop_reason {
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "refusal" => FinishReason::Refusal,
        // end_turn, stop_sequence, pause_turn and anything new: a normal stop
        _ => FinishReason::Stop,
    }
}

#[async_trait]
impl ProviderAdapter for AnthropicAdapter {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }
    fn model_id(&self) -> &str {
        &self.cfg.model
    }

    async fn call(&self, req: &Request) -> Result<ProviderReply, ProviderError> {
        let Request::Text(p) = req else {
            return Err(ProviderError::Other);
        };
        let key = self
            .creds
            .get(&self.cfg.credential_name)
            .map_err(|_| ProviderError::Auth)?;
        let max_tokens = if p.max_tokens == 0 {
            self.cfg.default_max_tokens
        } else {
            p.max_tokens
        };
        // worst case: ~1 token per 3 bytes of prompt (conservative), and the whole output allowance
        let worst = self
            .cfg
            .pricing
            .worst_case(p.prompt.len() as u64 / 3 + 1, u64::from(max_tokens));
        let reservation = self.ledger.reserve(worst)?;

        let body = json!({
            "model": self.cfg.model,
            "max_tokens": max_tokens,
            "temperature": p.temperature,
            "messages": [{"role": "user", "content": p.prompt}],
        });
        let bytes = serde_json::to_vec(&body).map_err(|_| ProviderError::Other)?;
        let bytes_sent = bytes.len() as u64;
        let sent = self
            .http
            .post(self.url("/v1/messages"))
            .header("x-api-key", key.expose())
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .body(bytes)
            .send()
            .await;
        let resp = match sent {
            Ok(r) => r,
            Err(e) => {
                self.ledger.release(reservation);
                return Err(transport_error(e));
            }
        };
        let status = resp.status().as_u16();
        let retry = retry_after(resp.headers());
        let text = match resp.text().await {
            Ok(t) => t,
            Err(e) => {
                self.ledger.release(reservation);
                return Err(transport_error(e));
            }
        };
        let body: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if status != 200 {
            self.ledger.release(reservation);
            return Err(classify(status, retry, &body));
        }

        let usage = Usage {
            input_tokens: body["usage"]["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: body["usage"]["output_tokens"].as_u64().unwrap_or(0),
            cache_read_tokens: body["usage"]["cache_read_input_tokens"]
                .as_u64()
                .unwrap_or(0),
            cache_write_tokens: body["usage"]["cache_creation_input_tokens"]
                .as_u64()
                .unwrap_or(0),
        };
        // a 200 is billed, a refusal included
        self.ledger
            .settle(reservation, self.cfg.pricing.cost(&usage));
        let completion: String = body["content"]
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|b| b["type"] == "text")
                    .filter_map(|b| b["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        let served = body["model"]
            .as_str()
            .unwrap_or(&self.cfg.model)
            .to_string();
        Ok(ProviderReply {
            output: Output::Text(InferenceResponse {
                completion,
                total_tokens: (usage.input_tokens + usage.output_tokens) as u32,
                prompt_tokens: usage.input_tokens as u32,
                completion_tokens: usage.output_tokens as u32,
                inference_ms: 0,
                model_version: served.clone(),
                quantization: "n/a (cloud)".into(),
                session_id: Default::default(),
                request_id: String::new(),
            }),
            finish: finish_of(body["stop_reason"].as_str().unwrap_or("end_turn")),
            model_served: served,
            bytes_sent,
            bytes_received: text.len() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::InMemoryCredentials;
    use crate::egress::{
        AgeBand, Label, Policy, ProviderMatch, Purpose, Rule, SourceKind, Subject, VecSink,
    };
    use crate::error::AiKitError;
    use crate::provider::{run_chain, ChainConfig, ScriptedProvider};
    use crate::types::InferenceParams;
    use std::sync::Mutex;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // example prices for the tests only (micro-dollars per million tokens)
    const PRICING: Pricing = Pricing {
        input_micro_per_mtok: 3_000_000,
        output_micro_per_mtok: 15_000_000,
        cache_read_micro_per_mtok: 300_000,
        cache_write_micro_per_mtok: 3_750_000,
    };
    const KEY: &str = "sk-ant-TEST-ONLY-NOT-A-REAL-KEY";

    fn adapter(server: &MockServer, budget: u64) -> (AnthropicAdapter, Arc<Ledger>) {
        let ledger = Arc::new(Ledger::new(budget));
        let mut cfg = AnthropicConfig::new("anthropic-api-key", "claude-test-model", PRICING);
        cfg.base_url = server.uri();
        let a = AnthropicAdapter::new(
            cfg,
            Arc::new(InMemoryCredentials::with("anthropic-api-key", KEY)),
            ledger.clone(),
        )
        .unwrap();
        (a, ledger)
    }
    fn req(prompt: &str) -> Request {
        Request::Text(InferenceParams {
            model_id: "m".into(),
            prompt: prompt.into(),
            temperature: 0.0,
            top_p: 1.0,
            max_tokens: 100,
            session_id: None,
            preferred_quantization: None,
            request_id: None,
        })
    }
    fn ok_body(stop: &str, text: &str) -> Value {
        json!({"id":"msg_1","type":"message","role":"assistant","model":"claude-test-model-2026",
               "content":[{"type":"text","text":text}],"stop_reason":stop,
               "usage":{"input_tokens":1000,"output_tokens":500,"cache_read_input_tokens":2000,"cache_creation_input_tokens":4000}})
    }
    async fn serve(server: &MockServer, status: u16, body: Value) {
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(server)
            .await;
    }
    fn text_of(r: &ProviderReply) -> String {
        match &r.output {
            Output::Text(t) => t.completion.clone(),
            o => panic!("{o:?}"),
        }
    }

    #[tokio::test]
    async fn a_normal_answer_is_settled_from_the_reported_usage_including_cache_fields() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", KEY))
            .and(header("anthropic-version", API_VERSION))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("end_turn", "hello")))
            .expect(1)
            .mount(&s)
            .await;
        let (a, ledger) = adapter(&s, 10_000_000);
        let r = a.call(&req("hi")).await.unwrap();
        assert_eq!(
            (text_of(&r), r.finish),
            ("hello".to_string(), FinishReason::Stop)
        );
        assert_eq!(
            r.model_served, "claude-test-model-2026",
            "what was actually served is recorded"
        );
        // 1000 in + 500 out + 2000 cache read + 4000 cache write, at the example prices
        let expected = PRICING.cost(&Usage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_read_tokens: 2000,
            cache_write_tokens: 4000,
        });
        assert_eq!(ledger.spent(), expected);
        assert_eq!(
            ledger.available(),
            10_000_000 - expected,
            "the reservation is gone"
        );
        assert!(r.bytes_sent > 0 && r.bytes_received > 0);
    }

    #[tokio::test]
    async fn a_refusal_is_a_billed_result_not_an_error() {
        let s = MockServer::start().await;
        serve(&s, 200, ok_body("refusal", "I can't help with that.")).await;
        let (a, ledger) = adapter(&s, 10_000_000);
        let r = a.call(&req("hi")).await.unwrap();
        assert_eq!(r.finish, FinishReason::Refusal);
        assert!(ledger.spent() > 0, "a refusal is billed");
    }

    #[tokio::test]
    async fn a_refusal_never_leads_to_a_second_hop_in_a_chain() {
        let s = MockServer::start().await;
        serve(&s, 200, ok_body("refusal", "no")).await;
        let (a, _) = adapter(&s, 10_000_000);
        let fallback =
            ScriptedProvider::new("gemini", Locality::Saas, &[RequestKind::TextGeneration]);
        let chain: Vec<Arc<dyn ProviderAdapter>> = vec![Arc::new(a), fallback.clone()];
        let policy = Policy {
            version: 1,
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[])],
        };
        let label = Label::new(
            SourceKind::HouseholdFile,
            &["notes"],
            &[Subject::person("alice", AgeBand::Adult)],
        );
        let (reply, _) = run_chain(
            &chain,
            &policy,
            &label,
            &req("hi"),
            &ChainConfig::new(Purpose::Answer, 1),
            &VecSink::default(),
        )
        .await
        .unwrap();
        assert_eq!(reply.finish, FinishReason::Refusal);
        assert_eq!(fallback.calls(), 0);
    }

    #[tokio::test]
    async fn tool_use_and_length_stops_are_mapped() {
        let s = MockServer::start().await;
        serve(&s, 200, ok_body("tool_use", "")).await;
        let (a, _) = adapter(&s, 10_000_000);
        assert_eq!(
            a.call(&req("x")).await.unwrap().finish,
            FinishReason::ToolCalls
        );
        assert_eq!(finish_of("max_tokens"), FinishReason::Length);
        assert_eq!(
            finish_of("model_context_window_exceeded"),
            FinishReason::Length
        );
    }

    #[tokio::test]
    async fn errors_map_to_classes_and_release_the_reservation() {
        let cases: Vec<(u16, Option<&str>, Value, ProviderError)> = vec![
            (
                401,
                None,
                json!({"error":{"type":"authentication_error","message":"invalid x-api-key SECRET-UPSTREAM-TEXT"}}),
                ProviderError::Auth,
            ),
            (
                402,
                None,
                json!({"error":{"type":"billing_error","message":"x"}}),
                ProviderError::BudgetExceeded,
            ),
            (
                429,
                Some("7"),
                json!({"error":{"type":"rate_limit_error","message":"slow down"}}),
                ProviderError::RateLimited {
                    retry_after_s: Some(7),
                },
            ),
            (
                429,
                None,
                json!({"error":{"type":"rate_limit_error","message":"cap","details":{"error_code":"enforced_spend_limit_reached"}}}),
                ProviderError::BudgetExceeded,
            ),
            (
                400,
                None,
                json!({"error":{"type":"invalid_request_error","message":"You have reached your specified API usage limits."}}),
                ProviderError::BudgetExceeded,
            ),
            (
                400,
                None,
                json!({"error":{"type":"invalid_request_error","message":"prompt is too long: 300000 tokens"}}),
                ProviderError::ContextOverflow,
            ),
            (
                400,
                None,
                json!({"error":{"type":"invalid_request_error","message":"something else"}}),
                ProviderError::Other,
            ),
            (
                529,
                None,
                json!({"error":{"type":"overloaded_error","message":"x"}}),
                ProviderError::Overloaded,
            ),
            (
                500,
                None,
                json!({"error":{"type":"api_error","message":"x"}}),
                ProviderError::Unavailable,
            ),
            (
                504,
                None,
                json!({"error":{"type":"timeout_error","message":"x"}}),
                ProviderError::Timeout,
            ),
        ];
        for (status, retry, body, want) in cases {
            let s = MockServer::start().await;
            let mut t = ResponseTemplate::new(status).set_body_json(body);
            if let Some(r) = retry {
                t = t.insert_header("retry-after", r);
            }
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .respond_with(t)
                .mount(&s)
                .await;
            let (a, ledger) = adapter(&s, 10_000_000);
            let got = a.call(&req("x")).await.unwrap_err();
            assert_eq!(got, want, "HTTP {status}");
            assert_eq!(
                ledger.available(),
                10_000_000,
                "HTTP {status}: an unbilled failure must release the reservation"
            );
            assert_eq!(ledger.spent(), 0);
        }
    }

    #[tokio::test]
    async fn the_upstream_body_never_reaches_the_caller() {
        let s = MockServer::start().await;
        serve(&s, 401, json!({"error":{"type":"authentication_error","message":"invalid key SECRET-UPSTREAM-TEXT"}})).await;
        let (a, _) = adapter(&s, 10_000_000);
        let chain: Vec<Arc<dyn ProviderAdapter>> = vec![Arc::new(a)];
        let policy = Policy {
            version: 1,
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[])],
        };
        let label = Label::new(
            SourceKind::HouseholdFile,
            &["notes"],
            &[Subject::household()],
        );
        let e = run_chain(
            &chain,
            &policy,
            &label,
            &req("x"),
            &ChainConfig::new(Purpose::Answer, 1),
            &VecSink::default(),
        )
        .await
        .unwrap_err();
        let shown = format!("{e} {e:?}");
        assert!(matches!(e, AiKitError::RemoteBackendError { .. }));
        assert!(!shown.contains("SECRET") && !shown.contains(KEY), "{shown}");
    }

    #[tokio::test]
    async fn a_call_that_does_not_fit_the_budget_sends_nothing() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("end_turn", "x")))
            .expect(0)
            .mount(&s)
            .await;
        let (a, ledger) = adapter(&s, 10); // 10 micro-dollars: nothing fits
        assert_eq!(
            a.call(&req("hello")).await.unwrap_err(),
            ProviderError::BudgetExceeded
        );
        assert_eq!(ledger.available(), 10);
        // wiremock verifies expect(0) when the server drops
    }

    #[tokio::test]
    async fn a_daily_cap_below_the_worst_case_sends_nothing_even_with_a_large_total_budget() {
        struct Fixed;
        impl crate::budget::Clock for Fixed {
            fn now(&self) -> u64 {
                0
            }
        }
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("end_turn", "x")))
            .expect(0)
            .mount(&s)
            .await;
        // $10 in total, but only 1,000 micro-dollars a day; the worst case for this call is well above 1,000
        let ledger = Arc::new(Ledger::with_daily_cap(10_000_000, 1_000, Arc::new(Fixed)));
        let mut cfg = AnthropicConfig::new("anthropic-api-key", "claude-test-model", PRICING);
        cfg.base_url = s.uri();
        let a = AnthropicAdapter::new(
            cfg,
            Arc::new(InMemoryCredentials::with("anthropic-api-key", KEY)),
            ledger.clone(),
        )
        .unwrap();
        assert_eq!(
            a.call(&req("hello")).await.unwrap_err(),
            ProviderError::BudgetExceeded
        );
        assert_eq!((ledger.available(), ledger.spent()), (1_000, 0));
    }

    #[tokio::test]
    async fn a_missing_credential_means_no_adapter_and_the_key_is_read_per_call() {
        let s = MockServer::start().await;
        let ledger = Arc::new(Ledger::new(1));
        let mut cfg = AnthropicConfig::new("anthropic-api-key", "m", PRICING);
        cfg.base_url = s.uri();
        assert!(
            AnthropicAdapter::new(
                cfg.clone(),
                Arc::new(InMemoryCredentials::default()),
                ledger
            )
            .is_err(),
            "fail closed"
        );

        // rotation: the header carries whatever the source returns NOW
        struct Rotating(Mutex<String>);
        impl CredentialSource for Rotating {
            fn get(&self, _: &str) -> Result<crate::credentials::Secret, CredentialError> {
                Ok(crate::credentials::Secret::new(
                    self.0.lock().unwrap().clone(),
                ))
            }
        }
        let src = Arc::new(Rotating(Mutex::new("key-one".into())));
        Mock::given(method("POST"))
            .and(header("x-api-key", "key-two"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("end_turn", "rotated")))
            .expect(1)
            .mount(&s)
            .await;
        let a = AnthropicAdapter::new(cfg, src.clone(), Arc::new(Ledger::new(10_000_000))).unwrap();
        *src.0.lock().unwrap() = "key-two".into();
        assert_eq!(text_of(&a.call(&req("x")).await.unwrap()), "rotated");
    }

    #[test]
    fn anthropic_offers_no_embeddings_and_the_debug_output_has_no_key() {
        let ledger = Arc::new(Ledger::new(1));
        let cfg = AnthropicConfig::new("anthropic-api-key", "m", PRICING);
        let a = AnthropicAdapter::new(
            cfg,
            Arc::new(InMemoryCredentials::with("anthropic-api-key", KEY)),
            ledger,
        )
        .unwrap();
        assert_eq!(a.info().kinds, vec![RequestKind::TextGeneration]);
        assert_eq!(a.info().locality, Locality::Saas);
        assert!(!format!("{a:?}").contains("TEST-ONLY"));
    }

    #[tokio::test]
    async fn model_discovery_builds_cards_and_unknown_means_false() {
        let s = MockServer::start().await;
        let list = json!({"data":[
            {"id":"claude-a","max_input_tokens":200000,"max_tokens":8192,"retires_at":"2027-01-01","capabilities":{"image_input":{"supported":true},"structured_outputs":false}},
            {"id":"claude-b"}
        ]});
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(list))
            .mount(&s)
            .await;
        let (a, _) = adapter(&s, 1);
        let cards = a.discover_models("2026-10-11").await.unwrap();
        assert_eq!(cards.len(), 2);
        assert_eq!(
            (
                cards[0].image_input,
                cards[0].structured_outputs,
                cards[0].tool_use
            ),
            (Cap::Yes, Cap::No, Cap::Unknown)
        );
        assert!(cards[0].fits_input(200_000) && !cards[0].fits_input(200_001));
        assert_eq!(
            cards[0].source,
            CardSource::Probed {
                on: "2026-10-11".into()
            }
        );
        assert!(
            !cards[1].image_input.is_yes() && !cards[1].fits_input(1),
            "a card with no facts claims nothing"
        );
    }
}
