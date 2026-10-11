//! Google Gemini (the Developer API: `generateContent` and `batchEmbedContents`) as a [`ProviderAdapter`]. **Tested against mocks only; no key exists yet.**
//!
//! What it does (each point is a wiremock test):
//! - the key is read from the [`CredentialSource`] **on every call** and sent as `x-goog-api-key`; a missing credential means no adapter;
//! - before sending it **reserves the worst-case cost in the [`Ledger`]**; if that does not fit, nothing is sent;
//! - a **blocked or filtered answer** (a candidate `finishReason` of `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, ...,
//!   or a `promptFeedback.blockReason` with no candidate) is a *result*, [`FinishReason::ContentFilter`], and never causes a second hop;
//! - **thinking tokens are billed as output** (`thoughtsTokenCount` is added to the output count) and cached tokens are priced as cache reads;
//! - errors map to classes and **never carry the upstream body or any header**; a daily-quota 429 is `BudgetExceeded` (it will not clear by
//!   retrying), an ordinary rate limit is `RateLimited`, and the 400 that Google returns for a bad key is `Auth`;
//! - embeddings go through `batchEmbedContents`; `model_served` records which embedding model produced the vectors, because two Gemini
//!   embedding models have incompatible vector spaces and a silent swap would corrupt an index.
//!
//! **Read before relying on it.** The wire shapes below follow Google's public documentation as summarised by a research memo, **not a live
//! response**: re-check them with a real key. Specific unknowns: whether a 429 carries `Retry-After`, the exact body that tells a per-minute limit
//! from a daily quota, and the full `finishReason` list. And the **data terms depend on the project's billing state, not the key** (an unpaid
//! project's inputs may be used to improve Google's products and read by human reviewers): that is a registry attestation per credential
//! and project, outside this adapter; an unverifiable tier must be treated as the strictest terms.
//!
//! Not in this slice: streaming, function declarations, structured output, images, the Files API, the newer Interactions API, batch mode.

use super::{Budget, ConfigError};
use crate::budget::Usage;
use crate::credentials::CredentialSource;
use crate::egress::{Locality, ProviderInfo};
use crate::modality::{EmbeddingResponse, Output, Request, RequestKind};
use crate::model_card::{Cap, CardSource, ModelCard};
use crate::provider::{FinishReason, ProviderAdapter, ProviderError, ProviderReply};
use crate::types::InferenceResponse;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct GeminiConfig {
    pub provider_id: String,
    pub base_url: String,
    /// Credential NAME, never the key.
    pub credential_name: String,
    /// The model for text requests, for example `gemini-2.5-flash` (configured, never assumed).
    pub model: String,
    /// The model for embedding requests; if `None` the adapter does not offer embeddings.
    pub embedding_model: Option<String>,
    pub default_max_tokens: u32,
    pub timeout: Duration,
}

impl GeminiConfig {
    pub fn new(credential_name: &str, model: &str) -> Self {
        Self {
            provider_id: "google-gemini".into(),
            base_url: "https://generativelanguage.googleapis.com".into(),
            credential_name: credential_name.into(),
            model: model.into(),
            embedding_model: None,
            default_max_tokens: 1024,
            timeout: Duration::from_secs(120),
        }
    }
}

pub struct GeminiAdapter {
    info: ProviderInfo,
    cfg: GeminiConfig,
    creds: Arc<dyn CredentialSource>,
    budget: Budget,
    http: reqwest::Client,
}

impl fmt::Debug for GeminiAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiAdapter")
            .field("provider_id", &self.cfg.provider_id)
            .field("model", &self.cfg.model)
            .finish()
    }
}

impl GeminiAdapter {
    /// Build the adapter. **Fails closed** if the credential cannot be loaded now.
    pub fn new(
        cfg: GeminiConfig,
        creds: Arc<dyn CredentialSource>,
        budget: Budget,
    ) -> Result<Self, ConfigError> {
        creds
            .get(&cfg.credential_name)
            .map_err(|_| ConfigError::CredentialUnavailable)?;
        let mut kinds = vec![RequestKind::TextGeneration];
        if cfg.embedding_model.is_some() {
            kinds.push(RequestKind::Embedding);
        }
        let http = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|_| ConfigError::HttpClient)?;
        Ok(Self {
            info: ProviderInfo {
                provider_id: cfg.provider_id.clone(),
                locality: Locality::Saas,
                kinds,
            },
            cfg,
            creds,
            budget,
            http,
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.cfg.base_url.trim_end_matches('/'), path)
    }

    fn key(&self) -> Result<String, ProviderError> {
        self.creds
            .get(&self.cfg.credential_name)
            .map(|s| s.expose().to_string())
            .map_err(|_| ProviderError::Auth)
    }

    /// `GET /v1beta/models` as cards. A model that lists `generateContent` serves text, `embedContent` serves embeddings; capability flags the
    /// listing does not carry stay unknown. Only the first page is read. `today` is `YYYY-MM-DD`.
    pub async fn discover_models(&self, today: &str) -> Result<Vec<ModelCard>, ProviderError> {
        let key = self.key()?;
        let resp = self
            .http
            .get(self.url("/v1beta/models"))
            .header("x-goog-api-key", key)
            .send()
            .await
            .map_err(transport_error)?;
        let status = resp.status().as_u16();
        let retry = retry_after(resp.headers());
        let body: Value = serde_json::from_str(&resp.text().await.map_err(transport_error)?)
            .unwrap_or(Value::Null);
        if status != 200 {
            return Err(classify(status, retry, &body));
        }
        Ok(parse_models(&self.cfg.provider_id, &body, today))
    }

    async fn post(&self, path: &str, body: &Value) -> Result<(u64, Value, u64), ProviderError> {
        let key = self.key()?;
        let bytes = serde_json::to_vec(body).map_err(|_| ProviderError::Other)?;
        let sent = bytes.len() as u64;
        let resp = self
            .http
            .post(self.url(path))
            .header("x-goog-api-key", key)
            .header("content-type", "application/json")
            .body(bytes)
            .send()
            .await
            .map_err(transport_error)?;
        let status = resp.status().as_u16();
        let retry = retry_after(resp.headers());
        let text = resp.text().await.map_err(transport_error)?;
        let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if status != 200 {
            return Err(classify(status, retry, &parsed));
        }
        Ok((sent, parsed, text.len() as u64))
    }

    async fn generate(
        &self,
        p: &crate::types::InferenceParams,
    ) -> Result<ProviderReply, ProviderError> {
        let max_tokens = if p.max_tokens == 0 {
            self.cfg.default_max_tokens
        } else {
            p.max_tokens
        };
        let reservation = self.budget.ledger.reserve(
            self.budget
                .pricing
                .worst_case(p.prompt.len() as u64 / 3 + 1, u64::from(max_tokens)),
        )?;
        let body = json!({
            "contents": [{"role": "user", "parts": [{"text": p.prompt}]}],
            "generationConfig": {"temperature": p.temperature, "maxOutputTokens": max_tokens},
        });
        let path = format!("/v1beta/models/{}:generateContent", self.cfg.model);
        let (sent, resp, received) = match self.post(&path, &body).await {
            Ok(r) => r,
            Err(e) => {
                self.budget.ledger.release(reservation);
                return Err(e);
            }
        };
        let u = &resp["usageMetadata"];
        let cached = u["cachedContentTokenCount"].as_u64().unwrap_or(0);
        let usage = Usage {
            // promptTokenCount includes the cached part: bill the cached tokens at the cache-read rate and the rest as input
            input_tokens: u["promptTokenCount"]
                .as_u64()
                .unwrap_or(0)
                .saturating_sub(cached),
            // thinking tokens are billed as output
            output_tokens: u["candidatesTokenCount"].as_u64().unwrap_or(0)
                + u["thoughtsTokenCount"].as_u64().unwrap_or(0),
            cache_read_tokens: cached,
            cache_write_tokens: 0,
        };
        self.budget
            .ledger
            .settle(reservation, self.budget.pricing.cost(&usage));

        let cand = &resp["candidates"][0];
        let parts = cand["content"]["parts"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let blocked_prompt = !resp["promptFeedback"]["blockReason"].is_null() && cand.is_null();
        let finish = if blocked_prompt {
            FinishReason::ContentFilter
        } else if parts.iter().any(|x| x.get("functionCall").is_some()) {
            FinishReason::ToolCalls
        } else {
            match cand["finishReason"].as_str().unwrap_or("STOP") {
                "MAX_TOKENS" => FinishReason::Length,
                "STOP" => FinishReason::Stop,
                // SAFETY, RECITATION, BLOCKLIST, PROHIBITED_CONTENT, SPII, IMAGE_SAFETY, LANGUAGE, ...: blocked, not retried elsewhere
                "FINISH_REASON_UNSPECIFIED" | "OTHER" => FinishReason::Stop,
                _ => FinishReason::ContentFilter,
            }
        };
        let completion: String = parts
            .iter()
            .filter(|x| x["thought"] != true)
            .filter_map(|x| x["text"].as_str())
            .collect::<Vec<_>>()
            .join("");
        let served = resp["modelVersion"]
            .as_str()
            .unwrap_or(&self.cfg.model)
            .to_string();
        Ok(ProviderReply {
            output: Output::Text(InferenceResponse {
                completion,
                total_tokens: (usage.input_tokens + usage.output_tokens + usage.cache_read_tokens)
                    as u32,
                prompt_tokens: (usage.input_tokens + usage.cache_read_tokens) as u32,
                completion_tokens: usage.output_tokens as u32,
                inference_ms: 0,
                model_version: served.clone(),
                quantization: "n/a (cloud)".into(),
                session_id: Default::default(),
                request_id: String::new(),
            }),
            finish,
            model_served: served,
            bytes_sent: sent,
            bytes_received: received,
        })
    }

    async fn embed(
        &self,
        p: &crate::modality::EmbeddingParams,
    ) -> Result<ProviderReply, ProviderError> {
        let model = self
            .cfg
            .embedding_model
            .as_deref()
            .ok_or(ProviderError::Other)?;
        let requests: Vec<Value> = p.inputs.iter().map(|t| json!({"model": format!("models/{model}"), "content": {"parts": [{"text": t}]}})).collect();
        let (sent, resp, received) = self
            .post(
                &format!("/v1beta/models/{model}:batchEmbedContents"),
                &json!({"requests": requests}),
            )
            .await?;
        let mut vectors: Vec<Vec<f32>> = resp["embeddings"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|e| {
                        e["values"]
                            .as_array()
                            .map(|v| {
                                v.iter()
                                    .filter_map(Value::as_f64)
                                    .map(|x| x as f32)
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default();
        if vectors.len() != p.inputs.len() || vectors.iter().any(Vec::is_empty) {
            return Err(ProviderError::Other);
        }
        if p.normalize {
            for v in &mut vectors {
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if n > 0.0 {
                    v.iter_mut().for_each(|x| *x /= n);
                }
            }
        }
        let dimensions = vectors.first().map_or(0, Vec::len);
        Ok(ProviderReply {
            output: Output::Embeddings(EmbeddingResponse {
                vectors,
                dimensions,
                prompt_tokens: 0,
                inference_ms: 0,
                // the model the caller configured: vector spaces differ between Gemini embedding models
                model_version: model.to_string(),
                session_id: Default::default(),
                request_id: String::new(),
            }),
            finish: FinishReason::Stop,
            model_served: model.to_string(),
            bytes_sent: sent,
            bytes_received: received,
        })
    }
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

pub(crate) fn parse_models(provider_id: &str, body: &Value, today: &str) -> Vec<ModelCard> {
    body["models"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let name = m["name"]
                        .as_str()?
                        .trim_start_matches("models/")
                        .to_string();
                    let methods: Vec<&str> = m["supportedGenerationMethods"]
                        .as_array()
                        .map(|v| v.iter().filter_map(Value::as_str).collect())
                        .unwrap_or_default();
                    let mut kinds = Vec::new();
                    if methods.contains(&"generateContent") {
                        kinds.push(RequestKind::TextGeneration);
                    }
                    if methods.contains(&"embedContent") || methods.contains(&"batchEmbedContents")
                    {
                        kinds.push(RequestKind::Embedding);
                    }
                    Some(ModelCard {
                        provider_id: provider_id.to_string(),
                        model_id: name,
                        kinds,
                        max_input_tokens: m["inputTokenLimit"].as_u64(),
                        max_output_tokens: m["outputTokenLimit"].as_u64(),
                        retires_at: None,
                        image_input: Cap::Unknown,
                        structured_outputs: Cap::Unknown,
                        tool_use: Cap::Unknown,
                        source: CardSource::Probed {
                            on: today.to_string(),
                        },
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Map an error response to a class. The body is read only to tell a bad key, a context overflow and a daily quota apart; it is never kept.
pub(crate) fn classify(status: u16, retry_after: Option<u32>, body: &Value) -> ProviderError {
    let message = body["error"]["message"]
        .as_str()
        .unwrap_or("")
        .to_ascii_lowercase();
    let state = body["error"]["status"].as_str().unwrap_or("");
    match status {
        // Google answers a bad API key with 400 INVALID_ARGUMENT, not 401
        400 if message.contains("api key not valid") || message.contains("api key expired") => {
            ProviderError::Auth
        }
        400 if message.contains("token count")
            || message.contains("input token")
            || message.contains("too long") =>
        {
            ProviderError::ContextOverflow
        }
        401 | 403 => ProviderError::Auth,
        402 => ProviderError::BudgetExceeded,
        404 => ProviderError::ModelNotFound,
        // a daily quota will not clear by retrying; an ordinary rate limit will
        429 if message.contains("daily")
            || message.contains("per day")
            || (message.contains("quota") && !message.contains("per minute")) =>
        {
            ProviderError::BudgetExceeded
        }
        429 => ProviderError::RateLimited {
            retry_after_s: retry_after,
        },
        503 => ProviderError::Overloaded,
        504 => ProviderError::Timeout,
        500..=599 if state == "UNAVAILABLE" => ProviderError::Overloaded,
        500..=599 => ProviderError::Unavailable,
        _ => ProviderError::Other,
    }
}

#[async_trait]
impl ProviderAdapter for GeminiAdapter {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }
    fn model_id(&self) -> &str {
        &self.cfg.model
    }
    async fn call(&self, req: &Request) -> Result<ProviderReply, ProviderError> {
        match req {
            Request::Text(p) => self.generate(p).await,
            Request::Embedding(p) => self.embed(p).await,
            _ => Err(ProviderError::Other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{Ledger, Pricing};
    use crate::credentials::InMemoryCredentials;
    use crate::egress::{
        AgeBand, Label, Policy, ProviderMatch, Purpose, Rule, SourceKind, Subject, VecSink,
    };
    use crate::modality::EmbeddingParams;
    use crate::provider::{run_chain, ChainConfig, ScriptedProvider};
    use crate::types::InferenceParams;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const PRICING: Pricing = Pricing {
        input_micro_per_mtok: 1_000_000,
        output_micro_per_mtok: 4_000_000,
        cache_read_micro_per_mtok: 250_000,
        cache_write_micro_per_mtok: 0,
    };
    const KEY: &str = "AIza-TEST-ONLY-NOT-A-REAL-KEY";

    fn adapter(s: &MockServer, budget: u64) -> (GeminiAdapter, Arc<Ledger>) {
        let ledger = Arc::new(Ledger::new(budget));
        let mut cfg = GeminiConfig::new("gemini-api-key", "gemini-test");
        cfg.base_url = s.uri();
        let a = GeminiAdapter::new(
            cfg,
            Arc::new(InMemoryCredentials::with("gemini-api-key", KEY)),
            Budget {
                ledger: ledger.clone(),
                pricing: PRICING,
            },
        )
        .unwrap();
        (a, ledger)
    }
    fn text(prompt: &str) -> Request {
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
    fn ok(reason: &str, t: &str) -> Value {
        json!({"candidates":[{"content":{"role":"model","parts":[{"text":t}]},"finishReason":reason}],
               "usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":200,"thoughtsTokenCount":50,"cachedContentTokenCount":400},
               "modelVersion":"gemini-test-001"})
    }
    async fn serve(s: &MockServer, status: u16, body: Value) {
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-test:generateContent"))
            .respond_with(ResponseTemplate::new(status).set_body_json(body))
            .mount(s)
            .await;
    }
    fn out(r: &ProviderReply) -> String {
        match &r.output {
            Output::Text(t) => t.completion.clone(),
            o => panic!("{o:?}"),
        }
    }

    #[tokio::test]
    async fn a_normal_answer_is_settled_with_thinking_as_output_and_cached_tokens_at_the_cache_rate(
    ) {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-test:generateContent"))
            .and(header("x-goog-api-key", KEY))
            .and(body_partial_json(
                json!({"generationConfig":{"maxOutputTokens":100}}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok("STOP", "hello")))
            .expect(1)
            .mount(&s)
            .await;
        let (a, ledger) = adapter(&s, 10_000_000);
        let r = a.call(&text("hi")).await.unwrap();
        assert_eq!(
            (out(&r), r.finish),
            ("hello".to_string(), FinishReason::Stop)
        );
        assert_eq!(r.model_served, "gemini-test-001");
        // prompt 1000 = 400 cached + 600 fresh; output 200 + 50 thinking
        let expected = PRICING.cost(&Usage {
            input_tokens: 600,
            output_tokens: 250,
            cache_read_tokens: 400,
            cache_write_tokens: 0,
        });
        assert_eq!(ledger.spent(), expected);
        assert_eq!(ledger.available(), 10_000_000 - expected);
    }

    #[tokio::test]
    async fn thought_parts_are_not_part_of_the_answer_and_function_calls_are_tool_calls() {
        let s = MockServer::start().await;
        serve(&s, 200, json!({"candidates":[{"content":{"parts":[{"text":"private reasoning","thought":true},{"text":"the answer"}]},"finishReason":"STOP"}],"usageMetadata":{}})).await;
        let (a, _) = adapter(&s, 10_000_000);
        assert_eq!(out(&a.call(&text("q")).await.unwrap()), "the answer");
        let s2 = MockServer::start().await;
        serve(&s2, 200, json!({"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}}]},"finishReason":"STOP"}],"usageMetadata":{}})).await;
        let (a2, _) = adapter(&s2, 10_000_000);
        assert_eq!(
            a2.call(&text("q")).await.unwrap().finish,
            FinishReason::ToolCalls
        );
    }

    #[tokio::test]
    async fn finish_reasons_map_and_blocks_are_content_filter_results() {
        for (reason, want) in [
            ("MAX_TOKENS", FinishReason::Length),
            ("STOP", FinishReason::Stop),
            ("SAFETY", FinishReason::ContentFilter),
            ("RECITATION", FinishReason::ContentFilter),
            ("PROHIBITED_CONTENT", FinishReason::ContentFilter),
            ("OTHER", FinishReason::Stop),
        ] {
            let s = MockServer::start().await;
            serve(&s, 200, ok(reason, "x")).await;
            let (a, _) = adapter(&s, 10_000_000);
            assert_eq!(a.call(&text("q")).await.unwrap().finish, want, "{reason}");
        }
        // the prompt itself was blocked: no candidate, only promptFeedback
        let s = MockServer::start().await;
        serve(&s, 200, json!({"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":5}})).await;
        let (a, _) = adapter(&s, 10_000_000);
        let r = a.call(&text("q")).await.unwrap();
        assert_eq!(
            (r.finish, out(&r)),
            (FinishReason::ContentFilter, String::new())
        );
    }

    #[tokio::test]
    async fn a_blocked_answer_never_leads_to_a_second_hop() {
        let s = MockServer::start().await;
        serve(&s, 200, ok("SAFETY", "")).await;
        let (a, _) = adapter(&s, 10_000_000);
        let fallback =
            ScriptedProvider::new("claude", Locality::Saas, &[RequestKind::TextGeneration]);
        let chain: Vec<Arc<dyn ProviderAdapter>> = vec![Arc::new(a), fallback.clone()];
        let policy = Policy {
            version: 1,
            rules: vec![Rule::allow("all", ProviderMatch::Any, &[])],
        };
        let label = Label::new(
            SourceKind::HouseholdFile,
            &["notes"],
            &[Subject::person("a", AgeBand::Adult)],
        );
        let (reply, _) = run_chain(
            &chain,
            &policy,
            &label,
            &text("q"),
            &ChainConfig::new(Purpose::Answer, 1),
            &VecSink::default(),
        )
        .await
        .unwrap();
        assert_eq!(reply.finish, FinishReason::ContentFilter);
        assert_eq!(fallback.calls(), 0);
    }

    #[tokio::test]
    async fn errors_map_to_classes_release_the_reservation_and_never_carry_the_body() {
        let cases: Vec<(u16, Option<&str>, Value, ProviderError)> = vec![
            (
                400,
                None,
                json!({"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}),
                ProviderError::Auth,
            ),
            (
                400,
                None,
                json!({"error":{"message":"The input token count (300000) exceeds the maximum number of tokens allowed","status":"INVALID_ARGUMENT"}}),
                ProviderError::ContextOverflow,
            ),
            (
                400,
                None,
                json!({"error":{"message":"something else"}}),
                ProviderError::Other,
            ),
            (
                403,
                None,
                json!({"error":{"status":"PERMISSION_DENIED","message":"SECRET-UPSTREAM-TEXT"}}),
                ProviderError::Auth,
            ),
            (
                402,
                None,
                json!({"error":{"message":"Prepay credit balance is depleted"}}),
                ProviderError::BudgetExceeded,
            ),
            (
                404,
                None,
                json!({"error":{"status":"NOT_FOUND","message":"models/x is not found"}}),
                ProviderError::ModelNotFound,
            ),
            (
                429,
                Some("7"),
                json!({"error":{"status":"RESOURCE_EXHAUSTED","message":"Resource has been exhausted (e.g. check quota). per minute"}}),
                ProviderError::RateLimited {
                    retry_after_s: Some(7),
                },
            ),
            (
                429,
                None,
                json!({"error":{"status":"RESOURCE_EXHAUSTED","message":"You exceeded your daily quota"}}),
                ProviderError::BudgetExceeded,
            ),
            (
                503,
                None,
                json!({"error":{"status":"UNAVAILABLE","message":"The model is overloaded"}}),
                ProviderError::Overloaded,
            ),
            (
                504,
                None,
                json!({"error":{"status":"DEADLINE_EXCEEDED"}}),
                ProviderError::Timeout,
            ),
            (
                500,
                None,
                json!({"error":{"status":"INTERNAL"}}),
                ProviderError::Unavailable,
            ),
        ];
        for (status, retry, body, want) in cases {
            let s = MockServer::start().await;
            let mut t = ResponseTemplate::new(status).set_body_json(body);
            if let Some(r) = retry {
                t = t.insert_header("retry-after", r);
            }
            Mock::given(method("POST")).respond_with(t).mount(&s).await;
            let (a, ledger) = adapter(&s, 10_000_000);
            assert_eq!(a.call(&text("x")).await.unwrap_err(), want, "HTTP {status}");
            assert_eq!(
                (ledger.available(), ledger.spent()),
                (10_000_000, 0),
                "HTTP {status}: an unbilled failure releases the reservation"
            );
        }
        let s = MockServer::start().await;
        serve(
            &s,
            403,
            json!({"error":{"status":"PERMISSION_DENIED","message":"SECRET-UPSTREAM-TEXT"}}),
        )
        .await;
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
            &text("x"),
            &ChainConfig::new(Purpose::Answer, 1),
            &VecSink::default(),
        )
        .await
        .unwrap_err();
        let shown = format!("{e} {e:?}");
        assert!(!shown.contains("SECRET") && !shown.contains(KEY), "{shown}");
    }

    #[tokio::test]
    async fn a_call_that_does_not_fit_the_budget_sends_nothing() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok("STOP", "x")))
            .expect(0)
            .mount(&s)
            .await;
        let (a, ledger) = adapter(&s, 10);
        assert_eq!(
            a.call(&text("hello")).await.unwrap_err(),
            ProviderError::BudgetExceeded
        );
        assert_eq!(ledger.available(), 10);
    }

    #[tokio::test]
    async fn embeddings_use_the_batch_endpoint_and_record_which_model_made_them() {
        let s = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/embed-test:batchEmbedContents"))
            .and(body_partial_json(json!({"requests":[{"model":"models/embed-test","content":{"parts":[{"text":"a"}]}},{"model":"models/embed-test","content":{"parts":[{"text":"b"}]}}]})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embeddings":[{"values":[3.0,4.0]},{"values":[0.0,2.0]}]})))
            .expect(1)
            .mount(&s)
            .await;
        let mut cfg = GeminiConfig::new("gemini-api-key", "gemini-test");
        cfg.base_url = s.uri();
        cfg.embedding_model = Some("embed-test".into());
        let a = GeminiAdapter::new(
            cfg,
            Arc::new(InMemoryCredentials::with("gemini-api-key", KEY)),
            Budget {
                ledger: Arc::new(Ledger::new(1)),
                pricing: PRICING,
            },
        )
        .unwrap();
        assert_eq!(
            a.info().kinds,
            vec![RequestKind::TextGeneration, RequestKind::Embedding]
        );
        let r = a
            .call(&Request::Embedding(EmbeddingParams {
                model_id: "x".into(),
                inputs: vec!["a".into(), "b".into()],
                normalize: true,
                session_id: None,
                request_id: None,
            }))
            .await
            .unwrap();
        assert_eq!(r.model_served, "embed-test");
        match r.output {
            Output::Embeddings(e) => {
                assert_eq!(
                    (e.vectors.len(), e.dimensions, e.model_version.as_str()),
                    (2, 2, "embed-test")
                );
                assert!(
                    (e.vectors[0][0] - 0.6).abs() < 1e-6 && (e.vectors[0][1] - 0.8).abs() < 1e-6
                );
            }
            o => panic!("{o:?}"),
        }
        // a short reply is an error, not a silently short result
        let s2 = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"embeddings":[{"values":[1.0]}]})),
            )
            .mount(&s2)
            .await;
        let mut cfg2 = GeminiConfig::new("gemini-api-key", "gemini-test");
        cfg2.base_url = s2.uri();
        cfg2.embedding_model = Some("embed-test".into());
        let a2 = GeminiAdapter::new(
            cfg2,
            Arc::new(InMemoryCredentials::with("gemini-api-key", KEY)),
            Budget {
                ledger: Arc::new(Ledger::new(1)),
                pricing: PRICING,
            },
        )
        .unwrap();
        let two = Request::Embedding(EmbeddingParams {
            model_id: "x".into(),
            inputs: vec!["a".into(), "b".into()],
            normalize: false,
            session_id: None,
            request_id: None,
        });
        assert_eq!(a2.call(&two).await.unwrap_err(), ProviderError::Other);
    }

    #[test]
    fn a_missing_credential_means_no_adapter_and_the_key_is_not_in_debug_output() {
        let cfg = GeminiConfig::new("gemini-api-key", "m");
        let b = || Budget {
            ledger: Arc::new(Ledger::new(1)),
            pricing: PRICING,
        };
        assert_eq!(
            GeminiAdapter::new(cfg.clone(), Arc::new(InMemoryCredentials::default()), b()).err(),
            Some(ConfigError::CredentialUnavailable)
        );
        let a = GeminiAdapter::new(
            cfg,
            Arc::new(InMemoryCredentials::with("gemini-api-key", KEY)),
            b(),
        )
        .unwrap();
        assert!(!format!("{a:?}").contains("TEST-ONLY"));
        assert_eq!(a.info().locality, Locality::Saas);
        assert_eq!(
            a.info().kinds,
            vec![RequestKind::TextGeneration],
            "no embedding model configured: no embeddings claimed"
        );
    }

    #[tokio::test]
    async fn model_discovery_maps_methods_to_kinds_and_unknown_means_false() {
        let s = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .and(header("x-goog-api-key", KEY))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models":[
                {"name":"models/gemini-a","inputTokenLimit":1048576,"outputTokenLimit":65536,"supportedGenerationMethods":["generateContent","countTokens"]},
                {"name":"models/embed-a","inputTokenLimit":2048,"supportedGenerationMethods":["embedContent"]},
                {"name":"models/odd"}]})))
            .mount(&s)
            .await;
        let (a, _) = adapter(&s, 1);
        let cards = a.discover_models("2026-10-11").await.unwrap();
        assert_eq!(cards.len(), 3);
        assert_eq!(
            (
                cards[0].model_id.as_str(),
                cards[0].kinds.clone(),
                cards[0].max_input_tokens
            ),
            (
                "gemini-a",
                vec![RequestKind::TextGeneration],
                Some(1_048_576)
            )
        );
        assert_eq!(cards[1].kinds, vec![RequestKind::Embedding]);
        assert!(
            cards[2].kinds.is_empty() && !cards[2].fits_input(1),
            "a model that lists nothing claims nothing"
        );
        assert!(
            cards
                .iter()
                .all(|c| !c.image_input.is_yes() && !c.tool_use.is_yes()),
            "capabilities the listing does not carry stay unknown"
        );
    }
}
