//! Every real adapter, run through the one provider contract in `crate::conformance`. Each fixture says what its vendor's mock upstream
//! returns, on the wire, for each scenario; the suite does the asserting. A new adapter earns its place by adding a fixture here.

#![allow(dead_code, unused_imports)]

use crate::budget::{Ledger, Pricing};
use crate::conformance::{assert_conforms, Fixture, Scenario, Setup, RETRY_AFTER_S, SECRET};
use crate::credentials::InMemoryCredentials;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const KEY: &str = "conformance-TEST-ONLY-credential";
const PRICING: Pricing = Pricing {
    input_micro_per_mtok: 1_000_000,
    output_micro_per_mtok: 4_000_000,
    cache_read_micro_per_mtok: 250_000,
    cache_write_micro_per_mtok: 0,
};

/// The response a mock should give: status, optional retry-after, JSON body.
struct Wire {
    status: u16,
    retry_after: Option<&'static str>,
    body: Value,
}

fn wire(status: u16, body: Value) -> Wire {
    Wire {
        status,
        retry_after: None,
        body,
    }
}

async fn mount(server: &MockServer, http_method: &str, p: &str, w: Wire, expect_zero: bool) {
    let mut t = ResponseTemplate::new(w.status).set_body_json(w.body);
    if let Some(r) = w.retry_after {
        t = t.insert_header("retry-after", r);
    }
    let mut m = Mock::given(method(http_method))
        .and(path(p))
        .respond_with(t);
    if expect_zero {
        m = m.expect(0);
    }
    m.mount(server).await;
}

#[cfg(feature = "provider-anthropic")]
mod anthropic {
    use super::*;
    use crate::providers::anthropic::{AnthropicAdapter, AnthropicConfig};

    pub struct F;
    fn ok(stop: &str, text: &str) -> Value {
        json!({"id":"m","type":"message","role":"assistant","model":"claude-test-2026","content":[{"type":"text","text":text}],"stop_reason":stop,
               "usage":{"input_tokens":10,"output_tokens":5}})
    }
    #[async_trait]
    impl Fixture for F {
        fn name(&self) -> &'static str {
            "anthropic"
        }
        async fn setup(&self, s: Scenario) -> Setup {
            let server = MockServer::start().await;
            let w = match s {
                Scenario::Ok => wire(200, ok("end_turn", "hello")),
                Scenario::Refusal => wire(200, ok("refusal", "no")),
                Scenario::RateLimited => Wire {
                    retry_after: Some("7"),
                    ..wire(
                        429,
                        json!({"error":{"type":"rate_limit_error","message":"slow down"}}),
                    )
                },
                Scenario::Auth => wire(
                    401,
                    json!({"error":{"type":"authentication_error","message":"invalid x-api-key"}}),
                ),
                Scenario::ServerError => {
                    wire(500, json!({"error":{"type":"api_error","message":"oops"}}))
                }
                Scenario::ContextOverflow => wire(
                    400,
                    json!({"error":{"type":"invalid_request_error","message":"prompt is too long: 300000 tokens"}}),
                ),
                Scenario::LeakyErrorBody => wire(
                    401,
                    json!({"error":{"type":"authentication_error","message":SECRET}}),
                ),
                Scenario::BudgetTooSmall => wire(200, ok("end_turn", "x")),
            };
            mount(
                &server,
                "POST",
                "/v1/messages",
                w,
                s == Scenario::BudgetTooSmall,
            )
            .await;
            let ledger = Arc::new(Ledger::new(if s == Scenario::BudgetTooSmall {
                10
            } else {
                100_000_000
            }));
            let mut cfg = AnthropicConfig::new("k", "claude-test", PRICING);
            cfg.base_url = server.uri();
            let a = AnthropicAdapter::new(
                cfg,
                Arc::new(InMemoryCredentials::with("k", KEY)),
                ledger.clone(),
            )
            .unwrap();
            Setup {
                adapter: Arc::new(a),
                ledger: Some(ledger),
                guard: Box::new(server),
                credential: Some(KEY),
            }
        }
    }
    #[tokio::test]
    async fn anthropic_conforms_to_the_provider_contract() {
        assert_conforms(&F).await;
    }
}

#[cfg(feature = "provider-gemini")]
mod gemini {
    use super::*;
    use crate::budget::Ledger;
    use crate::providers::gemini::{GeminiAdapter, GeminiConfig};
    use crate::providers::Budget;

    pub struct F;
    const P: &str = "/v1beta/models/gemini-test:generateContent";
    fn ok(reason: &str, text: &str) -> Value {
        json!({"candidates":[{"content":{"role":"model","parts":[{"text":text}]},"finishReason":reason}],
               "usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5},"modelVersion":"gemini-test-001"})
    }
    #[async_trait]
    impl Fixture for F {
        fn name(&self) -> &'static str {
            "gemini"
        }
        async fn setup(&self, s: Scenario) -> Setup {
            let server = MockServer::start().await;
            let w = match s {
                Scenario::Ok => wire(200, ok("STOP", "hello")),
                Scenario::Refusal => wire(200, ok("SAFETY", "")),
                Scenario::RateLimited => Wire {
                    retry_after: Some("7"),
                    ..wire(
                        429,
                        json!({"error":{"status":"RESOURCE_EXHAUSTED","message":"Too many requests per minute"}}),
                    )
                },
                Scenario::Auth => wire(
                    400,
                    json!({"error":{"status":"INVALID_ARGUMENT","message":"API key not valid. Please pass a valid API key."}}),
                ),
                Scenario::ServerError => {
                    wire(500, json!({"error":{"status":"INTERNAL","message":"oops"}}))
                }
                Scenario::ContextOverflow => wire(
                    400,
                    json!({"error":{"status":"INVALID_ARGUMENT","message":"The input token count (300000) exceeds the maximum"}}),
                ),
                Scenario::LeakyErrorBody => wire(
                    403,
                    json!({"error":{"status":"PERMISSION_DENIED","message":SECRET}}),
                ),
                Scenario::BudgetTooSmall => wire(200, ok("STOP", "x")),
            };
            mount(&server, "POST", P, w, s == Scenario::BudgetTooSmall).await;
            let ledger = Arc::new(Ledger::new(if s == Scenario::BudgetTooSmall {
                10
            } else {
                100_000_000
            }));
            let mut cfg = GeminiConfig::new("k", "gemini-test");
            cfg.base_url = server.uri();
            let a = GeminiAdapter::new(
                cfg,
                Arc::new(InMemoryCredentials::with("k", KEY)),
                Budget {
                    ledger: ledger.clone(),
                    pricing: PRICING,
                },
            )
            .unwrap();
            Setup {
                adapter: Arc::new(a),
                ledger: Some(ledger),
                guard: Box::new(server),
                credential: Some(KEY),
            }
        }
    }
    #[tokio::test]
    async fn gemini_conforms_to_the_provider_contract() {
        assert_conforms(&F).await;
    }
}

#[cfg(feature = "provider-openai-compat")]
mod openai_compat {
    use super::*;
    use crate::providers::openai_compat::{OpenAiCompatAdapter, OpenAiCompatConfig};

    pub struct F;
    fn ok(content: &str, finish: &str) -> Value {
        json!({"model":"m:served","choices":[{"message":{"role":"assistant","content":content},"finish_reason":finish}],"usage":{"prompt_tokens":10,"completion_tokens":5}})
    }
    #[async_trait]
    impl Fixture for F {
        fn name(&self) -> &'static str {
            "openai-compatible (local, no budget)"
        }
        /// A local server has no budget to exceed.
        fn supports(&self, s: Scenario) -> bool {
            s != Scenario::BudgetTooSmall
        }
        async fn setup(&self, s: Scenario) -> Setup {
            let server = MockServer::start().await;
            let w = match s {
                Scenario::Ok => wire(200, ok("hello", "stop")),
                Scenario::Refusal => wire(
                    200,
                    json!({"model":"m","choices":[{"message":{"role":"assistant","content":null,"refusal":"I can't do that"},"finish_reason":"stop"}],"usage":{}}),
                ),
                Scenario::RateLimited => Wire {
                    retry_after: Some("7"),
                    ..wire(429, json!({"error":{"message":"slow"}}))
                },
                Scenario::Auth => wire(401, json!({"error":{"message":"bad key"}})),
                Scenario::ServerError => wire(500, json!({"error":"boom"})),
                Scenario::ContextOverflow => wire(
                    400,
                    json!({"error":{"message":"request exceeds the maximum context length"}}),
                ),
                Scenario::LeakyErrorBody => wire(401, json!({"error":{"message":SECRET}})),
                Scenario::BudgetTooSmall => wire(200, ok("x", "stop")),
            };
            mount(&server, "POST", "/v1/chat/completions", w, false).await;
            let mut cfg = OpenAiCompatConfig::ollama("m");
            cfg.base_url = server.uri();
            let a = OpenAiCompatAdapter::new(cfg, None, None).unwrap();
            Setup {
                adapter: Arc::new(a),
                ledger: None,
                guard: Box::new(server),
                credential: None,
            }
        }
    }
    #[tokio::test]
    async fn openai_compatible_conforms_to_the_provider_contract() {
        assert_conforms(&F).await;
    }
}
