//! Minimal OpenRouter chat client, logprob-first.
//!
//! One call, one capture: [`Gateway::chat`] posts a [`ChatSpec`], captures
//! the exact response bytes into a [`ProviderCapture`](crate::capture::ProviderCapture),
//! and hands back the parsed pieces an instrument needs (content, the FULL
//! array of answer-position top-logprobs, usage). It does not classify
//! refusals, does not pick a "best" logprob position, and does not decide
//! what counts as an answer — that judgement belongs to
//! [`crate::instrument`]. The gateway's only job is faithful transport with
//! provenance.
//!
//! Provider payload shape variance (which field carries the token budget,
//! whether logprobs are requested, whether a JSON response format is
//! demanded) is a property of [`PayloadShape`], not of call sites: every
//! [`ChatSpec`] renders to a payload the same way regardless of which model
//! or provider quirk is in play, via [`chat_payload`]. Redesigned from the
//! diamond2 `cardinal-harness-v2` quarry's `ChatCompletionRequestConfig` /
//! `chat_completion_payload` and
//! `provider_top_logprobs_from_chat_completion_response` (position
//! disambiguation, left to instruments here) and `visible_top_logprob_mass`
//! clamp patterns.

use crate::capture::ProviderCapture;
use crate::ontology::ContentId;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

/// Default OpenRouter API base (chat-completions endpoint lives under it).
const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// The one request path this gateway speaks.
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
/// Domain tag for the request-fingerprint content id (see [`ContentId::derive`]).
const REQUEST_FINGERPRINT_DOMAIN: &str = "seriate/gateway-request";
/// Fixed backoff before the single retry on a 5xx or transport timeout.
const RETRY_BACKOFF: Duration = Duration::from_millis(300);
/// Overall per-request timeout.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Providers cap top-logprobs; OpenRouter/OpenAI-compatible APIs at 20.
const MAX_TOP_LOGPROBS: u8 = 20;

/// A single chat request, provider-agnostic.
///
/// Rendered to a provider payload by [`chat_payload`] under a
/// [`PayloadShape`]; none of the provider-specific field naming lives here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatSpec {
    /// Provider model identifier, e.g. `"openai/gpt-4.1-mini"`.
    pub model: String,
    /// System prompt.
    pub system: String,
    /// User prompt.
    pub user: String,
    /// Sampling temperature.
    pub temperature: f64,
    /// Completion token budget.
    pub max_tokens: u32,
    /// When `Some(k)`, request top-`k` logprobs at every completion-token
    /// position (clamped to `MAX_TOP_LOGPROBS`). `None` requests no logprobs.
    pub top_logprobs: Option<u8>,
    /// Whether to demand a JSON-object response format.
    pub response_format_json: bool,
}

/// Which field name carries the completion-token budget in the provider
/// payload. OpenAI's newer models want `max_completion_tokens`; OpenRouter's
/// OpenAI-compatible surface (and most providers behind it) still take
/// `max_tokens`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxTokensField {
    /// `"max_tokens"` — the OpenRouter / classic OpenAI-compatible shape.
    MaxTokens,
    /// `"max_completion_tokens"` — newer OpenAI-native shape.
    MaxCompletionTokens,
}

/// Provider payload shape configuration, isolated from call sites so a new
/// provider quirk is one struct edit, not a hunt through every [`ChatSpec`]
/// construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadShape {
    /// Which field name carries the completion-token budget.
    pub max_tokens_field: MaxTokensField,
}

impl Default for PayloadShape {
    /// OpenRouter's shape: `max_tokens`.
    fn default() -> Self {
        Self {
            max_tokens_field: MaxTokensField::MaxTokens,
        }
    }
}

/// Render a [`ChatSpec`] into the JSON body a chat-completions-shaped
/// provider expects, under the given [`PayloadShape`]. Pure and
/// network-free so payload shape is testable without a server.
pub fn chat_payload(spec: &ChatSpec, shape: PayloadShape) -> Value {
    let mut payload = json!({
        "model": spec.model,
        "messages": [
            {"role": "system", "content": spec.system},
            {"role": "user", "content": spec.user},
        ],
        "temperature": spec.temperature,
    });
    let max_tokens_key = match shape.max_tokens_field {
        MaxTokensField::MaxTokens => "max_tokens",
        MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
    };
    payload[max_tokens_key] = json!(spec.max_tokens);
    if let Some(top_k) = spec.top_logprobs {
        payload["logprobs"] = json!(true);
        payload["top_logprobs"] = json!(top_k.min(MAX_TOP_LOGPROBS));
    }
    if spec.response_format_json {
        payload["response_format"] = json!({"type": "json_object"});
    }
    payload
}

/// One completion-token position's logprob, plus its top-k alternatives.
///
/// The gateway keeps the WHOLE array of positions from
/// `choices[0].logprobs.content[..]` — deciding which position is "the
/// answer" is a parsing concern for `crate::instrument`, not a transport
/// concern here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// The token the provider actually emitted at this position.
    pub token: String,
    /// Its logprob.
    pub logprob: f64,
    /// Up to top-k `(token, logprob)` alternatives the provider showed for
    /// this position, in provider order.
    pub top: Vec<(String, f64)>,
}

/// Token and cost accounting for one chat call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt tokens billed.
    pub input_tokens: u32,
    /// Completion tokens billed.
    pub output_tokens: u32,
    /// Cost in nanodollars (10^-9 USD): integer, no float drift downstream.
    pub cost_nanodollars: i64,
    /// True when the provider did not report a cost and `cost_nanodollars`
    /// is a zero placeholder rather than a real figure.
    pub cost_is_estimate: bool,
}

/// Everything one [`Gateway::chat`] call produces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatOutcome {
    /// The raw-bytes provenance anchor for this call.
    pub capture: ProviderCapture,
    /// `choices[0].message.content` (or `.refusal` as a passthrough
    /// fallback), verbatim. Refusals are not classified here.
    pub content: String,
    /// All answer-position top-logprobs, if the provider returned any.
    /// `None` means no logprob content was present — not an error.
    pub answer_logprobs: Option<Vec<TokenLogprob>>,
    /// Token and cost accounting for the call.
    pub usage: Usage,
}

/// Everything that can go wrong making a chat call.
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    /// `OPENROUTER_API_KEY` was not set (only from [`Gateway::from_env`]).
    #[error("OPENROUTER_API_KEY is not set")]
    MissingApiKey,
    /// The `ChatSpec` was not sendable (non-finite temperature, zero budget).
    #[error("invalid chat spec: {0}")]
    InvalidSpec(String),
    /// Transport failure (connect, timeout, TLS, ...), after the retry.
    #[error("http transport error: {0}")]
    Http(#[source] reqwest::Error),
    /// Provider answered with a non-2xx status, after the retry.
    #[error("provider returned http {status}: {body}")]
    Provider {
        /// HTTP status code.
        status: u16,
        /// Response body (for diagnostics; not re-parsed).
        body: String,
    },
    /// The response body was not valid JSON.
    #[error("failed to decode provider response body as json: {0}")]
    Decode(#[source] serde_json::Error),
    /// The response was valid JSON but missing an expected shape (e.g. no
    /// `choices[0].message`).
    #[error("malformed provider response: {0}")]
    MalformedResponse(String),
}

/// A minimal OpenRouter-shaped chat client. Cheap to clone-construct; holds
/// one [`reqwest::Client`] (connection pool) per instance.
#[derive(Clone, Debug)]
pub struct Gateway {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    payload_shape: PayloadShape,
}

impl Gateway {
    /// Build a gateway against an explicit base URL (bypassing env vars —
    /// this is the constructor tests should use, so tests don't race on
    /// process-wide environment state).
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .expect("reqwest client builds with a fixed timeout and no exotic config"),
            api_key: api_key.into(),
            base_url: base_url.into(),
            payload_shape: PayloadShape::default(),
        }
    }

    /// Build a gateway from `OPENROUTER_API_KEY` (required) and
    /// `OPENROUTER_BASE_URL` (optional override, e.g. to point at a mock
    /// server in an integration test harness).
    pub fn from_env() -> Result<Self, GatewayError> {
        let api_key =
            std::env::var("OPENROUTER_API_KEY").map_err(|_| GatewayError::MissingApiKey)?;
        let base_url =
            std::env::var("OPENROUTER_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_owned());
        Ok(Self::new(api_key, base_url))
    }

    /// Override the provider payload shape (default: OpenRouter's).
    pub fn with_payload_shape(mut self, shape: PayloadShape) -> Self {
        self.payload_shape = shape;
        self
    }

    /// Send one chat request. Retries exactly once, after a fixed backoff,
    /// on a 5xx response or a transport timeout/connect failure; any other
    /// outcome (including a second failure) is returned as-is. Refusals are
    /// passed through as ordinary content, not classified as errors.
    pub async fn chat(&self, spec: &ChatSpec) -> Result<ChatOutcome, GatewayError> {
        if !spec.temperature.is_finite() {
            return Err(GatewayError::InvalidSpec(
                "temperature must be finite".into(),
            ));
        }
        if spec.max_tokens == 0 {
            return Err(GatewayError::InvalidSpec(
                "max_tokens must be greater than zero".into(),
            ));
        }

        let payload = chat_payload(spec, self.payload_shape);
        let payload_bytes = serde_json::to_vec(&payload).expect("chat payload always serializes");
        let request_fingerprint = ContentId::derive(REQUEST_FINGERPRINT_DOMAIN, &payload_bytes);
        let url = format!(
            "{}{CHAT_COMPLETIONS_PATH}",
            self.base_url.trim_end_matches('/')
        );

        let (status, body) = self.send_with_one_retry(&url, &payload).await?;

        if !status.is_success() {
            return Err(GatewayError::Provider {
                status: status.as_u16(),
                body,
            });
        }

        let capture = ProviderCapture::new(
            body.clone(),
            request_fingerprint,
            spec.model.clone(),
            CHAT_COMPLETIONS_PATH,
            now_ms(),
        );

        let parsed: Value = serde_json::from_str(&body).map_err(GatewayError::Decode)?;
        let message = extract_message(&parsed)?;
        let content = extract_content(message);
        let answer_logprobs = extract_logprobs(&parsed);
        let usage = extract_usage(&parsed);

        Ok(ChatOutcome {
            capture,
            content,
            answer_logprobs,
            usage,
        })
    }

    /// POST once; on a 5xx status or a retryable transport error, sleep
    /// [`RETRY_BACKOFF`] and try exactly one more time.
    async fn send_with_one_retry(
        &self,
        url: &str,
        payload: &Value,
    ) -> Result<(reqwest::StatusCode, String), GatewayError> {
        let mut retried = false;
        loop {
            match self.post_once(url, payload).await {
                Ok((status, body)) => {
                    if status.is_server_error() && !retried {
                        retried = true;
                        tokio::time::sleep(RETRY_BACKOFF).await;
                        continue;
                    }
                    return Ok((status, body));
                }
                Err(err) => {
                    if !retried && is_retryable(&err) {
                        retried = true;
                        tokio::time::sleep(RETRY_BACKOFF).await;
                        continue;
                    }
                    return Err(GatewayError::Http(err));
                }
            }
        }
    }

    async fn post_once(
        &self,
        url: &str,
        payload: &Value,
    ) -> Result<(reqwest::StatusCode, String), reqwest::Error> {
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(payload)
            .send()
            .await?;
        let status = response.status();
        let body = response.text().await?;
        Ok((status, body))
    }
}

/// Retryable transport failures: connect and timeout, not e.g. a body
/// decode error, which retrying can't fix.
fn is_retryable(err: &reqwest::Error) -> bool {
    err.is_timeout() || err.is_connect()
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn extract_message(response: &Value) -> Result<&Value, GatewayError> {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .ok_or_else(|| GatewayError::MalformedResponse("missing choices[0].message".into()))
}

/// Content, verbatim; falls back to a structured `refusal` field so a
/// refusal-shaped response still yields text instead of an error. This is
/// passthrough, not classification: callers decide what a refusal means.
fn extract_content(message: &Value) -> String {
    message
        .get("content")
        .and_then(Value::as_str)
        .or_else(|| message.get("refusal").and_then(Value::as_str))
        .unwrap_or_default()
        .to_owned()
}

/// `choices[0].logprobs.content[..]`, kept whole. `None` when the provider
/// did not send logprobs at all (not requested, or unsupported by the
/// model) — this is expected, not an error.
fn extract_logprobs(response: &Value) -> Option<Vec<TokenLogprob>> {
    let positions = response
        .get("choices")?
        .as_array()?
        .first()?
        .get("logprobs")?
        .get("content")?
        .as_array()?;
    let out = positions
        .iter()
        .filter_map(|position| {
            let token = position.get("token")?.as_str()?.to_owned();
            let logprob = position.get("logprob")?.as_f64()?;
            let top = position
                .get("top_logprobs")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| {
                            let token = item.get("token")?.as_str()?.to_owned();
                            let logprob = item.get("logprob")?.as_f64()?;
                            Some((token, logprob))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Some(TokenLogprob {
                token,
                logprob,
                top,
            })
        })
        .collect();
    Some(out)
}

/// Usage/cost mapping. Cost prefers `usage.cost` (OpenRouter's own figure),
/// falling back to `usage.cost_details.upstream_inference_cost`; absent
/// either, cost is a zero estimate rather than a fabricated number
/// (salvaged fallback chain from the diamond2 `cost_ledger_from_chat_completion_response`).
fn extract_usage(response: &Value) -> Usage {
    let usage = response.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let cost_usd = usage
        .and_then(|u| u.get("cost"))
        .or_else(|| {
            usage
                .and_then(|u| u.get("cost_details"))
                .and_then(|d| d.get("upstream_inference_cost"))
        })
        .and_then(Value::as_f64);
    match cost_usd {
        Some(usd) => Usage {
            input_tokens,
            output_tokens,
            cost_nanodollars: (usd * 1e9).round() as i64,
            cost_is_estimate: false,
        },
        None => Usage {
            input_tokens,
            output_tokens,
            cost_nanodollars: 0,
            cost_is_estimate: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::CaptureId;
    use std::sync::Mutex;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn spec() -> ChatSpec {
        ChatSpec {
            model: "openai/gpt-4.1-mini".into(),
            system: "system prompt".into(),
            user: "user prompt".into(),
            temperature: 0.0,
            max_tokens: 32,
            top_logprobs: Some(5),
            response_format_json: false,
        }
    }

    fn success_json(content: &str) -> String {
        json!({
            "choices": [{
                "message": {"content": content},
                "logprobs": {
                    "content": [{
                        "token": "B",
                        "logprob": (-0.05f64),
                        "top_logprobs": [
                            {"token": "B", "logprob": -0.05},
                            {"token": "C", "logprob": -3.1},
                        ],
                    }],
                },
            }],
            "usage": {"prompt_tokens": 120, "completion_tokens": 3, "cost": 0.000_42},
        })
        .to_string()
    }

    #[test]
    fn payload_requests_logprobs_and_uses_openrouter_max_tokens_field() {
        let payload = chat_payload(&spec(), PayloadShape::default());
        assert_eq!(payload["logprobs"], json!(true));
        assert_eq!(payload["top_logprobs"], json!(5));
        assert_eq!(payload["max_tokens"], json!(32));
        assert!(payload.get("max_completion_tokens").is_none());
        assert!(payload.get("response_format").is_none());
    }

    #[test]
    fn payload_shape_can_switch_to_max_completion_tokens() {
        let shape = PayloadShape {
            max_tokens_field: MaxTokensField::MaxCompletionTokens,
        };
        let payload = chat_payload(&spec(), shape);
        assert_eq!(payload["max_completion_tokens"], json!(32));
        assert!(payload.get("max_tokens").is_none());
    }

    #[test]
    fn payload_json_response_format_is_call_site_controlled() {
        let mut s = spec();
        s.response_format_json = true;
        s.top_logprobs = None;
        let payload = chat_payload(&s, PayloadShape::default());
        assert_eq!(payload["response_format"]["type"], json!("json_object"));
        assert!(payload.get("logprobs").is_none());
        assert!(payload.get("top_logprobs").is_none());
    }

    #[test]
    fn top_logprobs_is_clamped_to_provider_ceiling() {
        let mut s = spec();
        s.top_logprobs = Some(200);
        let payload = chat_payload(&s, PayloadShape::default());
        assert_eq!(payload["top_logprobs"], json!(20));
    }

    #[tokio::test]
    async fn capture_id_binds_to_exact_response_bytes() {
        let server = MockServer::start().await;
        let raw = success_json("hello");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(raw.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let outcome = gw.chat(&spec()).await.expect("mocked call succeeds");

        assert_eq!(outcome.capture.raw, raw);
        assert!(outcome.capture.verify(), "id binds to the whole event");
        assert_eq!(outcome.capture.model, "openai/gpt-4.1-mini");
        assert_eq!(outcome.capture.url_path, "/chat/completions");
        assert_eq!(outcome.content, "hello");
    }

    #[tokio::test]
    async fn usage_maps_tokens_and_cost_from_provider_figure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(success_json("ok")))
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let outcome = gw.chat(&spec()).await.expect("mocked call succeeds");

        assert_eq!(outcome.usage.input_tokens, 120);
        assert_eq!(outcome.usage.output_tokens, 3);
        assert_eq!(outcome.usage.cost_nanodollars, 420_000);
        assert!(!outcome.usage.cost_is_estimate);
    }

    #[tokio::test]
    async fn usage_is_marked_an_estimate_when_provider_omits_cost() {
        let server = MockServer::start().await;
        let body = json!({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })
        .to_string();
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let outcome = gw.chat(&spec()).await.expect("mocked call succeeds");

        assert_eq!(outcome.usage.cost_nanodollars, 0);
        assert!(outcome.usage.cost_is_estimate);
    }

    #[tokio::test]
    async fn retries_once_on_server_error_then_succeeds() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("upstream hiccup"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(success_json("recovered")))
            .expect(1)
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let outcome = gw
            .chat(&spec())
            .await
            .expect("second attempt after retry succeeds");

        assert_eq!(outcome.content, "recovered");
    }

    #[tokio::test]
    async fn a_second_server_error_is_returned_not_retried_again() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("still down"))
            .expect(2)
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let err = gw
            .chat(&spec())
            .await
            .expect_err("stays failed after one retry");
        match err {
            GatewayError::Provider { status, .. } => assert_eq!(status, 500),
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn client_error_is_not_retried() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .expect(1)
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let err = gw.chat(&spec()).await.expect_err("4xx is not retried");
        match err {
            GatewayError::Provider { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_logprobs_response_yields_none_not_error() {
        let server = MockServer::start().await;
        let body = json!({
            "choices": [{"message": {"content": "no logprobs here"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        })
        .to_string();
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let outcome = gw
            .chat(&spec())
            .await
            .expect("missing logprobs is not an error");

        assert_eq!(outcome.content, "no logprobs here");
        assert!(outcome.answer_logprobs.is_none());
    }

    #[tokio::test]
    async fn refusal_field_passes_through_when_content_is_absent() {
        let server = MockServer::start().await;
        let body = json!({
            "choices": [{"message": {"refusal": "I can't help with that."}}],
        })
        .to_string();
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let outcome = gw
            .chat(&spec())
            .await
            .expect("refusal is passthrough, not an error");

        assert_eq!(outcome.content, "I can't help with that.");
    }

    #[tokio::test]
    async fn malformed_response_is_a_distinct_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"choices":[]}"#))
            .mount(&server)
            .await;

        let gw = Gateway::new("test-key", server.uri());
        let err = gw
            .chat(&spec())
            .await
            .expect_err("empty choices is malformed");
        assert!(matches!(err, GatewayError::MalformedResponse(_)));
    }

    #[test]
    fn invalid_spec_is_rejected_before_any_network_call() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let gw = Gateway::new("test-key", "http://127.0.0.1:1");
        let mut s = spec();
        s.max_tokens = 0;
        let err = rt
            .block_on(gw.chat(&s))
            .expect_err("zero budget is invalid");
        assert!(matches!(err, GatewayError::InvalidSpec(_)));

        let mut s = spec();
        s.temperature = f64::NAN;
        let err = rt
            .block_on(gw.chat(&s))
            .expect_err("non-finite temperature is invalid");
        assert!(matches!(err, GatewayError::InvalidSpec(_)));
    }

    // `from_env` mutates process-wide environment variables, so this test is
    // serialized against any sibling test doing the same (there are none
    // today, but the lock keeps it that way safely if one is ever added).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn from_env_reads_key_and_base_url_override() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("OPENROUTER_API_KEY", "env-key");
        std::env::set_var("OPENROUTER_BASE_URL", "http://example.invalid");
        let gw = Gateway::from_env().expect("both vars set");
        assert_eq!(gw.api_key, "env-key");
        assert_eq!(gw.base_url, "http://example.invalid");
        std::env::remove_var("OPENROUTER_BASE_URL");
        let gw = Gateway::from_env().expect("base url falls back to default");
        assert_eq!(gw.base_url, DEFAULT_BASE_URL);
        std::env::remove_var("OPENROUTER_API_KEY");
        assert!(matches!(
            Gateway::from_env(),
            Err(GatewayError::MissingApiKey)
        ));
    }
}
