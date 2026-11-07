mod decision;
mod vision;
use crate::{
    output::{anthropic_blocks, Part},
    supervisor::{ApiError, Begin, Event as ModelEvent, Request, Service},
};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures_util::Stream;
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::mpsc;

#[derive(Clone, Copy, PartialEq)]
enum Api {
    Anthropic,
    OpenAi,
}

/// Output tokens used when a request omits an explicit limit.
pub(crate) const DEFAULT_MAX_TOKENS: u64 = 512;

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(json!({"type":"error","error":{"type":public_error_type(self.status),"code":self.code,"message":self.message}}))).into_response()
    }
}

fn public_error_type(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ => "api_error",
    }
}

/// Serve the declared routes with an already loaded service.
///
/// The load gate owns the only route table, so the library and CLI surfaces
/// cannot declare different routes.
pub fn router(service: Arc<Service>) -> Router {
    router_with_gate(Gate::published(service))
}
/// The HTTP surface before model loading publishes a ready service.
pub struct Gate {
    service: std::sync::RwLock<Option<Arc<Service>>>,
}

impl Gate {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            service: std::sync::RwLock::new(None),
        })
    }

    /// A gate that already publishes a loaded service.
    pub fn published(service: Arc<Service>) -> Arc<Self> {
        let gate = Self::new();
        gate.publish(service);
        gate
    }

    /// Publish the loaded service. Requests before this point see the loading
    /// surface; requests after it use the declared routes.
    pub fn publish(&self, service: Arc<Service>) {
        *self
            .service
            .write()
            .expect("The load gate lock is poisoned.") = Some(service);
    }

    fn service(&self) -> Option<Arc<Service>> {
        self.service
            .read()
            .expect("The load gate lock is poisoned.")
            .clone()
    }
}

fn loading_error() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"type":"error","error":{"type":"overloaded_error","code":"model_unavailable","message":"The model is still loading."}})),
    )
        .into_response()
}

fn loading_readiness() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({"ready":false,"phase":"loading"})),
    )
        .into_response()
}

/// Serve the declared routes while the worker is still loading.
///
/// Health reports that the process is alive, readiness reports the loading
/// phase, and every other route answers HTTP 503 `model_unavailable` until
/// `Gate::publish` installs the ready service.
pub fn router_with_gate(gate: Arc<Gate>) -> Router {
    Router::new()
        .route(
            "/healthz",
            get(|| async { Json(json!({"status":"alive"})) }),
        )
        .route("/readyz", get(gated_ready))
        .route("/v1/models", get(gated_models))
        .route("/v1/messages", post(gated_messages))
        .route("/v1/messages/count_tokens", post(gated_count_tokens))
        .route("/v1/chat/completions", post(gated_chat))
        .route("/v1/systemone", post(gated_systemone))
        .route("/metrics", get(gated_metrics))
        .layer(DefaultBodyLimit::max(1_000_000))
        .with_state(gate)
}

async fn gated_ready(State(gate): State<Arc<Gate>>) -> Response {
    match gate.service() {
        Some(service) => ready(State(service)).await,
        None => loading_readiness(),
    }
}

async fn gated_models(State(gate): State<Arc<Gate>>) -> Response {
    match gate.service() {
        Some(service) => models(State(service)).await.into_response(),
        None => loading_error(),
    }
}

async fn gated_messages(
    State(gate): State<Arc<Gate>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match gate.service() {
        Some(service) => messages(State(service), headers, body).await,
        None => loading_error(),
    }
}

async fn gated_chat(State(gate): State<Arc<Gate>>, headers: HeaderMap, body: Bytes) -> Response {
    match gate.service() {
        Some(service) => chat(State(service), headers, body).await,
        None => loading_error(),
    }
}

async fn gated_count_tokens(
    State(gate): State<Arc<Gate>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    match gate.service() {
        Some(service) => count_tokens(State(service), headers, body).await,
        None => loading_error(),
    }
}

async fn gated_systemone(State(gate): State<Arc<Gate>>, body: Bytes) -> Response {
    match gate.service() {
        Some(service) => decision::systemone(State(service), body).await,
        None => loading_error(),
    }
}

async fn gated_metrics(State(gate): State<Arc<Gate>>) -> Response {
    match gate.service() {
        Some(service) => metrics(State(service)).await.into_response(),
        None => loading_error(),
    }
}

async fn ready(State(service): State<Arc<Service>>) -> Response {
    let (admission_allowed, host_pressure) = service.admission_snapshot();
    let worker_available = service.available.load(Ordering::Acquire);
    let status = if worker_available && admission_allowed {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let sampled = service.sampling_enabled();
    let tools = match service.output_profile() {
        crate::output::OutputProfile::MiniCpm5 => "minicpm5_xml",
        crate::output::OutputProfile::Qwen => "qwen_xml_or_json",
    };
    let mut body = json!({"ready":status==StatusCode::OK,"worker_available":worker_available,"host_pressure":host_pressure,"model":service.model_id,"worker":service.ready_snapshot(),"profile":public_profile(&service),"public_capabilities":{"anthropic_messages":true,"openai_chat_completions":true,"tools":tools,"tool_choice":["auto","none"],"thinking":false,"sampling":sampled,"images":false,"sessions":false,"batching":false,"prompt_cache":false,"output_effort":"advisory_only"}});
    if service.image_enabled() {
        body["public_capabilities"]["images"] = json!(true);
        body["public_capabilities"]["image_input_restriction"] =
            json!("anthropic_base64_png_user_and_tool_result");
        body["public_capabilities"]["openai_image_input_restriction"] =
            json!("openai_base64_png_user_only");
        body["public_capabilities"]["media_limits"] =
            service.ready["capabilities"]["media_limits"].clone();
    }
    if service.decision_enabled() {
        body["public_capabilities"] = json!({"workload":"decision","systemone":true,"text":true,"noul":true,"choice":true,"score":true,"images":false,"streaming":false,"token_events":false,"kv_reuse":false,"max_context":service.ready["capabilities"]["max_context"],"max_output_tokens":0,"max_active_requests":1,"cancel_granularity":"process_exit","sessions":false,"batching":false,"prompt_cache":false,"anthropic_messages":false,"openai_chat_completions":false,"count_tokens":false,"state_schema":"none"});
    }
    if service.model_thinking() {
        body["public_capabilities"]["model_thinking"] = json!("initial_hidden");
    }
    if service.bounded_thinking() {
        body["public_capabilities"]["initial_reasoning_budget_tokens"] = json!(512);
    }
    (status, Json(body)).into_response()
}

fn public_profile(service: &Service) -> &'static str {
    if service.decision_enabled() {
        "apxinf-clef-text-v0.1"
    } else if service.ready["protocol"] == crate::contracts::NATIVE_PREPARED_PROTOCOL {
        "apxinf-native-prepared-v0.1"
    } else if service.image_enabled() {
        "apxinf-minicpm-vision-v0.1"
    } else if service.bounded_thinking() {
        "apxinf-native-bounded-thinking-v0.1"
    } else if service.model_thinking() {
        "apxinf-native-thinking-v0.1"
    } else if service.sampling_enabled() {
        "apxinf-native-sampled-v0.1"
    } else {
        "apxinf-serial-v0.1"
    }
}
async fn models(State(service): State<Arc<Service>>) -> Json<Value> {
    Json(
        json!({"object":"list","data":[{"id":service.model_id,"object":"model","created":0,"owned_by":"apxinf","model_revision":service.ready["model_revision"]}],"has_more":false,"first_id":service.model_id,"last_id":service.model_id}),
    )
}
async fn metrics(State(service): State<Arc<Service>>) -> impl IntoResponse {
    (
        [("content-type", "text/plain; version=0.0.4")],
        service.metrics_text(),
    )
}
async fn messages(
    State(service): State<Arc<Service>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle(service, headers, body, Api::Anthropic, false)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}
async fn chat(State(service): State<Arc<Service>>, headers: HeaderMap, body: Bytes) -> Response {
    handle(service, headers, body, Api::OpenAi, false)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}
async fn count_tokens(
    State(service): State<Arc<Service>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle(service, headers, body, Api::Anthropic, true)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

struct CancelGuard(Arc<AtomicBool>);
impl Drop for CancelGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

async fn handle(
    service: Arc<Service>,
    _headers: HeaderMap,
    body: Bytes,
    api: Api,
    count_only: bool,
) -> Result<Response, ApiError> {
    if service.decision_enabled() {
        return Err(ApiError::new(
            400,
            "unsupported_feature",
            "The selected Clef provider supports only /v1/systemone.",
        ));
    }
    let value = crate::contracts::parse_document(&body)
        .map_err(|e| ApiError::invalid(format!("The request is not valid JSON: {e}")))?;
    let (request, stream) = normalize(&value, api, count_only, &service)?;
    let preparation = json!({"protocol":service.ready["protocol"],"kind":"prepare_input","worker_epoch":service.ready["worker_epoch"],"command_id":uuid::Uuid::new_v4().to_string(),"model_revision":service.ready["model_revision"],"messages":request.messages,"tools":request.tools,"template_options":{"enable_thinking":service.model_thinking()}});
    crate::contracts::encode_frame(&preparation)
        .map_err(|e| ApiError::invalid(format!("The normalized input is invalid: {e}")))?;
    let ticket = service.enqueue(request)?;
    let guard = CancelGuard(ticket.cancel);
    let begin = ticket.begin.await.map_err(|_| {
        ApiError::new(
            503,
            "worker_lost",
            "The worker stopped before request admission.",
        )
    })??;
    if count_only {
        return Ok(Json(json!({"input_tokens":begin.input_tokens})).into_response());
    }
    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", begin.request_id.parse().unwrap());
    headers.insert(
        "x-apxinf-model-revision",
        service.ready["model_revision"]
            .as_str()
            .unwrap_or("")
            .parse()
            .unwrap(),
    );
    headers.insert(
        "x-apxinf-profile",
        public_profile(&service).parse().unwrap(),
    );
    if stream {
        let include_usage = value["stream_options"]["include_usage"] == true;
        let stream = PublicStream::new(
            ticket.output,
            guard,
            api,
            begin,
            service.model_id.clone(),
            include_usage,
        );
        Ok((
            headers,
            Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(5))),
        )
            .into_response())
    } else {
        let mut receiver = ticket.output;
        let mut parts = Vec::new();
        while let Some(event) = receiver.recv().await {
            match event {
                ModelEvent::Decision(_) => {
                    return Err(ApiError::new(
                        500,
                        "protocol_fault",
                        "A text request received a decision result.",
                    ))
                }
                ModelEvent::Part(part) => parts.push(part),
                ModelEvent::Error(error) => return Err(error),
                ModelEvent::Done {
                    cause,
                    input_tokens,
                    output_tokens,
                    matched_stop,
                } => {
                    return Ok((
                        headers,
                        Json(response_body(
                            api,
                            &begin,
                            &service.model_id,
                            &parts,
                            &cause,
                            input_tokens,
                            output_tokens,
                            matched_stop,
                        )),
                    )
                        .into_response());
                }
            }
        }
        drop(guard);
        Err(ApiError::new(
            503,
            "worker_lost",
            "The output stream ended without a terminal result.",
        ))
    }
}

fn unknown(value: &Value, allowed: &[&str], label: &str) -> Result<(), ApiError> {
    let object = value
        .as_object()
        .ok_or_else(|| ApiError::invalid(format!("{label} must be an object.")))?;
    for field in object.keys() {
        if !allowed.contains(&field.as_str()) {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                format!("{label} field {field} is unsupported."),
            ));
        }
    }
    Ok(())
}
fn text(value: &Value, label: &str) -> Result<String, ApiError> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| ApiError::invalid(format!("{label} must be a string.")))
}
fn array<'a>(value: &'a Value, label: &str) -> Result<&'a Vec<Value>, ApiError> {
    value
        .as_array()
        .ok_or_else(|| ApiError::invalid(format!("{label} must be an array.")))
}

/// Accept the OpenAI fields whose default value leaves generation unchanged.
///
/// Local clients often send these fields explicitly. A value that would need
/// behavior this profile does not implement remains an `unsupported_feature`
/// error instead of being silently ignored.
fn openai_compatibility(value: &Value, service: &Service) -> Result<(), ApiError> {
    let unsupported = |message: &str| ApiError::new(400, "unsupported_feature", message);
    if let Some(seed) = value.get("seed") {
        // The request parser rejects integers above the safe range, so this
        // bound only protects callers that normalize a JSON value directly.
        if !seed
            .as_u64()
            .is_some_and(|seed| seed <= crate::contracts::MAX_SAFE_INTEGER)
        {
            return Err(ApiError::invalid(
                "seed must be a non-negative safe integer.",
            ));
        }
        if service.sampling_enabled() {
            return Err(unsupported(
                "This profile derives the sampling seed from the request ID.",
            ));
        }
    }
    if let Some(n) = value.get("n") {
        match n.as_u64() {
            Some(1) => {}
            // Zero is an invalid value rather than an unimplemented behavior,
            // so it keeps the same error class as other malformed fields.
            Some(0) => return Err(ApiError::invalid("n must be a positive integer.")),
            Some(_) => return Err(unsupported("This profile returns exactly one choice.")),
            None => return Err(ApiError::invalid("n must be a positive integer.")),
        }
    }
    for key in ["presence_penalty", "frequency_penalty"] {
        if let Some(penalty) = value.get(key) {
            match penalty.as_f64() {
                Some(0.0) => {}
                Some(_) => return Err(unsupported(&format!("This profile does not apply {key}."))),
                None => return Err(ApiError::invalid(format!("{key} must be a number."))),
            }
        }
    }
    if let Some(logprobs) = value.get("logprobs") {
        match logprobs {
            Value::Bool(false) => {}
            Value::Bool(true) => {
                return Err(unsupported(
                    "This profile does not return log probabilities.",
                ))
            }
            _ => return Err(ApiError::invalid("logprobs must be a Boolean.")),
        }
    }
    if let Some(format) = value.get("response_format") {
        let format = format
            .as_object()
            .ok_or_else(|| ApiError::invalid("response_format must be an object."))?;
        match format.get("type").and_then(Value::as_str) {
            Some("text") => {}
            Some(_) => {
                return Err(unsupported(
                    "This profile does not provide structured output.",
                ))
            }
            None => return Err(ApiError::invalid("response_format requires a type.")),
        }
        // The accepted shape has no behavior behind any other key, so an extra
        // key is an unsupported request instead of a silently ignored one.
        if format.keys().any(|key| key != "type") {
            return Err(unsupported(
                "response_format accepts no keys besides the type.",
            ));
        }
    }
    if let Some(parallel) = value.get("parallel_tool_calls") {
        match parallel {
            Value::Bool(true) => {}
            Value::Bool(false) => {
                return Err(unsupported(
                    "This profile cannot enforce one tool call per generation.",
                ))
            }
            _ => return Err(ApiError::invalid("parallel_tool_calls must be a Boolean.")),
        }
    }
    Ok(())
}

fn normalize(
    value: &Value,
    api: Api,
    count_only: bool,
    service: &Service,
) -> Result<(Request, bool), ApiError> {
    let allowed = if api == Api::Anthropic {
        vec![
            "model",
            "max_tokens",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "stream",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "metadata",
            "thinking",
            "output_config",
            "service_tier",
            "cache_control",
        ]
    } else {
        vec![
            "model",
            "max_tokens",
            "max_completion_tokens",
            "messages",
            "tools",
            "tool_choice",
            "stream",
            "temperature",
            "top_p",
            "stop",
            "stream_options",
            "user",
            "n",
            "logprobs",
            "presence_penalty",
            "frequency_penalty",
            "response_format",
            "parallel_tool_calls",
            "seed",
        ]
    };
    unknown(value, &allowed, "Request")?;
    if value["model"].as_str() != Some(&service.model_id) {
        return Err(ApiError::new(
            404,
            "model_not_found",
            "Select the model listed by /v1/models.",
        ));
    }
    if api == Api::OpenAi {
        openai_compatibility(value, service)?;
    }
    let selection = if service.image_enabled() {
        crate::sampling::Selection::from_vision_request(value)?
    } else {
        crate::sampling::Selection::from_request(value, service.sampling_enabled())?
    };
    if let Some(thinking) = value.get("thinking") {
        if thinking != &json!({"type":"disabled"}) {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                "This profile requires thinking to be disabled.",
            ));
        }
    }
    if let Some(output) = value.get("output_config") {
        unknown(output, &["effort"], "output_config")?;
        if let Some(effort) = output.get("effort") {
            if !["low", "medium", "high", "xhigh", "max"].contains(&effort.as_str().unwrap_or("")) {
                return Err(ApiError::invalid("The output effort value is invalid."));
            }
        }
    }
    if let Some(tier) = value.get("service_tier") {
        if tier != "auto" && tier != "standard_only" {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                "This profile supports only the standard service tier.",
            ));
        }
    }
    if let Some(options) = value.get("stream_options") {
        unknown(options, &["include_usage"], "stream_options")?;
        if options
            .get("include_usage")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(ApiError::invalid("include_usage must be a Boolean."));
        }
    }
    let stream = match value.get("stream") {
        None => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| ApiError::invalid("stream must be a Boolean."))?,
    };
    if !stream && value.get("stream_options").is_some() {
        return Err(ApiError::invalid("stream_options requires stream=true."));
    }
    if value.get("max_tokens").is_some() && value.get("max_completion_tokens").is_some() {
        return Err(ApiError::invalid(
            "Set either max_tokens or max_completion_tokens.",
        ));
    }
    let max_value = value
        .get("max_tokens")
        .or_else(|| value.get("max_completion_tokens"));
    let output_limit = service.ready["capabilities"]["max_output_tokens"]
        .as_u64()
        .unwrap_or(0);
    let max_tokens = if count_only {
        0
    } else if let Some(value) = max_value {
        let tokens = value
            .as_u64()
            .ok_or_else(|| ApiError::invalid("max_tokens must be a non-negative integer."))?;
        if tokens == 0 {
            return Err(ApiError::invalid("max_tokens must be at least 1."));
        }
        if tokens > output_limit {
            return Err(ApiError::invalid(
                "max_tokens exceeds the deployment output limit.",
            ));
        }
        tokens as usize
    } else {
        // The default must never exceed what this deployment can emit, so an
        // omitted limit stays usable on a deployment with a small output budget.
        let tokens = DEFAULT_MAX_TOKENS.min(output_limit);
        if tokens == 0 {
            return Err(ApiError::invalid(
                "This deployment allows no output tokens.",
            ));
        }
        tokens as usize
    };
    let stops = value.get(if api == Api::Anthropic {
        "stop_sequences"
    } else {
        "stop"
    });
    let stops = match stops {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) if api == Api::OpenAi => vec![s.clone()],
        Some(v) => array(v, "Stop sequences")?
            .iter()
            .map(|v| text(v, "Stop sequence"))
            .collect::<Result<Vec<_>, _>>()?,
    };
    if stops.len() > 16 || stops.iter().any(|s| s.is_empty() || s.len() > 1024) {
        return Err(ApiError::invalid(
            "Use at most 16 non-empty stop sequences of at most 1024 bytes.",
        ));
    }
    let mut tools = Vec::new();
    if let Some(items) = value.get("tools") {
        for tool in array(items, "Tools")? {
            if api == Api::Anthropic {
                unknown(
                    tool,
                    &[
                        "name",
                        "description",
                        "input_schema",
                        "cache_control",
                        "type",
                    ],
                    "Tool",
                )?;
                if tool.get("type").is_some_and(|v| v != "custom") {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        "Server tools are unsupported.",
                    ));
                }
                let name = text(&tool["name"], "Tool name")?;
                if !tool["input_schema"].is_object() {
                    return Err(ApiError::invalid("Tool input_schema must be an object."));
                }
                let mut function = json!({"name":name,"parameters":tool["input_schema"]});
                if let Some(description) = tool.get("description") {
                    function["description"] = json!(text(description, "Tool description")?);
                }
                tools.push(json!({"type":"function","function":function}));
            } else {
                // Validate the OpenAI declaration here so a shape error is
                // reported with an actionable message, not as a generic
                // prepared-input contract failure.
                let function = tool["function"].as_object().ok_or_else(|| {
                    ApiError::invalid("Each OpenAI tool requires a function object.")
                })?;
                if function.get("strict").is_some_and(|value| value == true) {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        "This profile does not provide strict schema generation.",
                    ));
                }
                if tool["type"] != "function" {
                    return Err(ApiError::invalid("OpenAI tools must use type function."));
                }
                if tool.as_object().is_some_and(|object| {
                    object.keys().any(|key| key != "type" && key != "function")
                }) {
                    return Err(ApiError::invalid(
                        "OpenAI tools accept only type and function fields.",
                    ));
                }
                if !function
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| !name.is_empty())
                {
                    return Err(ApiError::invalid(
                        "Tool function name must be a non-empty string.",
                    ));
                }
                if function
                    .get("description")
                    .is_some_and(|value| !value.is_string())
                {
                    return Err(ApiError::invalid(
                        "Tool function description must be a string.",
                    ));
                }
                if function
                    .get("parameters")
                    .is_some_and(|value| !value.is_object())
                {
                    return Err(ApiError::invalid("Tool parameters must be an object."));
                }
                if function
                    .get("strict")
                    .is_some_and(|value| !value.is_boolean())
                {
                    return Err(ApiError::invalid("Tool strict must be a Boolean."));
                }
                if function.keys().any(|key| {
                    !["name", "description", "parameters", "strict"].contains(&key.as_str())
                }) {
                    return Err(ApiError::invalid(
                        "Tool functions accept only name, description, parameters, and strict fields.",
                    ));
                }
                tools.push(tool.clone());
            }
        }
    }
    if tools.len() > 256 {
        return Err(ApiError::invalid("The tool count exceeds 256."));
    }
    if let Some(choice) = value.get("tool_choice") {
        let kind = if api == Api::Anthropic {
            unknown(
                choice,
                &["type", "disable_parallel_tool_use"],
                "tool_choice",
            )?;
            choice["type"].as_str()
        } else {
            choice.as_str()
        };
        match kind {
            Some("auto") => {}
            Some("none") => tools.clear(),
            _ => {
                return Err(ApiError::new(
                    400,
                    "unsupported_feature",
                    "This profile supports only auto or none tool choice.",
                ))
            }
        }
        if choice
            .get("disable_parallel_tool_use")
            .is_some_and(|v| !v.is_boolean())
        {
            return Err(ApiError::invalid(
                "disable_parallel_tool_use must be a Boolean.",
            ));
        }
        if choice["disable_parallel_tool_use"] == true {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                "This profile cannot enforce one tool call per generation.",
            ));
        }
    }
    let mut messages = Vec::new();
    if let Some(system) = value.get("system") {
        messages.push(json!({"role":"system","content":content_text(system)?}));
    }
    for message in array(&value["messages"], "Messages")? {
        if api == Api::OpenAi {
            unknown(
                message,
                &["role", "content", "tool_calls", "tool_call_id", "name"],
                "Message",
            )?;
            let mut message = message.clone();
            if message["content"].is_null() && message.get("tool_calls").is_some() {
                message["content"] = json!("");
            }
            // Accept the OpenAI role vocabulary and report message shape
            // problems here instead of leaving them to the prepared-input
            // contract, which can only report a generic contract failure.
            let role = text(&message["role"], "Message role")?;
            if role == "developer" {
                // OpenAI developer messages carry the same instructions as a
                // system message, so both use the worker's system role.
                message["role"] = json!("system");
            } else if !matches!(role.as_str(), "system" | "user" | "assistant" | "tool") {
                return Err(ApiError::invalid(
                    "OpenAI messages require system, developer, user, assistant, or tool roles.",
                ));
            }
            for key in ["name", "tool_call_id"] {
                if message.get(key).is_some_and(|value| !value.is_string()) {
                    return Err(ApiError::invalid(format!(
                        "Message {key} must be a string."
                    )));
                }
            }
            if !message["content"].is_string() && !message["content"].is_array() {
                return Err(ApiError::invalid(
                    "Message content must be a string or a content-part array.",
                ));
            }
            if message["content"].is_array() {
                message["content"] = openai_content(&message["content"], &role, service)?;
            }
            if message.get("tool_calls").is_some() {
                if role != "assistant" {
                    return Err(ApiError::invalid(
                        "Only assistant messages can contain tool_calls.",
                    ));
                }
                let calls = message["tool_calls"]
                    .as_array_mut()
                    .ok_or_else(|| ApiError::invalid("Message tool_calls must be an array."))?;
                for call in calls {
                    if call["function"].as_object().is_none() {
                        return Err(ApiError::invalid("Tool calls require a function object."));
                    }
                    if call["type"] != "function" {
                        return Err(ApiError::invalid("Tool calls must use type function."));
                    }
                    if call.as_object().is_some_and(|object| {
                        object
                            .keys()
                            .any(|key| !["id", "type", "function"].contains(&key.as_str()))
                    }) {
                        return Err(ApiError::invalid(
                            "Tool calls accept only id, type, and function fields.",
                        ));
                    }
                    if !call["id"].as_str().is_some_and(|id| !id.is_empty()) {
                        return Err(ApiError::invalid(
                            "Tool call id must be a non-empty string.",
                        ));
                    }
                    let function = call["function"].as_object().expect("checked above");
                    if function
                        .keys()
                        .any(|key| !["name", "arguments"].contains(&key.as_str()))
                    {
                        return Err(ApiError::invalid(
                            "Tool call functions accept only name and arguments fields.",
                        ));
                    }
                    if !function
                        .get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|name| !name.is_empty())
                    {
                        return Err(ApiError::invalid(
                            "Tool call function name must be a non-empty string.",
                        ));
                    }
                    match function.get("arguments").unwrap_or(&Value::Null) {
                        Value::Object(_) => {}
                        Value::String(_) => {}
                        _ => {
                            return Err(ApiError::invalid(
                                "Tool call arguments must be an object or a JSON string.",
                            ))
                        }
                    }
                    // A JSON string must encode an object. Any other parsed
                    // value would reach the prepared-input contract and report
                    // its internal wording instead of this request error.
                    if let Some(arguments) = call["function"]["arguments"].as_str() {
                        let parsed = crate::contracts::parse_document(arguments.as_bytes())
                            .map_err(|_| {
                                ApiError::invalid(
                                    "Tool arguments must contain valid JSON without duplicate keys.",
                                )
                            })?;
                        if !parsed.is_object() {
                            return Err(ApiError::invalid(
                                "Tool call arguments must encode a JSON object.",
                            ));
                        }
                        call["function"]["arguments"] = parsed;
                    }
                }
            } else if role == "tool" {
                match message["tool_call_id"].as_str() {
                    Some(id) if !id.is_empty() => {}
                    Some(_) => {
                        return Err(ApiError::invalid(
                            "Tool message tool_call_id must be a non-empty string.",
                        ))
                    }
                    None => return Err(ApiError::invalid("Tool messages require a tool_call_id.")),
                }
            }
            messages.push(message);
            continue;
        }
        unknown(message, &["role", "content"], "Message")?;
        let role = text(&message["role"], "Message role")?;
        if role != "user" && role != "assistant" {
            return Err(ApiError::invalid(
                "Messages require user or assistant roles.",
            ));
        }
        if let Some(content) = message["content"].as_str() {
            messages.push(json!({"role":role,"content":content}));
            continue;
        }
        let mut text_parts = String::new();
        let mut ordered_parts = Vec::new();
        let mut calls = Vec::new();
        for block in array(&message["content"], "Message content")? {
            match block["type"].as_str() {
                Some("text") => {
                    unknown(block, &["type", "text", "cache_control"], "Text block")?;
                    text_parts.push_str(&text(&block["text"], "Text block text")?);
                    if service.image_enabled() && role == "user" {
                        ordered_parts.push(json!({"type":"text","text":block["text"]}));
                    }
                }
                Some("image") if service.image_enabled() && role == "user" => {
                    ordered_parts.push(vision::image_part(block)?);
                }
                Some("tool_use") if role == "assistant" => {
                    unknown(
                        block,
                        &["type", "id", "name", "input", "cache_control"],
                        "Tool use",
                    )?;
                    if !block["input"].is_object() {
                        return Err(ApiError::invalid("Tool input must be an object."));
                    }
                    calls.push(json!({"id":text(&block["id"],"Tool ID")?,"type":"function","function":{"name":text(&block["name"],"Tool name")?,"arguments":block["input"]}}));
                }
                Some("tool_result") if role == "user" => {
                    unknown(
                        block,
                        &[
                            "type",
                            "tool_use_id",
                            "content",
                            "is_error",
                            "cache_control",
                        ],
                        "Tool result",
                    )?;
                    if !text_parts.is_empty() || !ordered_parts.is_empty() {
                        return Err(ApiError::invalid(
                            "Tool results must precede text within the user message.",
                        ));
                    }
                    if block.get("is_error").is_some_and(|v| !v.is_boolean()) {
                        return Err(ApiError::invalid("Tool result is_error must be a Boolean."));
                    }
                    if service.image_enabled() {
                        messages.push(json!({"role":"tool","tool_call_id":text(&block["tool_use_id"],"Tool result ID")?,"content":vision::tool_content(block)?}));
                        continue;
                    }
                    let mut content = if block.get("content").is_none() {
                        String::new()
                    } else {
                        content_text(&block["content"])?
                    };
                    if block["is_error"] == true {
                        content = format!("[tool_result is_error=true]\n{content}");
                    }
                    messages.push(json!({"role":"tool","tool_call_id":text(&block["tool_use_id"],"Tool result ID")?,"content":content}));
                }
                _ => {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        "This profile supports text, tool_use, and tool_result blocks.",
                    ))
                }
            }
        }
        if !text_parts.is_empty() || !calls.is_empty() || !ordered_parts.is_empty() {
            let mut normalized = json!({"role":role,"content":text_parts});
            if !ordered_parts.is_empty() {
                normalized["content"] = json!(ordered_parts);
            }
            if !calls.is_empty() {
                normalized["tool_calls"] = json!(calls);
            }
            messages.push(normalized);
        }
    }
    if messages.is_empty() || messages.len() > 256 {
        return Err(ApiError::invalid(
            "Use between 1 and 256 normalized messages.",
        ));
    }
    if service.image_enabled() {
        vision::check_media_limits(&messages)?;
    }
    Ok((
        Request {
            selection,
            messages,
            tools,
            max_tokens,
            stops,
            count_only,
        },
        stream,
    ))
}

/// Normalize the OpenAI content-part array into the serial profile shape.
///
/// Text parts join into one string unless the vision profile must preserve
/// order, in which case they stay ordered with the image parts.
fn openai_content(content: &Value, role: &str, service: &Service) -> Result<Value, ApiError> {
    let ordered = service.image_enabled() && role == "user";
    let mut joined = String::new();
    let mut parts = Vec::new();
    for part in array(content, "Message content")? {
        match part["type"].as_str() {
            Some("text") => {
                unknown(part, &["type", "text"], "Text block")?;
                let value = text(&part["text"], "Text block text")?;
                if ordered {
                    parts.push(json!({"type":"text","text":value}));
                } else {
                    joined.push_str(&value);
                }
            }
            Some("image_url") => {
                if !service.image_enabled() {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        "OpenAI image inputs are unsupported.",
                    ));
                }
                if !ordered {
                    return Err(ApiError::new(
                        400,
                        "unsupported_feature",
                        "OpenAI images require user content.",
                    ));
                }
                parts.push(vision::openai_image_part(part)?);
            }
            _ => {
                return Err(ApiError::new(
                    400,
                    "unsupported_feature",
                    "OpenAI messages support text content parts.",
                ))
            }
        }
    }
    Ok(if parts.is_empty() {
        json!(joined)
    } else {
        json!(parts)
    })
}

fn content_text(content: &Value) -> Result<String, ApiError> {
    if let Some(s) = content.as_str() {
        return Ok(s.into());
    }
    let mut result = String::new();
    for block in array(content, "Text content")? {
        unknown(
            block,
            &["type", "text", "cache_control"],
            "Text content block",
        )?;
        if block["type"] != "text" {
            return Err(ApiError::new(
                400,
                "unsupported_feature",
                "Only text content is supported.",
            ));
        }
        result.push_str(&text(&block["text"], "Content text")?);
    }
    Ok(result)
}

fn finish_reason(api: Api, cause: &str) -> &'static str {
    match (api, cause) {
        (Api::Anthropic, "tool_calls") => "tool_use",
        (Api::Anthropic, "length") => "max_tokens",
        (Api::Anthropic, "stop_sequence") => "stop_sequence",
        (Api::Anthropic, _) => "end_turn",
        (Api::OpenAi, "tool_calls") => "tool_calls",
        (Api::OpenAi, "length") => "length",
        _ => "stop",
    }
}
fn response_body(
    api: Api,
    begin: &Begin,
    model: &str,
    parts: &[Part],
    cause: &str,
    input: u64,
    output: u64,
    stop: Option<String>,
) -> Value {
    if api == Api::Anthropic {
        json!({"id":format!("msg_{}",begin.request_id),"type":"message","role":"assistant","model":model,"content":anthropic_blocks(parts),"stop_reason":finish_reason(api,cause),"stop_sequence":stop,"usage":{"input_tokens":input,"output_tokens":output}})
    } else {
        let mut text = String::new();
        let mut tools = Vec::new();
        for part in parts {
            match part { Part::Text(t)=>text.push_str(t),Part::Tool{id,name,input}=>tools.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":input.to_string()}})) }
        }
        let mut message = json!({"role":"assistant","content":text});
        if !tools.is_empty() {
            message["tool_calls"] = json!(tools);
        }
        json!({"id":format!("chatcmpl-{}",begin.request_id),"object":"chat.completion","created":0,"model":model,"choices":[{"index":0,"message":message,"finish_reason":finish_reason(api,cause)}],"usage":{"prompt_tokens":input,"completion_tokens":output,"total_tokens":input+output}})
    }
}

struct PublicStream {
    receiver: mpsc::Receiver<ModelEvent>,
    _guard: CancelGuard,
    api: Api,
    id: String,
    model: String,
    pending: std::collections::VecDeque<Event>,
    index: usize,
    text_open: bool,
    done: bool,
    tools: usize,
    include_usage: bool,
}
impl PublicStream {
    fn new(
        receiver: mpsc::Receiver<ModelEvent>,
        guard: CancelGuard,
        api: Api,
        begin: Begin,
        model: String,
        include_usage: bool,
    ) -> Self {
        let mut stream = Self {
            receiver,
            _guard: guard,
            api,
            id: begin.request_id.clone(),
            model,
            pending: Default::default(),
            index: 0,
            text_open: false,
            done: false,
            tools: 0,
            include_usage,
        };
        if api == Api::Anthropic {
            stream.emit("message_start",json!({"type":"message_start","message":{"id":format!("msg_{}",begin.request_id),"type":"message","role":"assistant","content":[],"model":stream.model,"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":begin.input_tokens,"output_tokens":0}}}));
        } else {
            stream.chunk(json!({"role":"assistant","content":""}), Value::Null);
        }
        stream
    }
    fn emit(&mut self, name: &str, value: Value) {
        self.pending
            .push_back(Event::default().event(name).data(value.to_string()));
    }
    fn chunk(&mut self, delta: Value, finish: Value) {
        let mut v = json!({"id":format!("chatcmpl-{}",self.id),"object":"chat.completion.chunk","created":0,"model":self.model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
        if self.include_usage {
            v["usage"] = Value::Null;
        }
        self.pending.push_back(Event::default().data(v.to_string()));
    }
    fn close_text(&mut self) {
        if self.text_open {
            self.emit(
                "content_block_stop",
                json!({"type":"content_block_stop","index":self.index}),
            );
            self.index += 1;
            self.text_open = false;
        }
    }
    fn accept(&mut self, event: ModelEvent) {
        match event {
            ModelEvent::Decision(_) => self.accept(ModelEvent::Error(ApiError::new(
                500,
                "protocol_fault",
                "A text stream received a decision result.",
            ))),
            ModelEvent::Part(Part::Text(text)) => {
                if !text.is_empty() {
                    if self.api == Api::OpenAi {
                        self.chunk(json!({"content":text}), Value::Null);
                    } else {
                        if !self.text_open {
                            self.emit("content_block_start",json!({"type":"content_block_start","index":self.index,"content_block":{"type":"text","text":""}}));
                            self.text_open = true;
                        }
                        self.emit("content_block_delta",json!({"type":"content_block_delta","index":self.index,"delta":{"type":"text_delta","text":text}}));
                    }
                }
            }
            ModelEvent::Part(Part::Tool { id, name, input }) => {
                if self.api == Api::OpenAi {
                    self.chunk(json!({"tool_calls":[{"index":self.tools,"id":id,"type":"function","function":{"name":name,"arguments":input.to_string()}}]}),Value::Null);
                } else {
                    self.close_text();
                    self.emit("content_block_start",json!({"type":"content_block_start","index":self.index,"content_block":{"type":"tool_use","id":id,"name":name,"input":{}}}));
                    self.emit("content_block_delta",json!({"type":"content_block_delta","index":self.index,"delta":{"type":"input_json_delta","partial_json":input.to_string()}}));
                    self.emit(
                        "content_block_stop",
                        json!({"type":"content_block_stop","index":self.index}),
                    );
                    self.index += 1;
                }
                self.tools += 1;
            }
            ModelEvent::Done {
                cause,
                input_tokens,
                output_tokens,
                matched_stop,
            } => {
                if self.api == Api::Anthropic {
                    self.close_text();
                    self.emit("message_delta",json!({"type":"message_delta","delta":{"stop_reason":finish_reason(self.api,&cause),"stop_sequence":matched_stop},"usage":{"output_tokens":output_tokens}}));
                    self.emit("message_stop", json!({"type":"message_stop"}));
                } else {
                    self.chunk(json!({}), json!(finish_reason(self.api, &cause)));
                    if self.include_usage {
                        let usage = json!({"id":format!("chatcmpl-{}",self.id),"object":"chat.completion.chunk","created":0,"model":self.model,"choices":[],
                            "usage":{"prompt_tokens":input_tokens,"completion_tokens":output_tokens,"total_tokens":input_tokens+output_tokens}});
                        self.pending
                            .push_back(Event::default().data(usage.to_string()));
                    }
                    self.pending.push_back(Event::default().data("[DONE]"));
                }
                self.done = true;
            }
            ModelEvent::Error(error) => {
                self.emit("error",json!({"type":"error","error":{"type":public_error_type(error.status),"code":error.code,"message":error.message}}));
                self.done = true;
            }
        }
    }
}
impl Stream for PublicStream {
    type Item = Result<Event, Infallible>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            if self.done {
                return Poll::Ready(None);
            }
            match self.receiver.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(event)) => self.accept(event),
                Poll::Ready(None) => self.accept(ModelEvent::Error(ApiError::new(
                    503,
                    "worker_lost",
                    "The worker ended the stream without a terminal result.",
                ))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{gateway_test_service, gateway_test_service_with_max_output};

    #[tokio::test]
    async fn prepared_readiness_advertises_the_selected_native_tool_parser() {
        let (service, _) = crate::supervisor::gateway_test_service_with_protocol(
            crate::contracts::NATIVE_PREPARED_PROTOCOL);
        let response = ready(State(service.clone())).await;
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 10000).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["public_capabilities"]["tools"], "minicpm5_xml");
        assert_eq!(body["public_capabilities"]["sampling"], false);
        assert_eq!(body["profile"], "apxinf-native-prepared-v0.1");
        service.shutdown();
    }

    #[tokio::test]
    async fn sampled_profile_preserves_policy_through_public_admission() {
        use crate::sampling::Selection;
        let (service, mut observed) = crate::supervisor::gateway_test_service_with_protocol(
            crate::contracts::NATIVE_SAMPLED_PROTOCOL,
        );
        for api in [Api::OpenAi, Api::Anthropic] {
            for (parameters, expected) in [
                (
                    json!({}),
                    Selection::Random {
                        temperature: 1.0,
                        top_p: 0.95,
                    },
                ),
                (json!({"temperature":0}), Selection::Greedy),
                (
                    json!({"temperature":0.7,"top_p":0.8}),
                    Selection::Random {
                        temperature: 0.7,
                        top_p: 0.8,
                    },
                ),
                (
                    json!({"top_p":0.6}),
                    Selection::Random {
                        temperature: 1.0,
                        top_p: 0.6,
                    },
                ),
            ] {
                let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
                value
                    .as_object_mut()
                    .unwrap()
                    .extend(parameters.as_object().unwrap().clone());
                let error = request_error(service.clone(), value, api).await;
                assert_eq!(error.message, "Test coordinator observed admission.");
                assert_eq!(observed.try_recv().unwrap().selection, expected);
            }
            let mut invalid = openai_request(json!([{"role":"user","content":"hello"}]));
            invalid["top_p"] = json!(0);
            assert_eq!(
                request_error(service.clone(), invalid, api).await.code,
                "invalid_request"
            );
            assert!(observed.try_recv().is_err());
        }
        let response = ready(State(service.clone())).await;
        let body = axum::body::to_bytes(response.into_body(), 10000)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["profile"], "apxinf-native-sampled-v0.1");
        assert_eq!(body["public_capabilities"]["sampling"], true);
        assert_eq!(body["public_capabilities"]["tools"], "minicpm5_xml");
        assert_eq!(body["public_capabilities"]["thinking"], false);
        service.shutdown();
    }

    #[tokio::test]
    async fn model_thinking_profile_is_distinct_from_extended_thinking() {
        for (protocol, profile, budget) in [
            (
                crate::contracts::NATIVE_THINKING_PROTOCOL,
                "apxinf-native-thinking-v0.1",
                None,
            ),
            (
                crate::contracts::NATIVE_BOUNDED_THINKING_PROTOCOL,
                "apxinf-native-bounded-thinking-v0.1",
                Some(512),
            ),
        ] {
            let (service, mut observed) =
                crate::supervisor::gateway_test_service_with_protocol(protocol);
            let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
            value["thinking"] = json!({"type":"disabled"});
            let error = request_error(service.clone(), value.clone(), Api::Anthropic).await;
            assert_eq!(error.message, "Test coordinator observed admission.");
            assert!(matches!(
                observed.try_recv().unwrap().selection,
                crate::sampling::Selection::Random { .. }
            ));
            for thinking in [
                json!({"type":"enabled","budget_tokens":1024}),
                json!({"type":"adaptive"}),
            ] {
                value["thinking"] = thinking;
                assert_eq!(
                    request_error(service.clone(), value.clone(), Api::Anthropic)
                        .await
                        .code,
                    "unsupported_feature"
                );
                assert!(observed.try_recv().is_err());
            }
            let response = ready(State(service.clone())).await;
            let body = axum::body::to_bytes(response.into_body(), 10000)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["profile"], profile);
            assert_eq!(
                body["public_capabilities"]["initial_reasoning_budget_tokens"].as_u64(),
                budget
            );
            assert_eq!(
                body["public_capabilities"]["model_thinking"],
                "initial_hidden"
            );
            assert_eq!(body["public_capabilities"]["thinking"], false);
            assert_eq!(body["public_capabilities"]["sampling"], true);
            service.shutdown();
        }
    }

    #[tokio::test]
    async fn direct_admission_rejects_unsupported_or_invalid_selection() {
        use crate::sampling::Selection;
        for protocol in [
            crate::contracts::PROTOCOL,
            crate::contracts::NATIVE_PROTOCOL,
            crate::contracts::NATIVE_SAMPLED_PROTOCOL,
            crate::contracts::NATIVE_THINKING_PROTOCOL,
            crate::contracts::NATIVE_BOUNDED_THINKING_PROTOCOL,
        ] {
            let (service, mut observed) =
                crate::supervisor::gateway_test_service_with_protocol(protocol);
            let value = openai_request(json!([{"role":"user","content":"hello"}]));
            let (mut request, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
            request.selection = Selection::Random {
                temperature: f32::NAN,
                top_p: 0.95,
            };
            assert!(service.enqueue(request).is_err());
            if !service.sampling_enabled() {
                let (mut request, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
                request.selection = Selection::Random {
                    temperature: 1.0,
                    top_p: 0.95,
                };
                assert!(service.enqueue(request).is_err());
            }
            assert!(observed.try_recv().is_err());
            service.shutdown();
        }
    }

    #[tokio::test]
    async fn readiness_separates_host_pressure_from_worker_availability() {
        use crate::host_pressure::{HostPressure, HostPressurePolicy, Pressure};
        let pressure = Arc::new(HostPressure::new(HostPressurePolicy::Macos));
        pressure.record(Ok(Pressure::Warning), std::time::Instant::now());
        let (service, mut observed) =
            crate::supervisor::gateway_test_service_with_pressure(pressure.clone());
        let response = ready(State(service.clone())).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(response.into_body(), 10000)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["ready"], false);
        assert_eq!(body["worker_available"], true);
        assert_eq!(body["host_pressure"]["policy"], "macos");
        assert_eq!(body["host_pressure"]["state"], "warning");
        assert_eq!(
            body["worker"]["worker_epoch"],
            service.ready_snapshot()["worker_epoch"]
        );
        let error = request_error(
            service.clone(),
            openai_request(json!([
                {"role":"user","content":"original pressure request"}
            ])),
            Api::OpenAi,
        )
        .await;
        assert_eq!(
            (error.status, error.code.as_str()),
            (503, "capacity_unavailable")
        );
        assert!(observed.try_recv().is_err());
        service.shutdown();
        let now = std::time::Instant::now();
        pressure.record(Ok(Pressure::Normal), now - Duration::from_secs(1));
        pressure.record(Ok(Pressure::Normal), now);
        let response = ready(State(service)).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    fn openai_request(messages: Value) -> Value {
        json!({"model": "test-model", "max_tokens": 4, "messages": messages})
    }

    async fn request_error(service: Arc<Service>, value: Value, api: Api) -> ApiError {
        tokio::time::timeout(
            Duration::from_secs(1),
            handle(
                service,
                HeaderMap::new(),
                Bytes::from(value.to_string()),
                api,
                false,
            ),
        )
        .await
        .expect("The request must not wait for admission.")
        .expect_err("The request must fail.")
    }

    #[tokio::test]
    async fn invalid_normalized_inputs_never_reach_admission() {
        let (service, mut observed) = gateway_test_service();
        let cases = [
            openai_request(json!([{"role": 17, "content": "hello"}])),
            openai_request(json!([{"role": "user", "content": 17}])),
            openai_request(json!([{"role": "user"}])),
            openai_request(json!([{"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "read", "arguments": {}}, "unknown": true}
            ]}])),
            openai_request(json!([{"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "read", "arguments": []}}
            ]}])),
            json!({"model": "test-model", "max_tokens": 4, "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {"name": "read", "parameters": [], "extra": true}}]}),
        ];
        for value in cases {
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(error.status, 400, "{}", error.message);
            assert_eq!(error.code, "invalid_request");
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
            assert!(service.available.load(Ordering::Acquire));
            assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
            assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
        }

        // This control proves that the observation queue is live.
        let valid = openai_request(json!([{"role": "user", "content": "hello"}]));
        let error = request_error(service.clone(), valid, Api::OpenAi).await;
        assert_eq!(error.message, "Test coordinator observed admission.");
        let admitted = observed
            .try_recv()
            .expect("Valid input must reach admission.");
        assert_eq!(
            admitted.messages,
            vec![json!({"role": "user", "content": "hello"})]
        );
        assert!(service.available.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn openai_developer_messages_use_the_system_role() {
        let (service, mut observed) = gateway_test_service();
        let value = openai_request(json!([
            {"role":"developer","content":"Be brief."},
            {"role":"user","content":"hello"}
        ]));
        let (request, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(
            request.messages,
            vec![
                json!({"role":"system","content":"Be brief."}),
                json!({"role":"user","content":"hello"}),
            ]
        );
        // Content parts on a developer message collapse into the serial form,
        // because the worker contract only accepts text for a system message.
        let value = openai_request(json!([{"role":"developer","content":[
            {"type":"text","text":"Be "},{"type":"text","text":"brief."}
        ]}]));
        let (request, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(
            request.messages,
            vec![json!({"role":"system","content":"Be brief."})]
        );
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn openai_message_shape_errors_are_actionable() {
        let (service, mut observed) = gateway_test_service();
        for (message, expected) in [
            (
                json!({"role":"function","content":"hello"}),
                "system, developer, user, assistant, or tool",
            ),
            (json!({"role":17,"content":"hello"}), "Message role"),
            (json!({"content":"hello"}), "Message role"),
            (
                json!({"role":"user","content":{"text":"hello"}}),
                "a string or a content-part array",
            ),
            (
                json!({"role":"user","tool_call_id":17,"content":"hello"}),
                "tool_call_id must be a string",
            ),
        ] {
            let value = openai_request(json!([message]));
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "invalid_request"),
                "{}",
                error.message
            );
            assert!(error.message.contains(expected), "{}", error.message);
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
        assert!(service.available.load(Ordering::Acquire));
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn conflicting_output_limits_fail_before_admission() {
        let (service, mut observed) = gateway_test_service();
        let mut value = openai_request(json!([{"role": "user", "content": "hello"}]));
        value["max_completion_tokens"] = json!(1);
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "invalid_request");
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn omitted_output_limit_stays_within_the_deployment_budget() {
        let (service, _) = gateway_test_service();
        let value = json!({"model":"test-model","messages":[{"role":"user","content":"hello"}]});
        let (request, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(request.max_tokens, 512);

        let (small, _) = gateway_test_service_with_max_output(64);
        let (request, _) = normalize(&value, Api::OpenAi, false, &small).unwrap();
        assert_eq!(request.max_tokens, 64);
        let explicit = json!({"model":"test-model","max_tokens":8,"messages":[{"role":"user","content":"hello"}]});
        let (request, _) = normalize(&explicit, Api::OpenAi, false, &small).unwrap();
        assert_eq!(request.max_tokens, 8);

        // Counting tokens never reserves output capacity and ignores the limit.
        let (request, _) = normalize(&value, Api::Anthropic, true, &small).unwrap();
        assert_eq!(request.max_tokens, 0);
    }

    #[tokio::test]
    async fn output_limits_reject_zero_and_an_exhausted_deployment() {
        let (service, mut observed) = gateway_test_service();
        for api in [Api::OpenAi, Api::Anthropic] {
            let value = json!({"model":"test-model","max_tokens":0,"messages":[{"role":"user","content":"hello"}]});
            let error = request_error(service.clone(), value, api).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "invalid_request")
            );
            assert!(error.message.contains("at least 1"), "{}", error.message);
        }
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        let (small, mut observed) = gateway_test_service_with_max_output(64);
        let over = json!({"model":"test-model","max_tokens":65,"messages":[{"role":"user","content":"hello"}]});
        let error = request_error(small.clone(), over, Api::OpenAi).await;
        assert_eq!(
            (error.status, error.code.as_str()),
            (400, "invalid_request")
        );
        assert_eq!(
            error.message,
            "max_tokens exceeds the deployment output limit."
        );

        let (exhausted, mut observed_exhausted) = gateway_test_service_with_max_output(0);
        let omitted = json!({"model":"test-model","messages":[{"role":"user","content":"hello"}]});
        let error = request_error(exhausted.clone(), omitted.clone(), Api::OpenAi).await;
        assert_eq!(
            (error.status, error.code.as_str()),
            (400, "invalid_request")
        );
        // Counting tokens still works without an output budget.
        let (request, _) = normalize(&omitted, Api::Anthropic, true, &exhausted).unwrap();
        assert_eq!(request.max_tokens, 0);
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(matches!(
            observed_exhausted.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert_eq!(small.metrics.failed.load(Ordering::Relaxed), 0);
        assert_eq!(exhausted.metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn openai_compatibility_fields_are_accepted_only_at_noops() {
        let (service, mut observed) = gateway_test_service();
        let value = json!({
            "model":"test-model","max_tokens":4,
            "messages":[{"role":"user","content":"hello"}],
            "n":1,"presence_penalty":0,"frequency_penalty":0,"logprobs":false,
            "response_format":{"type":"text"},"parallel_tool_calls":true
        });
        let (request, stream) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(request.max_tokens, 4);
        assert!(!stream);
        assert_eq!(
            request.messages,
            vec![json!({"role":"user","content":"hello"})]
        );
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        for field in [
            ("n", json!(2)),
            ("presence_penalty", json!(0.5)),
            ("frequency_penalty", json!(-0.5)),
            ("logprobs", json!(true)),
            ("response_format", json!({"type":"json_object"})),
            (
                "response_format",
                json!({"type":"json_schema","json_schema":{}}),
            ),
            // A text response format cannot carry unrelated behavior.
            ("response_format", json!({"type":"text","json_schema":{}})),
            ("response_format", json!({"type":"text","unsupported":true})),
            ("parallel_tool_calls", json!(false)),
        ] {
            let (name, replacement) = field;
            let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
            value[name] = replacement;
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "unsupported_feature"),
                "{name}"
            );
        }
        for field in [
            ("n", json!(0)),
            ("n", json!("1")),
            ("n", json!(1.5)),
            ("presence_penalty", json!("0")),
            ("frequency_penalty", json!(null)),
            ("logprobs", json!(0)),
            ("response_format", json!({"type":3})),
            ("response_format", json!({})),
            ("response_format", json!({"json_schema":{}})),
            ("parallel_tool_calls", json!("true")),
        ] {
            let (name, replacement) = field;
            let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
            value[name] = replacement;
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "invalid_request"),
                "{name}"
            );
        }
        // The compatibility relaxation is specific to the OpenAI route.
        let error = request_error(service.clone(), value, Api::Anthropic).await;
        assert_eq!(
            (error.status, error.code.as_str()),
            (400, "unsupported_feature")
        );
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(service.available.load(Ordering::Acquire));
        assert_eq!(service.metrics.queued.load(Ordering::Relaxed), 0);
        assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn parallel_tool_option_requires_a_boolean() {
        let (service, mut observed) = gateway_test_service();
        for invalid in [json!("false"), json!(0), json!(null)] {
            let value = json!({"model":"test-model", "max_tokens":4,
                "messages":[{"role":"user", "content":"hello"}],
                "tool_choice":{"type":"auto", "disable_parallel_tool_use":invalid}});
            let error = request_error(service.clone(), value, Api::Anthropic).await;
            assert_eq!(error.status, 400);
            assert_eq!(error.code, "invalid_request");
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
    }

    #[tokio::test]
    async fn loading_surface_answers_before_a_service_is_published() {
        let gate = Gate::new();
        let response = gated_ready(State(gate.clone())).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 10000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ready"], false);
        assert_eq!(body["phase"], "loading");

        let body = Bytes::from_static(b"{}");
        for response in [
            gated_models(State(gate.clone())).await,
            gated_metrics(State(gate.clone())).await,
            gated_messages(State(gate.clone()), HeaderMap::new(), body.clone()).await,
            gated_chat(State(gate.clone()), HeaderMap::new(), body.clone()).await,
            gated_count_tokens(State(gate.clone()), HeaderMap::new(), body.clone()).await,
            gated_systemone(State(gate.clone()), body.clone()).await,
        ] {
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            let body: Value = serde_json::from_slice(
                &axum::body::to_bytes(response.into_body(), 10000)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(body["error"]["code"], "model_unavailable");
        }

        let (service, _) = gateway_test_service();
        gate.publish(service);
        let response = gated_ready(State(gate)).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 100000)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["ready"], true);
    }

    #[tokio::test]
    async fn published_gate_serves_every_declared_route() {
        let (service, _) = gateway_test_service();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = router(service);
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        // Every declared route must answer through the published gate. A route
        // that only one router declares would answer 404 from the other.
        for (method, path, payload) in [
            ("GET", "/healthz", ""),
            ("GET", "/readyz", ""),
            ("GET", "/v1/models", ""),
            ("GET", "/metrics", ""),
            ("POST", "/v1/messages", "{"),
            ("POST", "/v1/messages/count_tokens", "{"),
            ("POST", "/v1/chat/completions", "{"),
            ("POST", "/v1/systemone", "{"),
        ] {
            let (status, response) = route_request(address, method, path, payload).await;
            assert_ne!(status, 404, "{method} {path} is not declared: {response}");
            assert_ne!(
                status, 405,
                "{method} {path} has no method binding: {response}"
            );
        }
        // Framework responses stay outside the declared set.
        let (status, response) = route_request(address, "GET", "/not-a-route", "").await;
        assert_eq!(status, 404, "{response}");
        let (status, response) = route_request(address, "GET", "/v1/messages", "").await;
        assert_eq!(status, 405, "{response}");
        server.abort();
    }

    async fn route_request(
        address: std::net::SocketAddr,
        method: &str,
        path: &str,
        payload: &str,
    ) -> (u16, String) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let text = String::from_utf8_lossy(&response).into_owned();
        let status = text
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u16>()
            .unwrap();
        (status, text)
    }

    #[tokio::test]
    async fn openai_seed_is_a_noop_only_where_generation_is_deterministic() {
        let (service, mut observed) = gateway_test_service();
        let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
        value["seed"] = json!(42);
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.message, "Test coordinator observed admission.");
        let admitted = observed
            .try_recv()
            .expect("A greedy seed must reach admission.");
        assert!(matches!(
            admitted.selection,
            crate::sampling::Selection::Greedy
        ));

        for seed in [json!("42"), json!(-1), json!(1.5), json!(true)] {
            let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
            value["seed"] = seed.clone();
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "invalid_request"),
                "{seed}"
            );
        }

        // The request parser rejects unsafe integers first, so the normalizer's
        // own bound is checked directly against a JSON value.
        let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
        value["seed"] = json!(crate::contracts::MAX_SAFE_INTEGER);
        let (request, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert!(matches!(
            request.selection,
            crate::sampling::Selection::Greedy
        ));
        let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
        value["seed"] = json!(crate::contracts::MAX_SAFE_INTEGER + 1);
        let error = normalize(&value, Api::OpenAi, false, &service)
            .err()
            .expect("An unsafe seed must fail.");
        assert_eq!(
            (error.status, error.code.as_str(), error.message.as_str()),
            (
                400,
                "invalid_request",
                "seed must be a non-negative safe integer."
            )
        );

        // A sampling profile derives the seed from the request ID.
        let (sampled, _) = crate::supervisor::gateway_test_service_with_protocol(
            crate::contracts::NATIVE_SAMPLED_PROTOCOL,
        );
        let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
        value["seed"] = json!(42);
        let error = request_error(sampled, value, Api::OpenAi).await;
        assert_eq!(
            (error.status, error.code.as_str()),
            (400, "unsupported_feature")
        );
        assert!(error.message.contains("request ID"));
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn openai_tool_shapes_fail_before_admission_with_actionable_errors() {
        let (service, mut observed) = gateway_test_service();
        let tool = |function: Value| json!([{"type":"function","function":function}]);
        for (tools, message) in [
            (
                json!([{"type":"function"}]),
                "Each OpenAI tool requires a function object.",
            ),
            (
                tool(json!({"name":17})),
                "Tool function name must be a non-empty string.",
            ),
            (
                tool(json!({"name":""})),
                "Tool function name must be a non-empty string.",
            ),
            (
                tool(json!({"name":"read","parameters":[]})),
                "Tool parameters must be an object.",
            ),
            (
                json!([{"type":"other","function":{"name":"read"}}]),
                "OpenAI tools must use type function.",
            ),
            (
                json!([{"type":"function","function":{"name":"read"},"extra":true}]),
                "OpenAI tools accept only type and function fields.",
            ),
        ] {
            let mut value = openai_request(json!([{"role":"user","content":"hello"}]));
            value["tools"] = tools;
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "invalid_request"),
                "{message}"
            );
            assert_eq!(error.message, message);
        }

        let call = |call: Value| json!([call]);
        let history = |calls: Value| {
            json!([
                {"role":"assistant","content":null,"tool_calls":calls},
                {"role":"tool","tool_call_id":"call-a","content":"x"}
            ])
        };
        for (messages, message) in [
            (
                history(call(
                    json!({"id":17,"type":"function","function":{"name":"read","arguments":{}}}),
                )),
                "Tool call id must be a non-empty string.",
            ),
            (
                history(call(
                    json!({"id":"call-a","type":"other","function":{"name":"read","arguments":{}}}),
                )),
                "Tool calls must use type function.",
            ),
            (
                history(call(
                    json!({"id":"call-a","type":"function","function":{"arguments":{}}}),
                )),
                "Tool call function name must be a non-empty string.",
            ),
            (
                history(call(
                    json!({"id":"call-a","type":"function","function":{"name":"read","arguments":[]}}),
                )),
                "Tool call arguments must be an object or a JSON string.",
            ),
            (
                history(call(
                    json!({"id":"call-a","type":"function","function":{"name":"read","arguments":{}},"extra":1}),
                )),
                "Tool calls accept only id, type, and function fields.",
            ),
            (
                history(call(
                    json!({"id":"call-a","type":"function","function":{"name":"read","arguments":"[]"}}),
                )),
                "Tool call arguments must encode a JSON object.",
            ),
            (
                history(call(
                    json!({"id":"call-a","type":"function","function":{"name":"read","arguments":"null"}}),
                )),
                "Tool call arguments must encode a JSON object.",
            ),
            (
                history(call(
                    json!({"id":"call-a","type":"function","function":{"name":"read","arguments":"5"}}),
                )),
                "Tool call arguments must encode a JSON object.",
            ),
            (
                history(call(
                    json!({"id":"call-a","type":"function","function":{"name":"read","arguments":"\"text\""}}),
                )),
                "Tool call arguments must encode a JSON object.",
            ),
        ] {
            let error = request_error(service.clone(), openai_request(messages), Api::OpenAi).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "invalid_request"),
                "{message}"
            );
            assert_eq!(error.message, message);
        }

        for (messages, message) in [
            (
                json!([{"role":"user","content":"x","tool_calls":[{"id":"call-a","type":"function","function":{"name":"read","arguments":{}}}]}]),
                "Only assistant messages can contain tool_calls.",
            ),
            (
                json!([{"role":"tool","content":"x"}]),
                "Tool messages require a tool_call_id.",
            ),
            (
                json!([{"role":"tool","tool_call_id":"","content":"x"}]),
                "Tool message tool_call_id must be a non-empty string.",
            ),
        ] {
            let error = request_error(service.clone(), openai_request(messages), Api::OpenAi).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "invalid_request"),
                "{message}"
            );
            assert_eq!(error.message, message);
        }
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        // A well-formed declaration and tool history still reaches admission.
        let value = json!({
            "model":"test-model","max_tokens":4,
            "tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],
            "messages":[
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call-a","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}]},
                {"role":"tool","tool_call_id":"call-a","content":"file"}
            ]
        });
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.message, "Test coordinator observed admission.");
        let admitted = observed
            .try_recv()
            .expect("A valid tool history must reach admission.");
        assert_eq!(admitted.messages[1]["content"], "file");
        assert_eq!(
            admitted.messages[0]["tool_calls"][0]["function"]["arguments"],
            json!({"path":"a"})
        );

        // An object arguments value stays accepted as well.
        let value = json!({
            "model":"test-model","max_tokens":4,
            "messages":[
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call-b","type":"function","function":{"name":"read","arguments":{"path":"b"}}} ]},
                {"role":"tool","tool_call_id":"call-b","content":"file"}
            ]
        });
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.message, "Test coordinator observed admission.");
        let admitted = observed
            .try_recv()
            .expect("An object arguments value must reach admission.");
        assert_eq!(
            admitted.messages[0]["tool_calls"][0]["function"]["arguments"],
            json!({"path":"b"})
        );
    }

    #[tokio::test]
    async fn strict_tools_fail_before_admission() {
        let (service, mut observed) = gateway_test_service();
        let mut value = openai_request(json!([{"role": "user", "content": "hello"}]));
        value["tools"] = json!([{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object"}, "strict": true
        }}]);
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.status, 400);
        assert_eq!(error.code, "unsupported_feature");
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        assert!(service.available.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn duplicate_keys_inside_argument_strings_fail_before_admission() {
        let (service, mut observed) = gateway_test_service();
        for arguments in [
            r#"{"path":"A","path":"B"}"#,
            r#"{"config":{"mode":1,"mode":2}}"#,
        ] {
            let value = openai_request(json!([{
                "role": "assistant", "content": null,
                "tool_calls": [{"id": "call-a", "type": "function", "function": {"name": "read", "arguments": arguments}}]
            }]));
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(error.status, 400);
            assert!(error.message.contains("duplicate keys"));
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
        assert!(service.available.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn openai_content_part_arrays_reach_admission_as_text() {
        let (service, mut observed) = gateway_test_service();
        let value = json!({"model":"test-model","max_tokens":4,"messages":[
            {"role":"user","content":[{"type":"text","text":"hello "},{"type":"text","text":"world"}]}
        ]});
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.message, "Test coordinator observed admission.");
        let admitted = observed
            .try_recv()
            .expect("Content parts must reach admission.");
        assert_eq!(
            admitted.messages,
            vec![json!({"role":"user","content":"hello world"})]
        );

        // Tool and assistant content arrays keep their tool identity.
        let value = json!({"model":"test-model","max_tokens":4,"messages":[
            {"role":"assistant","content":[{"type":"text","text":"calling"}],"tool_calls":[
                {"id":"call-a","type":"function","function":{"name":"read","arguments":{}}}]},
            {"role":"tool","tool_call_id":"call-a","content":[{"type":"text","text":"file "},{"type":"text","text":"text"}]}
        ]});
        let error = request_error(service.clone(), value, Api::OpenAi).await;
        assert_eq!(error.message, "Test coordinator observed admission.");
        let admitted = observed
            .try_recv()
            .expect("Tool content parts must reach admission.");
        assert_eq!(admitted.messages[0]["content"], "calling");
        assert_eq!(
            admitted.messages[1],
            json!({"role":"tool","tool_call_id":"call-a","content":"file text"})
        );

        // Unknown part types and images stay explicit request errors.
        for content in [
            json!([{"type":"input_audio","input_audio":{"data":"x","format":"wav"}}]),
            json!([{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]),
        ] {
            let value = json!({"model":"test-model","max_tokens":4,"messages":[{"role":"user","content":content}]});
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "unsupported_feature")
            );
        }
        assert!(matches!(
            observed.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test]
    async fn openai_tool_history_preserves_typed_arguments() {
        let (service, _) = gateway_test_service();
        let value = openai_request(json!([
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"001\",\"count\":2,\"enabled\":false}"}}
            ]},
            {"role": "tool", "tool_call_id": "call-a", "content": "file text"}
        ]));
        let (request, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(request.messages[0]["content"], "");
        assert_eq!(
            request.messages[0]["tool_calls"][0]["function"]["arguments"],
            json!({"path": "001", "count": 2, "enabled": false})
        );
        assert_eq!(
            request.messages[1],
            json!({"role": "tool", "tool_call_id": "call-a", "content": "file text"})
        );
    }

    #[tokio::test]
    async fn anthropic_history_preserves_error_results_and_message_order() {
        let (service, _) = gateway_test_service();
        let value = json!({
            "model": "test-model", "max_tokens": 4,
            "system": [{"type": "text", "text": "Read local files."}],
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Inspect both files."},
                    {"type": "tool_use", "id": "call-a", "name": "read", "input": {"path": "A"}},
                    {"type": "tool_use", "id": "call-b", "name": "read", "input": {"path": "B"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call-a", "is_error": true,
                        "content": [{"type": "text", "text": "Access denied."}]},
                    {"type": "tool_result", "tool_use_id": "call-b", "is_error": false, "content": "File B text."},
                    {"type": "text", "text": "Continue with file B."}
                ]}
            ]
        });
        let (request, _) = normalize(&value, Api::Anthropic, false, &service).unwrap();
        assert_eq!(request.messages.len(), 5);
        assert_eq!(
            request.messages[0],
            json!({"role": "system", "content": "Read local files."})
        );
        assert_eq!(request.messages[1]["content"], "Inspect both files.");
        assert_eq!(
            request.messages[1]["tool_calls"][0],
            json!({"id": "call-a", "type": "function", "function": {"name": "read", "arguments": {"path": "A"}}})
        );
        assert_eq!(request.messages[1]["tool_calls"][1]["id"], "call-b");
        assert_eq!(
            request.messages[2],
            json!({"role": "tool", "tool_call_id": "call-a", "content": "[tool_result is_error=true]\nAccess denied."})
        );
        assert_eq!(
            request.messages[3],
            json!({"role": "tool", "tool_call_id": "call-b", "content": "File B text."})
        );
        assert_eq!(
            request.messages[4],
            json!({"role": "user", "content": "Continue with file B."})
        );
    }

    #[tokio::test]
    async fn anthropic_invalid_result_order_and_error_flag_fail_before_admission() {
        let (service, mut observed) = gateway_test_service();
        for content in [
            json!([{"type": "text", "text": "Continue."}, {"type": "tool_result", "tool_use_id": "call-a", "content": "text"}]),
            json!([{"type": "tool_result", "tool_use_id": "call-a", "is_error": "true", "content": "text"}]),
        ] {
            let value = json!({"model": "test-model", "max_tokens": 4, "messages": [{"role": "user", "content": content}]});
            let error = request_error(service.clone(), value, Api::Anthropic).await;
            assert_eq!(error.status, 400);
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
    }

    fn completed(cause: &str) -> ModelEvent {
        ModelEvent::Done {
            cause: cause.into(),
            input_tokens: 7,
            output_tokens: 9,
            matched_stop: None,
        }
    }

    async fn collect_stream(api: Api, events: Vec<ModelEvent>) -> (String, Arc<AtomicBool>) {
        collect_stream_with_usage(api, events, false).await
    }

    async fn collect_stream_with_usage(
        api: Api,
        events: Vec<ModelEvent>,
        include_usage: bool,
    ) -> (String, Arc<AtomicBool>) {
        let (sender, receiver) = mpsc::channel(32);
        for event in events {
            assert!(sender.try_send(event).is_ok());
        }
        drop(sender);
        let cancelled = Arc::new(AtomicBool::new(false));
        let stream = PublicStream::new(
            receiver,
            CancelGuard(cancelled.clone()),
            api,
            Begin {
                request_id: "request-a".into(),
                input_tokens: 7,
            },
            "test-model".into(),
            include_usage,
        );
        let body = Sse::new(stream).into_response().into_body();
        let bytes = tokio::time::timeout(Duration::from_secs(1), axum::body::to_bytes(body, 65536))
            .await
            .expect("The stream must end.")
            .unwrap();
        (String::from_utf8(bytes.to_vec()).unwrap(), cancelled)
    }

    fn decode_sse(text: &str) -> Vec<(String, Value)> {
        text.split("\n\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                let name = block
                    .lines()
                    .find_map(|line| line.strip_prefix("event:"))
                    .unwrap_or("")
                    .trim()
                    .to_owned();
                let data = block
                    .lines()
                    .find_map(|line| line.strip_prefix("data:"))
                    .expect("SSE data is required.")
                    .trim();
                let value = if data == "[DONE]" {
                    json!(data)
                } else {
                    serde_json::from_str(data).unwrap()
                };
                (name, value)
            })
            .collect()
    }

    #[tokio::test]
    async fn anthropic_stream_closes_each_block_before_the_final_result() {
        let (text, cancelled) = collect_stream(
            Api::Anthropic,
            vec![
                ModelEvent::Part(Part::Text("A".into())),
                ModelEvent::Part(Part::Text("B".into())),
                ModelEvent::Part(Part::Tool {
                    id: "call-a".into(),
                    name: "read".into(),
                    input: json!({"path": "A"}),
                }),
                ModelEvent::Part(Part::Tool {
                    id: "call-b".into(),
                    name: "read".into(),
                    input: json!({"path": "B"}),
                }),
                ModelEvent::Part(Part::Text("C".into())),
                completed("tool_calls"),
            ],
        )
        .await;
        let events = decode_sse(&text);
        let names: Vec<_> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        for (start, stop, index) in [(1, 4, 0), (5, 7, 1), (8, 10, 2), (11, 13, 3)] {
            assert_eq!(events[start].1["index"], index);
            assert_eq!(events[stop].1["index"], index);
        }
        assert_eq!(events[5].1["content_block"]["id"], "call-a");
        assert_eq!(events[8].1["content_block"]["id"], "call-b");
        let arguments: Value =
            serde_json::from_str(events[6].1["delta"]["partial_json"].as_str().unwrap()).unwrap();
        assert_eq!(arguments, json!({"path": "A"}));
        assert_eq!(
            events[0].1["message"]["usage"],
            json!({"input_tokens": 7, "output_tokens": 0})
        );
        assert_eq!(events[14].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(events[14].1["usage"]["output_tokens"], 9);
        assert!(cancelled.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn openai_stream_has_stable_tool_indices_and_one_final_result() {
        let (text, _) = collect_stream_with_usage(
            Api::OpenAi,
            vec![
                ModelEvent::Part(Part::Text("Inspect.".into())),
                ModelEvent::Part(Part::Tool {
                    id: "call-a".into(),
                    name: "read".into(),
                    input: json!({"path": "A"}),
                }),
                ModelEvent::Part(Part::Tool {
                    id: "call-b".into(),
                    name: "read".into(),
                    input: json!({"path": "B"}),
                }),
                completed("tool_calls"),
            ],
            true,
        )
        .await;
        let events = decode_sse(&text);
        assert_eq!(events.len(), 7);
        assert_eq!(events[0].1["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            events[2].1["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );
        assert_eq!(
            events[3].1["choices"][0]["delta"]["tool_calls"][0]["index"],
            1
        );
        assert_eq!(events[4].1["choices"][0]["finish_reason"], "tool_calls");
        assert!(events[..5]
            .iter()
            .all(|(_, value)| value.get("usage") == Some(&Value::Null)));
        assert_eq!(events[5].1["choices"], json!([]));
        assert_eq!(
            events[5].1["usage"],
            json!({"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16})
        );
        assert_eq!(events[5].1["id"], events[0].1["id"]);
        assert_eq!(events[6].1, "[DONE]");
        assert!(events[..4]
            .iter()
            .all(|(_, value)| value["choices"][0]["finish_reason"].is_null()));
    }

    #[tokio::test]
    async fn openai_stream_omits_usage_when_not_requested() {
        let (text, _) = collect_stream(
            Api::OpenAi,
            vec![
                ModelEvent::Part(Part::Text("Answer.".into())),
                completed("length"),
            ],
        )
        .await;
        let events = decode_sse(&text);
        assert_eq!(events.len(), 4);
        assert!(events.iter().all(|(_, value)| value.get("usage").is_none()));
        assert!(events[..3]
            .iter()
            .all(|(_, value)| value["choices"].as_array().unwrap().len() == 1));
        assert_eq!(events[2].1["choices"][0]["finish_reason"], "length");
        assert_eq!(events[3].1, "[DONE]");
    }

    #[tokio::test]
    async fn stream_options_require_streaming_and_a_boolean_usage_flag() {
        let (service, mut observed) = gateway_test_service();
        for (stream, options) in [
            (false, json!({"include_usage": true})),
            (true, json!({"include_usage": 1})),
        ] {
            let mut value = openai_request(json!([{"role": "user", "content": "hello"}]));
            value["stream"] = json!(stream);
            value["stream_options"] = options;
            let error = request_error(service.clone(), value, Api::OpenAi).await;
            assert_eq!(error.status, 400);
            assert!(matches!(
                observed.try_recv(),
                Err(mpsc::error::TryRecvError::Empty)
            ));
        }
        for options in [
            None,
            Some(json!({})),
            Some(json!({"include_usage": false})),
            Some(json!({"include_usage": true})),
        ] {
            let mut value = openai_request(json!([{"role": "user", "content": "hello"}]));
            value["stream"] = json!(true);
            if let Some(options) = options {
                value["stream_options"] = options;
            }
            assert!(normalize(&value, Api::OpenAi, false, &service).unwrap().1);
        }
    }

    #[tokio::test]
    async fn stream_error_suppresses_a_queued_success_result() {
        for api in [Api::Anthropic, Api::OpenAi] {
            let (text, _) = collect_stream_with_usage(
                api,
                vec![
                    ModelEvent::Part(Part::Text("Partial text.".into())),
                    ModelEvent::Error(ApiError::new(500, "internal_error", "Execution failed.")),
                    completed("eos"),
                ],
                true,
            )
            .await;
            let events = decode_sse(&text);
            assert_eq!(events.iter().filter(|(name, _)| name == "error").count(), 1);
            assert_eq!(events.last().unwrap().1["error"]["code"], "internal_error");
            assert!(!events.iter().any(|(name, value)| name == "message_stop"
                || name == "message_delta"
                || value == "[DONE]"));
            assert!(events
                .iter()
                .all(|(_, value)| value["choices"][0]["finish_reason"].is_null()));
            if api == Api::OpenAi {
                assert!(events.iter().all(|(_, value)| !value["usage"].is_object()));
            }
        }
    }

    #[tokio::test]
    async fn stream_eof_without_terminal_is_one_worker_error() {
        for api in [Api::Anthropic, Api::OpenAi] {
            let (text, _) = collect_stream(
                api,
                vec![ModelEvent::Part(Part::Text("Partial text.".into()))],
            )
            .await;
            let events = decode_sse(&text);
            assert_eq!(events.last().unwrap().0, "error");
            assert_eq!(events.last().unwrap().1["error"]["code"], "worker_lost");
            assert_eq!(events.iter().filter(|(name, _)| name == "error").count(), 1);
            assert!(!text.contains("message_stop") && !text.contains("[DONE]"));
        }
    }

    #[tokio::test]
    async fn dropping_an_unpolled_stream_cancels_the_request() {
        let (_sender, receiver) = mpsc::channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        let stream = PublicStream::new(
            receiver,
            CancelGuard(cancelled.clone()),
            Api::Anthropic,
            Begin {
                request_id: "request-a".into(),
                input_tokens: 7,
            },
            "test-model".into(),
            false,
        );
        assert!(!cancelled.load(Ordering::Acquire));
        drop(stream);
        assert!(cancelled.load(Ordering::Acquire));
    }
}
