use super::{array, content_text, text, unknown, ApiError};
use crate::owned_media::{MediaBatch, MediaError, OwnedPng};
use serde_json::{json, Value};

fn media_error(error: MediaError) -> ApiError {
    ApiError::new(400, error.code, error.message)
}

pub(super) fn image_part(block: &Value) -> Result<Value, ApiError> {
    unknown(block, &["type", "source", "cache_control"], "Image block")?;
    let image = OwnedPng::from_source(&block["source"]).map_err(media_error)?;
    Ok(json!({"type":"image","media":image.descriptor()}))
}

/// Convert an OpenAI `image_url` part into the owned PNG descriptor.
pub(super) fn openai_image_part(block: &Value) -> Result<Value, ApiError> {
    unknown(block, &["type", "image_url"], "Image URL block")?;
    let url = &block["image_url"];
    unknown(url, &["url", "detail"], "image_url")?;
    if let Some(detail) = url.get("detail") {
        if !matches!(detail.as_str(), Some("auto" | "low" | "high")) {
            return Err(ApiError::invalid(
                "image_url detail must be auto, low, or high.",
            ));
        }
    }
    let url = text(&url["url"], "image_url url")?;
    let unsupported = || {
        ApiError::new(
            400,
            "unsupported_feature",
            "OpenAI images require a base64 PNG data URL.",
        )
    };
    let data = url.strip_prefix("data:").ok_or_else(unsupported)?;
    let (media_type, encoded) = data.split_once(";base64,").ok_or_else(unsupported)?;
    if media_type != "image/png" {
        return Err(ApiError::new(
            400,
            "unsupported_feature",
            "This profile accepts only PNG images.",
        ));
    }
    let source = json!({"type":"base64","media_type":media_type,"data":encoded});
    let image = OwnedPng::from_source(&source).map_err(media_error)?;
    Ok(json!({"type":"image","media":image.descriptor()}))
}

pub(super) fn tool_content(block: &Value) -> Result<Value, ApiError> {
    let content = block.get("content");
    let prefix = "[tool_result is_error=true]\n";
    if content.is_none_or(Value::is_string) {
        let value = content.map(content_text).transpose()?.unwrap_or_default();
        return Ok(if block["is_error"] == true {
            json!([{"type":"text","text":prefix},{"type":"text","text":value}])
        } else {
            json!(value)
        });
    }
    let mut parts = Vec::new();
    if block["is_error"] == true {
        parts.push(json!({"type":"text","text":prefix}));
    }
    for part in array(content.unwrap(), "Tool result content")? {
        match part["type"].as_str() {
            Some("text") => {
                unknown(part, &["type", "text", "cache_control"], "Text block")?;
                parts.push(json!({"type":"text","text":text(&part["text"], "Text block text")?}));
            }
            Some("image") => parts.push(image_part(part)?),
            _ => {
                return Err(ApiError::new(
                    400,
                    "unsupported_feature",
                    "Tool results support text and PNG images.",
                ))
            }
        }
    }
    if parts.is_empty() {
        Ok(json!(""))
    } else {
        Ok(json!(parts))
    }
}

pub(super) fn check_media_limits(messages: &[Value]) -> Result<(), ApiError> {
    let media: Vec<_> = messages
        .iter()
        .filter_map(|message| message["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "image")
        .map(|part| part["media"].clone())
        .collect();
    MediaBatch::from_descriptors(&json!(media)).map_err(media_error)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{handle, normalize, ready, Api};
    use super::*;
    use crate::{contracts, sampling::Selection, supervisor::gateway_test_service_with_protocol};
    use axum::{
        body::{to_bytes, Bytes},
        extract::State,
        http::HeaderMap,
    };
    use base64::{engine::general_purpose::STANDARD, Engine};

    fn descriptor(name: &str) -> Value {
        let corpus: Value = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/serving/owned-media-v0.1.json"
        ))
        .unwrap();
        corpus["images"][name].clone()
    }

    fn image(name: &str) -> Value {
        json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":descriptor(name)["data_base64"]}})
    }

    fn request(messages: Value) -> Value {
        json!({"model":"test-model","max_tokens":2,"messages":messages})
    }

    #[tokio::test]
    async fn history_preserves_text_images_tool_ids_and_error_prefix() {
        let (service, _) = gateway_test_service_with_protocol(contracts::VISION_PROTOCOL);
        let value = request(json!([
            {"role":"user","content":[{"type":"text","text":"before"},image("rgb"),{"type":"text","text":"after"}]},
            {"role":"assistant","content":[{"type":"tool_use","id":"call_a","name":"draw","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_a","is_error":true,"content":[image("rgba"),{"type":"text","text":"failed"},image("rgb")]},{"type":"text","text":"retry"}]}
        ]));
        let (normalized, _) = normalize(&value, Api::Anthropic, false, &service).unwrap();
        assert_eq!(
            normalized.messages[0]["content"],
            json!([{"type":"text","text":"before"},{"type":"image","media":descriptor("rgb")},{"type":"text","text":"after"}])
        );
        assert_eq!(
            normalized.messages[2],
            json!({"role":"tool","tool_call_id":"call_a","content":[{"type":"text","text":"[tool_result is_error=true]\n"},{"type":"image","media":descriptor("rgba")},{"type":"text","text":"failed"},{"type":"image","media":descriptor("rgb")}]})
        );
        assert_eq!(
            normalized.messages[3],
            json!({"role":"user","content":[{"type":"text","text":"retry"}]})
        );
        assert!(matches!(
            normalized.selection,
            Selection::Random {
                temperature: 1.0,
                ..
            }
        ));
        let text_error = tool_content(&json!({"content":"failed","is_error":true})).unwrap();
        assert_eq!(
            text_error,
            json!([{"type":"text","text":"[tool_result is_error=true]\n"},{"type":"text","text":"failed"}])
        );
    }

    #[tokio::test]
    async fn image_shapes_and_profile_isolation_fail_before_admission() {
        let (vision, mut observed) = gateway_test_service_with_protocol(contracts::VISION_PROTOCOL);
        let mut bad_mime = image("rgb");
        bad_mime["source"]["media_type"] = json!("image/jpeg");
        let mut bad_png = image("rgb");
        bad_png["source"]["data"] = json!("aW52YWxpZA==");
        let cases = [
            (
                Api::Anthropic,
                request(json!([{"role":"assistant","content":[image("rgb")]}])),
            ),
            (
                Api::Anthropic,
                request(json!([{"role":"user","content":[bad_mime]}])),
            ),
            (
                Api::Anthropic,
                request(json!([{"role":"user","content":[bad_png]}])),
            ),
            (
                Api::Anthropic,
                request(
                    json!([{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.test/image.png"}}]}]),
                ),
            ),
            (
                Api::OpenAi,
                request(json!([{"role":"user","content":[image("rgb")]}])),
            ),
            (
                Api::OpenAi,
                request(
                    json!([{"role":"tool","tool_call_id":"a","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,invalid"}}]}]),
                ),
            ),
        ];
        for (api, value) in cases {
            let error = handle(
                vision.clone(),
                HeaderMap::new(),
                Bytes::from(value.to_string()),
                api,
                false,
            )
            .await
            .err()
            .unwrap();
            assert_eq!(error.status, 400);
            assert!(observed.try_recv().is_err());
        }
        let mut system = request(json!([{"role":"user","content":"hello"}]));
        system["system"] = json!([image("rgb")]);
        assert!(normalize(&system, Api::Anthropic, false, &vision).is_err());
        for protocol in [
            "apxinf-worker/2.0",
            contracts::NATIVE_PROTOCOL,
            contracts::NATIVE_SAMPLED_PROTOCOL,
            contracts::NATIVE_THINKING_PROTOCOL,
            contracts::NATIVE_BOUNDED_THINKING_PROTOCOL,
        ] {
            let (service, _) = gateway_test_service_with_protocol(protocol);
            assert!(normalize(
                &request(json!([{"role":"user","content":[image("rgb")]}])),
                Api::Anthropic,
                false,
                &service
            )
            .is_err());
            let (text, _) = normalize(&request(json!([{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}])), Api::Anthropic, false, &service).unwrap();
            assert_eq!(text.messages[0]["content"], "ab");
        }
    }

    fn openai_image(name: &str) -> Value {
        json!({"type":"image_url","image_url":{"url":format!(
            "data:image/png;base64,{}",
            descriptor(name)["data_base64"].as_str().unwrap()
        )}})
    }

    #[tokio::test]
    async fn openai_image_parts_keep_order_and_the_contract_shape() {
        let (service, _) = gateway_test_service_with_protocol(contracts::VISION_PROTOCOL);
        let value = request(json!([
            {"role":"user","content":[{"type":"text","text":"before"},openai_image("rgb"),{"type":"text","text":"after"}]}
        ]));
        let (normalized, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(
            normalized.messages[0]["content"],
            json!([{"type":"text","text":"before"},{"type":"image","media":descriptor("rgb")},{"type":"text","text":"after"}])
        );
        // The gateway encodes this preparation frame before admission.
        let preparation = json!({
            "protocol": service.ready["protocol"],
            "kind": "prepare_input",
            "worker_epoch": service.ready["worker_epoch"],
            "command_id": uuid::Uuid::new_v4().to_string(),
            "model_revision": service.ready["model_revision"],
            "messages": normalized.messages,
            "tools": normalized.tools,
            "template_options": {"enable_thinking": false},
        });
        assert!(contracts::encode_frame(&preparation).is_ok());

        let png = |name| descriptor(name)["data_base64"].as_str().unwrap().to_owned();
        for (part, code) in [
            (
                json!({"type":"image_url","image_url":{"url":"https://example.test/image.png"}}),
                "unsupported_feature",
            ),
            (
                json!({"type":"image_url","image_url":{"url":format!("data:image/jpeg;base64,{}", png("rgb"))}}),
                "unsupported_feature",
            ),
            (
                json!({"type":"image_url","image_url":{"url":"data:image/png;base64,aW52YWxpZA=="}}),
                "invalid_request",
            ),
            (
                json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}", png("rgb")),"detail":"huge"}}),
                "invalid_request",
            ),
        ] {
            let value = request(json!([{"role":"user","content":[part]}]));
            let error = normalize(&value, Api::OpenAi, false, &service)
                .err()
                .expect("The image part must fail.");
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, code),
                "{error:?}"
            );
        }

        // Images stay out of assistant and tool content.
        for message in [
            json!([{"role":"assistant","content":[openai_image("rgb")]}]),
            json!([{"role":"tool","tool_call_id":"a","content":[openai_image("rgb")]}]),
        ] {
            let error = normalize(&request(message), Api::OpenAi, false, &service)
                .err()
                .expect("The image role must fail.");
            assert_eq!(
                (error.status, error.code.as_str()),
                (400, "unsupported_feature")
            );
        }

        // Four images stay inside the declared media limit.
        let value = request(json!([{"role":"user","content":[
            openai_image("rgb"),openai_image("rgba"),openai_image("rgb"),openai_image("rgba")
        ]}]));
        let (normalized, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(
            normalized.messages[0]["content"].as_array().unwrap().len(),
            4
        );

        // A fifth image exceeds the same media limit on this route.
        let value = request(json!([{"role":"user","content":[
            openai_image("rgb"),openai_image("rgba"),openai_image("rgb"),
            openai_image("rgba"),openai_image("rgb")
        ]}]));
        let error = normalize(&value, Api::OpenAi, false, &service)
            .err()
            .expect("The fifth image must exceed the media limit.");
        assert_eq!(
            (error.status, error.code.as_str()),
            (400, "unsupported_feature")
        );
    }

    #[tokio::test]
    async fn openai_text_content_parts_keep_the_vision_order() {
        let (service, _) = gateway_test_service_with_protocol(contracts::VISION_PROTOCOL);
        let value = request(json!([
            {"role":"user","content":[{"type":"text","text":"before"},{"type":"text","text":"after"}]},
            {"role":"assistant","content":[{"type":"text","text":"reply "},{"type":"text","text":"text"}]}
        ]));
        let (normalized, _) = normalize(&value, Api::OpenAi, false, &service).unwrap();
        assert_eq!(
            normalized.messages[0]["content"],
            json!([{"type":"text","text":"before"},{"type":"text","text":"after"}])
        );
        assert_eq!(normalized.messages[1]["content"], "reply text");
    }

    #[tokio::test]
    async fn vision_profile_rejects_public_top_k_and_seed() {
        let (service, _) = gateway_test_service_with_protocol(contracts::VISION_PROTOCOL);
        let mut value = request(json!([{"role":"user","content":"hello"}]));
        value["top_k"] = json!(5);
        let error = normalize(&value, Api::Anthropic, false, &service)
            .err()
            .expect("The vision profile must reject top_k.");
        assert_eq!(
            (error.status, error.code.as_str()),
            (400, "unsupported_feature")
        );
        assert!(error.message.contains("top_k"), "{}", error.message);
        let mut value = request(json!([{"role":"user","content":"hello"}]));
        value["seed"] = json!(42);
        let error = normalize(&value, Api::OpenAi, false, &service)
            .err()
            .expect("The vision profile must reject seed.");
        assert_eq!(
            (error.status, error.code.as_str()),
            (400, "unsupported_feature")
        );
        assert!(error.message.contains("request ID"), "{}", error.message);
    }

    #[tokio::test]
    async fn image_limits_count_occurrences_across_history() {
        let (service, _) = gateway_test_service_with_protocol(contracts::VISION_PROTOCOL);
        let mut messages = vec![
            json!({"role":"user","content":[image("rgb"),image("rgba")]}),
            json!({"role":"user","content":[image("rgb"),image("rgba")]}),
        ];
        normalize(&request(json!(messages)), Api::Anthropic, false, &service).unwrap();
        messages.push(json!({"role":"user","content":[image("rgb")]}));
        assert_eq!(
            normalize(&request(json!(messages)), Api::Anthropic, false, &service)
                .err()
                .unwrap()
                .code,
            "unsupported_feature"
        );
        let mut png = STANDARD
            .decode(descriptor("rgb")["data_base64"].as_str().unwrap())
            .unwrap();
        let mut payload = vec![0; 87382 - png.len() - 12];
        payload[..14].copy_from_slice(&[73, 73, 42, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut chunk = (payload.len() as u32).to_be_bytes().to_vec();
        chunk.extend_from_slice(b"eXIf");
        chunk.extend_from_slice(&payload);
        chunk.extend_from_slice(&crc32fast::hash(&chunk[4..]).to_be_bytes());
        png.splice(33..33, chunk);
        let large = json!({"type":"image","source":{"type":"base64","media_type":"image/png","data":STANDARD.encode(png)}});
        let value = request(
            json!([{"role":"user","content":[large,large]},{"role":"user","content":[large]}]),
        );
        assert_eq!(
            normalize(&value, Api::Anthropic, false, &service)
                .err()
                .unwrap()
                .code,
            "unsupported_feature"
        );
    }

    #[tokio::test]
    async fn vision_readiness_names_restriction_and_qwen_tools() {
        let (service, _) = gateway_test_service_with_protocol(contracts::VISION_PROTOCOL);
        let response = ready(State(service)).await;
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 100000).await.unwrap()).unwrap();
        assert_eq!(body["profile"], "apxinf-minicpm-vision-v0.1");
        assert_eq!(body["public_capabilities"]["images"], true);
        assert_eq!(body["public_capabilities"]["sampling"], true);
        assert_eq!(body["public_capabilities"]["tools"], "qwen_xml_or_json");
        assert_eq!(
            body["public_capabilities"]["image_input_restriction"],
            "anthropic_base64_png_user_and_tool_result"
        );
        assert_eq!(
            body["public_capabilities"]["openai_image_input_restriction"],
            "openai_base64_png_user_only"
        );
        assert_eq!(
            body["public_capabilities"]["media_limits"],
            json!({"max_images":4,"max_image_bytes":131072,"max_total_bytes":262144,"max_dimension":1024,"max_host_array_bytes":67108864})
        );
    }
}
