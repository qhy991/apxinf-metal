use std::io::Write;
use std::path::Path;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_loader::safetensors;
use apxinf_model::{GeneralQwen35, LlmInput, LlmTrait, Qwen35Config};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};

use crate::assets::{self, Assets};
use crate::protocol::{validate_context, validate_prompt, Request, MAX_CONTEXT};
use crate::Result;

const FORMAT: &str = "apxinf-qwen35-specialized-v1";
const PROFILE: &str = "qwen35-0.8b-f32-metal-w8-head-mlp-v1";
const TOKENIZER_EOS: u32 = 248046;

fn fixed_tokenizer_eos(eos: Option<u32>) -> Result<u32> {
    match eos {
        Some(TOKENIZER_EOS) => Ok(TOKENIZER_EOS),
        _ => Err("This profile requires tokenizer EOS <|im_end|> (248046)".into()),
    }
}

pub struct Runtime {
    model: GeneralQwen35,
    tokenizer: Tokenizer,
    eos_token: u32,
    pub startup: Value,
}

impl Runtime {
    pub fn load(directory: &Path) -> Result<Self> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Err("This profile requires macOS on Apple Silicon".into());
        }
        let asset_started = Instant::now();
        let assets = Assets::load(directory)?;
        let asset_check_ms = milliseconds(asset_started.elapsed());

        let tokenizer_started = Instant::now();
        let tokenizer = Tokenizer::from_bytes(
            &assets.tokenizer,
            Some(&assets.tokenizer_config),
            Some(&assets.chat_template),
        )?;
        let tokenizer_load_ms = milliseconds(tokenizer_started.elapsed());

        let model_started = Instant::now();
        let config = Qwen35Config::from_json_str(std::str::from_utf8(&assets.config)?)?;
        let eos_token = fixed_tokenizer_eos(tokenizer.eos_token_id())?;
        if config.text.n_layers != 24 || !config.text.tie_word_embeddings {
            return Err("This profile requires 24 layers and tied embeddings".into());
        }
        let (weights, _) = safetensors::load_native_file_filtered(&assets.weights, |name| {
            name.starts_with("model.language_model.") || name == "lm_head.weight"
        })?;
        let model = GeneralQwen35::from_weights_with_metal_w8_mlp_blocks_and_lm_head(
            config,
            weights,
            Device::Cpu,
            MAX_CONTEXT,
        )?;
        let model_load_ms = milliseconds(model_started.elapsed());
        Ok(Self {
            model,
            tokenizer,
            eos_token,
            startup: json!({"asset_check_ms": asset_check_ms, "tokenizer_load_ms": tokenizer_load_ms, "model_load_ms": model_load_ms}),
        })
    }

    pub fn ready(&self) -> Value {
        json!({
            "format": FORMAT,
            "kind": "ready",
            "profile": PROFILE,
            "model_revision": assets::MODEL_REVISION,
            "asset_identity": assets::identity(),
            "max_context": MAX_CONTEXT,
            "precision": {
                "body": "CPU/Accelerate F32",
                "head": "Metal W8 top-4 with F32 rerank",
                "decode_mlp": "Metal W8 in all 24 layers",
            },
            "startup": self.startup,
        })
    }

    pub fn request(&mut self, request: &Request) -> Result<Value> {
        Ok(generate_request(
            &mut self.model,
            &self.tokenizer,
            self.eos_token,
            request,
            |_| Ok(()),
        )?
        .json())
    }

    pub fn request_plain(&mut self, request: &Request, output: &mut impl Write) -> Result<()> {
        let mut decoder = self.tokenizer.decode_stream();
        let completed = generate_request(
            &mut self.model,
            &self.tokenizer,
            self.eos_token,
            request,
            |token| {
                if let Some(text) = decoder.step(token)? {
                    output.write_all(text.as_bytes())?;
                    output.flush()?;
                }
                Ok(())
            },
        )?;
        let tail = decoder.finish(&completed.generated_token_ids)?;
        output.write_all(tail.as_bytes())?;
        output.write_all(b"\n")?;
        output.flush()?;
        Ok(())
    }
}

struct Completed {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    text: String,
    stop_reason: &'static str,
    max_tokens: usize,
    request_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
    total_request_ms: f64,
    generation_path_receipt: Value,
}

impl Completed {
    fn json(self) -> Value {
        let decode_tps = if self.generated_token_ids.len() > 1 && self.decode_ms > 0.0 {
            Some((self.generated_token_ids.len() - 1) as f64 * 1000.0 / self.decode_ms)
        } else {
            None
        };
        json!({
            "format": FORMAT,
            "kind": "result",
            "profile": PROFILE,
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "text": self.text,
            "stop_reason": self.stop_reason,
            "max_tokens": self.max_tokens,
            "timing": {
                "request_ms": self.request_ms,
                "ttft_ms": self.ttft_ms,
                "decode_ms": self.decode_ms,
                "decode_tps": decode_tps,
                "total_request_ms": self.total_request_ms,
            },
            "generation_path_receipt": self.generation_path_receipt,
        })
    }
}

fn generate_request(
    model: &mut impl LlmTrait,
    tokenizer: &Tokenizer,
    eos_token: u32,
    request: &Request,
    mut on_token: impl FnMut(u32) -> Result<()>,
) -> Result<Completed> {
    let total_started = Instant::now();
    validate_prompt(&request.prompt)?;
    let text = tokenizer.apply_chat_template(&[ChatMessage::user(&request.prompt)])?;
    let prompt_token_ids = tokenizer.encode(&text)?;
    validate_context(prompt_token_ids.len(), request.max_tokens)?;
    let mut first_token = None;
    let mut callback_count = 0;
    let mut output_error = None;
    let started = Instant::now();
    model.reset();
    let generated = model.generate_streaming(
        LlmInput::text(&prompt_token_ids),
        request.max_tokens,
        |token| {
            first_token.get_or_insert_with(Instant::now);
            callback_count += 1;
            if output_error.is_none() {
                if let Err(error) = on_token(token) {
                    output_error = Some(error);
                }
            }
        },
        Some(eos_token),
    );
    let ended = Instant::now();
    if let Some(error) = output_error {
        return Err(error);
    }
    let (generated_token_ids, _) = generated?;
    if generated_token_ids.is_empty() || callback_count != generated_token_ids.len() {
        return Err("Generation returned an inconsistent token callback count".into());
    }
    let first_token = first_token.ok_or("Generation returned no first-token callback")?;
    let stop_reason = if generated_token_ids.last() == Some(&eos_token) {
        "eos"
    } else if generated_token_ids.len() == request.max_tokens {
        "max_tokens"
    } else {
        return Err("Generation stopped without EOS or its token budget".into());
    };
    let text = tokenizer.decode(&generated_token_ids)?;
    let generation_path_receipt = model
        .generation_path_receipt()
        .ok_or("Generation returned no path receipt")?;
    Ok(Completed {
        prompt_token_ids,
        generated_token_ids,
        text,
        stop_reason,
        max_tokens: request.max_tokens,
        request_ms: milliseconds(ended.duration_since(started)),
        ttft_ms: milliseconds(first_token.duration_since(started)),
        decode_ms: milliseconds(ended.duration_since(first_token)),
        total_request_ms: milliseconds(total_started.elapsed()),
        generation_path_receipt,
    })
}

fn milliseconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::{Error, Tensor};
    use apxinf_loader::ModelConfig;
    use std::collections::HashMap;

    #[derive(Default)]
    struct TestModel {
        resets: usize,
        position: usize,
        failure_at: Option<usize>,
    }

    impl LlmTrait for TestModel {
        fn load(
            _: ModelConfig,
            _: HashMap<String, Tensor>,
            _: Device,
        ) -> apxinf_core::Result<Self> {
            Err(Error::Other("Test model has no weight loader".into()))
        }
        fn forward(&mut self, _: &[u32], _: u32) -> apxinf_core::Result<Tensor> {
            Err(Error::Other(
                "Test model requires its direct-token hooks".into(),
            ))
        }
        fn prefill_token_for_generation(
            &mut self,
            _: LlmInput<'_>,
        ) -> Option<apxinf_core::Result<u32>> {
            if self.failure_at == Some(0) {
                return Some(Err(Error::Other("Prefill failed".into())));
            }
            if self.position != 0 {
                return Some(Err(Error::Other("State was not reset".into())));
            }
            self.position = 1;
            Some(Ok(2))
        }
        fn decode_token(&mut self, _: u32, _: u32) -> Option<apxinf_core::Result<u32>> {
            if self.failure_at == Some(self.position) {
                return Some(Err(Error::Other("Decode failed".into())));
            }
            self.position += 1;
            Some(Ok(3))
        }
        fn reset(&mut self) {
            self.position = 0;
            self.resets += 1;
        }
        fn vocab_size(&self) -> usize {
            4
        }
        fn generation_path_receipt(&self) -> Option<Value> {
            Some(json!({"test_position": self.position}))
        }
    }

    fn tokenizer() -> Tokenizer {
        let bytes = serde_json::to_vec(&json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hello": 1, "answer": 2, "<eos>": 3}, "unk_token": "[UNK]"},
        })).unwrap();
        Tokenizer::from_bytes(
            &bytes,
            Some(br#"{"chat_template":"{{ messages[0].content }}","eos_token_id":3}"#),
            None,
        )
        .unwrap()
    }

    #[test]
    fn generation_uses_tokenizer_eos_when_model_configuration_eos_differs() {
        let model_configuration_eos = 248044;
        let tokenizer_eos = Some(248046);
        let generation_eos = fixed_tokenizer_eos(tokenizer_eos).unwrap();
        assert_eq!(generation_eos, 248046);
        assert_ne!(generation_eos, model_configuration_eos);
        assert!(fixed_tokenizer_eos(None).is_err());
        assert!(fixed_tokenizer_eos(Some(model_configuration_eos)).is_err());
    }

    #[test]
    fn repeated_requests_reset_state_and_report_eos_and_budget_stops() {
        let mut model = TestModel::default();
        let tokenizer = tokenizer();
        let request = Request {
            prompt: "hello".into(),
            max_tokens: 4,
        };
        let first = generate_request(&mut model, &tokenizer, 3, &request, |_| Ok(())).unwrap();
        let second = generate_request(&mut model, &tokenizer, 3, &request, |_| Ok(())).unwrap();
        assert_eq!(model.resets, 2);
        assert_eq!(first.generated_token_ids, [2, 3]);
        assert_eq!(first.generated_token_ids, second.generated_token_ids);
        assert_eq!(second.stop_reason, "eos");
        assert!(second.request_ms >= second.ttft_ms);
        assert!(second.total_request_ms >= second.request_ms);
        let budget = Request {
            prompt: "hello".into(),
            max_tokens: 1,
        };
        let result = generate_request(&mut model, &tokenizer, 3, &budget, |_| Ok(()))
            .unwrap()
            .json();
        assert_eq!(result["stop_reason"], "max_tokens");
        assert!(result["timing"]["decode_tps"].is_null());
    }

    #[test]
    fn context_overflow_fails_before_model_reset() {
        let mut model = TestModel::default();
        let request = Request {
            prompt: "hello ".repeat(2048),
            max_tokens: 1,
        };
        assert!(generate_request(&mut model, &tokenizer(), 3, &request, |_| Ok(())).is_err());
        assert_eq!(model.resets, 0);
    }

    #[test]
    fn output_errors_are_returned_and_stop_later_output_callbacks() {
        let mut model = TestModel::default();
        let mut callbacks = 0;
        let request = Request {
            prompt: "hello".into(),
            max_tokens: 4,
        };
        let error = generate_request(&mut model, &tokenizer(), 3, &request, |_| {
            callbacks += 1;
            Err("Output is closed".into())
        })
        .err()
        .unwrap();
        assert_eq!(callbacks, 1);
        assert_eq!(error.to_string(), "Output is closed");
    }

    #[test]
    fn inference_errors_before_and_after_first_token_never_return_a_completed_request() {
        for (failure_at, expected_callbacks, message) in
            [(0, 0, "Prefill failed"), (1, 1, "Decode failed")]
        {
            let mut model = TestModel {
                failure_at: Some(failure_at),
                ..TestModel::default()
            };
            let mut callbacks = 0;
            let request = Request {
                prompt: "hello".into(),
                max_tokens: 4,
            };
            let error = generate_request(&mut model, &tokenizer(), 3, &request, |_| {
                callbacks += 1;
                Ok(())
            })
            .err()
            .unwrap();
            assert!(error.to_string().contains(message));
            assert_eq!(callbacks, expected_callbacks);
        }
    }
}
