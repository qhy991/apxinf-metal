//! Compare build selections through one unchanged native model entry point.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Device};
use apxinf_model::{AutoModel, LlmInput, LoadOptions};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

struct Settings {
    model: PathBuf,
    cases: PathBuf,
    output: PathBuf,
    warmups: usize,
    repeats: usize,
    max_tokens: usize,
}

fn settings() -> Result<Settings, Box<dyn Error>> {
    let mut model = None;
    let mut cases = None;
    let mut output = None;
    let mut warmups = 2;
    let mut repeats = 3;
    let mut max_tokens = 64;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments.next().ok_or("Each flag requires one value")?;
        match flag.to_str().ok_or("Flags must contain UTF-8 text")? {
            "--model" => model = Some(PathBuf::from(value)),
            "--cases" => cases = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--warmups" => warmups = value.to_str().ok_or("Invalid warmup count")?.parse()?,
            "--repeats" => repeats = value.to_str().ok_or("Invalid repeat count")?.parse()?,
            "--max-tokens" => max_tokens = value.to_str().ok_or("Invalid token count")?.parse()?,
            _ => return Err(format!("Unknown flag: {}", flag.to_string_lossy()).into()),
        }
    }
    if repeats == 0 || max_tokens == 0 {
        return Err("Repeat and token counts must be positive".into());
    }
    Ok(Settings {
        model: model.ok_or("Missing --model")?,
        cases: cases.ok_or("Missing --cases")?,
        output: output.ok_or("Missing --output")?,
        warmups,
        repeats,
        max_tokens,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let settings = settings()?;
    let process_started = Instant::now();
    let model_path = settings.model.canonicalize()?;
    let model_type = AutoModel::detect_model_name(&model_path)?;
    if !matches!(model_type.as_str(), "qwen3_5" | "qwen35") {
        return Err("This experiment requires the Qwen3.5 text model".into());
    }
    let cases_bytes = std::fs::read(&settings.cases)?;
    let cases: Vec<Value> = serde_json::from_slice(&cases_bytes)?;
    if cases.is_empty() {
        return Err("The case list must not be empty".into());
    }
    let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))?;
    let mut prepared = Vec::new();
    for case in cases {
        let id = case["id"]
            .as_str()
            .ok_or("Each case requires an id")?
            .to_owned();
        let prompt = case["prompt"]
            .as_str()
            .ok_or("Each case requires a prompt")?;
        let text = tokenizer.apply_chat_template(&[ChatMessage::user(prompt)])?;
        let tokens = tokenizer.encode(&text)?;
        if tokens
            .len()
            .checked_add(settings.max_tokens)
            .ok_or("Context overflow")?
            > 2048
        {
            return Err("The prompt and output budget exceed context 2048".into());
        }
        prepared.push((id, tokens));
    }
    let options = LoadOptions {
        model_name: Some(model_type),
        text_weight_dtype: Some(DType::F32),
        max_context: Some(2048),
        metal_w8_lm_head: true,
        metal_w8_mlp_block: true,
        ..LoadOptions::default()
    };
    let load_started = Instant::now();
    let mut model = AutoModel::load_model(Device::Cpu, &model_path, &options)?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let config = std::fs::read(model_path.join("config.json"))?;
    let mut report = json!({
        "schema": "apxinf-specialization-generation-v1",
        "status": "running",
        "model": model_path,
        "config_sha256": format!("{:x}", Sha256::digest(config)),
        "cases_sha256": format!("{:x}", Sha256::digest(&cases_bytes)),
        "load_ms": load_ms,
        "settings": {
            "max_context": 2048,
            "max_tokens": settings.max_tokens,
            "warmups_per_case": settings.warmups,
            "repeats_per_case": settings.repeats,
            "batch": 1,
            "concurrency": 1,
            "eos_stopping": true,
            "eos_token_id": tokenizer.eos_token_id(),
            "metal_w8_lm_head": true,
            "metal_w8_mlp_block": true,
            "body": "CPU/Accelerate F32 with existing decode Metal W8 head and MLP",
            "streaming_output": false,
        },
        "timing": {
            "request_ms": "Reset through completed generation and returned token IDs.",
            "ttft_wall_ms": "Request start through the first token callback.",
            "decode_tps": "Output count minus one divided by time from first callback to generation return.",
            "excluded": "Tokenization, text decoding, result serialization, and report writes.",
            "load_ms": "Native model construction, including weight preparation and Metal initialization.",
        },
        "runs": [],
    });
    std::fs::write(&settings.output, serde_json::to_vec_pretty(&report)?)?;
    let mut first_request_ms = None;
    for (id, prompt_tokens) in prepared {
        for index in 0..settings.warmups + settings.repeats {
            let mut first_token_ms = None;
            let mut callbacks = 0;
            let started = Instant::now();
            model.reset()?;
            let (tokens, profile) = model.generate_streaming(
                LlmInput::text(&prompt_tokens),
                settings.max_tokens,
                |_| {
                    callbacks += 1;
                    if first_token_ms.is_none() {
                        first_token_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
                    }
                },
                tokenizer.eos_token_id(),
            )?;
            let request_ms = started.elapsed().as_secs_f64() * 1000.0;
            if tokens.is_empty() || callbacks != tokens.len() {
                return Err(
                    "The generation callback count does not match the returned tokens".into(),
                );
            }
            if first_request_ms.is_none() {
                first_request_ms = Some(request_ms);
            }
            let first = first_token_ms.ok_or("No first-token callback")?;
            let decode_ms = request_ms - first;
            let decode_tps = if tokens.len() > 1 && decode_ms > 0.0 {
                Some((tokens.len() - 1) as f64 * 1000.0 / decode_ms)
            } else {
                None
            };
            let record = json!({
                "case": id,
                "repetition": index,
                "warmup": index < settings.warmups,
                "prompt_token_ids": prompt_tokens,
                "generated_token_ids": tokens,
                "text": tokenizer.decode(&tokens)?,
                "request_ms": request_ms,
                "ttft_wall_ms": first,
                "decode_ms": decode_ms,
                "decode_tps": decode_tps,
                "profile": {
                    "ttft_ms": profile.ttft_ms(),
                    "tpot_ms": profile.tpot_ms(),
                    "total_latency_ms": profile.total_latency_ms(),
                    "input_tokens": profile.input_tokens(),
                    "output_tokens": profile.output_tokens(),
                },
                "generation_path_receipt": model.generation_path_receipt()?,
            });
            report["runs"]
                .as_array_mut()
                .ok_or("Missing run array")?
                .push(record);
            std::fs::write(&settings.output, serde_json::to_vec_pretty(&report)?)?;
        }
    }
    report["first_request_ms"] = json!(first_request_ms);
    report["process_body_ms"] = json!(process_started.elapsed().as_secs_f64() * 1000.0);
    report["status"] = json!("complete");
    std::fs::write(&settings.output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}
