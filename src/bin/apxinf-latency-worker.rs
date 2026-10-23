//! Latency worker for the mlx-model-baselines comparison driver
//! (protocol `mlx-baseline.worker.v1`).
//!
//! ```text
//! apxinf-latency-worker JOB.json
//! ```
//!
//! Loads one model once through `AutoModel`, runs the job's schedule in order
//! and appends one JSON record per request to `records.jsonl` in the job's
//! output directory. The driver holds the host locks and owns the receipts.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{
    AutoModel, DecisionInput, DecisionSchema, GenerationOptions, ImageRequest, LlmInput,
    LoadOptions, SamplingMode, SpeechRequest,
};
use apxinf_tokenizer::{ChatMessage, Tokenizer};
use serde_json::{json, Map, Value};

const TIMING_BOUNDARY: &str =
    "request preprocessing through synchronized host output; excludes load and artifact save";

type Fallible<T> = Result<T, String>;

fn field<'a>(value: &'a Value, key: &str) -> Fallible<&'a Value> {
    value.get(key).ok_or_else(|| format!("missing field {key}"))
}

fn text<'a>(value: &'a Value, key: &str) -> Fallible<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("{key} must be a string"))
}

fn unsigned(value: &Value, key: &str) -> Fallible<u64> {
    field(value, key)?
        .as_u64()
        .ok_or_else(|| format!("{key} must be a non-negative integer"))
}

/// The baseline suite's sampling settings must be the ones the family reads
/// from the checkpoint; the speech API takes no sampling parameters.
fn check_speech_settings(snapshot: &Path, settings: &Value) -> Fallible<Value> {
    let path = snapshot.join("generation_config.json");
    let generation: Value = serde_json::from_str(
        &std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?,
    )
    .map_err(|e| format!("parse {}: {e}", path.display()))?;
    for key in ["temperature", "top_k", "top_p", "repetition_penalty"] {
        let (want, have) = (
            field(settings, key)?.as_f64(),
            field(&generation, key)?.as_f64(),
        );
        if want.is_none() || want != have {
            return Err(format!(
                "suite {key}={want:?} differs from the checkpoint's {have:?}; the family cannot follow it"
            ));
        }
    }
    Ok(json!({
        "sampling_source": path.display().to_string(),
        "frame_limit": field(&generation, "max_new_tokens")?,
        "suite_max_tokens": field(settings, "max_tokens")?,
    }))
}

struct Request {
    payload: Value,
    arrays: Vec<(&'static str, &'static str, Vec<usize>, Vec<u8>)>,
}

/// Exactly `fixed_output_tokens` greedy tokens: EOS does not stop generation,
/// so every implementation does the same decode work.
fn generate_fixed(
    model: &mut apxinf_model::LoadedModel,
    tokenizer: &Tokenizer,
    case: &Value,
    settings: &Value,
) -> Fallible<(f64, Request)> {
    let count = unsigned(settings, "fixed_output_tokens")
        .map_err(|_| "the ApxInf worker measures text only with fixed_output_tokens".to_string())?
        as usize;
    let started = Instant::now();
    let messages = field(case, "messages")?
        .as_array()
        .ok_or("messages must be an array")?
        .iter()
        .map(|m| {
            let content = text(m, "content")?;
            match text(m, "role")? {
                "system" => Ok(ChatMessage::system(content)),
                "user" => Ok(ChatMessage::user(content)),
                "assistant" => Ok(ChatMessage::assistant(content)),
                other => Err(format!("unsupported message role {other}")),
            }
        })
        .collect::<Fallible<Vec<_>>>()?;
    let mut options = Map::new();
    let thinking = settings.get("enable_thinking").cloned();
    options.insert("enable_thinking".into(), thinking.unwrap_or(json!(false)));
    let prompt = tokenizer
        .apply_chat_template_with_options(&messages, &options)
        .map_err(|e| e.to_string())?;
    let prompt_ids = tokenizer.encode(&prompt).map_err(|e| e.to_string())?;
    if prompt_ids.len() + count > unsigned(settings, "max_context")? as usize {
        return Err("prompt plus output budget exceeds the declared context".into());
    }
    let generation = GenerationOptions {
        max_new_tokens: Some(count),
        eos_token_ids: Some(Vec::new()),
        sampling_mode: Some(SamplingMode::Greedy),
        return_logprob: Some(false),
        ..GenerationOptions::default()
    };
    let (mut tokens, mut first) = (Vec::with_capacity(count), None);
    model
        .generate_streaming_with_options(LlmInput::text(&prompt_ids), &generation, |token| {
            first.get_or_insert_with(|| started.elapsed());
            tokens.push(token.token_id);
        })
        .map_err(|e| e.to_string())?;
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    if tokens.len() != count {
        return Err(format!(
            "fixed-length generation returned {} of {count} tokens",
            tokens.len()
        ));
    }
    let eos = match model.generation_defaults().map(|d| d.eos_token_ids.clone()) {
        Ok(Some(ids)) => ids,
        _ => tokenizer.eos_token_id().into_iter().collect(),
    };
    let payload = json!({
        "text": tokenizer.decode(&tokens).map_err(|e| e.to_string())?,
        "token_ids": tokens,
        "prompt_tokens": prompt_ids.len(),
        "output_tokens": count,
        "finish_reason": "fixed_length",
        "ttft_ms": first.ok_or("no token was generated")?.as_secs_f64() * 1000.0,
        "eos_positions": tokens.iter().enumerate().filter(|(_, t)| eos.contains(t)).map(|(i, _)| i).collect::<Vec<_>>(),
    });
    Ok((
        elapsed,
        Request {
            payload,
            arrays: vec![],
        },
    ))
}

fn run(
    model: &mut apxinf_model::LoadedModel,
    tokenizer: Option<&Tokenizer>,
    task: &str,
    case: &Value,
    settings: &Value,
    seed: u64,
) -> Fallible<(f64, Request)> {
    match task {
        "text" => generate_fixed(
            model,
            tokenizer.ok_or("text needs the tokenizer")?,
            case,
            settings,
        ),
        "classification" => {
            let started = Instant::now();
            let schema = DecisionSchema::from_json(&field(case, "record")?.to_string())
                .map_err(|e| e.to_string())?;
            let output = model
                .decide(DecisionInput::Schema(&schema))
                .map_err(|e| e.to_string())?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            let (mut probabilities, mut decisions) = (Map::new(), Map::new());
            for answer in &output.answers {
                let id = answer
                    .question_id
                    .clone()
                    .ok_or("schema answer without a question id")?;
                let selected = answer.selected().map_err(|e| e.to_string())?;
                decisions.insert(id.clone(), json!(selected.code));
                probabilities.insert(
                    id,
                    answer
                        .scores
                        .iter()
                        .map(|s| (s.code.clone(), json!(s.probability as f64)))
                        .collect::<Map<_, _>>()
                        .into(),
                );
            }
            Ok((
                elapsed,
                Request {
                    payload: json!({"probabilities": probabilities, "decisions": decisions}),
                    arrays: vec![],
                },
            ))
        }
        "tts" => {
            let request = SpeechRequest {
                text: text(case, "text")?,
                voice: case
                    .get("speaker")
                    .and_then(Value::as_str)
                    .unwrap_or("Vivian"),
                language: text(case, "language")?,
                instruction: case.get("instruct").and_then(Value::as_str).unwrap_or(""),
                seed,
            };
            let started = Instant::now();
            let audio = model.synthesize(&request).map_err(|e| e.to_string())?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            if audio.channels != 1 {
                return Err(format!(
                    "expected mono audio, got {} channels",
                    audio.channels
                ));
            }
            let samples = audio.samples.len();
            let peak = audio.samples.iter().fold(0f32, |m, v| m.max(v.abs()));
            let bytes = audio.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
            let payload = json!({
                "sample_rate": audio.sample_rate, "samples": samples,
                "audio_seconds": samples as f64 / audio.sample_rate as f64,
                "peak_amplitude": peak as f64,
                "finish_reason": if audio.complete { "end_of_speech" } else { "frame_limit" },
                "quality": "requires transcript and listening review",
            });
            Ok((
                elapsed,
                Request {
                    payload,
                    arrays: vec![("audio", "<f4", vec![samples], bytes)],
                },
            ))
        }
        "image" => {
            if field(settings, "guidance")?.as_f64() != Some(1.0)
                || text(settings, "scheduler")? != "linear"
            {
                return Err("the family admits only guidance 1 with the linear scheduler".into());
            }
            let (width, height) = (
                unsigned(case, "width")? as usize,
                unsigned(case, "height")? as usize,
            );
            let request = ImageRequest {
                prompt: text(case, "prompt")?,
                width,
                height,
                seed,
                steps: unsigned(settings, "steps")? as usize,
            };
            let started = Instant::now();
            let image = model.generate_image(&request).map_err(|e| e.to_string())?;
            let elapsed = started.elapsed().as_secs_f64() * 1000.0;
            let payload = json!({"width": image.width, "height": image.height, "seed": seed});
            Ok((
                elapsed,
                Request {
                    payload,
                    arrays: vec![(
                        "image",
                        "|u1",
                        vec![image.height, image.width, 3],
                        image.rgb,
                    )],
                },
            ))
        }
        other => Err(format!("no ApxInf latency worker for task {other}")),
    }
}

fn dtype_name(descr: &str) -> &'static str {
    match descr {
        "<f4" => "float32",
        _ => "uint8",
    }
}

fn main_inner(job_path: &Path) -> Fallible<()> {
    let cache_limit = match std::env::var("APXINF_MLX_CACHE_LIMIT_BYTES") {
        Ok(raw) => {
            if raw.is_empty() || !raw.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(
                    "APXINF_MLX_CACHE_LIMIT_BYTES must be a nonnegative decimal byte count".into(),
                );
            }
            Some(
                raw.parse::<usize>()
                    .map_err(|e| format!("APXINF_MLX_CACHE_LIMIT_BYTES: {e}"))?,
            )
        }
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(format!("APXINF_MLX_CACHE_LIMIT_BYTES: {error}")),
    };
    let job: Value = serde_json::from_str(
        &std::fs::read_to_string(job_path).map_err(|e| format!("read job: {e}"))?,
    )
    .map_err(|e| format!("parse job: {e}"))?;
    if text(&job, "schema")? != "mlx-baseline.worker-job.v1" {
        return Err("expected an mlx-baseline.worker-job.v1 job".into());
    }
    let suite = field(&job, "suite")?;
    let task = text(suite, "task")?.to_owned();
    let settings = field(&job, "settings")?.clone();
    let snapshot = PathBuf::from(text(&job, "snapshot")?);
    let output = PathBuf::from(text(&job, "output")?);
    let cases: std::collections::HashMap<&str, &Value> = field(&job, "cases")?
        .as_array()
        .ok_or("cases must be an array")?
        .iter()
        .map(|c| Ok((text(c, "id")?, c)))
        .collect::<Fallible<_>>()?;
    let mut records = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("records.jsonl"))
        .map_err(|e| format!("open records: {e}"))?;
    let mut emit = |record: Value| -> Fallible<()> {
        writeln!(records, "{record}")
            .and_then(|_| records.flush())
            .map_err(|e| format!("write record: {e}"))
    };

    let tokenizer = if task == "text" {
        Some(
            Tokenizer::from_file(snapshot.join("tokenizer.json"))
                .map_err(|e| format!("tokenizer: {e}"))?,
        )
    } else {
        None
    };
    let checks = if task == "tts" {
        check_speech_settings(&snapshot, &settings)?
    } else {
        Value::Null
    };
    let model_variant = std::env::var("APXINF_MODEL_VARIANT").ok();
    // MiniCPM5 declares model_type llama, so its family must be named explicitly.
    let model_name_override = std::env::var("APXINF_MODEL_NAME").ok();
    let options = LoadOptions {
        model_name: model_name_override.clone(),
        model_variant: model_variant.clone(),
        ..LoadOptions::default()
    };
    let previous_cache_limit = cache_limit
        .map(apxinf_mlx::set_cache_limit)
        .transpose()
        .map_err(|error| format!("set APXINF_MLX_CACHE_LIMIT_BYTES={cache_limit:?}: {error}"))?;
    let started = Instant::now();
    let mut model = AutoModel::load_model(Device::Metal(0), &snapshot, &options)
        .map_err(|e| format!("load: {e}"))?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let model_name = model_name_override
        .unwrap_or_else(|| AutoModel::detect_model_name(&snapshot).unwrap_or_default());
    emit(json!({
        "type": "ready", "engine": "apxinf",
        "implementation": format!("AutoModel {model_name} on Metal, model_variant {}", model_variant.as_deref().unwrap_or("auto")),
        "baseline_kind": "apxinf-native",
        "precision": "checkpoint weights; ApxInf family-specific compute (see the family README)",
        "timing_boundary": TIMING_BOUNDARY,
        "process_policy": "one resident model, serial fresh requests",
        "identity": {"commit": option_env!("APXINF_LATENCY_WORKER_COMMIT"),
                     "features": option_env!("APXINF_LATENCY_WORKER_FEATURES"),
                     "model_name": model_name, "model_variant": model_variant, "checks": checks,
                     "allocator_cache_limit_bytes": cache_limit,
                     "previous_allocator_cache_limit_bytes": previous_cache_limit},
        "load_ms": load_ms,
    }))?;

    for (index, item) in field(&job, "schedule")?
        .as_array()
        .ok_or("schedule must be an array")?
        .iter()
        .enumerate()
    {
        let case = cases
            .get(text(item, "case")?)
            .ok_or("schedule names an unknown case")?;
        let seed = unsigned(item, "seed")?;
        let (latency_ms, request) =
            run(&mut model, tokenizer.as_ref(), &task, case, &settings, seed)?;
        let mut arrays = Map::new();
        for (key, descr, shape, bytes) in request.arrays {
            let name = format!("{index:05}-{key}.bin");
            File::create(output.join(&name))
                .and_then(|mut f| f.write_all(&bytes))
                .map_err(|e| format!("write {name}: {e}"))?;
            arrays.insert(
                key.into(),
                json!({"file": name, "dtype": dtype_name(descr), "shape": shape}),
            );
        }
        let mut record = item
            .as_object()
            .ok_or("schedule item must be an object")?
            .clone();
        record.insert("type".into(), json!("request"));
        record.insert("index".into(), json!(index));
        record.insert("status".into(), json!("ok"));
        record.insert("latency_ms".into(), json!(latency_ms));
        record.insert("preparation_ms".into(), json!(0.0));
        // Process-cumulative allocator peak, as the MLX workers report it.
        let memory = apxinf_mlx::memory_stats().map_err(|e| e.to_string())?;
        record.insert("peak_bytes".into(), json!(memory.peak_bytes));
        record.insert("active_bytes".into(), json!(memory.active_bytes));
        record.insert("cache_bytes".into(), json!(memory.cache_bytes));
        record.insert("payload".into(), request.payload);
        record.insert("arrays".into(), arrays.into());
        emit(record.into())?;
    }
    emit(json!({"type": "done"}))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: apxinf-latency-worker JOB.json");
        std::process::exit(2);
    }
    if let Err(error) = main_inner(Path::new(&args[1])) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
