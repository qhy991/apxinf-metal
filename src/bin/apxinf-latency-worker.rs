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
    AutoModel, DecisionImage, DecisionInput, DecisionSchema, GenerationOptions, ImageRequest,
    LlmInput, LoadOptions, SamplingMode, SpeechRequest,
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
    /// Diagnostics outside `payload`, so the baseline payload shape is untouched.
    /// Only populated for `classification` with `APXINF_CLEF_DEBUG=1`.
    debug: Option<Value>,
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
            debug: None,
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
            // The clock covers media decoding and preprocessing: the MLX
            // baseline worker opens and preprocesses the same PNGs inside its
            // request too (mlx_baselines.adapters.Clef.run).
            let started = Instant::now();
            let schema = classification_schema(field(case, "record")?)?;
            let (paths, bytes) = classification_images(field(case, "record")?)?;
            let mut images = Vec::with_capacity(paths.len());
            for path in &paths {
                images.push(DecisionImage::Path(path.as_path()));
            }
            for png in &bytes {
                images.push(DecisionImage::Png(png.as_slice()));
            }
            let input = if images.is_empty() {
                DecisionInput::Schema(&schema)
            } else {
                DecisionInput::SchemaWithImages {
                    schema: &schema,
                    images: &images,
                }
            };
            let output = model.decide(input).map_err(|e| e.to_string())?;
            let input_ids = model
                .decision_input_ids()
                .map_err(|e| e.to_string())?
                .map(|ids| ids.to_vec());
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
            let debug = std::env::var_os("APXINF_CLEF_DEBUG").map(|_| {
                json!({
                    "input_ids": input_ids,
                    "input_tokens": input_ids.as_ref().map_or(0, Vec::len),
                    "images": images.len(),
                })
            });
            Ok((
                elapsed,
                Request {
                    payload: json!({"probabilities": probabilities, "decisions": decisions}),
                    arrays: vec![],
                    debug,
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
                    debug: None,
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
                    debug: None,
                },
            ))
        }
        other => Err(format!("no ApxInf latency worker for task {other}")),
    }
}

/// The schema part of a classification record: `state` + `questions`. The
/// record may also carry `images`, which `DecisionSchema` rejects as an unknown
/// field, and `videos`, which this family does not support.
fn classification_schema(record: &Value) -> Fallible<DecisionSchema> {
    if let Some(videos) = record.get("videos") {
        if videos.as_array().is_some_and(|items| !items.is_empty()) {
            return Err("the Clef worker supports images, not video".into());
        }
    }
    let schema = json!({
        "state": field(record, "state")?,
        "questions": field(record, "questions")?,
    });
    DecisionSchema::from_json(&schema.to_string()).map_err(|e| e.to_string())
}

/// Images of a classification record, in request order: local PNG paths and
/// inline `data:image/...;base64,` payloads (the two forms the baseline accepts).
fn classification_images(record: &Value) -> Fallible<(Vec<PathBuf>, Vec<Vec<u8>>)> {
    let Some(value) = record.get("images") else {
        return Ok((Vec::new(), Vec::new()));
    };
    let Some(items) = value.as_array() else {
        return Err("record.images must be an array".into());
    };
    if items.len() > 4 {
        return Err("the Clef worker supports at most four images per request".into());
    }
    let mut paths = Vec::new();
    let mut bytes = Vec::new();
    for item in items {
        let Some(item) = item.as_str() else {
            return Err("record.images entries must be strings".into());
        };
        if let Some(rest) = item.strip_prefix("data:image/") {
            let Some((meta, data)) = rest.split_once(',') else {
                return Err("malformed data URL in record.images".into());
            };
            if !meta.ends_with(";base64") {
                return Err("record.images data URLs must be base64".into());
            }
            bytes.push(decode_base64(data)?);
        } else if item.is_empty() {
            return Err("record.images entry is empty".into());
        } else {
            paths.push(PathBuf::from(item));
        }
    }
    Ok((paths, bytes))
}

/// Standard-alphabet padded base64, so the worker needs no image or encoding
/// dependency to accept the demo's inline PNGs.
fn decode_base64(text: &str) -> Fallible<Vec<u8>> {
    fn value(byte: u8) -> Fallible<u32> {
        match byte {
            b'A'..=b'Z' => Ok((byte - b'A') as u32),
            b'a'..=b'z' => Ok((byte - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((byte - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            other => Err(format!("invalid base64 byte {other:?}")),
        }
    }
    let bytes = text.trim().as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let mut chunk = [0u8; 4];
    let mut filled = 0;
    let mut padding = 0;
    for &byte in bytes {
        if byte == b'=' {
            padding += 1;
            continue;
        }
        if padding > 0 {
            return Err("base64 padding is not at the end".into());
        }
        chunk[filled] = byte;
        filled += 1;
        if filled == 4 {
            let mut packed = 0u32;
            for &part in &chunk {
                packed = (packed << 6) | value(part)?;
            }
            out.extend_from_slice(&packed.to_be_bytes()[1..]);
            filled = 0;
        }
    }
    // Unpadded tails are legal base64 (`aGVsbG8` == `aGVsbG8=`); a padded
    // alphabet must still end with the right number of `=`.
    match (filled, padding) {
        (0, 0) => {}
        (2, 2) | (2, 0) => {
            let packed = (value(chunk[0])? << 6) | value(chunk[1])?;
            out.push((packed >> 4) as u8);
        }
        (3, 1) | (3, 0) => {
            let mut packed = 0u32;
            for &part in &chunk[..3] {
                packed = (packed << 6) | value(part)?;
            }
            // Three sextets are 18 bits; align them as the top 18 of 24.
            let bytes = (packed << 6).to_be_bytes();
            out.extend_from_slice(&bytes[1..3]);
        }
        _ => return Err("base64 has a ragged tail".into()),
    }
    Ok(out)
}

fn dtype_name(descr: &str) -> &'static str {
    match descr {
        "<f4" => "float32",
        _ => "uint8",
    }
}

/// `(peak_bytes, active_bytes, cache_bytes)` of the resident MLX allocator.
/// `apxinf_mlx::memory_stats` only exists on the native Apple target, which is
/// the only place these numbers are meaningful; the Linux feature-isolation
/// check compiles this binary without the MLX native exports, so it gets zeros
/// instead of a link-time failure.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn resident_memory() -> Fallible<(usize, usize, usize)> {
    let memory = apxinf_mlx::memory_stats().map_err(|e| e.to_string())?;
    Ok((memory.peak_bytes, memory.active_bytes, memory.cache_bytes))
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn resident_memory() -> Fallible<(usize, usize, usize)> {
    Ok((0, 0, 0))
}

fn main_inner(job_path: &Path) -> Fallible<()> {
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
                     "model_name": model_name, "model_variant": model_variant, "checks": checks},
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
        let Request {
            payload,
            arrays: request_arrays,
            debug,
        } = request;
        let mut arrays = Map::new();
        for (key, descr, shape, bytes) in request_arrays {
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
        let (peak_bytes, active_bytes, cache_bytes) = resident_memory()?;
        record.insert("peak_bytes".into(), json!(peak_bytes));
        record.insert("active_bytes".into(), json!(active_bytes));
        record.insert("cache_bytes".into(), json!(cache_bytes));
        record.insert("payload".into(), payload);
        record.insert("arrays".into(), arrays.into());
        if let Some(debug) = debug {
            record.insert("debug".into(), debug);
        }
        emit(record.into())?;
    }
    emit(json!({"type": "done"}))?;
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decodes_padded_and_unpadded_inputs() {
        // PNG signature, and "hello" with and without padding.
        assert_eq!(
            decode_base64("iVBORw0KGgo=").unwrap(),
            [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        );
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("aGVsbG8").unwrap(), b"hello");
        assert_eq!(decode_base64("aGVsbA==").unwrap(), b"hell");
        assert!(decode_base64("a").is_err());
        assert!(decode_base64("!").is_err());
    }

    #[test]
    fn classification_records_split_paths_and_inline_pngs() {
        let record = json!({
            "state": "s",
            "images": ["/tmp/a.png", "data:image/png;base64,iVBORw0KGgo="],
            "questions": {"q": {"type": "noul"}},
        });
        let (paths, bytes) = classification_images(&record).unwrap();
        assert_eq!(paths, vec![PathBuf::from("/tmp/a.png")]);
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0][..4], [0x89, 0x50, 0x4e, 0x47]);

        let schema = classification_schema(&record).unwrap();
        assert_eq!(schema.questions.len(), 1);
        assert_eq!(schema.state, json!("s"));

        assert!(classification_images(&json!({"images": ["data:image/png,abc"]})).is_err());
        assert!(classification_images(&json!({"images": [1]})).is_err());
        assert!(classification_images(&json!({"images": ["a", "b", "c", "d", "e"]})).is_err());
        assert!(
            classification_schema(&json!({"state": "s", "videos": ["v.mp4"],
                "questions": {"q": {"type": "noul"}}}))
            .is_err()
        );
    }
}
