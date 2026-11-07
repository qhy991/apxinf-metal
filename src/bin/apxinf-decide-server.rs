//! Resident Clef decision process for the Mario demo (`APXINF_DECIDE_CMD`).
//!
//! ```text
//! apxinf-decide-server --model <checkpoint> [--model-variant reference|auto]
//! ```
//!
//! Loads one checkpoint, prints one `ready` line on stdout, then serves JSONL
//! requests on stdin and answers one JSONL line each:
//!
//! ```text
//! {"id":1,"state":"...","questions":{"enemy":{"type":"noul","instructions":"..."}},"images":["<base64 png>",...]}
//! {"id":1,"answers":{"enemy":{"type":"noul","noul":0.42}},"latency_ms":2710.5}
//! ```
//!
//! The request clock covers image decoding and preprocessing through available
//! probabilities — the same boundary the benchmark worker and the MLX baseline
//! use. A malformed request answers `{"id":...,"error":"..."}` and the process
//! keeps serving; only a load failure, exhaused stdin or a closed stdout ends it.
//!
//! The decision path is the worker's: `AutoModel::load_model` +
//! `LoadedModel::decide(DecisionInput::{Schema,SchemaWithImages})`. The media
//! contract (base64 PNG, at most four images) is the same as
//! `apxinf-latency-worker`'s classification case, so the two stay comparable.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{
    AutoModel, DecisionImage, DecisionInput, DecisionOutput, DecisionSchema, LoadOptions,
    LoadedModel,
};
use serde_json::{json, Map, Value};

type Fallible<T> = Result<T, String>;

fn main() {
    match load() {
        Ok(mut model) => serve(&mut model),
        Err(error) => {
            // The demo client reads this as a failed start-up and shows the text.
            let _ = writeln!(
                std::io::stdout(),
                "{}",
                json!({"type": "error", "error": error})
            );
            let _ = std::io::stdout().flush();
            std::process::exit(1);
        }
    }
}

struct Options {
    model: PathBuf,
    variant: Option<String>,
    model_name: Option<String>,
}

fn parse_options() -> Fallible<Options> {
    let mut model = None;
    let mut variant = None;
    let mut model_name = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "--snapshot" => {
                model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?))
            }
            "--model-variant" => variant = Some(args.next().ok_or("--model-variant needs a value")?),
            "--model-name" => model_name = Some(args.next().ok_or("--model-name needs a value")?),
            "--help" | "-h" => {
                return Err(
                    "usage: apxinf-decide-server --model <checkpoint> [--model-variant reference|auto]"
                        .into(),
                )
            }
            other if model.is_none() && !other.starts_with('-') => {
                model = Some(PathBuf::from(other))
            }
            other => return Err(format!("unexpected argument {other}")),
        }
    }
    Ok(Options {
        model: model.ok_or("--model <checkpoint> is required")?,
        variant,
        model_name,
    })
}

/// Load the resident model and print the one `ready` line the client waits for.
fn load() -> Fallible<LoadedModel> {
    let options = parse_options()?;
    let started = Instant::now();
    let load_options = LoadOptions {
        model_name: options.model_name.clone(),
        model_variant: options.variant.clone(),
        ..LoadOptions::default()
    };
    let model = AutoModel::load_model(Device::Metal(0), &options.model, &load_options)
        .map_err(|e| format!("load: {e}"))?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let model_name = options
        .model_name
        .or_else(|| AutoModel::detect_model_name(&options.model).ok())
        .unwrap_or_else(|| "clef".into());
    let variant = options.variant.unwrap_or_else(|| "auto".into());
    writeln!(
        std::io::stdout(),
        "{}",
        json!({
            "type": "ready",
            "model": model_name,
            "variant": variant,
            "load_ms": load_ms,
        })
    )
    .map_err(|e| format!("write ready: {e}"))?;
    std::io::stdout()
        .flush()
        .map_err(|e| format!("flush ready: {e}"))?;
    Ok(model)
}

fn serve(model: &mut LoadedModel) {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("stdin: {error}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = handle(model, &line);
        if writeln!(stdout, "{response}").is_err() || stdout.flush().is_err() {
            break;
        }
    }
}

/// One request line in, one response line out; never fails the process.
fn handle(model: &mut LoadedModel, line: &str) -> Value {
    let request: Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(error) => {
            return json!({"id": Value::Null, "error": format!("request is not JSON: {error}")})
        }
    };
    let id = request.get("id").cloned().unwrap_or(Value::Null);
    match answer(model, &request) {
        Ok((answers, latency_ms)) => json!({
            "id": id,
            "answers": answers,
            "latency_ms": latency_ms,
        }),
        Err(error) => json!({"id": id, "error": error}),
    }
}

fn answer(model: &mut LoadedModel, request: &Value) -> Fallible<(Value, f64)> {
    // The clock starts before the media is decoded, as on the benchmark and
    // baseline side.
    let started = Instant::now();
    let schema = schema_from_request(request)?;
    let (paths, payloads) = images_from_request(request)?;
    let mut images = Vec::with_capacity(paths.len() + payloads.len());
    for path in &paths {
        images.push(DecisionImage::Path(path.as_path()));
    }
    for png in &payloads {
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
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok((answers_json(&schema, &output)?, latency_ms))
}

/// `state` + `questions` only: `images` is the worker's media field, not schema.
fn schema_from_request(request: &Value) -> Fallible<DecisionSchema> {
    let state = request.get("state").ok_or("request needs state")?;
    let questions = request.get("questions").ok_or("request needs questions")?;
    let schema = json!({"state": state, "questions": questions});
    DecisionSchema::from_json(&schema.to_string()).map_err(|e| e.to_string())
}

/// `[{"id":..,"answers":{"<question>":{"type":"noul","noul":p}|{"type":"choice","choice":code}|{"type":"score","score":expected}}}]`
fn answers_json(schema: &DecisionSchema, output: &DecisionOutput) -> Fallible<Value> {
    let mut answers = Map::new();
    for answer in &output.answers {
        let id = answer
            .question_id
            .clone()
            .ok_or("schema answer without a question id")?;
        let kind = schema
            .questions
            .iter()
            .find(|question| question.id == id)
            .map(|question| question.kind.clone())
            .unwrap_or_default();
        let entry = match kind.as_str() {
            "noul" => {
                let probability = answer
                    .scores
                    .iter()
                    .find(|score| score.code == "true")
                    .map(|score| score.probability)
                    .ok_or_else(|| format!("question {id} has no `true` option"))?;
                json!({"type": "noul", "noul": probability})
            }
            "choice" => {
                let selected = answer.selected().map_err(|e| e.to_string())?;
                json!({"type": "choice", "choice": selected.code})
            }
            "score" => {
                let mut expected = 0.0f64;
                let mut numeric = true;
                for score in &answer.scores {
                    match score.code.parse::<f64>() {
                        Ok(value) => expected += value * score.probability as f64,
                        Err(_) => {
                            numeric = false;
                            break;
                        }
                    }
                }
                if numeric {
                    json!({"type": "score", "score": expected})
                } else {
                    let selected = answer.selected().map_err(|e| e.to_string())?;
                    json!({"type": "score", "score": selected.code})
                }
            }
            other => return Err(format!("unsupported question type {other:?}")),
        };
        answers.insert(id, entry);
    }
    Ok(Value::Object(answers))
}

/// Images of a request in request order: local PNG paths, `data:image/...;base64,`
/// URLs or bare base64 PNG payloads (what the demo client sends).
fn images_from_request(request: &Value) -> Fallible<(Vec<PathBuf>, Vec<Vec<u8>>)> {
    let Some(value) = request.get("images") else {
        return Ok((Vec::new(), Vec::new()));
    };
    let items = value.as_array().ok_or("images must be an array")?;
    if items.len() > 4 {
        return Err("at most four images are supported per request".into());
    }
    let mut paths = Vec::new();
    let mut payloads = Vec::new();
    for item in items {
        let item = item.as_str().ok_or("images entries must be strings")?;
        if let Some(rest) = item.strip_prefix("data:image/") {
            let (meta, data) = rest.split_once(',').ok_or("malformed data URL")?;
            if !meta.ends_with(";base64") {
                return Err("image data URLs must be base64".into());
            }
            payloads.push(decode_base64(data)?);
        } else if item.is_empty() {
            return Err("images entry is empty".into());
        } else if item.starts_with('/') || item.starts_with("./") || item.ends_with(".png") {
            paths.push(PathBuf::from(item));
        } else {
            payloads.push(decode_base64(item)?);
        }
    }
    Ok((paths, payloads))
}

/// Standard-alphabet padded or unpadded base64 (the demo inlines PNG bytes).
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
            let bytes = (packed << 6).to_be_bytes();
            out.extend_from_slice(&bytes[1..3]);
        }
        _ => return Err("base64 has a ragged tail".into()),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decodes_padded_and_unpadded_payloads() {
        assert_eq!(
            decode_base64("iVBORw0KGgo=").unwrap(),
            [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
        );
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(decode_base64("aGVsbG8").unwrap(), b"hello");
        assert_eq!(decode_base64("aGVsbA==").unwrap(), b"hell");
        assert!(decode_base64("a").is_err());
    }

    #[test]
    fn request_media_forms_are_split_into_paths_and_payloads() {
        let request = json!({
            "state": "s",
            "images": ["/tmp/a.png", "data:image/png;base64,iVBORw0KGgo=", "iVBORw0KGgo="],
            "questions": {"q": {"type": "noul"}},
        });
        let (paths, payloads) = images_from_request(&request).unwrap();
        assert_eq!(paths, vec![PathBuf::from("/tmp/a.png")]);
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0][..4], [0x89, 0x50, 0x4e, 0x47]);

        let schema = schema_from_request(&request).unwrap();
        assert_eq!(schema.questions.len(), 1);
        assert_eq!(schema.state, json!("s"));
        assert!(images_from_request(&json!({"images": [1]})).is_err());
        assert!(images_from_request(&json!({"images": ["a", "b", "c", "d", "e"]})).is_err());
    }
}
