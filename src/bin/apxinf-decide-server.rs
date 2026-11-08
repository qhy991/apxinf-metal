//! Resident Clef decision process for the Mario demo (`APXINF_DECIDE_CMD`).
//!
//! ```text
//! apxinf-decide-server --model <checkpoint> [--model-variant reference|auto]
//!                                  [--warm-shapes <file>]
//!                                  [--listen unix:<path>|tcp:<host:port>]
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
//! With `--listen` (or `APXINF_DECIDE_LISTEN`) the process stays resident on a
//! Unix domain socket or TCP address instead of serving stdin/stdout: the
//! model is loaded (and warmed) once, each client connection first receives
//! the same `ready` line and then speaks the identical JSONL protocol, and
//! connections are served sequentially. The demo's per-episode spawn thus pays
//! no load or warmup: the proxy just connects. The pipe mode is unchanged when
//! the flag is absent.
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
        Ok((mut model, ready, listen)) => match listen {
            None => serve(&mut model),
            Some(addr) => {
                if let Err(error) = serve_listen(&mut model, &ready, &addr) {
                    eprintln!("listen: {error}");
                    std::process::exit(1);
                }
            }
        },
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
    /// `--warm-shapes <file>` (env `APXINF_CLEF_WARM_SHAPES`). `None` keeps
    /// the cold-start behaviour: no load-time warm, lazy per-shape compile.
    warm_shapes: Option<PathBuf>,
    /// `--listen unix:<path>` or `--listen tcp:<host:port>`. `None` keeps the
    /// pipe mode: stdin/stdout JSONL, exactly as before.
    listen: Option<ListenAddr>,
}

/// A resident listener address: a Unix domain socket path or a TCP host:port.
#[derive(Clone, Debug)]
enum ListenAddr {
    Unix(PathBuf),
    Tcp(String),
}

impl ListenAddr {
    fn parse(value: &str) -> Fallible<Self> {
        if let Some(path) = value.strip_prefix("unix:") {
            if path.is_empty() {
                return Err("--listen unix: needs a socket path".into());
            }
            return Ok(ListenAddr::Unix(PathBuf::from(path)));
        }
        if let Some(address) = value.strip_prefix("tcp:") {
            if !address.contains(':') {
                return Err("--listen tcp: needs host:port".into());
            }
            return Ok(ListenAddr::Tcp(address.to_owned()));
        }
        Err(format!(
            "--listen must be unix:<path> or tcp:<host:port>, not {value:?}"
        ))
    }
}

fn parse_options() -> Fallible<Options> {
    let mut model = None;
    let mut variant = None;
    let mut model_name = None;
    // The hypothesis gate is opt-in; the flag wins over the environment.
    let mut warm_shapes = std::env::var_os("APXINF_CLEF_WARM_SHAPES")
        .or_else(|| std::env::var_os("APXINF_DECIDE_WARM_SHAPES"))
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty());
    let mut listen = std::env::var_os("APXINF_DECIDE_LISTEN")
        .filter(|value| !value.is_empty())
        .map(|value| ListenAddr::parse(&value.to_string_lossy()))
        .transpose()?;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" | "--snapshot" => {
                model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?))
            }
            "--model-variant" => variant = Some(args.next().ok_or("--model-variant needs a value")?),
            "--model-name" => model_name = Some(args.next().ok_or("--model-name needs a value")?),
            "--warm-shapes" => {
                let value = args.next().ok_or("--warm-shapes needs a file")?;
                if value.is_empty() {
                    return Err("--warm-shapes needs a non-empty file".into());
                }
                warm_shapes = Some(PathBuf::from(value));
            }
            "--listen" => {
                let value = args.next().ok_or("--listen needs an address")?;
                if value.is_empty() {
                    return Err("--listen needs a non-empty address".into());
                }
                if listen.is_some() {
                    return Err("--listen may be given only once".into());
                }
                listen = Some(ListenAddr::parse(&value)?);
            }
            "--help" | "-h" => {
                return Err(
                    "usage: apxinf-decide-server --model <checkpoint> [--model-variant reference|auto] [--warm-shapes <file>] [--listen unix:<path>|tcp:<host:port>]".into(),
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
        warm_shapes,
        listen,
    })
}

/// Load the resident model and print the one `ready` line the client waits
/// for. Returns the model, the ready line as JSON (each listening client gets
/// the identical line) and the requested listen mode.
fn load() -> Fallible<(LoadedModel, Value, Option<ListenAddr>)> {
    let options = parse_options()?;
    // Parse the warm manifest before the checkpoint load so a broken explicit
    // warm request fails in milliseconds, not after a multi-second load.
    let warm_manifest = match &options.warm_shapes {
        Some(path) => Some(load_warm_manifest(path)?),
        None => None,
    };
    let started = Instant::now();
    let load_options = LoadOptions {
        model_name: options.model_name.clone(),
        model_variant: options.variant.clone(),
        ..LoadOptions::default()
    };
    let mut model = AutoModel::load_model(Device::Metal(0), &options.model, &load_options)
        .map_err(|e| format!("load: {e}"))?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut warm_ms = 0.0;
    let mut warm_rows = Vec::new();
    if let Some(manifest) = &warm_manifest {
        let (wall_ms, rows) = warm_shapes(&mut model, manifest)?;
        warm_ms = wall_ms;
        warm_rows = rows;
    }
    let model_name = options
        .model_name
        .or_else(|| AutoModel::detect_model_name(&options.model).ok())
        .unwrap_or_else(|| "clef".into());
    let variant = options.variant.unwrap_or_else(|| "auto".into());
    let ready = json!({
        "type": "ready",
        "model": model_name,
        "variant": variant,
        "load_ms": load_ms,
        "warm_ms": warm_ms,
        "warm_shapes": warm_rows,
    });
    writeln!(std::io::stdout(), "{ready}").map_err(|e| format!("write ready: {e}"))?;
    std::io::stdout()
        .flush()
        .map_err(|e| format!("flush ready: {e}"))?;
    Ok((model, ready, options.listen))
}

/// One warm manifest shape: the question id the demo uses, the exact crop
/// size the server receives, an optional crop image and the question body.
#[derive(Clone, Debug)]
struct WarmShape {
    id: String,
    width: usize,
    height: usize,
    image: Option<PathBuf>,
    question: Value,
}

#[derive(Clone, Debug)]
struct WarmManifest {
    state: String,
    shapes: Vec<WarmShape>,
}

/// Read and validate `--warm-shapes <file>`. A present but broken manifest is
/// a start-up error, never a silent cold start.
fn load_warm_manifest(path: &std::path::Path) -> Fallible<WarmManifest> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("warm shapes {path:?}: {error}"))?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| format!("warm shapes {path:?}: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| format!("warm shapes {path:?}: expected a JSON object"))?;
    if let Some(schema) = object.get("schema").and_then(Value::as_str) {
        if schema != "clef-warm-shapes/1" {
            return Err(format!(
                "warm shapes {path:?}: unsupported schema {schema:?}, want \"clef-warm-shapes/1\""
            ));
        }
    }
    let state = object
        .get("state")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("warm shapes {path:?}: state must be a string"))?
        .to_string();
    let shapes = object
        .get("shapes")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("warm shapes {path:?}: shapes must be an array"))?;
    if shapes.is_empty() {
        return Err(format!("warm shapes {path:?}: shapes is empty"));
    }
    let mut parsed = Vec::with_capacity(shapes.len());
    for shape in shapes {
        let shape = shape
            .as_object()
            .ok_or_else(|| format!("warm shapes {path:?}: each shape must be an object"))?;
        let id = shape
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| format!("warm shapes {path:?}: shape id must be a non-empty string"))?
            .to_string();
        let dimension = |key: &str| -> Fallible<usize> {
            let value = shape
                .get(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("warm shapes {path:?}: shape {id} needs integer {key}"))?;
            usize::try_from(value)
                .ok()
                .filter(|value| (1..=8192).contains(value))
                .ok_or_else(|| format!("warm shapes {path:?}: shape {id} {key} out of range"))
        };
        let width = dimension("width")?;
        let height = dimension("height")?;
        let question = shape
            .get("question")
            .cloned()
            .ok_or_else(|| format!("warm shapes {path:?}: shape {id} needs a question"))?;
        if !question.is_object() {
            return Err(format!("warm shapes {path:?}: shape {id} question must be an object"));
        }
        let image = shape
            .get("image")
            .and_then(Value::as_str)
            .filter(|image| !image.is_empty())
            .map(PathBuf::from);
        parsed.push(WarmShape {
            id,
            width,
            height,
            image,
            question,
        });
    }
    Ok(WarmManifest {
        state,
        shapes: parsed,
    })
}

/// One synthetic full decide per listed shape, discarding the answers: this
/// builds the per-shape `Compiled` profile and its Metal JIT before `ready`.
fn warm_shapes(model: &mut LoadedModel, manifest: &WarmManifest) -> Fallible<(f64, Vec<Value>)> {
    let started = Instant::now();
    let mut rows = Vec::with_capacity(manifest.shapes.len());
    for shape in &manifest.shapes {
        let png = match &shape.image {
            Some(path) => std::fs::read(path)
                .map_err(|error| format!("warm image {path:?}: {error}"))?,
            None => solid_png(shape.width, shape.height)?,
        };
        let mut questions = Map::new();
        questions.insert(shape.id.clone(), shape.question.clone());
        let request = json!({
            "state": manifest.state,
            "questions": Value::Object(questions),
            "images": [encode_base64(&png)],
        });
        let shape_started = Instant::now();
        answer(model, &request).map_err(|error| format!("warm shape {}: {error}", shape.id))?;
        let ms = shape_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "clef warm: {} {}x{} {ms:.1} ms",
            shape.id, shape.width, shape.height
        );
        rows.push(json!({
            "id": shape.id,
            "width": shape.width,
            "height": shape.height,
            "ms": ms,
        }));
    }
    Ok((started.elapsed().as_secs_f64() * 1000.0, rows))
}

/// A valid 8-bit grayscale PNG of the exact crop size: raw rows with filter 0,
/// a stored-block zlib stream and manual CRCs (no image crate).
fn solid_png(width: usize, height: usize) -> Fallible<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err("solid PNG needs a positive size".into());
    }
    let stride = width + 1;
    let mut raw = vec![0u8; stride * height];
    for row in raw.chunks_exact_mut(stride) {
        row[0] = 0;
        row[1..].fill(0x80);
    }
    let compressed = zlib_stored(&raw);
    let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 0, 0, 0, 0]);
    png_chunk(&mut png, b"IHDR", &ihdr);
    png_chunk(&mut png, b"IDAT", &compressed);
    png_chunk(&mut png, b"IEND", &[]);
    Ok(png)
}

fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc = 0xFFFF_FFFFu32;
    for byte in kind.iter().chain(data) {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                0xEDB8_8320 ^ (crc >> 1)
            } else {
                crc >> 1
            };
        }
    }
    out.extend_from_slice(&(!crc).to_be_bytes());
}

/// zlib stream of stored (uncompressed) deflate blocks, which the server's
/// system-zlib `uncompress` accepts.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let blocks: Vec<&[u8]> = raw.chunks(0xFFFF).collect();
    let mut out = Vec::with_capacity(raw.len() + blocks.len() * 5 + 6);
    out.extend_from_slice(&[0x78, 0x01]);
    for (index, block) in blocks.iter().enumerate() {
        out.push(if index + 1 == blocks.len() { 0x01 } else { 0x00 });
        out.extend_from_slice(&(block.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        out.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let mut packed = 0u32;
        for (index, &byte) in chunk.iter().enumerate() {
            packed |= (byte as u32) << (16 - 8 * index);
        }
        for index in 0..4 {
            if index <= chunk.len() {
                out.push(ALPHABET[((packed >> (18 - 6 * index)) & 0x3F) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
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

/// Resident listen mode: one process, the model loaded once, clients connect
/// and speak the same JSONL protocol as the pipe mode. Connections are served
/// sequentially on the request thread: the demo's episodes are sequential, so
/// a client queue is the correct concurrency policy and no `Send` bound on the
/// model is needed. Every connection first receives the startup `ready` line
/// verbatim, so a per-episode proxy behaves exactly as in the pipe mode.
#[cfg(unix)]
fn serve_listen(model: &mut LoadedModel, ready: &Value, addr: &ListenAddr) -> Fallible<()> {
    // Remove the socket file when the process exits cleanly; a SIGKILL leaves
    // it behind and the next bind removes it first.
    struct Cleanup(Option<PathBuf>);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            if let Some(path) = &self.0 {
                let _ = std::fs::remove_file(path);
            }
        }
    }

    match addr {
        // Bind after the model (and its warm shapes) is fully up: a client
        // that can connect is guaranteed the warmup already happened.
        ListenAddr::Unix(path) => {
            if path.exists() {
                std::fs::remove_file(path).map_err(|e| format!("remove stale {path:?}: {e}"))?;
            }
            let listener = std::os::unix::net::UnixListener::bind(path)
                .map_err(|e| format!("bind {path:?}: {e}"))?;
            eprintln!("clef decide-server: listening on unix:{path:?}");
            let _cleanup = Cleanup(Some(path.clone()));
            loop {
                let (stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) => {
                        eprintln!("accept: {error}");
                        continue;
                    }
                };
                eprintln!("clef decide-server: client connected (unix)");
                if let Err(error) = serve_client(model, ready, &stream, &stream) {
                    eprintln!("clef decide-server: unix client: {error}");
                }
                eprintln!("clef decide-server: unix client disconnected");
            }
        }
        ListenAddr::Tcp(address) => {
            let listener =
                std::net::TcpListener::bind(address).map_err(|e| format!("bind {address}: {e}"))?;
            eprintln!("clef decide-server: listening on tcp:{address}");
            let _cleanup = Cleanup(None);
            loop {
                let (stream, peer) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) => {
                        eprintln!("accept: {error}");
                        continue;
                    }
                };
                eprintln!("clef decide-server: client {peer} connected");
                if let Err(error) = serve_client(model, ready, &stream, &stream) {
                    eprintln!("clef decide-server: client {peer}: {error}");
                }
                eprintln!("clef decide-server: client {peer} disconnected");
            }
        }
    }
}

/// One connected client: the ready line, then JSONL until the peer closes its
/// write side or stops reading. The request clock and answers are identical to
/// the pipe mode. `&TcpStream`/`&UnixStream` implement both `Read` and
/// `Write`, so the caller passes the same peer reference for both halves.
#[cfg(unix)]
fn serve_client(
    model: &mut LoadedModel,
    ready: &Value,
    peer_in: impl std::io::Read,
    mut peer_out: impl std::io::Write,
) -> Fallible<()> {
    let mut reader = std::io::BufReader::new(peer_in);
    writeln!(peer_out, "{ready}").map_err(|e| format!("write ready: {e}"))?;
    peer_out.flush().map_err(|e| format!("flush ready: {e}"))?;
    let mut line = String::new();
    loop {
        line.clear();
        match std::io::BufRead::read_line(&mut reader, &mut line) {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(error) => return Err(format!("read: {error}")),
        }
        if line.trim().is_empty() {
            continue;
        }
        let response = handle(model, &line);
        writeln!(peer_out, "{response}").map_err(|e| format!("write: {e}"))?;
        peer_out.flush().map_err(|e| format!("flush: {e}"))?;
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
    fn listen_addr_parses_unix_and_tcp_and_rejects_the_rest() {
        match ListenAddr::parse("unix:/tmp/clef.sock").unwrap() {
            ListenAddr::Unix(path) => assert_eq!(path, PathBuf::from("/tmp/clef.sock")),
            ListenAddr::Tcp(_) => panic!("unix: must parse as Unix"),
        }
        match ListenAddr::parse("tcp:127.0.0.1:7878").unwrap() {
            ListenAddr::Tcp(address) => assert_eq!(address, "127.0.0.1:7878"),
            ListenAddr::Unix(_) => panic!("tcp: must parse as Tcp"),
        }
        for value in ["", "unix:", "tcp:", "tcp:nohost", "http://x", "clef.sock"] {
            assert!(ListenAddr::parse(value).is_err(), "accepted {value:?}");
        }
    }

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
    fn warm_manifest_parses_the_demo_shapes_and_rejects_broken_ones() {
        let dir = std::env::temp_dir().join(format!("clef-warm-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("warm.json");
        std::fs::write(
            &path,
            r#"{"schema":"clef-warm-shapes/1","state":"Answer each question using only the picture it refers to.",
                "shapes":[{"id":"enemy","width":320,"height":400,
                           "question":{"type":"noul","instructions":"Close-up:\nIs there an enemy?"}}]}"#,
        )
        .unwrap();
        let manifest = load_warm_manifest(&path).unwrap();
        assert_eq!(manifest.shapes.len(), 1);
        assert_eq!(manifest.shapes[0].id, "enemy");
        assert_eq!((manifest.shapes[0].width, manifest.shapes[0].height), (320, 400));
        assert!(manifest.shapes[0].image.is_none());

        for broken in [
            r#"{"state":"s","shapes":[]}"#,
            r#"{"state":"s","shapes":[{"id":"x","width":0,"height":10,"question":{}}]}"#,
            r#"{"state":"s","shapes":[{"id":"x","width":10,"height":10}]}"#,
            r#"{"schema":"other","state":"s","shapes":[{"id":"x","width":10,"height":10,"question":{}}]}"#,
        ] {
            std::fs::write(&path, broken).unwrap();
            assert!(load_warm_manifest(&path).is_err(), "accepted {broken}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn solid_png_is_a_well_formed_gray_png_and_round_trips_through_base64() {
        let png = solid_png(320, 400).unwrap();
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..24], &[0, 0, 1, 64, 0, 0, 1, 144]);
        assert_eq!(&png[24..29], &[8, 0, 0, 0, 0]);
        // The IDAT inflates to (width + 1) * height raw bytes; the server's
        // decoder accepts exactly this shape.
        assert!(png.len() > 320 * 400);
        let encoded = encode_base64(&png);
        assert_eq!(decode_base64(&encoded).unwrap(), png);
        assert_eq!(encode_base64(b"hello"), "aGVsbG8=");
        assert_eq!(encode_base64(b"hell"), "aGVsbA==");
        assert!(solid_png(0, 4).is_err());
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
