//! Resident d3-lite decision process for the Mario demo (`APXINF_DECIDE_CMD`).
//!
//! ```text
//! apxinf-decide-server --model <checkpoint> [--model-name NAME] [--device metal:N]
//!                       [--concurrency N] [--warm-shapes <file>]
//! ```
//!
//! One checkpoint, one `{"type":"ready",…}` line on stdout, then one JSONL
//! request → one JSONL response per stdin line:
//!
//! ```text
//! {"id":1,"state":"…","questions":{"enemy":{"type":"noul","instructions":"…"}},
//!  "images":["<base64 png>"]}
//! {"id":1,"answers":{"enemy":{"type":"noul","noul":0.42}},
//!  "p_bits":{"enemy":"3e20e560"},"latency_ms":184.2,"stages":{…}}
//! ```
//!
//! This is the Clef `apxinf-decide-server` protocol, so the demo's
//! `clef_local.ClefLocalWorker` drives it unchanged: no new bridge, and the
//! Clef bridge's behaviour is untouched. A malformed request answers
//! `{"id":…,"error":"…"}` and the process keeps serving; only a load failure,
//! exhausted stdin or a closed stdout ends it.
//!
//! **`p_bits`.** Every successful response also carries `p_bits`: one
//! 8-hex-digit `f32::to_bits` per question key in `answers`, taken from the
//! reported `noul` probability (`confidence` for `choice`, which reports no
//! `noul` field). It exists so two runs can be compared bitwise; it changes
//! no existing field.
//!
//! **Concurrency.** `--concurrency N` (env `APXINF_DECIDE_CONCURRENCY`,
//! default 1) starts N worker threads. Each worker builds and owns its own
//! `Server`/`LoadedModel` entirely on that thread — the model is `!Send`
//! (`Rc` handles, `RefCell` caches, thread-local MLX stream) and is never
//! shared or locked. The main thread reads stdin and dispatches request
//! `String`s into **per-question worker queues**: a request naming exactly one
//! manifest question goes to the worker that owns that question (worker `i`
//! owns shape index `i` mod N), while an unknown/unmapped question, a joint
//! body or malformed JSON falls back to the least-loaded worker, which answers
//! with the existing lazy `prepare`. Backpressure bounds the total queued at
//! N; one writer thread emits responses under a single stdout lock, in
//! completion order. Memory cost is N × model, which is expected. N = 1 stays
//! on the original serial path.
//!
//! **Warm start (`--warm-shapes` / `APXINF_DECIDE_WARM_SHAPES`).** The
//! manifest names the demo's shapes (question id, exact crop size, fixed
//! prompt) and a shared `state`. Before the single `ready` line, every worker
//! runs **one full synthetic `decide`** per shape it owns — serial (N = 1)
//! warms all three, each pooled worker warms exactly its own — discarding the
//! answer. A full decide is required because the vision tower is eager: a
//! zero-feature `prepare_image_profile` does not warm its first-use Metal JIT.
//! `ready.load_ms` then reports the wall of load + warm and the additive
//! `ready.warm_ms` splits the two off. With no manifest the server behaves
//! exactly as before (cold start, lazy prepare).
//!
//! **Image contract.** d3-lite writes one contiguous image-placeholder run
//! (P1b), so one request carries at most one image. The demo's separate
//! protocol (`MARIO_CLEF_PROTOCOL=separate`, the Clef reference's own startup
//! parameter) is one crop + one Noul question per request; `joint` would send
//! three images and is rejected by name.
//!
//! The request clock covers base64 + PNG decode, the checkpoint's
//! resize/normalize/patchify, prompt render + tokenize and the model call —
//! the same boundary the Clef server and `d3_lite_bench` use.
//!
//! The stage split (`stages`) is the runner's host wall-clock attribution:
//! `host_prep` (decode + preprocess + prompt encode), then `layout`, `vision`,
//! `forward` (text layers + FP32 readout) and `sync` from the model profile.
//! On the **first request of a shape** (a prepare ran during the call) the
//! additive `t1_ms`/`t2_ms`/`t3_ms` fields split that request further:
//! `t1` = first `call_and_eval` inside `prepare` (trace + compile + Metal JIT
//! + eval), `t2` = the replay (GPU only) and `t3` = the real request call, so
//! `t1 − t2` is the trace/compile cost. They are JSON `null` on every
//! steady-state or already-warm request.

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod live {
    use std::collections::{HashMap, VecDeque};
    use std::io::{BufRead, Write};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, RecvTimeoutError, Sender};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::time::{Duration, Instant};

    use apxinf_core::{Device, Tensor};
    use apxinf_model::metal_d3_lite::{preprocess, D3LiteFormat};
    use apxinf_model::{
        AutoModel, DecisionInput, DecisionOutput, DecisionQuestion, ImageInput, LoadOptions,
        LoadedModel,
    };
    use base64::decode as base64_decode;
    use serde_json::{json, Map, Value};

    type Fallible<T> = Result<T, String>;

    /// The compiled whole-request callable always runs over BF16 weights with
    /// the FP32 readout (`admit` accepts no other precision).
    const VARIANT: &str = "bf16-compiled";

    pub fn main() {
        let options = match parse_options() {
            Ok(options) => options,
            Err(error) => fail_start(&error),
        };
        let manifest = match load_warm_manifest(&options) {
            Ok(manifest) => manifest,
            Err(error) => fail_start(&error),
        };
        if options.concurrency == 1 {
            serial(&options, manifest.as_ref());
        } else {
            parallel(options, manifest);
        }
    }

    /// A start-up failure: print the demo's failed-start line and exit 1.
    fn fail_start(error: &str) -> ! {
        let _ = writeln!(
            std::io::stdout(),
            "{}",
            json!({"type": "error", "error": error})
        );
        let _ = std::io::stdout().flush();
        std::process::exit(1);
    }

    #[derive(Clone, Debug)]
    struct Options {
        model: PathBuf,
        model_name: Option<String>,
        device: Device,
        concurrency: usize,
        /// `--warm-shapes <file>` (env `APXINF_DECIDE_WARM_SHAPES`). `None`
        /// keeps the pre-warm behaviour exactly: no load-time warm, lazy
        /// prepare on the first request for each shape.
        warm_shapes: Option<PathBuf>,
    }

    /// One entry of the warm manifest: the question id, the exact crop
    /// dimensions the demo sends to the server (post-upscale, so the
    /// checkpoint's resize/grid and therefore the compiled profile match) and
    /// the fixed prompt the real request carries.
    #[derive(Clone, Debug)]
    struct WarmShape {
        id: String,
        width: usize,
        height: usize,
        instructions: String,
    }

    /// The `--warm-shapes` file: a shared `state` plus the shapes to warm.
    #[derive(Clone, Debug)]
    struct WarmManifest {
        state: String,
        shapes: Vec<WarmShape>,
    }

    impl WarmManifest {
        /// The worker that owns `shape_index` under question-affine
        /// ownership: worker `i` owns every shape whose index is `i` mod
        /// `workers`. For the campaign's N = 3 that is one shape each
        /// (enemy→0, obstacle→1, gap→2); serial (N = 1) owns all three; N = 2
        /// splits 2+1; extra workers (N > 3) own nothing and serve only the
        /// fallback path.
        fn owner(&self, shape_index: usize, workers: usize) -> usize {
            shape_index % workers
        }

        /// `question id -> owning worker`, the routing table for the pool.
        fn owners(&self, workers: usize) -> HashMap<String, usize> {
            self.shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| (shape.id.clone(), self.owner(index, workers)))
                .collect()
        }
    }

    fn parse_options() -> Fallible<Options> {
        let mut model = None;
        let mut model_name = None;
        let mut device = Device::Metal(0);
        // The flag wins over the environment; the environment wins over 1.
        let mut concurrency = match std::env::var("APXINF_DECIDE_CONCURRENCY") {
            Ok(value) => parse_concurrency("APXINF_DECIDE_CONCURRENCY", &value)?,
            Err(_) => 1,
        };
        // The manifest is opt-in; the flag wins over the environment, which
        // is how the attribution runs stay unwarmed.
        let mut warm_shapes = std::env::var_os("APXINF_DECIDE_WARM_SHAPES")
            .map(PathBuf::from)
            .filter(|path| !path.as_os_str().is_empty());
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" | "--snapshot" => {
                    model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?))
                }
                "--model-name" => {
                    model_name = Some(args.next().ok_or("--model-name needs a value")?)
                }
                "--device" => {
                    let value = args.next().ok_or("--device needs a value")?;
                    device = parse_device(&value)?
                }
                "--concurrency" => {
                    let value = args.next().ok_or("--concurrency needs a value")?;
                    concurrency = parse_concurrency("--concurrency", &value)?
                }
                "--warm-shapes" => {
                    let value = args.next().ok_or("--warm-shapes needs a file")?;
                    if value.is_empty() {
                        return Err("--warm-shapes needs a non-empty file".into());
                    }
                    warm_shapes = Some(PathBuf::from(value));
                }
                "--help" | "-h" => {
                    return Err("usage: apxinf-decide-server --model <checkpoint> \
                                 [--model-name NAME] [--device metal:N] \
                                 [--concurrency N] [--warm-shapes <file>]"
                        .into())
                }
                other if model.is_none() && !other.starts_with('-') => {
                    model = Some(PathBuf::from(other))
                }
                other => return Err(format!("unexpected argument {other}")),
            }
        }
        Ok(Options {
            model: model.ok_or("--model <checkpoint> is required")?,
            model_name,
            device,
            concurrency,
            warm_shapes,
        })
    }

    /// Read and validate `--warm-shapes <file>`. A present-but-broken
    /// manifest is a start-up error (an explicit warm request that cannot be
    /// honoured must fail loudly, not silently fall back to cold start).
    fn load_warm_manifest(options: &Options) -> Fallible<Option<WarmManifest>> {
        let Some(path) = &options.warm_shapes else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("warm shapes {path:?}: {error}"))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| format!("warm shapes {path:?}: {error}"))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("warm shapes {path:?}: expected a JSON object"))?;
        if let Some(schema) = object.get("schema").and_then(Value::as_str) {
            if schema != "d3-lite-warm-shapes/1" {
                return Err(format!(
                    "warm shapes {path:?}: unsupported schema {schema:?}, \
                     want \"d3-lite-warm-shapes/1\""
                ));
            }
        }
        let state = object
            .get("state")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("warm shapes {path:?}: needs a string `state`"))?;
        if state.is_empty() {
            return Err(format!("warm shapes {path:?}: `state` must not be empty"));
        }
        let raw_shapes = object
            .get("shapes")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("warm shapes {path:?}: needs a `shapes` array"))?;
        if raw_shapes.is_empty() {
            return Err(format!("warm shapes {path:?}: `shapes` must not be empty"));
        }
        let mut shapes = Vec::with_capacity(raw_shapes.len());
        let mut seen = std::collections::HashSet::new();
        for (index, raw) in raw_shapes.iter().enumerate() {
            let entry = raw
                .as_object()
                .ok_or_else(|| format!("warm shapes {path:?}: shape {index} is not an object"))?;
            let id = entry
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .ok_or_else(|| {
                    format!("warm shapes {path:?}: shape {index} needs a non-empty `id`")
                })?
                .to_string();
            if !seen.insert(id.clone()) {
                return Err(format!("warm shapes {path:?}: duplicate shape id {id:?}"));
            }
            let width = manifest_dimension(path, &id, entry, "width")?;
            let height = manifest_dimension(path, &id, entry, "height")?;
            let instructions = entry
                .get("instructions")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .ok_or_else(|| {
                    format!("warm shapes {path:?}: shape {id:?} needs non-empty `instructions`")
                })?
                .to_string();
            shapes.push(WarmShape {
                id,
                width,
                height,
                instructions,
            });
        }
        Ok(Some(WarmManifest {
            state: state.to_string(),
            shapes,
        }))
    }

    fn manifest_dimension(
        path: &Path,
        id: &str,
        entry: &Map<String, Value>,
        key: &str,
    ) -> Fallible<usize> {
        let value = entry
            .get(key)
            .and_then(Value::as_u64)
            .filter(|&value| value > 0 && value <= 16_384)
            .ok_or_else(|| {
                format!("warm shapes {path:?}: shape {id:?} needs `{key}` in 1..=16384")
            })?;
        Ok(value as usize)
    }

    fn parse_concurrency(flag: &str, value: &str) -> Fallible<usize> {
        let concurrency = value
            .parse::<usize>()
            .map_err(|_| format!("{flag} must be a positive integer, got {value:?}"))?;
        if concurrency == 0 {
            return Err(format!("{flag} must be at least 1"));
        }
        Ok(concurrency)
    }

    fn parse_device(value: &str) -> Fallible<Device> {
        let (kind, index) = value
            .split_once(':')
            .map_or((value, Some(0usize)), |(kind, index)| {
                (kind, index.parse::<usize>().ok())
            });
        match (kind, index) {
            ("metal", Some(index)) => Ok(Device::Metal(index)),
            ("metal", None) => Err("--device metal:N needs a numeric index".into()),
            _ => Err(format!(
                "unsupported device {value:?}; this server runs metal:N"
            )),
        }
    }

    /// Load the resident model and the prompt contract, and verify the
    /// package's code→token mapping against the tokenizer. Printing the
    /// `ready` line is the caller's job (one line, from one thread).
    struct Server {
        model: LoadedModel,
        format: D3LiteFormat,
        model_name: String,
    }

    fn build(options: &Options) -> Fallible<(Server, f64)> {
        let started = Instant::now();
        let format = D3LiteFormat::from_checkpoint(&options.model)
            .map_err(|error| format!("format: {error}"))?;
        // Prove the decision_config codes/token IDs are the tokenizer's own
        // result (`d3_format.answer_codes`), not a transcription, before the
        // model is loaded.
        let (probe_codes, probe_ids) = format
            .answer_codes()
            .map_err(|error| format!("answer-code probe: {error}"))?;
        if probe_codes != format.package_codes() || probe_ids != format.package_token_ids() {
            return Err(
                "d3_format.answer_codes disagrees with decision_config.json codes/token_ids".into(),
            );
        }
        let mut load_options = LoadOptions::default();
        load_options.model_name = options.model_name.clone();
        let model = AutoModel::load_model(options.device, &options.model, &load_options)
            .map_err(|error| format!("load: {error}"))?;
        let load_ms = started.elapsed().as_secs_f64() * 1e3;
        let model_name = load_options
            .model_name
            .or_else(|| AutoModel::detect_model_name(&options.model).ok())
            .unwrap_or_else(|| "metal_d3".into());
        Ok((
            Server {
                model,
                format,
                model_name,
            },
            load_ms,
        ))
    }

    /// The one `ready` line, printed by whichever thread owns the startup
    /// decision — the main thread in both the serial and pooled paths.
    /// `load_ms` is the wall of load **plus** the pre-ready warm (additive
    /// `warm_ms` separates the two so the demo can see what spawn→ready is
    /// made of).
    fn print_ready(model_name: &str, load_ms: f64, warm_ms: f64) -> Fallible<()> {
        writeln!(
            std::io::stdout(),
            "{}",
            json!({
                "type": "ready",
                "model": model_name,
                "variant": VARIANT,
                "load_ms": load_ms,
                "warm_ms": warm_ms,
                "codes_verified": true,
            })
        )
        .map_err(|error| format!("write ready: {error}"))?;
        std::io::stdout()
            .flush()
            .map_err(|error| format!("flush ready: {error}"))
    }

    /// N = 1: the original serial path — build on the main thread, warm every
    /// manifest shape (fairness: serial gets the *same* total warmup the
    /// pooled arm's three workers get, on the same clock), print `ready`, then
    /// read a line, answer it, write the response.
    fn serial(options: &Options, manifest: Option<&WarmManifest>) {
        let (mut server, load_ms) = match build(options) {
            Ok(loaded) => loaded,
            Err(error) => fail_start(&error),
        };
        let warm_ms = match manifest {
            Some(manifest) => match warm_owned(&mut server, manifest, 1, 0) {
                Ok(milliseconds) => milliseconds,
                Err(error) => fail_start(&error),
            },
            None => 0.0,
        };
        if let Err(error) = print_ready(&server.model_name, load_ms + warm_ms, warm_ms) {
            fail_start(&error);
        }
        serve(server);
    }

    /// One full synthetic `decide` for every manifest shape this worker owns,
    /// output discarded, and the wall of that warm. This is the load-time half
    /// of the fairness rule: with `workers = 1` (serial) it warms all three
    /// shapes; with `workers = 3` each pooled worker warms exactly its own.
    ///
    /// The request is *full*, not a zero-feature `prepare_image_profile`:
    /// the vision tower is eager, so only a real forward warms its first-use
    /// Metal JIT. The exact crop size reproduces the real request's
    /// resize/grid (hence the same compiled profile) and the fixed prompt
    /// reproduces its token length.
    fn warm_owned(
        server: &mut Server,
        manifest: &WarmManifest,
        workers: usize,
        index: usize,
    ) -> Fallible<f64> {
        let started = Instant::now();
        for (shape_index, shape) in manifest.shapes.iter().enumerate() {
            if manifest.owner(shape_index, workers) != index {
                continue;
            }
            let request = synthetic_request(shape, &manifest.state)
                .map_err(|error| format!("warm {}: {error}", shape.id))?;
            // The answer (and its latency/profile) is deliberately dropped:
            // the point is the first-use setup, not the value.
            let _ = answer(&mut server.model, &server.format, &request)
                .map_err(|error| format!("warm {}: {error}", shape.id))?;
        }
        Ok(started.elapsed().as_secs_f64() * 1e3)
    }

    /// The synthetic warm request: a solid-black PNG of the shape's exact
    /// crop size plus its fixed prompt, in the real request shape so it runs
    /// through the same decode → preprocess → encode → `decide` path.
    fn synthetic_request(shape: &WarmShape, state: &str) -> Fallible<Value> {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
            shape.width as u32,
            shape.height as u32,
            image::Rgb([0u8, 0u8, 0u8]),
        ));
        let mut buffer = std::io::Cursor::new(Vec::new());
        image
            .write_to(&mut buffer, image::ImageFormat::Png)
            .map_err(|error| format!("synthetic PNG: {error}"))?;
        let mut questions = Map::new();
        questions.insert(
            shape.id.clone(),
            json!({"type": "noul", "instructions": shape.instructions.clone()}),
        );
        Ok(json!({
            "id": 0,
            "state": state,
            "questions": Value::Object(questions),
            "images": [base64::encode(buffer.into_inner())],
        }))
    }

    fn serve(mut server: Server) {
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
            let response = handle(&mut server.model, &server.format, &server.model_name, &line);
            if writeln!(stdout, "{response}").is_err() || stdout.flush().is_err() {
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // Request-level concurrency (N > 1)
    // ------------------------------------------------------------------

    /// Shared pool state: one FIFO **per worker** (a worker's queue is its
    /// question's queue under question-affine ownership), the total queued
    /// count that bounds the producer, and the in-flight/live counters. One
    /// `Mutex` + condvar stands in for the multi-consumer bounded channel
    /// (std's `Receiver` cannot be cloned); the lock is held only for queue
    /// bookkeeping, never across a model call. `close()` is the shutdown
    /// signal. Requests whose question is unknown, unmapped, or whose body
    /// does not name exactly one question take the fallback path: they are
    /// pushed to the least-loaded worker, which answers with the existing
    /// lazy `prepare`.
    struct Pool {
        state: Mutex<PoolState>,
        closed: AtomicBool,
        changed: Condvar,
        /// Queued + currently processing; reaches 0 when all work is done.
        outstanding: AtomicUsize,
        /// Workers that have not yet exited; 0 means nobody can drain more.
        live: AtomicUsize,
        /// Set by the writer when stdout breaks.
        stdout_broken: AtomicBool,
        /// Question id → owning worker index; built from the warm manifest.
        owners: HashMap<String, usize>,
    }

    /// The lock-guarded half of [`Pool`]. `queued` is the global
    /// backpressure counter (the shared FIFO's old `len()`), so a producer
    /// still blocks once `workers` requests are waiting.
    struct PoolState {
        queues: Vec<VecDeque<String>>,
        queued: usize,
    }

    impl Pool {
        fn new(workers: usize, owners: HashMap<String, usize>) -> Self {
            Self {
                state: Mutex::new(PoolState {
                    queues: (0..workers).map(|_| VecDeque::new()).collect(),
                    queued: 0,
                }),
                closed: AtomicBool::new(false),
                changed: Condvar::new(),
                outstanding: AtomicUsize::new(0),
                live: AtomicUsize::new(workers),
                stdout_broken: AtomicBool::new(false),
                owners,
            }
        }

        /// Stop accepting work. Workers drain whatever is queued, then exit.
        fn close(&self) {
            self.closed.store(true, Ordering::SeqCst);
            self.changed.notify_all();
        }

        fn shutting_down(&self) -> bool {
            self.closed.load(Ordering::SeqCst)
                || self.stdout_broken.load(Ordering::SeqCst)
                || self.live.load(Ordering::SeqCst) == 0
        }
    }

    /// Lock the queues, recovering from a poisoned mutex left by a panicked
    /// worker: the queues themselves are still usable.
    fn state_lock(pool: &Pool) -> MutexGuard<'_, PoolState> {
        pool.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn state_wait<'a>(pool: &Pool, guard: MutexGuard<'a, PoolState>) -> MutexGuard<'a, PoolState> {
        pool.changed
            .wait(guard)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The owning worker for this line, or `None` for the fallback path.
    /// Exactly one question with a manifest entry routes question-affinely;
    /// anything else (joint body, unknown question, malformed JSON) falls
    /// back to any worker, which lazy-prepares the shape.
    fn routed_worker(pool: &Pool, line: &str) -> Option<usize> {
        let request: Value = serde_json::from_str(line).ok()?;
        let questions = request.get("questions")?.as_object()?;
        if questions.len() != 1 {
            return None;
        }
        let id = questions.keys().next()?;
        pool.owners.get(id).copied()
    }

    /// The fallback target: least queued, ties broken by lowest index so the
    /// choice is deterministic for a given state.
    fn fallback_worker(state: &PoolState) -> usize {
        (0..state.queues.len())
            .min_by_key(|&index| state.queues[index].len())
            .unwrap_or(0)
    }

    /// One popped job: decrements `outstanding` exactly once, including while
    /// unwinding a worker panic. It takes the queue lock before notifying so
    /// a drain waiter that has checked the counter can never miss the signal.
    struct Outstanding {
        pool: Arc<Pool>,
    }

    impl Outstanding {
        fn claim(pool: &Pool) {
            pool.outstanding.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl Drop for Outstanding {
        fn drop(&mut self) {
            let _state = state_lock(&self.pool);
            self.pool.outstanding.fetch_sub(1, Ordering::SeqCst);
            self.pool.changed.notify_all();
        }
    }

    /// One worker's load+warm handshake. `load_ms` is the load phase alone;
    /// the ready line adds `warm_ms` per worker before taking the max, so
    /// `ready.load_ms` is the wall of load+warm.
    struct Handshake {
        load_ms: f64,
        warm_ms: f64,
        model_name: String,
    }

    /// N worker threads. Each builds its own `Server`/`LoadedModel` here, on
    /// this thread, warms the shapes it owns, and uses it only here — the
    /// model is `!Send` and is never shared, moved, or wrapped in a `Mutex`.
    /// The single `ready` is printed by the main thread only after every
    /// worker's handshake (load **and** warm) has arrived.
    fn parallel(options: Options, manifest: Option<WarmManifest>) {
        let workers = options.concurrency;
        let owners = manifest
            .as_ref()
            .map(|manifest| manifest.owners(workers))
            .unwrap_or_default();
        let pool = Arc::new(Pool::new(workers, owners));
        let (handshake_tx, handshake_rx) = mpsc::channel::<Result<Handshake, String>>();
        let (response_tx, response_rx) = mpsc::channel::<String>();

        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let pool = Arc::clone(&pool);
            let handshake_tx = handshake_tx.clone();
            let response_tx = response_tx.clone();
            let options = options.clone();
            let manifest = manifest.clone();
            handles.push(
                std::thread::Builder::new()
                    .name(format!("decide-worker-{index}"))
                    .spawn(move || {
                        worker(index, options, manifest, pool, handshake_tx, response_tx)
                    })
                    .expect("spawn decide worker"),
            );
        }
        // Only the workers produce responses from here on.
        drop(handshake_tx);
        // The main thread never writes response lines; drop its sender so
        // the writer sees the channel close once every worker has exited.
        drop(response_tx);

        // Collect one handshake per worker (load + pre-ready warm). A worker
        // that dies before its handshake is noticed through the live counter
        // (or its panic line).
        let mut handshakes = Vec::with_capacity(workers);
        let mut received = 0usize;
        let mut failure: Option<String> = None;
        while received < workers && failure.is_none() {
            match handshake_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(Ok(handshake)) => {
                    handshakes.push(handshake);
                    received += 1;
                }
                Ok(Err(error)) => {
                    failure = Some(error);
                    received += 1;
                }
                Err(RecvTimeoutError::Timeout) => {
                    if pool.live.load(Ordering::SeqCst) + received < workers {
                        failure = Some("worker exited before handshake".into());
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    failure = Some("worker channel closed before handshake".into());
                }
            }
        }

        if let Some(error) = failure {
            pool.close();
            for handle in handles {
                let _ = handle.join();
            }
            fail_start(&error);
        }

        // All workers loaded **and warmed**: one `ready` line from the main
        // thread. Workers run concurrently, so the wall to ready is the slowest
        // worker's load+warm; `warm_ms` separates the two (additive).
        let model_name = handshakes
            .first()
            .map(|handshake| handshake.model_name.clone())
            .unwrap_or_else(|| "metal_d3".into());
        let load_ms = handshakes
            .iter()
            .map(|handshake| handshake.load_ms + handshake.warm_ms)
            .fold(0.0, f64::max);
        let warm_ms = handshakes
            .iter()
            .map(|handshake| handshake.warm_ms)
            .fold(0.0, f64::max);
        if let Err(error) = print_ready(&model_name, load_ms, warm_ms) {
            pool.close();
            for handle in handles {
                let _ = handle.join();
            }
            fail_start(&error);
        }

        // One writer owns the stdout lock for its whole life and emits
        // responses in completion order.
        let writer = {
            let pool = Arc::clone(&pool);
            std::thread::Builder::new()
                .name("decide-writer".into())
                .spawn(move || {
                    let stdout = std::io::stdout();
                    let mut out = stdout.lock();
                    for response in response_rx {
                        if writeln!(out, "{response}").is_err() || out.flush().is_err() {
                            // Broken stdout: stop the producer and the
                            // dispatcher; in-flight work still drains.
                            pool.stdout_broken.store(true, Ordering::SeqCst);
                            pool.close();
                            break;
                        }
                    }
                })
                .expect("spawn decide writer")
        };

        // Main thread: read stdin and dispatch into the bounded per-question
        // queues. Routing is computed before taking the lock (it parses the
        // request JSON, image and all).
        let capacity = workers;
        let stdin = std::io::stdin();
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
            let routed = routed_worker(&pool, &line);
            // Backpressure: block while `capacity` requests are waiting,
            // until shutdown. `queued` is the shared FIFO's old `len()`.
            let mut state = state_lock(&pool);
            while state.queued >= capacity && !pool.shutting_down() {
                state = state_wait(&pool, state);
            }
            if pool.shutting_down() {
                break;
            }
            // A mapped question goes to its owner; anything else (unknown
            // question/shape, joint body, bad JSON) falls back to the
            // least-loaded worker, which lazy-prepares.
            let target = routed.unwrap_or_else(|| fallback_worker(&state));
            state.queues[target].push_back(line);
            state.queued += 1;
            Outstanding::claim(&pool);
            // The woken worker may own a different question than the one
            // just queued, so wake all of them (N is small).
            pool.changed.notify_all();
        }

        // EOF (or shutdown): close the queues, workers drain them, then wait
        // for in-flight work to finish. If every worker is gone there is
        // nothing left to drain, so we do not wait forever.
        pool.close();
        {
            let mut state = state_lock(&pool);
            while pool.outstanding.load(Ordering::SeqCst) > 0
                && pool.live.load(Ordering::SeqCst) > 0
            {
                state = state_wait(&pool, state);
            }
        }

        for handle in handles {
            let _ = handle.join();
        }
        let _ = writer.join();
    }

    fn worker(
        index: usize,
        options: Options,
        manifest: Option<WarmManifest>,
        pool: Arc<Pool>,
        handshake: Sender<Result<Handshake, String>>,
        responses: Sender<String>,
    ) {
        // A panic anywhere in this worker must still decrement `live` and
        // wake the main thread's drain/dispatch waits.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut server, load_ms) = match build(&options) {
                Ok(loaded) => loaded,
                Err(error) => {
                    let _ = handshake.send(Err(error));
                    return;
                }
            };
            // Warm the shapes this worker owns BEFORE the single `ready`.
            // The main thread cannot print ready until every handshake
            // arrives, so the warm wall delays spawn→ready, not decision 1.
            // With no manifest this is 0 and behaviour is unchanged.
            let warm_ms = match &manifest {
                Some(manifest) => {
                    match warm_owned(&mut server, manifest, options.concurrency, index) {
                        Ok(milliseconds) => milliseconds,
                        Err(error) => {
                            let _ = handshake.send(Err(error));
                            return;
                        }
                    }
                }
                None => 0.0,
            };
            if handshake
                .send(Ok(Handshake {
                    load_ms,
                    warm_ms,
                    model_name: server.model_name.clone(),
                }))
                .is_err()
            {
                return;
            }
            serve_worker(index, server, &pool, responses);
        }));
        {
            let _state = state_lock(&pool);
            pool.live.fetch_sub(1, Ordering::SeqCst);
            pool.changed.notify_all();
        }
        if let Err(payload) = outcome {
            eprintln!("decide worker panicked: {}", panic_message(payload));
        }
    }

    /// Serve from this worker's own queue only. A closed queue still yields
    /// its queued work before the worker exits.
    fn serve_worker(index: usize, mut server: Server, pool: &Arc<Pool>, responses: Sender<String>) {
        loop {
            let job = {
                let mut state = state_lock(pool);
                loop {
                    if let Some(job) = state.queues[index].pop_front() {
                        state.queued -= 1;
                        pool.changed.notify_all(); // space for the producer
                        break Some(job);
                    }
                    if pool.closed.load(Ordering::SeqCst) {
                        break None;
                    }
                    state = state_wait(pool, state);
                }
            };
            let Some(line) = job else { break };
            let guard = Outstanding {
                pool: Arc::clone(pool),
            };
            let response =
                handle(&mut server.model, &server.format, &server.model_name, &line).to_string();
            if responses.send(response).is_err() {
                break; // writer gone; other workers drain the rest
            }
            drop(guard);
        }
    }

    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_string()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "non-string panic payload".into()
        }
    }

    // ------------------------------------------------------------------

    /// One request line in, one response line out; never fails the process.
    fn handle(
        model: &mut LoadedModel,
        format: &D3LiteFormat,
        model_name: &str,
        line: &str,
    ) -> Value {
        let request: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(error) => {
                return json!({"id": Value::Null, "error": format!("request is not JSON: {error}")})
            }
        };
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        match answer(model, format, &request) {
            Ok((answers, p_bits, latency_ms, stages)) => json!({
                "id": id,
                "answers": answers,
                "p_bits": p_bits,
                "latency_ms": latency_ms,
                "stages": stages,
                "model": format_package(format, model_name),
            }),
            Err(error) => json!({"id": id, "error": error}),
        }
    }

    fn format_package(format: &D3LiteFormat, model_name: &str) -> Value {
        json!({
            "name": model_name,
            "format_id": apxinf_model::metal_d3_lite::FORMAT_ID,
            "codes": format.package_codes().len(),
        })
    }

    /// One request: every question is one prompt; with an image, every
    /// question sees the request's single image (the separate protocol asks
    /// one question per image).
    ///
    /// Returns the answers, the matching `p_bits` object, the latency and
    /// the stage split.
    fn answer(
        model: &mut LoadedModel,
        format: &D3LiteFormat,
        request: &Value,
    ) -> Fallible<(Value, Value, f64, Value)> {
        let started = Instant::now();
        let state = request.get("state").ok_or("request needs state")?;
        let questions = request
            .get("questions")
            .and_then(Value::as_object)
            .ok_or("request needs a questions object")?;
        if questions.is_empty() {
            return Err("request needs at least one question".into());
        }
        let images = request.get("images").and_then(Value::as_array);
        if let Some(images) = images {
            if images.len() > 1 {
                return Err("d3-lite Metal writes one image run per request; use \
                             MARIO_CLEF_PROTOCOL=separate (one crop + one question)"
                    .into());
            }
        }
        let image = images
            .and_then(|images| images.first())
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| "images must be base64 PNG strings".to_string())
            })
            .transpose()?
            .map(decode_png)
            .transpose()?;

        let mut answers = Map::with_capacity(questions.len());
        let mut p_bits = Map::with_capacity(questions.len());
        let mut host_prep_ms = 0.0f64;
        let mut layout_ms = 0.0f64;
        let mut vision_ms = 0.0f64;
        let mut forward_ms = 0.0f64;
        let mut sync_ms = 0.0f64;
        // First-request attribution. `Some` sums the questions that prepared
        // during this call; `None` (→ JSON null) when none did, i.e. every
        // steady-state request and every request on an already-warm worker.
        let mut t1_ms = 0.0f64;
        let mut t2_ms = 0.0f64;
        let mut t3_ms = 0.0f64;
        let mut prepared_any = false;
        for (id, raw_question) in questions {
            let question = decision_question(id, raw_question)?;
            let prepared_at = Instant::now();
            let prepared = PreparedQuestion::build(format, state, &question, image.as_ref())?;
            host_prep_ms += prepared_at.elapsed().as_secs_f64() * 1e3;

            let output = model
                .decide(prepared.input())
                .map_err(|error| error.to_string())?;
            let profile = model
                .decision_mut()
                .map_err(|error| error.to_string())?
                .last_stage_profile()
                .ok_or_else(|| "d3-lite runner did not record a stage profile".to_string())?;
            layout_ms += profile.layout_ms;
            vision_ms += profile.vision_ms;
            forward_ms += profile.forward_ms;
            sync_ms += profile.sync_ms;
            if let (Some(t1), Some(t2), Some(t3)) = (profile.t1_ms, profile.t2_ms, profile.t3_ms) {
                t1_ms += t1;
                t2_ms += t2;
                t3_ms += t3;
                prepared_any = true;
            }
            let probabilities = prepared.option_probabilities(&output);
            let parsed = format
                .answer(prepared.normalized(), &probabilities)
                .map_err(|error| error.to_string())?;
            p_bits.insert(id.clone(), Value::String(probability_bits(&parsed)));
            answers.insert(id.clone(), parsed);
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1e3;
        // `t₁ − t₂` is the trace/compile cost of the first request of a shape.
        let stages = json!({
            "host_prep_ms": host_prep_ms,
            "layout_ms": layout_ms,
            "vision_ms": vision_ms,
            "forward_ms": forward_ms,
            "sync_ms": sync_ms,
            "t1_ms": prepared_any.then_some(t1_ms),
            "t2_ms": prepared_any.then_some(t2_ms),
            "t3_ms": prepared_any.then_some(t3_ms),
            "questions": questions.len(),
        });
        Ok((
            Value::Object(answers),
            Value::Object(p_bits),
            latency_ms,
            stages,
        ))
    }

    /// 8 hex digits of `f32::to_bits` for the probability a question
    /// reports: `noul` for noul questions, `confidence` for `choice` (which
    /// reports no `noul` field). Read back out of the answer object so the
    /// bits always describe the value that was actually emitted.
    fn probability_bits(answer: &Value) -> String {
        let probability = answer
            .get("noul")
            .or_else(|| answer.get("confidence"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0);
        format!("{:08x}", (probability as f32).to_bits())
    }

    /// A request question in the model-neutral schema shape.
    fn decision_question(id: &str, value: &Value) -> Fallible<DecisionQuestion> {
        let object = value
            .as_object()
            .ok_or_else(|| format!("question {id} must be an object"))?;
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("question {id} needs a `type`"))?
            .to_string();
        for key in object.keys() {
            if !matches!(key.as_str(), "type" | "instructions" | "criteria") {
                return Err(format!("question {id} has unknown field `{key}`"));
            }
        }
        Ok(DecisionQuestion {
            id: id.to_string(),
            kind,
            instructions: object.get("instructions").cloned(),
            criteria: object.get("criteria").cloned(),
        })
    }

    /// base64 → RGB8 + dimensions. The demo sends PNG; the checkpoint's
    /// preprocessor is bit-exact for PNG (P1b).
    fn decode_png(text: &str) -> Fallible<(Vec<u8>, usize, usize)> {
        if text.len() > 24 * 1024 * 1024 {
            return Err("image base64 exceeds 24 MiB".into());
        }
        let bytes =
            base64_decode(text).map_err(|error| format!("image is not valid base64: {error}"))?;
        let decoded =
            image::load_from_memory(&bytes).map_err(|error| format!("image decode: {error}"))?;
        let rgb = decoded.to_rgb8();
        let (width, height) = image::GenericImageView::dimensions(&rgb);
        Ok((rgb.into_raw(), width as usize, height as usize))
    }

    /// Everything one question needs to reach the model, owned so the
    /// borrowed [`DecisionInput`] stays valid across `model.decide`.
    enum PreparedQuestion {
        Text {
            encoded: apxinf_model::metal_d3_lite::EncodedRequest,
        },
        Image {
            encoded: apxinf_model::metal_d3_lite::EncodedImageRequest,
            pixels: Tensor,
            grids: [[u32; 3]; 1],
        },
    }

    impl PreparedQuestion {
        fn build(
            format: &D3LiteFormat,
            state: &Value,
            question: &DecisionQuestion,
            image: Option<&(Vec<u8>, usize, usize)>,
        ) -> Fallible<Self> {
            if let Some((rgb, width, height)) = image {
                // The checkpoint's resize/normalize/patchify contract runs on
                // the host; `PreparedImage` owns the pixel tensor and grid.
                let prepared = preprocess(
                    &format.config().preprocessor,
                    &format.config().vision,
                    rgb,
                    *width,
                    *height,
                )
                .map_err(|error| format!("preprocess: {error}"))?;
                let rows = prepared.grid_thw[1] as usize * prepared.grid_thw[2] as usize;
                let patch_dim = format.patch_dim();
                if prepared.pixel_values.len() != rows * patch_dim {
                    return Err(format!(
                        "preprocess produced {} values, expected {rows} x {patch_dim}",
                        prepared.pixel_values.len()
                    ));
                }
                let pixels = Tensor::from_f32(vec![rows, patch_dim], &prepared.pixel_values)
                    .map_err(|error| format!("pixel tensor: {error}"))?;
                let grids = [prepared.grid_thw];
                let encoded = format
                    .encode_image(state, question, &prepared)
                    .map_err(|error| error.to_string())?;
                Ok(Self::Image {
                    encoded,
                    pixels,
                    grids,
                })
            } else {
                let encoded = format
                    .encode_text(state, question)
                    .map_err(|error| error.to_string())?;
                Ok(Self::Text { encoded })
            }
        }

        fn input(&self) -> DecisionInput<'_> {
            match self {
                Self::Text { encoded } => {
                    DecisionInput::text(&encoded.token_ids, encoded.candidate_count)
                }
                Self::Image {
                    encoded,
                    pixels,
                    grids,
                    ..
                } => DecisionInput::with_vision(
                    &encoded.token_ids,
                    ImageInput::new(pixels, grids),
                    encoded.candidate_count,
                ),
            }
        }

        fn normalized(&self) -> &apxinf_model::metal_d3_lite::NormalizedQuestion {
            match self {
                Self::Text { encoded } => &encoded.normalized,
                Self::Image { encoded, .. } => &encoded.normalized,
            }
        }

        fn option_probabilities(&self, output: &DecisionOutput) -> Vec<f32> {
            match self {
                Self::Text { encoded } => {
                    apxinf_model::metal_d3_lite::D3LiteFormat::option_probabilities(encoded, output)
                }
                Self::Image { encoded, .. } => {
                    apxinf_model::metal_d3_lite::D3LiteFormat::image_option_probabilities(
                        encoded, output,
                    )
                }
            }
        }
    }
}

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
fn main() {
    live::main();
}

/// The resident server is Apple-Silicon + MLX only, matching the
/// `metal_d3_lite` module's gate. Other targets still compile this binary so
/// `cargo check --features mlx` (the Linux feature-isolation CI job) stays
/// green; the process refuses to serve.
#[cfg(not(all(feature = "mlx", target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!(
        "apxinf-decide-server requires --features mlx on Apple Silicon \
         (this build: feature={}, target={})",
        if cfg!(feature = "mlx") {
            "mlx"
        } else {
            "no-mlx"
        },
        std::env::consts::OS,
    );
    std::process::exit(1);
}
