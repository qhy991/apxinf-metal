//! Resident d3-lite decision process for the Mario demo (`APXINF_DECIDE_CMD`).
//!
//! ```text
//! apxinf-decide-server --model <checkpoint> [--model-name NAME] [--device metal:N]
//!                       [--concurrency N]
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
//! `String`s into a bounded queue of capacity N; one writer thread emits
//! responses under a single stdout lock, in completion order. Memory cost is
//! N × model, which is expected. N = 1 stays on the original serial path.
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

#[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
mod live {
    use std::collections::VecDeque;
    use std::io::{BufRead, Write};
    use std::path::PathBuf;
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
        if options.concurrency == 1 {
            serial(&options);
        } else {
            parallel(options);
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
                "--help" | "-h" => {
                    return Err("usage: apxinf-decide-server --model <checkpoint> \
                                 [--model-name NAME] [--device metal:N] \
                                 [--concurrency N]"
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
        })
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
    fn print_ready(model_name: &str, load_ms: f64) -> Fallible<()> {
        writeln!(
            std::io::stdout(),
            "{}",
            json!({
                "type": "ready",
                "model": model_name,
                "variant": VARIANT,
                "load_ms": load_ms,
                "codes_verified": true,
            })
        )
        .map_err(|error| format!("write ready: {error}"))?;
        std::io::stdout()
            .flush()
            .map_err(|error| format!("flush ready: {error}"))
    }

    /// N = 1: the original serial path — build on the main thread, print
    /// `ready`, then read a line, answer it, write the response.
    fn serial(options: &Options) {
        let (server, load_ms) = match build(options) {
            Ok(loaded) => loaded,
            Err(error) => fail_start(&error),
        };
        if let Err(error) = print_ready(&server.model_name, load_ms) {
            fail_start(&error);
        }
        serve(server);
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

    /// Shared pool state: a bounded FIFO of request lines, the in-flight
    /// counter, and the live-worker counter. A `Mutex<VecDeque>` + condvar
    /// stands in for a multi-consumer bounded mpsc (std's `Receiver` cannot
    /// be cloned); the lock is held only for queue bookkeeping, never across
    /// a model call. `close()` is the shutdown signal.
    struct Pool {
        queue: Mutex<VecDeque<String>>,
        closed: AtomicBool,
        changed: Condvar,
        /// Queued + currently processing; reaches 0 when all work is done.
        outstanding: AtomicUsize,
        /// Workers that have not yet exited; 0 means nobody can drain more.
        live: AtomicUsize,
        /// Set by the writer when stdout breaks.
        stdout_broken: AtomicBool,
    }

    impl Pool {
        fn new(workers: usize) -> Self {
            Self {
                queue: Mutex::new(VecDeque::new()),
                closed: AtomicBool::new(false),
                changed: Condvar::new(),
                outstanding: AtomicUsize::new(0),
                live: AtomicUsize::new(workers),
                stdout_broken: AtomicBool::new(false),
            }
        }

        /// Stop accepting work. Workers drain whatever is queued, then exit.
        fn close(&self) {
            self.closed.store(true, Ordering::SeqCst);
            self.changed.notify_all();
        }
    }

    /// Lock the queue, recovering from a poisoned mutex left by a panicked
    /// worker: the queue itself is still usable.
    fn queue_lock(pool: &Pool) -> MutexGuard<'_, VecDeque<String>> {
        pool.queue
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn queue_wait<'a>(
        pool: &Pool,
        guard: MutexGuard<'a, VecDeque<String>>,
    ) -> MutexGuard<'a, VecDeque<String>> {
        pool.changed
            .wait(guard)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
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
            let _state = queue_lock(&self.pool);
            self.pool.outstanding.fetch_sub(1, Ordering::SeqCst);
            self.pool.changed.notify_all();
        }
    }

    /// N worker threads. Each builds its own `Server`/`LoadedModel` here, on
    /// this thread, and uses it only here — the model is `!Send` and is never
    /// shared, moved, or wrapped in a `Mutex`.
    fn parallel(options: Options) {
        let workers = options.concurrency;
        let pool = Arc::new(Pool::new(workers));
        let (handshake_tx, handshake_rx) = mpsc::channel::<Result<(f64, String), String>>();
        let (response_tx, response_rx) = mpsc::channel::<String>();

        let mut handles = Vec::with_capacity(workers);
        for index in 0..workers {
            let pool = Arc::clone(&pool);
            let handshake_tx = handshake_tx.clone();
            let response_tx = response_tx.clone();
            let options = options.clone();
            handles.push(
                std::thread::Builder::new()
                    .name(format!("decide-worker-{index}"))
                    .spawn(move || worker(options, pool, handshake_tx, response_tx))
                    .expect("spawn decide worker"),
            );
        }
        // Only the workers produce responses from here on.
        drop(handshake_tx);
        // The main thread never writes response lines; drop its sender so
        // the writer sees the channel close once every worker has exited.
        drop(response_tx);

        // Collect one handshake per worker. A worker that dies before its
        // handshake is noticed through the live counter (or its panic line).
        let mut load_times = Vec::with_capacity(workers);
        let mut model_names = Vec::with_capacity(workers);
        let mut received = 0usize;
        let mut failure: Option<String> = None;
        while received < workers && failure.is_none() {
            match handshake_rx.recv_timeout(Duration::from_millis(200)) {
                Ok(Ok((load_ms, model_name))) => {
                    load_times.push(load_ms);
                    model_names.push(model_name);
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

        // All workers loaded: one `ready` line from the main thread, with
        // the slowest load (all workers load concurrently).
        let model_name = model_names
            .first()
            .cloned()
            .unwrap_or_else(|| "metal_d3".into());
        let load_ms = load_times.iter().copied().fold(0.0, f64::max);
        if let Err(error) = print_ready(&model_name, load_ms) {
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

        // Main thread: read stdin and dispatch into the bounded queue.
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
            // Backpressure: block while the queue is full, until shutdown.
            let mut state = queue_lock(&pool);
            while state.len() >= capacity
                && !pool.closed.load(Ordering::SeqCst)
                && !pool.stdout_broken.load(Ordering::SeqCst)
                && pool.live.load(Ordering::SeqCst) > 0
            {
                state = queue_wait(&pool, state);
            }
            if pool.closed.load(Ordering::SeqCst)
                || pool.stdout_broken.load(Ordering::SeqCst)
                || pool.live.load(Ordering::SeqCst) == 0
            {
                break;
            }
            state.push_back(line);
            Outstanding::claim(&pool);
            pool.changed.notify_one();
        }

        // EOF (or shutdown): close the queue, workers drain it, then wait
        // for in-flight work to finish. If every worker is gone there is
        // nothing left to drain, so we do not wait forever.
        pool.close();
        {
            let mut state = queue_lock(&pool);
            while pool.outstanding.load(Ordering::SeqCst) > 0
                && pool.live.load(Ordering::SeqCst) > 0
            {
                state = queue_wait(&pool, state);
            }
        }

        for handle in handles {
            let _ = handle.join();
        }
        let _ = writer.join();
    }

    fn worker(
        options: Options,
        pool: Arc<Pool>,
        handshake: Sender<Result<(f64, String), String>>,
        responses: Sender<String>,
    ) {
        // A panic anywhere in this worker must still decrement `live` and
        // wake the main thread's drain/dispatch waits.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (server, load_ms) = match build(&options) {
                Ok(loaded) => loaded,
                Err(error) => {
                    let _ = handshake.send(Err(error));
                    return;
                }
            };
            if handshake
                .send(Ok((load_ms, server.model_name.clone())))
                .is_err()
            {
                return;
            }
            serve_worker(server, &pool, responses);
        }));
        {
            let _state = queue_lock(&pool);
            pool.live.fetch_sub(1, Ordering::SeqCst);
            pool.changed.notify_all();
        }
        if let Err(payload) = outcome {
            eprintln!("decide worker panicked: {}", panic_message(payload));
        }
    }

    fn serve_worker(mut server: Server, pool: &Arc<Pool>, responses: Sender<String>) {
        loop {
            // Pop one job, or exit once the queue is closed *and* drained:
            // a closed queue still yields queued work first.
            let job = {
                let mut state = queue_lock(pool);
                loop {
                    if let Some(job) = state.pop_front() {
                        pool.changed.notify_all(); // space for the producer
                        break Some(job);
                    }
                    if pool.closed.load(Ordering::SeqCst) {
                        break None;
                    }
                    state = queue_wait(pool, state);
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
            let probabilities = prepared.option_probabilities(&output);
            let parsed = format
                .answer(prepared.normalized(), &probabilities)
                .map_err(|error| error.to_string())?;
            p_bits.insert(id.clone(), Value::String(probability_bits(&parsed)));
            answers.insert(id.clone(), parsed);
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1e3;
        let stages = json!({
            "host_prep_ms": host_prep_ms,
            "layout_ms": layout_ms,
            "vision_ms": vision_ms,
            "forward_ms": forward_ms,
            "sync_ms": sync_ms,
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
