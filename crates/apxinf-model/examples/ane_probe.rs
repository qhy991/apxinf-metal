//! ANE physical-quantity probe for Issue #134 (T2 ANE bandwidth, T3
//! ANE<->GPU shared-event round-trip, T4 ANE small-shape matmul).
//!
//! Core ML `Model` is not `Send` and placement facts only say *anticipated*
//! device, so the ANE work and the GPU streaming probe have to share one
//! process with an explicit schedule; that is why this is a Rust example and
//! not a Python one. Weights are synthetic constants of the frozen shapes: the
//! ticket measures bytes/round-trip/MACs, not model quality, and synthetic
//! weights keep the package cache hit across windows.
//!
//! Build (lock-free; no Metal lease held by apxinf-tool):
//!   SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
//!   MLX_ROOT=$HOME/mlx-sdk \
//!   cargo build --release -p apxinf-model --features coreml --example ane_probe
//!
//! Modes (all write one JSON line to stdout, schema frozen in doc §2.3-§2.5):
//!   placement --cell <Mxd>   placement facts for one T4 cell (no timing)
//!   gpu --reps <n>           T2 GPU-only streaming probe (MLX 3.4576 GB copy+mean)
//!   ane-bw --reps <n>        T2 ANE-only: conv weight-read GB/s
//!   both --reps <n>          T2 GPU+ANE overlapped; GPU thread + ANE thread
//!   matmul --cell <Mxd> --reps <n> --batch <k>   T4 ANE arm (M rows, batch=k M's)
//!   event --reps <n> --rounds <k>                T3 shared-event round-trip
//!
//! --out <path> additionally writes the same JSON object to <path>.
#![cfg_attr(not(all(target_os = "macos", target_arch = "aarch64")), allow(dead_code))]

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native {
    use std::process;
    use apxinf_core::{Error, Result};
    use apxinf_coreml::{
        ComputeDeviceKind, ComputeUnits, Operand, Package, ProgramBuilder, ProgramCache,
    };
    use apxinf_mlx::{Array, MlxDType, Stream};
    use half::f16;
    use serde_json::{json, Value};
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Doc §2.1 / §2.3: same array as m4_bw_probe.rs, so T2's GPU arm is
    /// byte-for-byte comparable with the A0 denominator.
    const ARRAY_BYTES: usize = 3_457_635_448;
    /// Doc §2.3: ANE model constant weights, >=1 GB as frozen.
    const CONV_WEIGHT_COLS: usize = 32_768;
    const CONV_WEIGHT_ROWS: usize = 16_384;
    const CONV_IN: usize = 1;
    /// Core ML conv expects [batch, channels, height, width].
    const CONV_SHAPE: [usize; 4] = [1, CONV_IN, 1, 1];
    const CONV_WEIGHT_SHAPE: [usize; 4] = [CONV_WEIGHT_COLS, CONV_IN, 1, 1];
    /// F16 activation element count for one conv predict (constant 1.0).
    const CONV_ACT_ELEMS: usize = CONV_IN;
    /// Doc §2.5: d in {2048, 4096}, M in {1,4,8,16,32,64}.
    const MATMUL_MS: [usize; 6] = [1, 4, 8, 16, 32, 64];
    const MATMUL_DS: [usize; 2] = [2048, 4096];
    const PLACEMENT_TIMEOUT_S: u64 = 120;

    fn err(message: impl Into<String>) -> Error {
        Error::Other(message.into())
    }

    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 1 {
            sorted[mid]
        } else {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        }
    }

    /// A/A: (max-min)/median, matching run-bw-util.py::_spread.
    fn spread(values: &[f64]) -> f64 {
        let med = median(values);
        if values.len() < 2 || med <= 0.0 {
            return 0.0;
        }
        (values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().cloned().fold(f64::INFINITY, f64::min))
            / med
    }

    fn parse_cell(cell: &str) -> Result<(usize, usize)> {
        let (m, d) = cell
            .split_once('x')
            .ok_or_else(|| err(format!("cell must be Mxd, got {cell:?}")))?;
        let m: usize = m
            .parse()
            .map_err(|_| err(format!("cell M is not a number: {m:?}")))?;
        let d: usize = d
            .parse()
            .map_err(|_| err(format!("cell d is not a number: {d:?}")))?;
        if !MATMUL_MS.contains(&m) {
            return Err(err(format!("M={m} not in frozen grid {MATMUL_MS:?}")));
        }
        if !MATMUL_DS.contains(&d) {
            return Err(err(format!("d={d} not in frozen grid {MATMUL_DS:?}")));
        }
        Ok((m, d))
    }

    fn usage_kind(
        op: &apxinf_coreml::OperationPlacement,
    ) -> (&'static str, Option<&'static str>) {
        let usage = match op.usage.as_ref() {
            Some(usage) => usage,
            None => return ("missing", None),
        };
        let preferred = usage
            .preferred
            .as_ref()
            .map(|device| match device.kind {
                ComputeDeviceKind::NeuralEngine => "ANE",
                ComputeDeviceKind::Gpu => "GPU",
                ComputeDeviceKind::Cpu => "CPU",
                ComputeDeviceKind::Unknown => "?",
            });
        let supported = usage
            .supported
            .iter()
            .map(|device| match device.kind {
                ComputeDeviceKind::NeuralEngine => "ANE",
                ComputeDeviceKind::Gpu => "GPU",
                ComputeDeviceKind::Cpu => "CPU",
                ComputeDeviceKind::Unknown => "?",
            })
            .collect::<Vec<_>>()
            .join("|");
        ("present", Some(if supported.is_empty() {
            "none"
        } else {
            Box::leak(supported.into_boxed_str())
        }))
    }

    fn collect_placement(facts: &apxinf_coreml::ExpectedPlacement) -> Value {
        let functions = facts
            .functions
            .iter()
            .map(|function| {
                let operations = function
                    .operations
                    .iter()
                    .map(|op| {
                        let (usage_state, supported) = usage_kind(op);
                        let preferred = op.usage.as_ref().and_then(|usage| {
                            usage.preferred.as_ref().map(|device| match device.kind {
                                ComputeDeviceKind::NeuralEngine => "ANE",
                                ComputeDeviceKind::Gpu => "GPU",
                                ComputeDeviceKind::Cpu => "CPU",
                                ComputeDeviceKind::Unknown => "?",
                            })
                        });
                        json!({
                            "path": op.path,
                            "operator": op.operator,
                            "usage_state": usage_state,
                            "supported": supported,
                            "preferred": preferred,
                            "child_operation_counts": op.child_operation_counts,
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "name": function.name,
                    "root_operation_count": function.root_operation_count,
                    "operations": operations,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "compiled_path": facts.compiled_path,
            "compute_units": format!("{:?}", facts.compute_units),
            "functions": functions,
        })
    }

    fn count_preferred(facts: &Value, device: &str) -> usize {
        facts["functions"]
            .as_array()
            .map(|functions| {
                functions
                    .iter()
                    .flat_map(|function| function["operations"].as_array().into_iter().flatten())
                    .filter(|op| op["preferred"] == json!(device))
                    .count()
            })
            .unwrap_or(0)
    }

    fn cache_dir() -> PathBuf {
        env::var_os("APXINF_ANE_PROBE_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/apxinf-ane-probe-cache"))
    }

    /// T2's ANE-only model: one `conv_fp16` whose weight blob is the
    /// dominant constant (32768*16384*2 = 1,073,741,824 B = exactly 1 GiB).
    /// Core ML reads the weight from unified memory on every predict; the
    /// activation is a single constant F16 1.0, so the predict's host input
    /// traffic is negligible and weight bytes are the denominator.
    fn conv_package() -> Result<Package> {
        let mut builder = ProgramBuilder::new();
        let x = builder.input("x", &CONV_SHAPE)?;
        let weight = vec![f16::from_f32(0.01); CONV_WEIGHT_COLS * CONV_WEIGHT_ROWS];
        let y = builder.conv_fp16(&x, &weight, &CONV_WEIGHT_SHAPE)?;
        let y = builder.identity(&y, Some("y"))?;
        builder.finish(&[&y])
    }

    fn matmul_package(m: usize, d: usize) -> Result<Package> {
        let mut builder = ProgramBuilder::new();
        let x = builder.input("x", &[m, d])?;
        // The doc §2.5 y is a constant [d, d] weight. ProgramBuilder::matmul
        // takes two Var activations and has no const-weight overload, so the
        // weight is a second input; the frozen kernel arithmetic (2*M*d*d)
        // is unaffected, but placement facts are interpreted with that in
        // mind (constant-fold opportunity not exercised).
        let y = builder.input("w", &[d, d])?;
        let out = builder.matmul(&x, &y, false)?;
        let out = builder.identity(&out, Some("o"))?;
        builder.finish(&[&out])
    }

    fn load_with_facts(package: &Package, units: ComputeUnits) -> Result<(apxinf_coreml::Model, Value)> {
        let cache = ProgramCache::new(cache_dir());
        let mut placement = None;
        let model = cache.load_with_placement(
            package,
            units,
            Duration::from_secs(PLACEMENT_TIMEOUT_S),
            |facts| {
                placement = Some(collect_placement(facts));
                Ok(())
            },
        )?;
        let placement = placement.ok_or_else(|| err("placement facts not collected"))?;
        Ok((model, placement))
    }

    /// Doc §2.1 / §2.3: MLX streaming copy (x + 0, reads 2*ARRAY_BYTES,
    /// writes ARRAY_BYTES) and mean-reduce over the 3.4576 GB array. The
    /// timing fence is `eval(&[out])` + `synchronize`, which is what
    /// m4_bw_probe.rs uses; MLX `eval` on a stream synchronizes that stream
    /// (verified by m4_bw_probe's physically-sane ~100 GB/s readings).
    fn run_gpu(reps: usize) -> Result<Value> {
        let stream = Stream::metal(0)?;
        let elems = ARRAY_BYTES / 2;
        let x = Array::zeros(&stream, &[elems], MlxDType::BF16)?;
        stream.eval(&[x.clone()])?;
        let zero = Array::scalar(&stream, 0.0, MlxDType::BF16)?;
        let mut copy_ms = Vec::with_capacity(reps);
        let mut reduce_ms = Vec::with_capacity(reps);
        // warmup + compile
        let copy = x.add(&zero)?;
        stream.eval(&[copy])?;
        let reduce = x.mean(&[0], false)?;
        stream.eval(&[reduce])?;
        for _ in 0..reps {
            let start = Instant::now();
            let out = x.add(&zero)?;
            stream.eval(&[out.clone()])?;
            stream.synchronize()?;
            copy_ms.push(start.elapsed().as_secs_f64() * 1e3);

            let start = Instant::now();
            let out = x.mean(&[0], false)?;
            stream.eval(&[out.clone()])?;
            stream.synchronize()?;
            reduce_ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let copy_med = median(&copy_ms);
        let reduce_med = median(&reduce_ms);
        let copy_gbps = ARRAY_BYTES as f64 / (copy_med / 1e3) / 1e9;
        let reduce_gbps = ARRAY_BYTES as f64 / (reduce_med / 1e3) / 1e9;
        Ok(json!({
            "probe": "gpu_mlx_streaming",
            "bytes": ARRAY_BYTES,
            "reps": reps,
            "copy_ms": copy_ms,
            "copy_ms_median": copy_med,
            "copy_spread": spread(&copy_ms),
            "copy_gbps": copy_gbps,
            "reduce_ms": reduce_ms,
            "reduce_ms_median": reduce_med,
            "reduce_spread": spread(&reduce_ms),
            "reduce_gbps": reduce_gbps,
        }))
    }

    /// Doc §2.3: ANE-only weight-read rate. Each `predict` reads the 1 GiB
    /// constant conv weight; GB/s = weight_bytes / (median_predict_s).
    /// Placement facts prove Core ML *anticipates* ANE for the conv
    /// (README: not execution proof; the doc requires us to say so).
    fn run_ane_bw(reps: usize) -> Result<Value> {
        let weight_bytes = (CONV_WEIGHT_COLS * CONV_WEIGHT_ROWS * 2) as u64;
        let (model, placement) = load_with_facts(&conv_package()?, ComputeUnits::CpuAndNeuralEngine)?;
        let x = vec![f16::ONE; CONV_ACT_ELEMS];
        let feed = [("x", &x[..], &CONV_SHAPE[..])];
        let want = [("y", &CONV_SHAPE[..])];
        for _ in 0..2 {
            model.predict(&feed, &want)?;
        }
        let mut predict_ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let out = model.predict(&feed, &want)?;
            predict_ms.push(start.elapsed().as_secs_f64() * 1e3);
            if out.is_empty() {
                return Err(err("ANE conv returned no output"));
            }
        }
        let med = median(&predict_ms);
        Ok(json!({
            "probe": "ane_conv_weight_read",
            "weight_bytes": weight_bytes,
            "conv_weight_shape": CONV_WEIGHT_SHAPE,
            "predict_ms": predict_ms,
            "predict_ms_median": med,
            "predict_spread": spread(&predict_ms),
            "gbps": weight_bytes as f64 / (med / 1e3) / 1e9,
            "compute_units": "CpuAndNeuralEngine",
            "placement": placement,
            "placement_note": "anticipated placement, not execution proof (apxinf-coreml README)",
        }))
    }

    /// Doc §2.3: GPU+ANE overlapped. Two threads share the process: the GPU
    /// thread runs the same MLX streaming probe as `run_gpu`; the ANE thread
    /// runs the same conv loop as `run_ane_bw`. Each reports its own
    /// medians; the analyzer adds them for total_gbps and compares against
    /// GPU-only. MLX stream and Core ML are independent devices on the
    /// shared-memory fabric, so overlap is the measurement.
    fn run_both(reps: usize) -> Result<Value> {
        let weight_bytes = (CONV_WEIGHT_COLS * CONV_WEIGHT_ROWS * 2) as u64;
        let gpu = thread::spawn(move || run_gpu(reps));
        // Give the GPU thread a head start so both are steady-state.
        thread::sleep(Duration::from_millis(50));
        let (model, placement) =
            load_with_facts(&conv_package()?, ComputeUnits::CpuAndNeuralEngine)?;
        let x = vec![f16::ONE; CONV_ACT_ELEMS];
        let feed = [("x", &x[..], &CONV_SHAPE[..])];
        let want = [("y", &CONV_SHAPE[..])];
        for _ in 0..2 {
            model.predict(&feed, &want)?;
        }
        let mut predict_ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let out = model.predict(&feed, &want)?;
            predict_ms.push(start.elapsed().as_secs_f64() * 1e3);
            if out.is_empty() {
                return Err(err("ANE conv returned no output"));
            }
        }
        let gpu = gpu
            .join()
            .map_err(|_| err("GPU probe thread panicked"))??;
        let med = median(&predict_ms);
        Ok(json!({
            "probe": "gpu_and_ane_overlap",
            "weight_bytes": weight_bytes,
            "predict_ms": predict_ms,
            "predict_ms_median": med,
            "predict_spread": spread(&predict_ms),
            "ane_gbps": weight_bytes as f64 / (med / 1e3) / 1e9,
            "compute_units": "CpuAndNeuralEngine",
            "placement": placement,
            "placement_note": "anticipated placement, not execution proof (apxinf-coreml README)",
            "gpu_arm": gpu,
        }))
    }

    /// Doc §2.5 T4: ANE arm of one cell, `batch` predicts of the [M,d]x[d,d]
    /// matmul, warmup 2 then `reps` timed. Reports median ms, MAC/s and the
    /// placement facts for the matmul operator. If the matmul's preferred
    /// device is not ANE the cell is blocked (analyzer reports BLOCKED).
    fn run_matmul(m: usize, d: usize, reps: usize, batch: usize) -> Result<Value> {
        let package = matmul_package(m, d)?;
        let (model, placement) =
            load_with_facts(&package, ComputeUnits::CpuAndNeuralEngine)?;
        let x = vec![f16::ONE; m * d];
        let w = vec![f16::from_f32(0.01); d * d];
        let feed = [("x", &x[..], &[m, d][..]), ("w", &w[..], &[d, d][..])];
        let want = [("o", &[m, d][..])];
        for _ in 0..2 {
            model.predict(&feed, &want)?;
        }
        let mut predict_ms = Vec::with_capacity(reps * batch);
        for _ in 0..reps {
            let start = Instant::now();
            for _ in 0..batch {
                let out = model.predict(&feed, &want)?;
                if out.is_empty() {
                    return Err(err("matmul returned no output"));
                }
            }
            predict_ms.push(start.elapsed().as_secs_f64() * 1e3 / batch as f64);
        }
        let med = median(&predict_ms);
        let macs = (2 * m * d * d) as f64;
        let matmul_preferred = placement["functions"]
            .as_array()
            .and_then(|functions| {
                functions.iter().find_map(|function| {
                    function["operations"].as_array().and_then(|operations| {
                        operations
                            .iter()
                            .find(|op| op["operator"] == json!("matmul"))
                            .map(|op| op["preferred"].clone())
                    })
                })
            })
            .unwrap_or(Value::Null);
        Ok(json!({
            "probe": "ane_matmul",
            "m": m,
            "d": d,
            "batch": batch,
            "reps": reps,
            "predict_ms": predict_ms,
            "predict_ms_median": med,
            "matmul_ms_median": med,
            "predict_spread": spread(&predict_ms),
            "macs": macs,
            "tmacs": macs / (med / 1e3) / 1e12,
            "matmul_preferred": matmul_preferred,
            "compute_units": "CpuAndNeuralEngine",
            "placement": placement,
            "placement_note": "anticipated placement, not execution proof (apxinf-coreml README)",
        }))
    }

    /// Doc §2.4 T3: one ANE eval + one MTLSharedEvent round-trip, then
    /// `rounds` serial round-trips (2*L = 48 for L=24). Because Core ML
    /// `Model::predict` is host-synchronous and its internal command buffer
    /// is not exposed, the honest "ANE eval + GPU wait" primitive is:
    ///   1. enqueue a GPU wait on the shared event (signalled value 0),
    ///   2. run the ANE predict (host blocks until Core ML completes),
    ///   3. signal the event from the host after the predict,
    ///   4. GPU wait is released.
    /// The measured interval is the host wall around one predict plus the
    /// wait/signal bookkeeping; this is the sync cost a layer-interleaved
    /// split would pay *in addition to* the ANE compute it replaces. The
    /// JSON reports both the per-round-trip wall and the predicted total for
    /// 48 rounds against the single-token reference the analyzer supplies.
    fn run_event(reps: usize, rounds: usize) -> Result<Value> {
        let device = MetalDevice::system_default().ok_or_else(|| err("no Metal device"))?;
        let event = device.new_shared_event();
        let queue = device.new_command_queue();
        // Minimal GPU workload so the wait has a real command buffer.
        let library = device
            .new_library_with_source(
                "kernel void noop(device float* out [[buffer(0)]]) { out[0] = 1.0f; }",
                &CompileOptions::new(),
            )
            .map_err(|error| err(format!("shader compile: {error:?}")))?;
        let function = library
            .get_function("noop", None)
            .map_err(|error| err(format!("get_function: {error:?}")))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(function.as_ref())
            .map_err(|error| err(format!("pipeline: {error:?}")))?;
        let out = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        // Build the ANE eval once (same conv as T2 so the placement facts
        // are the same package; its predict is the "empty/minimal ANE eval"
        // the Gate asks for).
        let (model, placement) =
            load_with_facts(&conv_package()?, ComputeUnits::CpuAndNeuralEngine)?;
        let x = vec![f16::ONE; CONV_ACT_ELEMS];
        let feed = [("x", &x[..], &CONV_SHAPE[..])];
        let want = [("y", &CONV_SHAPE[..])];
        for _ in 0..2 {
            model.predict(&feed, &want)?;
        }

        // Round-119: `event.signaled_value()` is set by the host, so
        // polling it does NOT prove the GPU-side wait completed. We keep a
        // reference to the committed wait command buffer and drain it with
        // `wait_until_completed()` after signalling; that is the honest
        // "GPU side waited" half of the round-trip.
        fn enqueue_wait<'q>(
            queue: &'q CommandQueue,
            event: &SharedEvent,
            value: u64,
            pipeline: &ComputePipelineState,
            buffer: &metal::Buffer,
        ) -> Result<&'q metal::CommandBufferRef> {
            let command_buffer = queue.new_command_buffer();
            command_buffer.encode_wait(event, value);
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(buffer.as_ref()), 0);
            encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            encoder.end_encoding();
            command_buffer.commit();
            Ok(command_buffer)
        }
        let mut single_ms = Vec::with_capacity(reps);
        let mut counter: u64 = 0;
        for _ in 0..reps {
            let cb = enqueue_wait(&queue, &event, counter, &pipeline, &out)?;
            let start = Instant::now();
            let out = model.predict(&feed, &want)?;
            if out.is_empty() {
                return Err(err("ANE eval returned no output"));
            }
            counter += 1;
            event.signal_value(counter);
            // Wait for the GPU-side buffer to actually observe the signal.
            cb.wait_until_completed();
            single_ms.push(start.elapsed().as_secs_f64() * 1e3);
        }

        let mut rounds_ms = Vec::new();
        let start_all = Instant::now();
        for _ in 0..rounds {
            let cb = enqueue_wait(&queue, &event, counter, &pipeline, &out)?;
            let out = model.predict(&feed, &want)?;
            if out.is_empty() {
                return Err(err("ANE eval returned no output"));
            }
            counter += 1;
            event.signal_value(counter);
            cb.wait_until_completed();
        }
        rounds_ms.push(start_all.elapsed().as_secs_f64() * 1e3);

        let single_med = median(&single_ms);
        Ok(json!({
            "probe": "ane_gpu_shared_event",
            "reps": reps,
            "rounds": rounds,
            "single_roundtrip_ms": single_ms,
            "single_roundtrip_ms_median": single_med,
            "single_roundtrip_spread": spread(&single_ms),
            "rounds_ms": rounds_ms,
            "rounds_ms_median": median(&rounds_ms),
            "predicted_48_rounds_ms": single_med * rounds as f64,
            "compute_units": "CpuAndNeuralEngine",
            "placement": placement,
            "placement_note": "anticipated placement, not execution proof (apxinf-coreml README)",
        }))
    }

    fn grid() -> Value {
        json!({
            "matmul_ds": MATMUL_DS,
            "matmul_ms": MATMUL_MS,
            "array_bytes": ARRAY_BYTES,
            "conv_weight_bytes": (CONV_WEIGHT_COLS * CONV_WEIGHT_ROWS * 2) as u64,
        })
    }

    pub fn run() -> Result<()> {
        let args: Vec<String> = env::args().skip(1).collect();
        let mut mode = None;
        let mut cell = None;
        let mut reps = 3;
        let mut rounds = 48;
        let mut batch = 1;
        let mut out: Option<PathBuf> = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--cell" => {
                    i += 1;
                    cell = Some(
                        args.get(i)
                            .ok_or_else(|| err("--cell needs a value"))?
                            .clone(),
                    );
                }
                "--reps" => {
                    i += 1;
                    reps = args
                        .get(i)
                        .ok_or_else(|| err("--reps needs a value"))?
                        .parse()
                        .map_err(|_| err("--reps is not a number"))?;
                }
                "--rounds" => {
                    i += 1;
                    rounds = args
                        .get(i)
                        .ok_or_else(|| err("--rounds needs a value"))?
                        .parse()
                        .map_err(|_| err("--rounds is not a number"))?;
                }
                "--batch" => {
                    i += 1;
                    batch = args
                        .get(i)
                        .ok_or_else(|| err("--batch needs a value"))?
                        .parse()
                        .map_err(|_| err("--batch is not a number"))?;
                }
                "--out" => {
                    i += 1;
                    out = Some(PathBuf::from(
                        args.get(i).ok_or_else(|| err("--out needs a value"))?,
                    ));
                }
                other if other.starts_with("--") => {
                    return Err(err(format!("unknown flag {other:?}")));
                }
                other => mode = Some(other.to_owned()),
            }
            i += 1;
        }
        let mode = mode.ok_or_else(|| err("mode required (gpu|ane-bw|both|matmul|event|placement|grid)"))?;
        let payload = match mode.as_str() {
            "grid" => grid(),
            "gpu" => run_gpu(reps)?,
            "ane-bw" => run_ane_bw(reps)?,
            "both" => run_both(reps)?,
            "event" => run_event(reps, rounds)?,
            "placement" => {
                let cell = cell.as_deref().ok_or_else(|| err("placement needs --cell"))?;
                let (m, d) = parse_cell(cell)?;
                let (model, placement) =
                    load_with_facts(&matmul_package(m, d)?, ComputeUnits::CpuAndNeuralEngine)?;
                // One predict so the model is actually loaded/compiled; the
                // placement facts (not the timing) are the payload.
                let x = vec![f16::ONE; m * d];
                let w = vec![f16::from_f32(0.01); d * d];
                let feed = [("x", &x[..], &[m, d][..]), ("w", &w[..], &[d, d][..])];
                let want = [("o", &[m, d][..])];
                let _ = model.predict(&feed, &want)?;
                json!({"probe": "placement", "m": m, "d": d, "placement": placement,
                       "placement_note": "anticipated placement, not execution proof (apxinf-coreml README)"})
            }
            "matmul" => {
                let cell = cell.as_deref().ok_or_else(|| err("matmul needs --cell"))?;
                let (m, d) = parse_cell(cell)?;
                run_matmul(m, d, reps, batch)?
            }
            other => return Err(err(format!("unknown mode {other:?}"))),
        };
        let text = serde_json::to_string(&payload)?;
        if let Some(path) = out {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, format!("{text}\n"))?;
        }
        println!("{text}");
        Ok(())
    }

    // ---- Metal shared-event primitives (round-119: use the `metal` crate's
    // own objc wrappers; objc2-metal's generated snake_case names never
    // matched this file and `ProtocolObject` casting was broken, which made
    // the original T3 block un-compilable. metal 0.33 already exposes
    // signaled_value/set_signaled_value on SharedEventRef and
    // encode_wait_for_event on CommandBufferRef; event semantics are
    // identical. objc2-metal stays in dev-deps for future tracing work. ----
    use metal::{
        CompileOptions, CommandQueue, ComputePipelineState, Device, Library,
        MTLResourceOptions, MTLSize, SharedEvent,
    };

    struct MetalDevice(Device);
    impl MetalDevice {
        fn system_default() -> Option<Self> {
            Device::system_default().map(Self)
        }
        fn new_shared_event(&self) -> SharedEvent {
            self.0.new_shared_event()
        }
        fn new_command_queue(&self) -> CommandQueue {
            self.0.new_command_queue()
        }
        fn new_library_with_source(
            &self,
            source: &str,
            options: &CompileOptions,
        ) -> Result<Library> {
            self.0
                .new_library_with_source(source, options)
                .map_err(|error| err(format!("library: {error:?}")))
        }
        fn new_buffer(&self, length: usize, options: MTLResourceOptions) -> metal::Buffer {
            self.0.new_buffer(length as u64, options)
        }
    }

    trait SharedEventExt {
        fn signal_value(&self, value: u64);
        fn signaled_value(&self) -> u64;
    }
    impl SharedEventExt for SharedEvent {
        fn signal_value(&self, value: u64) {
            self.set_signaled_value(value);
        }
        fn signaled_value(&self) -> u64 {
            SharedEvent::signaled_value(self)
        }
    }

    trait CommandBufferExt {
        fn encode_wait(&self, event: &SharedEvent, value: u64);
    }
    impl CommandBufferExt for metal::CommandBufferRef {
        fn encode_wait(&self, event: &SharedEvent, value: u64) {
            self.encode_wait_for_event(event, value);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = native::run() {
        eprintln!("ane_probe: {error:?}");
        process::exit(1);
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("ane_probe requires Apple Silicon macOS");
    process::exit(1);
}
