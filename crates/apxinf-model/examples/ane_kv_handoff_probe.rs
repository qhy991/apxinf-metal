//! P2 state-handoff probe for Issue #161 (doc/ane-prefill-offload.md §1.1).
//! Three arms, one request state, one MLX stream:
//!
//!   copy  - same-layout BF16 copy of the full decoder-consumable state
//!           (attention KV slab + conv windows). This is arm C: the number
//!           the P2 gate compares against engine-basis prefill_ms.
//!   host  - `Array::to_bytes_ready` on the same state (arm H). Worst-case
//!           unification path: one device->host transfer per layer. Reported
//!           only; it does not decide the gate.
//!   iosurface - arm I: shells out to the standalone `iosurface_probe`
//!           binary for the IOSurface feasibility check (framework
//!           availability + allocator size), then runs the GPU-side layout
//!           conversion of the ANE-layout KV slab ([1,4,C,128] ->
//!           [1,C,4,128] + contiguous copy) at each requested P. The AE
//!           half (real ANE write into the shared buffer) is out of scope
//!           for P2 and deferred to P3.
//!
//! Build (lock-free; no Metal lease held):
//!   SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
//!   MLX_ROOT=$HOME/mlx-sdk \
//!   cargo build --release -p apxinf-model --features coreml --example ane_kv_handoff_probe
//!
//! `--out <path>` additionally writes the JSON object to <path>.
#![cfg_attr(
    not(all(target_os = "macos", target_arch = "aarch64")),
    allow(dead_code)
)]

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native {
    use apxinf_core::{Error, Result};
    use apxinf_mlx::{Array, MlxDType, Stream};
    use serde_json::{json, Value};
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    /// Doc §1.1 gate: the state bytes the decoder actually reuses after one
    /// prefill. Layer geometry from metal_lfm3 config (1.7B: 6 full + 5
    /// sliding attention layers, 13 conv layers, 4 kv_heads, head_dim 128,
    /// conv_state_len 6). `S(P) = 12288*P + 2940928`.
    const FULL_ATTN_LAYERS: usize = 6;
    const SLIDING_ATTN_LAYERS: usize = 5;
    const KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 128;
    const CONV_LAYERS: usize = 13;
    const HIDDEN: usize = 2048;
    const CONV_STATE_LEN: usize = 6;
    /// Sliding-window length (metal config window=256); the sliding KV slab
    /// is dense C rows but only the window is logically live.
    const SLIDING_WINDOW: usize = 256;
    const DEFAULT_P: &[usize] = &[512, 2048, 8192];
    const DEFAULT_REPS: usize = 21;
    const DEFAULT_WARMUP: usize = 5;

    /// Doc §1.1 frozen table, used as the probe's own gate.
    const EXPECTED_S: [(usize, usize); 3] =
        [(512, 9_232_384), (2048, 28_106_752), (8192, 103_604_224)];

    fn err(message: impl Into<String>) -> Error {
        Error::Other(message.into())
    }

    /// Same arithmetic as doc §1.1. Attention KV:
    /// 2(K,V) * KV_HEADS * HEAD_DIM * 2 B = 2048 B per layer per token;
    /// full = 6 * 2048 * P, sliding = 5 * 2048 * window, conv = 13 * HIDDEN
    /// * CONV_STATE_LEN * 2.
    fn state_bytes(p: usize) -> usize {
        let kv_per_token = 2 * KV_HEADS * HEAD_DIM * 2;
        let full = FULL_ATTN_LAYERS * kv_per_token * p;
        let sliding = SLIDING_ATTN_LAYERS * kv_per_token * SLIDING_WINDOW;
        let conv = CONV_LAYERS * HIDDEN * CONV_STATE_LEN * 2;
        full + sliding + conv
    }

    fn state_shape(p: usize) -> Vec<usize> {
        // Concatenated decoder-consumable layout, same row order for every
        // layer: full KV rows [0,p) and sliding KV rows [0,window), then the
        // conv windows. The source layout of the handoff is the ANE natural
        // layout; the destination layout is byte-identical (doc §1.1: C arm
        // is a pure copy, no transpose).
        let kv_rows = FULL_ATTN_LAYERS * p + SLIDING_ATTN_LAYERS * SLIDING_WINDOW;
        vec![2 * KV_HEADS * HEAD_DIM * kv_rows + CONV_LAYERS * HIDDEN * CONV_STATE_LEN]
    }

    fn state_bytes_expected(p: usize) -> Result<usize> {
        let expected = state_bytes(p);
        if let Some((_, frozen)) = EXPECTED_S.iter().find(|(q, _)| *q == p) {
            if expected != *frozen {
                return Err(err(format!(
                    "state byte arithmetic mismatch for P={p}: computed {expected}, doc frozen {frozen}"
                )));
            }
        }
        if expected != state_shape(p).iter().product::<usize>() * MlxDType::BF16.size_in_bytes() {
            return Err(err(format!(
                "state byte arithmetic mismatch for P={p}: formula {expected}, shape {} elements",
                state_shape(p).iter().product::<usize>()
            )));
        }
        Ok(expected)
    }

    /// Deterministic mock state: `[0, p)` rows carry a low-amplitude pattern
    /// so a copy/readback is observable; `[p, capacity)` rows are exactly
    /// zero. Rows are uploaded once, evaluated once, and then reused for
    /// every rep (the A/A gate is about launch variance, not about the
    /// contents).
    fn mock_state(stream: &Stream, p: usize) -> Result<Array> {
        let shape = state_shape(p);
        let elems: usize = shape.iter().product();
        let live = state_shape(p)[0] / 2; // one K or V row half of the slab
        let mut data = vec![0u8; elems * 2];
        // BF16 bit pattern for a small non-zero value: encode 1.0 = 0x3c00.
        let live_bits = 0x3c00u16;
        let half = elems / 2;
        for i in 0..half.min(live) {
            data[2 * i..2 * i + 2].copy_from_slice(&live_bits.to_le_bytes());
        }
        // Keep the V half zero so the readback distinguishes K from V.
        let a = Array::from_bytes(stream, &shape, MlxDType::BF16, &data)?;
        a.eval()?;
        stream.synchronize()?;
        Ok(a)
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

    fn spread(values: &[f64]) -> f64 {
        let med = median(values);
        if values.len() < 2 || med <= 0.0 {
            return 0.0;
        }
        (values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().cloned().fold(f64::INFINITY, f64::min))
            / med
    }

    /// Arm C: the decoder-layout copy. `Array::copy()` is MLX's explicit
    /// device-to-device copy (op 33), so it reads and writes the full slab.
    fn run_copy(p: usize, reps: usize, warmup: usize) -> Result<Value> {
        let bytes = state_bytes_expected(p)?;
        let stream = Stream::metal(0)?;
        let source = mock_state(&stream, p)?;
        for _ in 0..warmup {
            let dst = source.copy()?;
            stream.eval(&[dst.clone()])?;
            stream.synchronize()?;
        }
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let dst = source.copy()?;
            stream.eval(&[dst.clone()])?;
            stream.synchronize()?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let med = median(times.as_slice());
        Ok(json!({
            "probe": "kv_handoff_copy",
            "arm": "C",
            "p": p,
            "bytes": bytes,
            "reps": reps,
            "warmup": warmup,
            "times_ms": times,
            "median_ms": med,
            "spread": spread(times.as_slice()),
            "gbps": bytes as f64 / (med / 1e3) / 1e9,
            "state_shape": state_shape(p),
            "bytes_from_formula": state_bytes(p),
            "bytes_match_formula": bytes == state_bytes(p),
        }))
    }

    /// Arm H: device -> host round trip of the whole state. `to_bytes_ready`
    /// is what `metal_lfm3::state_bytes` uses, so the mechanism matches the
    /// reference helper. Reported only; not a gate input.
    fn run_host(p: usize, reps: usize, warmup: usize) -> Result<Value> {
        let bytes = state_bytes_expected(p)?;
        let stream = Stream::metal(0)?;
        let source = mock_state(&stream, p)?;
        for _ in 0..warmup {
            let _ = source.to_bytes_ready()?;
        }
        let mut times = Vec::with_capacity(reps);
        let mut last_len = 0usize;
        for _ in 0..reps {
            let start = Instant::now();
            let out = source.to_bytes_ready()?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
            last_len = out.len();
        }
        let med = median(times.as_slice());
        Ok(json!({
            "probe": "kv_handoff_host",
            "arm": "H",
            "p": p,
            "bytes": bytes,
            "readback_bytes": last_len,
            "reps": reps,
            "warmup": warmup,
            "times_ms": times,
            "median_ms": med,
            "spread": spread(times.as_slice()),
            "gbps": bytes as f64 / (med / 1e3) / 1e9,
            "note": "mechanism accounting only; not a P2 gate input",
            "state_shape": state_shape(p),
        }))
    }

    /// Arm I (GPU side): the layout conversion the decoder would pay if the
    /// ANE wrote into its natural layout. The ANE slab is `[1, 4, C, 128]`;
    /// the decoder-consumable layout is `[1, C, 4, 128]`. The transpose is a
    /// view; the subsequent `contiguous()` (MLX op 3) performs the physical
    /// layout conversion. The AE half (real ANE write into a shared buffer)
    /// is P3.
    fn run_layout(p: usize, reps: usize, warmup: usize) -> Result<Value> {
        // The I-arm converts the *full* ANE-layout KV slab for one request,
        // which is the 11 attention layers' rows, not the whole S(P) state
        // (conv windows are not transposed).
        let kv_rows = FULL_ATTN_LAYERS * p + SLIDING_ATTN_LAYERS * SLIDING_WINDOW;
        let slab_elems = 2 * KV_HEADS * HEAD_DIM * kv_rows;
        let slab_bytes = slab_elems * 2;
        let stream = Stream::metal(0)?;
        // ANE natural layout: [1, KV_HEADS, rows, HEAD_DIM].
        let shape = vec![1, KV_HEADS, kv_rows, HEAD_DIM];
        let mut data = vec![0u8; slab_elems * 2];
        let live = KV_HEADS * p * HEAD_DIM; // full-attn rows only
        let live_bits = 0x3c00u16;
        for i in 0..live.min(slab_elems) {
            data[2 * i..2 * i + 2].copy_from_slice(&live_bits.to_le_bytes());
        }
        let source = Array::from_bytes(&stream, &shape, MlxDType::BF16, &data)?;
        source.eval()?;
        stream.synchronize()?;

        // [1, 4, rows, 128] -> [1, rows, 4, 128]: move the head axis behind
        // the row axis. `contiguous()` performs the physical conversion the
        // decoder would consume.
        let transposed = source.transpose(&[0, 2, 1, 3])?;
        let expected_shape = vec![1, kv_rows, KV_HEADS, HEAD_DIM];

        for _ in 0..warmup {
            let dst = transposed.contiguous()?;
            stream.eval(&[dst.clone()])?;
            stream.synchronize()?;
        }
        let mut times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let start = Instant::now();
            let dst = transposed.contiguous()?;
            stream.eval(&[dst.clone()])?;
            stream.synchronize()?;
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let med = median(times.as_slice());
        Ok(json!({
            "probe": "kv_handoff_layout",
            "arm": "I",
            "p": p,
            "slab_shape_ane": shape,
            "slab_shape_decoder": expected_shape,
            "transpose_axes": [0, 2, 1, 3],
            "bytes": slab_bytes,
            "reps": reps,
            "warmup": warmup,
            "times_ms": times,
            "median_ms": med,
            "spread": spread(times.as_slice()),
            "gbps": slab_bytes as f64 / (med / 1e3) / 1e9,
            "ae_half": "deferred to P3 (real ANE write into shared buffer)",
        }))
    }

    /// Convenience: run one mode across every requested P in one process.
    fn run_grid(mode: &str, ps: &[usize], reps: usize, warmup: usize) -> Result<Value> {
        let mut arms = Vec::with_capacity(ps.len());
        for &p in ps {
            let arm = match mode {
                "copy" => run_copy(p, reps, warmup)?,
                "host" => run_host(p, reps, warmup)?,
                "iosurface" => run_layout(p, reps, warmup)?,
                other => return Err(err(format!("unknown mode {other:?}"))),
            };
            arms.push(arm);
        }
        Ok(json!({
            "probe": "kv_handoff_grid",
            "mode": mode,
            "ps": ps,
            "reps": reps,
            "warmup": warmup,
            "arms": arms,
            "state_bytes_table": EXPECTED_S.iter().map(|(p, s)| json!({"p": p, "bytes": s})).collect::<Vec<_>>(),
        }))
    }

    /// IOSurface feasibility half of arm I: invoke the standalone
    /// `iosurface_probe` binary (dlopen-only, no IOSurface state in this
    /// process) and merge its result with this process's GPU layout arms.
    /// `--iosurface-probe <path>` overrides the default sibling-binary
    /// location.
    fn run_iosurface(
        shell_out: Option<PathBuf>,
        ps: &[usize],
        reps: usize,
        warmup: usize,
    ) -> Result<Value> {
        let bin =
            shell_out.unwrap_or_else(|| PathBuf::from("target/release/examples/iosurface_probe"));
        let output = Command::new(&bin)
            .output()
            .map_err(|e| err(format!("spawn {}: {e}", bin.display())))?;
        if !output.status.success() {
            return Err(err(format!(
                "iosurface_probe exited {:?}: {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            )));
        }
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let feasibility: Value =
            serde_json::from_str(&text).map_err(|e| err(format!("iosurface_probe json: {e}")))?;
        // GPU layout conversion at every requested P; feasibility is one
        // report (fixed 8192-row slab), so it is merged once.
        let grid = run_grid("iosurface", ps, reps, warmup)?;
        Ok(json!({
            "probe": "kv_handoff_iosurface",
            "arm": "I",
            "ps": ps,
            "reps": reps,
            "warmup": warmup,
            "feasibility": feasibility,
            "gpu_layout_arms": grid["arms"],
            "note": "GPU-side layout conversion only; real ANE write into shared buffer is P3",
        }))
    }

    fn parse_u32s(list: &str) -> Result<Vec<usize>> {
        list.split(',')
            .map(|s| {
                s.trim()
                    .parse()
                    .map_err(|_| err(format!("not a list of integers: {list:?}")))
            })
            .collect()
    }

    pub fn run() -> Result<()> {
        let args: Vec<String> = env::args().skip(1).collect();
        let mut mode: Option<String> = None;
        let mut p_list: Vec<usize> = DEFAULT_P.to_vec();
        let mut reps = DEFAULT_REPS;
        let mut warmup = DEFAULT_WARMUP;
        let mut out: Option<PathBuf> = None;
        let mut iosurface_probe: Option<PathBuf> = None;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--p" => {
                    i += 1;
                    p_list = parse_u32s(args.get(i).ok_or_else(|| err("--p needs a value"))?)?;
                }
                "--reps" => {
                    i += 1;
                    reps = args
                        .get(i)
                        .ok_or_else(|| err("--reps needs a value"))?
                        .parse()
                        .map_err(|_| err("--reps is not a number"))?;
                }
                "--warmup" => {
                    i += 1;
                    warmup = args
                        .get(i)
                        .ok_or_else(|| err("--warmup needs a value"))?
                        .parse()
                        .map_err(|_| err("--warmup is not a number"))?;
                }
                "--out" => {
                    i += 1;
                    out = Some(PathBuf::from(
                        args.get(i).ok_or_else(|| err("--out needs a value"))?,
                    ));
                }
                "--iosurface-probe" => {
                    i += 1;
                    iosurface_probe = Some(PathBuf::from(
                        args.get(i)
                            .ok_or_else(|| err("--iosurface-probe needs a value"))?,
                    ));
                }
                other if other.starts_with("--") => {
                    return Err(err(format!("unknown flag {other:?}")));
                }
                other => mode = Some(other.to_owned()),
            }
            i += 1;
        }
        let mode = mode.ok_or_else(|| {
            err("mode required (copy|host|iosurface); --p/--reps/--warmup/--out/--iosurface-probe")
        })?;
        let payload = match mode.as_str() {
            "copy" | "host" => run_grid(&mode, &p_list, reps, warmup)?,
            "iosurface" => run_iosurface(iosurface_probe)?,
            other => return Err(err(format!("unknown mode {other:?}"))),
        };
        let text =
            serde_json::to_string(&payload).map_err(|error| err(format!("serde_json: {error}")))?;
        if let Some(path) = out {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, format!("{text}\n"))?;
        }
        println!("{text}");
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    if let Err(error) = native::run() {
        eprintln!("ane_kv_handoff_probe: {error:?}");
        std::process::exit(1);
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("ane_kv_handoff_probe requires Apple Silicon macOS");
    std::process::exit(1);
}
