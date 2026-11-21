//! LFM3 MLP column-share throughput probe for the GPU (issue #145, preregister
//! `doc/lfm3-ane-mlp-preregister.md`). The MLX baseline against which the ANE
//! probe (`apxinf-coreml/examples/lfm3_mlp_probe.rs`) is compared: the same
//! MAC count `3 * T * 2048 * S` per cell, as the chain the LFM3 engine runs
//! today — gate/up projections, `silu(g) * u`, down projection.
//!
//! Two arms per cell: BF16 matmuls (the certified #121 engine precision) and
//! Q4 group-64 `quantized_matmul` (the packed 4-bit premise of #144). Weights
//! are transposed/quantized once before timing, as a load would. Warmup 3, 20
//! timed evaluations, median reported.
//!
//! ```sh
//! DYLD_LIBRARY_PATH=$MLX_ROOT/lib cargo run -p apxinf-mlx --features native \
//!   --example lfm3_mlp_gpu_probe --release -- [out.json]
//! ```

use apxinf_core::Result;
use apxinf_mlx::{Array, MlxDType, Stream};
use std::time::Instant;

const HIDDEN: usize = 2048;
const WARMUP: usize = 3;
const RUNS: usize = 20;

fn normal(stream: &Stream, shape: &[usize], seed: u64, scale: f32) -> Result<Array> {
    Array::random_normal(shape, MlxDType::F32, &Array::random_key(stream, seed)?)?
        .mul(&Array::scalar(stream, scale, MlxDType::F32)?)?
        .cast(MlxDType::BF16)
}

struct Cell {
    s: usize,
    t: usize,
    arm: &'static str,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    /// GPU rate in TFLOP/s: one multiply-add counts as 2 FLOPs (multiply +
    /// add), while the ANE probe's `tmacs` counts it as 1 MAC. Do not divide
    /// `tmacs` by `tflops` directly; use `2 * tmacs / tflops` or `gpu_ms / ane_ms`.
    tflops: f64,
}

fn time_chain(
    stream: &Stream,
    iters: usize,
    mut chain: impl FnMut() -> Result<Array>,
) -> Result<(f64, f64, f64)> {
    for _ in 0..WARMUP {
        stream.eval(&[chain()?])?;
    }
    let mut rows = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        stream.eval(&[chain()?])?;
        rows.push(started.elapsed().as_secs_f64());
    }
    rows.sort_by(f64::total_cmp);
    Ok((rows[rows.len() / 2], rows[0], rows[rows.len() - 1]))
}

fn probe_cell(stream: &Stream, s: usize, t: usize, arm: &'static str) -> Result<Cell> {
    let x = normal(stream, &[1, t, HIDDEN], 11, 1.0)?;
    let wg = normal(stream, &[s, HIDDEN], 7, 0.02)?;
    let wu = normal(stream, &[s, HIDDEN], 13, 0.02)?;
    let wd = normal(stream, &[HIDDEN, s], 17, 0.02)?;
    let flops = 2.0 * 3.0 * t as f64 * HIDDEN as f64 * s as f64;

    let (median, min, max) = match arm {
        "bf16" => {
            // A load transposes once; the timed chain sees the ready operand.
            let wg_t = wg.transpose(&[1, 0])?;
            let wu_t = wu.transpose(&[1, 0])?;
            let wd_t = wd.transpose(&[1, 0])?;
            stream.eval(&[wg_t.clone(), wu_t.clone(), wd_t.clone(), x.clone()])?;
            time_chain(stream, RUNS, || {
                let g = x.matmul(&wg_t)?;
                let u = x.matmul(&wu_t)?;
                let h = g.mul(&g.sigmoid()?)?.mul(&u)?;
                h.matmul(&wd_t)
            })?
        }
        "q4" => {
            let [qg, sg, bg] = wg.quantize(64, 4)?;
            let [qu, su, bu] = wu.quantize(64, 4)?;
            let [qd, sd, bd] = wd.quantize(64, 4)?;
            stream.eval(&[
                qg.clone(), sg.clone(), bg.clone(),
                qu.clone(), su.clone(), bu.clone(),
                qd.clone(), sd.clone(), bd.clone(),
                x.clone(),
            ])?;
            time_chain(stream, RUNS, || {
                let g = x.quantized_matmul(&qg, &sg, &bg, true, 64, 4)?;
                let u = x.quantized_matmul(&qu, &su, &bu, true, 64, 4)?;
                let h = g.mul(&g.sigmoid()?)?.mul(&u)?;
                h.quantized_matmul(&qd, &sd, &bd, true, 64, 4)
            })?
        }
        other => panic!("unknown arm {other}"),
    };
    Ok(Cell {
        s,
        t,
        arm,
        median_ms: median * 1e3,
        min_ms: min * 1e3,
        max_ms: max * 1e3,
        tflops: flops / median / 1e12,
    })
}

fn main() -> Result<()> {
    let stream = Stream::metal(0)?;
    let out_path = std::env::args().nth(1);
    let mut cells = Vec::new();
    for s in [2048usize, 3584, 4096, 5632, 7680] {
        for t in [128usize, 512, 2048] {
            for arm in ["bf16", "q4"] {
                let cell = probe_cell(&stream, s, t, arm)?;
                println!(
                    "lfm3-mlp-gpu-probe: S={s} T={t} {arm} median={:.3}ms min={:.3} max={:.3} \
                     rate={:.2}TFLOP/s",
                    cell.median_ms, cell.min_ms, cell.max_ms, cell.tflops
                );
                cells.push(cell);
            }
        }
    }
    let mut json = format!(
        "{{\"kind\":\"lfm3-mlp-gpu-probe\",\"arch\":\"{}\",\"cells\":[",
        std::env::consts::ARCH
    );
    let mut first = true;
    for cell in &cells {
        if !first {
            json.push(',');
        }
        first = false;
        json.push_str(&format!(
            "{{\"s\":{},\"t\":{},\"arm\":\"{}\",\"median_ms\":{:.4},\"min_ms\":{:.4},\
             \"max_ms\":{:.4},\"tflops\":{:.4}}}",
            cell.s, cell.t, cell.arm, cell.median_ms, cell.min_ms, cell.max_ms, cell.tflops
        ));
    }
    json.push_str("]}");
    if let Some(path) = out_path {
        std::fs::write(&path, &json).expect("write probe json");
        println!("wrote {path}");
    }
    Ok(())
}
