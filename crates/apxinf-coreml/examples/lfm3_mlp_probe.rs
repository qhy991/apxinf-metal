//! LFM3 MLP column-share throughput probe for the Neural Engine (issue #145,
//! preregister `doc/lfm3-ane-mlp-preregister.md`).
//!
//! Rebuilds the Clef GPU+ANE column-split MLP program operation for operation
//! at the LFM3 geometry (hidden 2048, intermediate shares of 7680): INT8
//! per-row 1x1 convolutions for gate and up, `0.5 v (1 + tanh(v/2))` SiLU, the
//! activation-max normalization chain, and the down projection as K-split INT8
//! convolutions. Weights are synthetic deterministic INT8 (per-row FP16 scales
//! 2e-5, so `factor_cap = 16` like the shipped Clef regime): this probe
//! measures throughput only, under the preregister's "weights already INT8"
//! premise (blocked by #144 for anything end-to-end).
//!
//! Cells: S in {2048, 3584, 4096, 5632, 7680} x T in {128, 512, 2048}, plus
//! the deployment pattern T=2048 as four sequential 512-token tiles. Every
//! cell loads with placement evidence and requires every conv op on the
//! Neural Engine (gate G2). Warmup 3, 20 timed predictions, median reported.
//!
//! ```sh
//! cargo run -p apxinf-coreml --features native --example lfm3_mlp_probe --release -- [out.json]
//! ```

use apxinf_core::Result;
use apxinf_coreml::{
    ComputeDeviceKind, ComputeUnits, Operand, Package, ProgramBuilder, ProgramCache, Var,
};
use half::f16;
use std::time::{Duration, Instant};

const HIDDEN: usize = 2048;
const WARMUP: usize = 3;
const RUNS: usize = 20;
const DOWN_PART: usize = 4096;
const FACTOR_SCALE: f64 = 16.0;
const DOWN_ABS_SUM_LIMIT: f64 = 8192.0;

/// Deterministic xorshift64 so the synthetic weights are reproducible.
fn next_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Synthetic per-row INT8 weights with FP16 scales 2e-5: values are
/// throughput-neutral (the conv does not exploit them) but the encoding,
/// shapes and scales are the shipped recipe's. The scale keeps every down
/// row's absolute sum under 8192/256, so `factor_cap` lands on 16.
fn int8_weights(rows: usize, cols: usize, seed: u64) -> (Vec<i8>, Vec<f16>) {
    let mut state = seed | 1;
    let q = (0..rows * cols)
        .map(|_| ((next_u64(&mut state) % 255) as i16 - 127) as i8)
        .collect();
    let s = vec![f16::from_f32(2e-5); rows];
    (q, s)
}

/// The Clef factor cap (metal_clef/ane.rs): bounds the down dot products so
/// the activation normalization stays in FP16 range. With scale 1e-4 synthetic
/// weights this lands on 16 for every split below, the shipped regime.
fn factor_cap(down: &[(Vec<i8>, Vec<f16>)], width: usize) -> f64 {
    let mut max_row_sum = 1f64;
    for r in 0..HIDDEN {
        let sum: f64 = down
            .iter()
            .map(|(qd, sd)| {
                f64::from(sd[r].to_f32())
                    * qd[r * width..(r + 1) * width]
                        .iter()
                        .map(|&v| f64::from(v).abs())
                        .sum::<f64>()
            })
            .sum();
        max_row_sum = max_row_sum.max(sum);
    }
    2f64.powf(
        FACTOR_SCALE
            .min((DOWN_ABS_SUM_LIMIT / max_row_sum).sqrt())
            .log2()
            .floor(),
    )
}

/// One layer's LFM3 MLP column share: `S` intermediate columns out of 7680,
/// `T` tokens per tile. Mirrors metal_clef/ane.rs `program()` with
/// HIDDEN 4096 -> 2048 and the share as a parameter.
fn program(s: usize, t: usize) -> Result<Package> {
    let parts = s.div_ceil(DOWN_PART);
    let width = s / parts;
    assert_eq!(width * parts, s, "the down split must cover S evenly");
    let (qg, sg) = int8_weights(s, HIDDEN, 0x9e3779b97f4a7c15 ^ (s as u64) << 32 | t as u64);
    let (qu, su) = int8_weights(s, HIDDEN, 0xc2b2ae3d27d4eb4f ^ (s as u64) << 32 | t as u64);
    let down: Vec<(Vec<i8>, Vec<f16>)> = (0..parts)
        .map(|p| int8_weights(HIDDEN, width, 0x165667b19e3779f9 ^ (p as u64)))
        .collect();
    let cap = factor_cap(&down, width);
    assert_eq!(cap, FACTOR_SCALE, "synthetic weights left the shipped factor regime");

    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, HIDDEN, 1, t])?;
    let wg = b.constexpr_blockwise_shift_scale(&qg, &[s, HIDDEN, 1, 1], &sg, &[s, 1, 1, 1])?;
    let g = b.conv(&x, &wg)?;
    let wu = b.constexpr_blockwise_shift_scale(&qu, &[s, HIDDEN, 1, 1], &su, &[s, 1, 1, 1])?;
    let u = b.conv(&x, &wu)?;
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let th = b.tanh(&half)?;
    let t1 = b.add(&th, Operand::Scalar(f16::ONE), None)?;
    let silu = b.mul(&half, Operand::Var(&t1), None)?;
    let sa = b.abs(&silu)?;
    let s_max = b.reduce_max(&sa, &[1], true)?;
    let ua = b.abs(&u)?;
    let u_max = b.reduce_max(&ua, &[1], true)?;
    let floor = f16::from_f64(1.0 / FACTOR_SCALE);
    let sm = b.mul(&s_max, Operand::Scalar(f16::from_f64(1.0 / cap)), None)?;
    let s_scale = b.maximum(&sm, Operand::Scalar(floor), Some("gate_scale"))?;
    let um = b.mul(&u_max, Operand::Scalar(f16::from_f64(1.0 / cap)), None)?;
    let u_scale = b.maximum(&um, Operand::Scalar(floor), Some("up_scale"))?;
    let sn = b.real_div(&silu, Operand::Var(&s_scale), None)?;
    let un = b.real_div(&u, Operand::Var(&u_scale), None)?;
    let a = b.mul(&sn, Operand::Var(&un), None)?;
    let xs = if parts > 1 {
        b.split(&a, &vec![width; parts], 1)?
    } else {
        vec![a]
    };
    let mut y: Option<Var> = None;
    for (xi, (qd, sd)) in xs.iter().zip(&down) {
        let wd =
            b.constexpr_blockwise_shift_scale(qd, &[HIDDEN, width, 1, 1], sd, &[HIDDEN, 1, 1, 1])?;
        let yi = b.conv(xi, &wd)?;
        y = Some(match y {
            None => yi,
            Some(acc) => b.add(&acc, Operand::Var(&yi), None)?,
        });
    }
    let y = b.identity(&y.expect("at least one down part"), Some("y"))?;
    b.finish(&[&y, &s_scale, &u_scale])
}

fn device_of(op: &apxinf_coreml::OperationPlacement) -> &'static str {
    match op.usage.as_ref().and_then(|u| u.preferred.as_ref()) {
        Some(device) => match device.kind {
            ComputeDeviceKind::NeuralEngine => "ANE",
            ComputeDeviceKind::Gpu => "GPU",
            ComputeDeviceKind::Cpu => "CPU",
            ComputeDeviceKind::Unknown => "?",
        },
        None => "-",
    }
}

struct Cell {
    s: usize,
    tokens: usize,
    mode: &'static str,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    /// ANE rate in TMAC/s: one multiply-add counts as 1 MAC. A GPU probe's
    /// `tflops` counts the same operation as 2 FLOPs, so use `2 * tmacs / tflops`.
    tmacs: f64,
    placement_ok: bool,
    placement: String,
}

/// Times `repeat` sequential predictions per sample; `tokens` is the total
/// token count the sample covers (for the MAC rate), so the 4x512 tiled mode
/// reports the whole 2048-token window.
fn probe_cell(cache: &ProgramCache, s: usize, tile: usize, tokens: usize, mode: &'static str, repeat: usize) -> Result<Cell> {
    let macs = 3.0 * tokens as f64 * HIDDEN as f64 * s as f64;
    let mut devices = Vec::new();
    let model = cache.load_with_placement(
        &program(s, tile)?,
        ComputeUnits::CpuAndNeuralEngine,
        Duration::from_secs(300),
        |facts| {
            devices.extend(facts.functions.iter().flat_map(|f| {
                f.operations
                    .iter()
                    .filter(|op| op.operator != "const")
                    .map(|op| format!("{}:{}", op.operator, device_of(op)))
            }));
            Ok(())
        },
    )?;
    let placement = devices.join(" ");
    let placement_ok = devices
        .iter()
        .filter(|d| d.starts_with("conv:"))
        .all(|d| d == "conv:ANE");

    let x = vec![f16::from_f32(0.01); HIDDEN * tile];
    let shape = [1usize, HIDDEN, 1, tile];
    let scale_shape = [1usize, 1, 1, tile];
    let feed = [("x", &x[..], &shape[..])];
    let want: [(&str, &[usize]); 3] = [
        ("y", &shape),
        ("gate_scale", &scale_shape),
        ("up_scale", &scale_shape),
    ];
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        let mut last = None;
        for _ in 0..repeat {
            last = Some(model.predict(&feed, &want)?);
        }
        samples.push(run.elapsed().as_secs_f64());
        assert!(
            last.expect("one prediction").first().map_or(true, |y| y[0].to_f32().is_finite()),
            "the probe went non-finite"
        );
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[RUNS / 2];
    Ok(Cell {
        s,
        tokens,
        mode,
        median_ms: median * 1e3,
        min_ms: samples[0] * 1e3,
        max_ms: samples[RUNS - 1] * 1e3,
        tmacs: macs / median / 1e12,
        placement_ok,
        placement,
    })
}

fn main() -> Result<()> {
    let dir = std::env::var("APXINF_LFM3_PROBE_CACHE")
        .unwrap_or_else(|_| "/tmp/lfm3-mlp-probe-cache".to_owned());
    std::fs::create_dir_all(&dir).ok();
    let cache = ProgramCache::new(dir);
    let out_path = std::env::args().nth(1);

    // APXINF_LFM3_PROBE_ONLY="S,T" runs one single-tile cell (smoke test).
    let only: Option<(usize, usize)> = std::env::var("APXINF_LFM3_PROBE_ONLY")
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once(',')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        });

    let mut cells = Vec::new();
    for s in [2048usize, 3584, 4096, 5632, 7680] {
        for t in [128usize, 512, 2048] {
            if let Some((os, ot)) = only {
                if (s, t) != (os, ot) {
                    continue;
                }
            }
            let cell = probe_cell(&cache, s, t, t, "single", 1)?;
            println!(
                "lfm3-mlp-probe: S={s} T={t} single median={:.3}ms min={:.3} max={:.3} \
                 rate={:.2}TMAC/s placement_ok={}",
                cell.median_ms, cell.min_ms, cell.max_ms, cell.tmacs, cell.placement_ok
            );
            cells.push(cell);
        }
        // The deployment pattern: a 2048-token prefill as four sequential
        // 512-token tiles on the same per-layer program.
        if only.is_some() {
            continue;
        }
        let cell = probe_cell(&cache, s, 512, 2048, "4x512", 4)?;
        println!(
            "lfm3-mlp-probe: S={s} T=2048 4x512 median={:.3}ms min={:.3} max={:.3} \
             rate={:.2}TMAC/s placement_ok={}",
            cell.median_ms, cell.min_ms, cell.max_ms, cell.tmacs, cell.placement_ok
        );
        cells.push(cell);
    }
    for cell in &cells {
        println!("lfm3-mlp-probe: placement S={} T={} {} => {}", cell.s, cell.tokens, cell.mode, cell.placement);
    }

    let mut json = format!(
        "{{\"kind\":\"lfm3-mlp-ane-probe\",\"arch\":\"{}\",\"cells\":[",
        std::env::consts::ARCH
    );
    let mut first = true;
    for cell in &cells {
        if !first {
            json.push(',');
        }
        first = false;
        json.push_str(&format!(
            "{{\"s\":{},\"tokens\":{},\"mode\":\"{}\",\"median_ms\":{:.4},\"min_ms\":{:.4},\
             \"max_ms\":{:.4},\"tmacs\":{:.4},\"placement_ok\":{},\"placement\":\"{}\"}}",
            cell.s, cell.tokens, cell.mode, cell.median_ms, cell.min_ms, cell.max_ms,
            cell.tmacs, cell.placement_ok, cell.placement
        ));
    }
    json.push_str("]}");
    if let Some(path) = out_path {
        std::fs::write(&path, &json).expect("write probe json");
        println!("wrote {path}");
    }
    Ok(())
}
