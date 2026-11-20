//! LFM3 MLP ANE INT8/FP16 precision probe with real checkpoint weights and
//! real activations (issue #150, preregister `doc/lfm3-ane-int8-parity.md`).
//!
//! Rebuilds Clef's MLP program (`metal_clef/ane.rs::program`) operation for
//! operation at the LFM3 geometry (hidden 2048, intermediate 7680): per-row
//! symmetric INT8 1x1 convs for gate/up, `0.5 v (1 + tanh(v/2))` SiLU, gate/up
//! row-max normalization, the normalized product, and the down projection as
//! K-split INT8 convolutions over 512-token tiles. Weights are quantized on the
//! host from the BF16 checkpoint exactly as Clef's `int8_per_row` does, and the
//! input is a real post-`ffn_norm` activation captured from the MLX BF16 engine
//! by `devlocal/lfm3-ane-int8-parity/scripts/export_inputs.py`.
//!
//! The probe restores each share (`y * gate_scale * up_scale`) and writes the
//! restored output, the normalized SwiGLU product (`a`), and the two scales as
//! FP32 [tokens, width] files for the GPU reference and the analysis. Arms:
//!
//! * `int8` at S = 7680 (whole MLP on the engine) and S = 5632 (columns
//!   `[2048, 7680)`); S = 5632 is the deployment share at rho ~ 2.5.
//! * `fp16` at S = 7680 (the fallback's weight precision), same chain with
//!   FP16 constant convolutions and the down K-split for shape reasons.
//!
//! ```sh
//! cargo run -p apxinf-coreml --features native --example lfm3_mlp_parity_probe \
//!   --release -- <inputs-dir> <outputs-dir>
//! ```

use apxinf_core::Result;
use apxinf_coreml::{
    ComputeDeviceKind, ComputeUnits, Operand, Package, ProgramBuilder, ProgramCache, Var,
};
use half::f16;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 7680;
const TILE: usize = 512;
const WARMUP: usize = 3;
const RUNS: usize = 20;
const DOWN_PART: usize = 4096;
const FACTOR_SCALE: f64 = 16.0;
const DOWN_ABS_SUM_LIMIT: f64 = 8192.0;
const LAYERS: [usize; 4] = [0, 4, 11, 22];
const SHARE: usize = 5632;
/// Deployment column split: the engine runs `[2048, 7680)` of the intermediate.
const SHARE_START: usize = INTERMEDIATE - SHARE;

fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .map_err(|error| apxinf_core::Error::Other(format!("read {}: {error}", path.display())))?;
    if bytes.len() != expected * 4 {
        return Err(apxinf_core::Error::Other(format!(
            "{} holds {} bytes, expected {expected} f32 values",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn write_f32(path: &Path, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes)
        .map_err(|error| apxinf_core::Error::Other(format!("write {}: {error}", path.display())))
}

/// Clef's `int8_per_row`: symmetric per-row INT8, `scale = f16(max|w| / 127)`,
/// `q = round_ties_even(w / scale)` clamped to `[-127, 127]`. `w` is FP16 as
/// Clef sees it after dequantization.
fn int8_per_row(w: &[f16], cols: usize) -> (Vec<i8>, Vec<f16>) {
    let mut q = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(w.len() / cols);
    for row in w.chunks_exact(cols) {
        let max = row.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
        let scale = f16::from_f32(max / 127.0);
        let s = scale.to_f32();
        scales.push(scale);
        q.extend(row.iter().map(|v| {
            if s == 0.0 {
                0
            } else {
                (v.to_f32() / s).round_ties_even().clamp(-127.0, 127.0) as i8
            }
        }));
    }
    (q, scales)
}

/// Rows `[start, start + rows)` of a row-major `[.., cols]` f32 matrix.
fn slice_rows(w: &[f32], cols: usize, start: usize, rows: usize) -> Vec<f32> {
    w[start * cols..(start + rows) * cols].to_vec()
}

/// Columns `[start, start + width)` of a row-major `[rows, cols]` f32 matrix.
fn slice_columns(w: &[f32], cols: usize, start: usize, width: usize) -> Vec<f32> {
    w.chunks_exact(cols)
        .flat_map(|row| row[start..start + width].iter().copied())
        .collect()
}

fn to_f16(w: &[f32]) -> Vec<f16> {
    w.iter().map(|&v| f16::from_f32(v)).collect()
}

/// Columns `[start, start + width)` of a row-major `[rows, cols]` f16 matrix.
fn slice_columns_f16(w: &[f16], cols: usize, start: usize, width: usize) -> Vec<f16> {
    w.chunks_exact(cols)
        .flat_map(|row| row[start..start + width].iter().copied())
        .collect()
}

/// Clef's `factor_cap` over the down projection's quantized (INT8 or FP16)
/// parts: bound the largest restored row sum so the normalization stays in
/// FP16 range.
fn factor_cap(parts: &[(Vec<f16>, usize)]) -> f64 {
    // Each part: (weights as FP16, width).
    let mut row_sums = vec![0f64; HIDDEN];
    for (weights, width) in parts {
        for (r, sum) in row_sums.iter_mut().enumerate() {
            let abs: f64 = weights[r * width..(r + 1) * width]
                .iter()
                .map(|v| f64::from(v.to_f32()).abs())
                .sum();
            *sum += abs;
        }
    }
    let max_row_sum = row_sums.iter().fold(1f64, |m, &v| m.max(v));
    2f64.powf(
        FACTOR_SCALE
            .min((DOWN_ABS_SUM_LIMIT / max_row_sum).sqrt())
            .log2()
            .floor(),
    )
}

fn cap_from_int8(parts: &[(Vec<i8>, Vec<f16>, usize)]) -> f64 {
    let mut row_sums = vec![0f64; HIDDEN];
    for (q, s, width) in parts {
        for (r, sum) in row_sums.iter_mut().enumerate() {
            let abs: f64 = q[r * width..(r + 1) * width]
                .iter()
                .map(|&v| f64::from(v).abs())
                .sum();
            *sum += f64::from(s[r].to_f32()) * abs;
        }
    }
    let max_row_sum = row_sums.iter().fold(1f64, |m, &v| m.max(v));
    2f64.powf(
        FACTOR_SCALE
            .min((DOWN_ABS_SUM_LIMIT / max_row_sum).sqrt())
            .log2()
            .floor(),
    )
}

struct Arm {
    package: Package,
    width: usize,
    cap: f64,
    placement: Vec<String>,
}

/// The INT8 arm: exact Clef `program()` over intermediate range
/// `[start, start + s)`, with the probe's extra `a` output.
fn int8_arm(
    wg: &[f32],
    wu: &[f32],
    wd: &[f32],
    start: usize,
    s: usize,
    tile: usize,
) -> Result<Arm> {
    let (qg, sg) = int8_per_row(&to_f16(&slice_rows(wg, HIDDEN, start, s)), HIDDEN);
    let (qu, su) = int8_per_row(&to_f16(&slice_rows(wu, HIDDEN, start, s)), HIDDEN);
    let wd_s = slice_columns(wd, INTERMEDIATE, start, s);
    let parts = s.div_ceil(DOWN_PART);
    let width = s / parts;
    assert_eq!(width * parts, s, "the down split must cover S evenly");
    let down: Vec<(Vec<i8>, Vec<f16>)> = (0..parts)
        .map(|p| int8_per_row(&to_f16(&slice_columns(&wd_s, s, p * width, width)), width))
        .collect();
    let cap = cap_from_int8(
        &down
            .iter()
            .map(|(q, s)| (q.clone(), s.clone(), width))
            .collect::<Vec<_>>(),
    );
    if cap < 1.0 {
        return Err(apxinf_core::Error::Contract(
            "down weights need factor scales outside the FP16 range".into(),
        ));
    }
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, HIDDEN, 1, tile])?;
    let wg_v = b.constexpr_blockwise_shift_scale(&qg, &[s, HIDDEN, 1, 1], &sg, &[s, 1, 1, 1])?;
    let g = b.conv(&x, &wg_v)?;
    let wu_v = b.constexpr_blockwise_shift_scale(&qu, &[s, HIDDEN, 1, 1], &su, &[s, 1, 1, 1])?;
    let u = b.conv(&x, &wu_v)?;
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let t = b.tanh(&half)?;
    let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
    let silu = b.mul(&half, Operand::Var(&t1), None)?;
    let sa = b.abs(&silu)?;
    let s_max = b.reduce_max(&sa, &[1], true)?;
    let ua = b.abs(&u)?;
    let u_max = b.reduce_max(&ua, &[1], true)?;
    let inverse_cap = f16::from_f64(1.0 / cap);
    let floor = f16::from_f64(1.0 / FACTOR_SCALE);
    let sm = b.mul(&s_max, Operand::Scalar(inverse_cap), None)?;
    let s_scale = b.maximum(&sm, Operand::Scalar(floor), Some("gate_scale"))?;
    let um = b.mul(&u_max, Operand::Scalar(inverse_cap), None)?;
    let u_scale = b.maximum(&um, Operand::Scalar(floor), Some("up_scale"))?;
    let sn = b.real_div(&silu, Operand::Var(&s_scale), None)?;
    let un = b.real_div(&u, Operand::Var(&u_scale), None)?;
    let a = b.mul(&sn, Operand::Var(&un), None)?;
    let a_out = b.identity(&a, Some("a"))?;
    let xs = if parts > 1 {
        b.split(&a, &vec![width; parts], 1)?
    } else {
        vec![a]
    };
    let mut y: Option<Var> = None;
    for (xi, (qd, sd)) in xs.iter().zip(&down) {
        let wd_v =
            b.constexpr_blockwise_shift_scale(qd, &[HIDDEN, width, 1, 1], sd, &[HIDDEN, 1, 1, 1])?;
        let yi = b.conv(xi, &wd_v)?;
        y = Some(match y {
            None => yi,
            Some(acc) => b.add(&acc, Operand::Var(&yi), None)?,
        });
    }
    let y = b.identity(&y.expect("at least one down part"), Some("y"))?;
    Ok(Arm {
        package: b.finish(&[&y, &s_scale, &u_scale, &a_out])?,
        width: s,
        cap,
        placement: Vec::new(),
    })
}

/// The FP16 fallback arm: the same chain with FP16 constant convolutions. The
/// down projection is K-split into `ceil(S / DOWN_PART)` convolutions that are
/// summed (algebraically exact), because one `[2048, 7680]` FP16 conv is a
/// larger shape than the shipped neuron split.
fn fp16_arm(wg: &[f32], wu: &[f32], wd: &[f32], s: usize, tile: usize) -> Result<Arm> {
    let fg = to_f16(&slice_rows(wg, HIDDEN, 0, s));
    let fu = to_f16(&slice_rows(wu, HIDDEN, 0, s));
    let wd_s = to_f16(&slice_columns(wd, INTERMEDIATE, 0, s));
    let parts = s.div_ceil(DOWN_PART);
    let width = s / parts;
    let down: Vec<Vec<f16>> = (0..parts)
        .map(|p| slice_columns_f16(&wd_s, s, p * width, width))
        .collect();
    let cap = factor_cap(&down.iter().map(|d| (d.clone(), width)).collect::<Vec<_>>());
    if cap < 1.0 {
        return Err(apxinf_core::Error::Contract(
            "down weights need factor scales outside the FP16 range".into(),
        ));
    }
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, HIDDEN, 1, tile])?;
    let g = b.conv_fp16(&x, &fg, &[s, HIDDEN, 1, 1])?;
    let u = b.conv_fp16(&x, &fu, &[s, HIDDEN, 1, 1])?;
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let t = b.tanh(&half)?;
    let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
    let silu = b.mul(&half, Operand::Var(&t1), None)?;
    let sa = b.abs(&silu)?;
    let s_max = b.reduce_max(&sa, &[1], true)?;
    let ua = b.abs(&u)?;
    let u_max = b.reduce_max(&ua, &[1], true)?;
    let inverse_cap = f16::from_f64(1.0 / cap);
    let floor = f16::from_f64(1.0 / FACTOR_SCALE);
    let sm = b.mul(&s_max, Operand::Scalar(inverse_cap), None)?;
    let s_scale = b.maximum(&sm, Operand::Scalar(floor), Some("gate_scale"))?;
    let um = b.mul(&u_max, Operand::Scalar(inverse_cap), None)?;
    let u_scale = b.maximum(&um, Operand::Scalar(floor), Some("up_scale"))?;
    let sn = b.real_div(&silu, Operand::Var(&s_scale), None)?;
    let un = b.real_div(&u, Operand::Var(&u_scale), None)?;
    let a = b.mul(&sn, Operand::Var(&un), None)?;
    let a_out = b.identity(&a, Some("a"))?;
    let xs = if parts > 1 {
        b.split(&a, &vec![width; parts], 1)?
    } else {
        vec![a]
    };
    let mut y: Option<Var> = None;
    for (xi, fd) in xs.iter().zip(&down) {
        let yi = b.conv_fp16(xi, fd, &[HIDDEN, width, 1, 1])?;
        y = Some(match y {
            None => yi,
            Some(acc) => b.add(&acc, Operand::Var(&yi), None)?,
        });
    }
    let y = b.identity(&y.expect("one down part"), Some("y"))?;
    Ok(Arm {
        package: b.finish(&[&y, &s_scale, &u_scale, &a_out])?,
        width: s,
        cap,
        placement: Vec::new(),
    })
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
    layer: usize,
    arm: &'static str,
    s: usize,
    median_ms: f64,
    four_tile_ms: f64,
    tmacs: f64,
    cap: f64,
    placement_ok: bool,
    placement: String,
}

/// Runs the arm on the real 512-token activation tile. `tiles` > 1 times the
/// deployment's sequential 512-token tiles (4x512 = a 2048-token prefill).
fn run_arm(
    cache: &ProgramCache,
    mut arm: Arm,
    x: &[f32],
    tiles: usize,
    out: Option<(&Path, &str)>,
) -> Result<(f64, f64, bool, String)> {
    let t = TILE;
    let mut devices = Vec::new();
    let model = cache.load_with_placement(
        &arm.package,
        ComputeUnits::CpuAndNeuralEngine,
        Duration::from_secs(600),
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
    arm.placement = devices;
    let placement_ok = arm
        .placement
        .iter()
        .filter(|d| d.starts_with("conv:"))
        .all(|d| d == "conv:ANE");
    // Real activation [tokens, HIDDEN] -> the program's [1, HIDDEN, 1, T] tile.
    let xf: Vec<f16> = (0..HIDDEN)
        .flat_map(|h| (0..t).map(move |token| f16::from_f32(x[token * HIDDEN + h])))
        .collect();
    let in_shape = [1usize, HIDDEN, 1, t];
    let y_shape = [1usize, HIDDEN, 1, t];
    let a_shape = [1usize, arm.width, 1, t];
    let scale_shape = [1usize, 1, 1, t];
    let feed = [("x", &xf[..], &in_shape[..])];
    let want: [(&str, &[usize]); 4] = [
        ("y", &y_shape),
        ("gate_scale", &scale_shape),
        ("up_scale", &scale_shape),
        ("a", &a_shape),
    ];
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    let mut last = None;
    for _ in 0..RUNS {
        let started = Instant::now();
        let mut prediction = None;
        for _ in 0..tiles {
            prediction = Some(model.predict(&feed, &want)?);
        }
        samples.push(started.elapsed().as_secs_f64());
        last = prediction;
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[RUNS / 2];
    if let Some((dir, stem)) = out {
        let out_vals = last.expect("one prediction");
        let [y_raw, gs_raw, us_raw, a_raw] =
            [&out_vals[0], &out_vals[1], &out_vals[2], &out_vals[3]];
        // Restore per token: `y * gate_scale * up_scale` is the MLP output
        // Clef's merge kernel produces; `a * gate_scale * up_scale` is the
        // SwiGLU product before the down projection.
        let mut y = vec![0f32; TILE * HIDDEN];
        let mut a = vec![0f32; TILE * arm.width];
        let mut gs = vec![0f32; TILE];
        let mut us = vec![0f32; TILE];
        for token in 0..TILE {
            gs[token] = gs_raw[token].to_f32();
            us[token] = us_raw[token].to_f32();
            let restore = gs[token] * us[token];
            for c in 0..HIDDEN {
                y[token * HIDDEN + c] = y_raw[c * TILE + token].to_f32() * restore;
            }
            for c in 0..arm.width {
                a[token * arm.width + c] = a_raw[c * TILE + token].to_f32() * restore;
            }
        }
        write_f32(&dir.join(format!("{stem}_y.f32")), &y)?;
        write_f32(&dir.join(format!("{stem}_a.f32")), &a)?;
        write_f32(&dir.join(format!("{stem}_gs.f32")), &gs)?;
        write_f32(&dir.join(format!("{stem}_us.f32")), &us)?;
    }
    let macs = 3.0 * TILE as f64 * HIDDEN as f64 * arm.width as f64;
    Ok((
        median * 1e3,
        macs / median / 1e12,
        placement_ok,
        arm.placement.join(" "),
    ))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let inputs = PathBuf::from(args.next().expect("usage: <inputs-dir> <outputs-dir>"));
    let out = PathBuf::from(args.next().expect("usage: <inputs-dir> <outputs-dir>"));
    std::fs::create_dir_all(&out).ok();
    let dir = std::env::var("APXINF_LFM3_PARITY_CACHE")
        .unwrap_or_else(|_| "/tmp/lfm3-mlp-parity-cache".to_owned());
    std::fs::create_dir_all(&dir).ok();
    let cache = ProgramCache::new(dir);

    // One layer per process is the M4 workaround for the engine's intermittent
    // "compute plan internal failure": the failure appears after several
    // programs have been loaded in one process.
    let selected: Vec<usize> = std::env::var("APXINF_LFM3_PARITY_LAYERS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| LAYERS.to_vec());
    let mut cells = Vec::new();
    for layer in selected {
        let x = read_f32(&inputs.join(format!("layer{layer}_x.f32")), TILE * HIDDEN)?;
        let w1 = read_f32(
            &inputs.join(format!("layer{layer}_w1.f32")),
            INTERMEDIATE * HIDDEN,
        )?;
        let w3 = read_f32(
            &inputs.join(format!("layer{layer}_w3.f32")),
            INTERMEDIATE * HIDDEN,
        )?;
        let w2 = read_f32(
            &inputs.join(format!("layer{layer}_w2.f32")),
            HIDDEN * INTERMEDIATE,
        )?;

        let arm = int8_arm(&w1, &w3, &w2, 0, INTERMEDIATE, TILE)?;
        let cap = arm.cap;
        let (ms, tmacs, ok, placement) = run_arm(
            &cache,
            arm,
            &x,
            1,
            Some((&out, &format!("layer{layer}_int8_s7680"))),
        )?;
        println!(
            "lfm3-mlp-parity-probe: layer={layer} arm=int8 s=7680 median={ms:.3}ms \
             rate={tmacs:.2}TMAC/s cap={cap:.3} placement_ok={ok}"
        );
        let arm = int8_arm(&w1, &w3, &w2, 0, INTERMEDIATE, TILE)?;
        let (four, _, _, _) = run_arm(&cache, arm, &x, 4, None)?;
        println!("lfm3-mlp-parity-probe: layer={layer} arm=int8 s=7680 four_tile={four:.3}ms");
        cells.push(Cell {
            layer,
            arm: "int8",
            s: INTERMEDIATE,
            median_ms: ms,
            four_tile_ms: four,
            tmacs,
            cap,
            placement_ok: ok,
            placement,
        });

        let arm = int8_arm(&w1, &w3, &w2, SHARE_START, SHARE, TILE)?;
        let cap = arm.cap;
        let (ms, tmacs, ok, placement) = run_arm(
            &cache,
            arm,
            &x,
            1,
            Some((&out, &format!("layer{layer}_int8_s5632"))),
        )?;
        println!(
            "lfm3-mlp-parity-probe: layer={layer} arm=int8 s=5632 median={ms:.3}ms \
             rate={tmacs:.2}TMAC/s cap={cap:.3} placement_ok={ok}"
        );
        cells.push(Cell {
            layer,
            arm: "int8",
            s: SHARE,
            median_ms: ms,
            four_tile_ms: 0.0,
            tmacs,
            cap,
            placement_ok: ok,
            placement,
        });

        let arm = fp16_arm(&w1, &w3, &w2, INTERMEDIATE, TILE)?;
        let cap = arm.cap;
        let (ms, tmacs, ok, placement) = run_arm(
            &cache,
            arm,
            &x,
            1,
            Some((&out, &format!("layer{layer}_fp16_s7680"))),
        )?;
        let arm = fp16_arm(&w1, &w3, &w2, INTERMEDIATE, TILE)?;
        let (four, _, _, _) = run_arm(&cache, arm, &x, 4, None)?;
        println!(
            "lfm3-mlp-parity-probe: layer={layer} arm=fp16 s=7680 median={ms:.3}ms \
             four_tile={four:.3}ms rate={tmacs:.2}TMAC/s cap={cap:.3} placement_ok={ok}"
        );
        cells.push(Cell {
            layer,
            arm: "fp16",
            s: INTERMEDIATE,
            median_ms: ms,
            four_tile_ms: four,
            tmacs,
            cap,
            placement_ok: ok,
            placement,
        });
    }

    let mut json = format!(
        "{{\"kind\":\"lfm3-mlp-parity-ane\",\"arch\":\"{}\",\"cells\":[",
        std::env::consts::ARCH
    );
    let mut first = true;
    for cell in &cells {
        if !first {
            json.push(',');
        }
        first = false;
        json.push_str(&format!(
            "{{\"layer\":{},\"arm\":\"{}\",\"s\":{},\"median_ms\":{:.4},\"four_tile_ms\":{:.4},\
             \"tmacs\":{:.4},\"factor_cap\":{:.6},\"placement_ok\":{},\"placement\":\"{}\"}}",
            cell.layer,
            cell.arm,
            cell.s,
            cell.median_ms,
            cell.four_tile_ms,
            cell.tmacs,
            cell.cap,
            cell.placement_ok,
            cell.placement
        ));
    }
    json.push_str("]}");
    let name = format!(
        "ane-{}.json",
        std::env::var("APXINF_LFM3_PARITY_LAYERS").unwrap_or_else(|_| "all".into())
    );
    std::fs::write(out.join(&name), json).expect("write ane json");
    println!("wrote {}", out.join(&name).display());
    Ok(())
}
