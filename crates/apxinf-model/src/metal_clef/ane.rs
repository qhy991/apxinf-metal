//! The Neural Engine share of each Clef MLP in the GPU+ANE column
//! composition. Intermediate columns `[GPU_COLUMNS, 12288)` run as one Core ML
//! program per layer; the GPU computes the rest with the GPU composition's MLP,
//! and the two shares are summed on the GPU. Portions adapted from EngineTailor
//! (MIT); see NOTICE.
//!
//! The program: gate and up as per-row symmetric INT8 1x1 convolutions, SiLU
//! as `0.5 v (1 + tanh(v / 2))`, gate and up normalized by their row maxima
//! before the product, down as K-split INT8 convolutions, 512-token tiles.
//! Weights are quantized on the host from the Q4 checkpoint.
use super::weights::{Mlp, Q4};
use apxinf_core::{Error, Result};
use apxinf_coreml::{ComputeUnits, Model, Operand, Package, ProgramBuilder, ProgramCache, Var};
use apxinf_mlx::{Array, MlxDType, Stream};
use half::f16;

/// Hidden width and MLP width of the only geometry this split was built for.
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 12288;
/// `12288 - round(12288 * 5632 / 12288 / 256) * 256`: the GPU keeps the first
/// 6656 columns, the Neural Engine runs the other 5632.
pub const GPU_COLUMNS: usize = 6656;
/// Column depths the campaign may run on the Neural Engine: the GPU keeps the
/// first `C` of 12288 intermediate columns, the Neural Engine runs the rest.
/// Each is a multiple of 64 so the packed Q4 down columns
/// (`weight` axis 1 length `C / 8`, scales/biases `C / 64`) stay whole.
const ANE_DEEPER_COLUMNS: &[usize] = &[6656, 5632, 4608];

/// The Neural Engine deeper-column split: `APXINF_CLEF_ANE_COLUMNS` overrides
/// the GPU's column share for the GPU+ANE column composition. Unset (or empty)
/// keeps [`GPU_COLUMNS`], the qualified default. Only a whole campaign-qualified
/// depth is accepted, so an arbitrary value cannot reach an unverified program.
/// Applies only to the GPU+ANE column composition; other compositions ignore it.
pub fn gpu_columns() -> Result<usize> {
    match std::env::var("APXINF_CLEF_ANE_COLUMNS").ok().as_deref() {
        None | Some("") => Ok(GPU_COLUMNS),
        Some(value) => value
            .parse::<usize>()
            .ok()
            .filter(|columns| ANE_DEEPER_COLUMNS.contains(columns))
            .ok_or_else(|| {
                Error::Contract(
                    "APXINF_CLEF_ANE_COLUMNS must be one of 6656, 5632 or 4608",
                )
            }),
    }
}

/// `(ane columns, down parts, down part width)` for a GPU column share. The
/// Neural Engine runs `[gpu, 12288)`; the down weight is K-split into at most
/// [`DOWN_PART`]-wide parts that are convolved and summed. All campaign depths
/// give two parts of a 128-multiple width, which the quantized layout needs.
pub fn ane_split_geometry(gpu: usize) -> Result<(usize, usize, usize)> {
    if gpu >= INTERMEDIATE || gpu % 64 != 0 {
        return Err(Error::Contract(
            "Clef GPU+ANE columns need a 64-multiple below 12288",
        ));
    }
    let n = INTERMEDIATE - gpu;
    let parts = n.div_ceil(DOWN_PART);
    if n % parts != 0 {
        return Err(Error::Contract(
            "Clef GPU+ANE down parts must divide the Neural Engine columns",
        ));
    }
    Ok((n, parts, n / parts))
}

const TILE: usize = 512;
const FACTOR_SCALE: f64 = 16.0;
const DOWN_ABS_SUM_LIMIT: f64 = 8192.0;
const DOWN_PART: usize = 4096;

/// FP16 values of rows `rows` and columns `cols` of a Q4 group-64 weight,
/// dequantized on the device as `mx.dequantize` does (BF16 scales), then
/// rounded to FP16.
fn dense(q: &Q4, rows: (usize, usize), cols: (usize, usize)) -> Result<Vec<f16>> {
    let slice = |a: &Array, per: usize| -> Result<Array> {
        a.slice_axis(0, rows.0, rows.1)?
            .slice_axis(1, cols.0 / per, cols.1 / per)
    };
    let w = slice(&q.weight, 8)?
        .dequantize(&slice(&q.scales, 64)?, &slice(&q.biases, 64)?, 64, 4)?
        .cast(MlxDType::F16)?;
    Ok(w.to_bytes()?
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// Symmetric per-row INT8 with FP16 scales: `scale = max|w| / 127` in FP32,
/// rounded to FP16; `q = clip(rint(w / scale), -127, 127)`.
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

/// Columns `[start, start + width)` of a row-major `[rows, cols]` matrix.
fn columns(w: &[f16], cols: usize, start: usize, width: usize) -> Vec<f16> {
    w.chunks_exact(cols)
        .flat_map(|row| row[start..start + width].iter().copied())
        .collect()
}

/// One layer's Core ML program for the Neural Engine share of `mlp`, the
/// intermediate columns `[gpu_columns, 12288)`.
pub fn program_with_columns(mlp: &Mlp, gpu_columns: usize) -> Result<Package> {
    let (n, parts, width) = ane_split_geometry(gpu_columns)?;
    if mlp.gate.rows() != INTERMEDIATE || mlp.down.rows() != HIDDEN {
        return Err(Error::Contract(
            "Clef GPU+ANE needs the 4096 x 12288 MLP geometry",
        ));
    }
    let (qg, sg) = int8_per_row(
        &dense(&mlp.gate, (gpu_columns, INTERMEDIATE), (0, HIDDEN))?,
        HIDDEN,
    );
    let (qu, su) = int8_per_row(
        &dense(&mlp.up, (gpu_columns, INTERMEDIATE), (0, HIDDEN))?,
        HIDDEN,
    );
    let wd = dense(&mlp.down, (0, HIDDEN), (gpu_columns, INTERMEDIATE))?;
    let down: Vec<(Vec<i8>, Vec<f16>)> = (0..parts)
        .map(|p| int8_per_row(&columns(&wd, n, p * width, width), width))
        .collect();
    // Bound the dot products of the INT8 weights the Neural Engine uses.
    let mut row_sums = vec![0f64; HIDDEN];
    for (qd, sd) in &down {
        for (r, sum) in row_sums.iter_mut().enumerate() {
            let abs: f64 = qd[r * width..(r + 1) * width]
                .iter()
                .map(|&v| f64::from(v).abs())
                .sum();
            *sum += f64::from(sd[r].to_f32()) * abs;
        }
    }
    let max_row_sum = row_sums.iter().fold(1f64, |m, &v| m.max(v));
    let factor_cap = 2f64.powf(
        FACTOR_SCALE
            .min((DOWN_ABS_SUM_LIMIT / max_row_sum).sqrt())
            .log2()
            .floor(),
    );
    if factor_cap < 1.0 {
        return Err(Error::Contract(
            "Clef GPU+ANE down weights need factor scales outside the FP16 range",
        ));
    }

    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, HIDDEN, 1, TILE]);
    let wg = b.constexpr_blockwise_shift_scale(&qg, &[n, HIDDEN, 1, 1], &sg, &[n, 1, 1, 1])?;
    let g = b.conv(&x, &wg)?;
    let wu = b.constexpr_blockwise_shift_scale(&qu, &[n, HIDDEN, 1, 1], &su, &[n, 1, 1, 1])?;
    let u = b.conv(&x, &wu)?;
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let t = b.tanh(&half);
    let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
    let s = b.mul(&half, Operand::Var(&t1), None)?;
    let sa = b.abs(&s);
    let s_max = b.reduce_max(&sa, &[1], true)?;
    let ua = b.abs(&u);
    let u_max = b.reduce_max(&ua, &[1], true)?;
    let inverse_cap = Operand::Scalar(f16::from_f64(1.0 / factor_cap));
    let floor = f16::from_f64(1.0 / FACTOR_SCALE);
    let sm = b.mul(&s_max, inverse_cap, None)?;
    let s_scale = b.maximum(&sm, Operand::Scalar(floor), Some("gate_scale"))?;
    let um = b.mul(
        &u_max,
        Operand::Scalar(f16::from_f64(1.0 / factor_cap)),
        None,
    )?;
    let u_scale = b.maximum(&um, Operand::Scalar(floor), Some("up_scale"))?;
    // Normalize before the product so a large SwiGLU value cannot overflow first.
    let sn = b.real_div(&s, Operand::Var(&s_scale), None)?;
    let un = b.real_div(&u, Operand::Var(&u_scale), None)?;
    let a = b.mul(&sn, Operand::Var(&un), None)?;
    let xs = if parts > 1 {
        b.split(&a, &vec![width; parts], 1)?
    } else {
        vec![a]
    };
    let mut y = None;
    for (xi, (qd, sd)) in xs.iter().zip(&down) {
        let wd =
            b.constexpr_blockwise_shift_scale(qd, &[HIDDEN, width, 1, 1], sd, &[HIDDEN, 1, 1, 1])?;
        let yi = b.conv(xi, &wd)?;
        y = Some(match y {
            None => yi,
            Some(acc) => b.add(&acc, Operand::Var(&yi), None)?,
        });
    }
    let y = b.identity(&y.expect("at least one down part"), Some("y"));
    b.finish(&[&y, &s_scale, &u_scale])
}

/// One layer's Core ML program for the Neural Engine share of `mlp` at the
/// qualified [`GPU_COLUMNS`] depth.
pub fn program(mlp: &Mlp) -> Result<Package> {
    program_with_columns(mlp, GPU_COLUMNS)
}

/// The neuron split: per MLP, the `NEURON_INT8` neurons whose int8 encoding
/// adds the least error run on the Neural Engine as per-channel int8, the
/// next `NEURON_FP16` as FP16, and the rest on the GPU, in `NEURON_TILE`-token
/// tiles.
pub const NEURON_INT8: usize = 5632;
pub const NEURON_FP16: usize = 1248;
pub const NEURON_TILE: usize = 128;

/// numpy 2.x `einsum("ij,ij->i", r, r, dtype=float64)` on Apple Silicon: one
/// two-lane accumulator, each 8-element block fused-multiply-added from its
/// last pair to its first, the tail in pairs; the lanes are summed last.
fn einsum_squares(r: &[f64]) -> f64 {
    let mut acc = [0f64; 2];
    let blocks = r.len() / 8;
    for b in 0..blocks {
        for k in (0..4).rev() {
            for (l, a) in acc.iter_mut().enumerate() {
                let v = r[8 * b + 2 * k + l];
                *a = v.mul_add(v, *a);
            }
        }
    }
    for pair in r[8 * blocks..].chunks(2) {
        for (a, &v) in acc.iter_mut().zip(pair) {
            *a = v.mul_add(v, *a);
        }
    }
    acc[0] + acc[1]
}

/// numpy's pairwise summation of a contiguous float64 array (`sum`).
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0, |s, &v| s + v);
    }
    if n <= 128 {
        let mut r: [f64; 8] = a[..8].try_into().unwrap();
        let whole = n - n % 8;
        for block in a[8..whole].chunks_exact(8) {
            for (x, &v) in r.iter_mut().zip(block) {
                *x += v;
            }
        }
        let head = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        return a[whole..].iter().fold(head, |s, &v| s + v);
    }
    let half = n / 2 - (n / 2) % 8;
    pairwise_sum(&a[..half]) + pairwise_sum(&a[half..])
}

/// Per row, the squared norm and the squared error of its
/// per-channel int8 encoding (FP32 arithmetic, float64 sums).
fn row_stats(w: &[f16], cols: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut norm, mut err) = (Vec::new(), Vec::new());
    for row in w.chunks_exact(cols) {
        let values: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
        let max = values.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = f16::from_f32(max / 127.0).to_f32();
        let wide: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
        let e: Vec<f64> = values
            .iter()
            .map(|&v| {
                let q = if s == 0.0 {
                    0.0
                } else {
                    (v / s).round_ties_even().clamp(-127.0, 127.0)
                };
                f64::from(v - q * s)
            })
            .collect();
        norm.push(einsum_squares(&wide));
        err.push(einsum_squares(&e));
    }
    (norm, err)
}

/// For a `[rows, cols]` down weight: per column the squared
/// norm (column sums in 16 row blocks, each block summed row by row, the
/// blocks added in order) and the mean squared rounding error per-row int8
/// steps add to any column.
fn column_stats(w: &[f16], cols: usize) -> (Vec<f64>, f64) {
    let rows = w.len() / cols;
    let mut norm = vec![0f64; cols];
    let mut steps = Vec::with_capacity(rows);
    // np.array_split(w, 16): the first rows % 16 blocks take one more row.
    let (base, extra) = (rows / 16, rows % 16);
    let mut start = 0;
    for b in 0..16 {
        let end = start + base + usize::from(b < extra);
        let mut block = vec![0f64; cols];
        for row in w[start * cols..end * cols].chunks_exact(cols) {
            for (s, v) in block.iter_mut().zip(row) {
                let v = f64::from(v.to_f32());
                *s = v.mul_add(v, *s);
            }
            let max = row.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
            steps.push(f64::from(max) / 127.0);
        }
        for (n, s) in norm.iter_mut().zip(&block) {
            *n += s;
        }
        start = end;
    }
    let squares: Vec<f64> = steps.iter().map(|s| s * s).collect();
    (norm, pairwise_sum(&squares) / 12.0)
}

/// FP16 values of a whole Q4 group-64 weight.
fn dense_all(q: &Q4) -> Result<Vec<f16>> {
    let (rows, cols) = (q.rows(), q.weight.shape()[1] * 8);
    dense(q, (0, rows), (0, cols))
}

/// The neuron split of one MLP: (GPU, Neural Engine
/// int8, Neural Engine FP16) intermediate indices, each ascending. Neurons are
/// ranked by the error their int8 encoding would add,
/// `|d|^2 (|u|^2 e(g) + |g|^2 e(u)) + |g|^2 |u|^2 e(d)`, with a stable sort.
pub struct NeuronSplit {
    pub gpu: Vec<usize>,
    pub int8: Vec<usize>,
    pub fp16: Vec<usize>,
}

/// The three dense weights of one MLP, decoded once for the split and the
/// program: gate and up `[INTERMEDIATE, HIDDEN]`, down `[HIDDEN, INTERMEDIATE]`.
pub struct DenseMlp {
    gate: Vec<f16>,
    up: Vec<f16>,
    down: Vec<f16>,
}

impl DenseMlp {
    pub fn new(mlp: &Mlp) -> Result<Self> {
        if mlp.gate.rows() != INTERMEDIATE || mlp.down.rows() != HIDDEN {
            return Err(Error::Contract(
                "Clef GPU+ANE needs the 4096 x 12288 MLP geometry",
            ));
        }
        Ok(Self {
            gate: dense_all(&mlp.gate)?,
            up: dense_all(&mlp.up)?,
            down: dense_all(&mlp.down)?,
        })
    }

    pub fn split(&self) -> NeuronSplit {
        let (ng, eg) = row_stats(&self.gate, HIDDEN);
        let (nu, eu) = row_stats(&self.up, HIDDEN);
        let (nd, ed) = column_stats(&self.down, INTERMEDIATE);
        let score: Vec<f64> = (0..INTERMEDIATE)
            .map(|j| nd[j] * (nu[j] * eg[j] + ng[j] * eu[j]) + ng[j] * nu[j] * ed)
            .collect();
        let mut order: Vec<usize> = (0..INTERMEDIATE).collect();
        order.sort_by(|&a, &b| score[a].total_cmp(&score[b]));
        let sorted = |s: &[usize]| {
            let mut v = s.to_vec();
            v.sort_unstable();
            v
        };
        NeuronSplit {
            int8: sorted(&order[..NEURON_INT8]),
            fp16: sorted(&order[NEURON_INT8..NEURON_INT8 + NEURON_FP16]),
            gpu: sorted(&order[NEURON_INT8 + NEURON_FP16..]),
        }
    }

    /// Rows `neurons` of a `[INTERMEDIATE, HIDDEN]` weight.
    fn rows(w: &[f16], neurons: &[usize]) -> Vec<f16> {
        neurons
            .iter()
            .flat_map(|&n| w[n * HIDDEN..(n + 1) * HIDDEN].iter().copied())
            .collect()
    }

    /// Columns `neurons` of the down weight, `[HIDDEN, neurons.len()]`.
    fn down_columns(&self, neurons: &[usize]) -> Vec<f16> {
        self.down
            .chunks_exact(INTERMEDIATE)
            .flat_map(|row| neurons.iter().map(move |&n| row[n]))
            .collect()
    }

    /// The Neural Engine program for this MLP's int8 and FP16 neurons, as
    /// coremltools 9.0 writes it: the FP16 gate conv and the following `* 0.5`
    /// fused into one conv with a zero bias.
    pub fn program(&self, split: &NeuronSplit) -> Result<Package> {
        let (n, nf) = (split.int8.len(), split.fp16.len());
        let (qg, sg) = int8_per_row(&Self::rows(&self.gate, &split.int8), HIDDEN);
        let (qu, su) = int8_per_row(&Self::rows(&self.up, &split.int8), HIDDEN);
        let fg = Self::rows(&self.gate, &split.fp16);
        let fu = Self::rows(&self.up, &split.fp16);
        let fd = self.down_columns(&split.fp16);
        let wd = self.down_columns(&split.int8);
        let parts = n.div_ceil(DOWN_PART);
        let width = n / parts;
        let down: Vec<(Vec<i8>, Vec<f16>)> = (0..parts)
            .map(|p| int8_per_row(&columns(&wd, n, p * width, width), width))
            .collect();
        // Bound the dot products of the int8 and FP16 weights the Neural Engine uses.
        let mut row_sums = vec![0f64; HIDDEN];
        for (qd, sd) in &down {
            for (r, sum) in row_sums.iter_mut().enumerate() {
                let abs: Vec<f64> = qd[r * width..(r + 1) * width]
                    .iter()
                    .map(|&v| f64::from(v).abs())
                    .collect();
                *sum += f64::from(sd[r].to_f32()) * pairwise_sum(&abs);
            }
        }
        for (r, sum) in row_sums.iter_mut().enumerate() {
            let abs: Vec<f64> = fd[r * nf..(r + 1) * nf]
                .iter()
                .map(|v| f64::from(v.to_f32()).abs())
                .collect();
            *sum += pairwise_sum(&abs);
        }
        let max_row_sum = row_sums.iter().fold(1f64, |m, &v| m.max(v));
        let factor_cap = 2f64.powf(
            FACTOR_SCALE
                .min((DOWN_ABS_SUM_LIMIT / max_row_sum).sqrt())
                .log2()
                .floor(),
        );
        if factor_cap < 1.0 {
            return Err(Error::Contract(
                "Clef GPU+ANE down weights need factor scales outside the FP16 range",
            ));
        }

        let half = Operand::Scalar(f16::from_f32(0.5));
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, HIDDEN, 1, NEURON_TILE]);
        let wg = b.constexpr_blockwise_shift_scale(&qg, &[n, HIDDEN, 1, 1], &sg, &[n, 1, 1, 1])?;
        let g = b.conv(&x, &wg)?;
        let wu = b.constexpr_blockwise_shift_scale(&qu, &[n, HIDDEN, 1, 1], &su, &[n, 1, 1, 1])?;
        let u = b.conv(&x, &wu)?;
        // swish(gf)'s `* 0.5` is the program's third mul (after swish(g)'s two).
        let gf_half =
            b.conv_fp16_scaled(&x, &fg, &[nf, HIDDEN, 1, 1], f16::from_f32(0.5), "mul_2")?;
        let uf = b.conv_fp16(&x, &fu, &[nf, HIDDEN, 1, 1])?;
        let swish = |b: &mut ProgramBuilder, half: &Var| -> Result<Var> {
            let t = b.tanh(half);
            let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
            b.mul(half, Operand::Var(&t1), None)
        };
        let g_half = b.mul(&g, half, None)?;
        let s = swish(&mut b, &g_half)?;
        let sf = swish(&mut b, &gf_half)?;
        let peak = |b: &mut ProgramBuilder, v: &Var, w: &Var| -> Result<Var> {
            let va = b.abs(v);
            let vm = b.reduce_max(&va, &[1], true)?;
            let wa = b.abs(w);
            let wm = b.reduce_max(&wa, &[1], true)?;
            b.maximum(&vm, Operand::Var(&wm), None)
        };
        let s_max = peak(&mut b, &s, &sf)?;
        let u_max = peak(&mut b, &u, &uf)?;
        let floor = Operand::Scalar(f16::from_f64(1.0 / FACTOR_SCALE));
        let sm = b.mul(
            &s_max,
            Operand::Scalar(f16::from_f64(1.0 / factor_cap)),
            None,
        )?;
        let s_scale = b.maximum(&sm, floor, Some("gate_scale"))?;
        let um = b.mul(
            &u_max,
            Operand::Scalar(f16::from_f64(1.0 / factor_cap)),
            None,
        )?;
        let u_scale = b.maximum(
            &um,
            Operand::Scalar(f16::from_f64(1.0 / FACTOR_SCALE)),
            Some("up_scale"),
        )?;
        // Normalize before the product so a large SwiGLU value cannot overflow first.
        let normalized = |b: &mut ProgramBuilder, s: &Var, u: &Var| -> Result<Var> {
            let sn = b.real_div(s, Operand::Var(&s_scale), None)?;
            let un = b.real_div(u, Operand::Var(&u_scale), None)?;
            b.mul(&sn, Operand::Var(&un), None)
        };
        let a = normalized(&mut b, &s, &u)?;
        let af = normalized(&mut b, &sf, &uf)?;
        let xs = if parts > 1 {
            b.split(&a, &vec![width; parts], 1)?
        } else {
            vec![a]
        };
        let mut ys = Vec::with_capacity(parts + 1);
        for (xi, (qd, sd)) in xs.iter().zip(&down) {
            let wd = b.constexpr_blockwise_shift_scale(
                qd,
                &[HIDDEN, width, 1, 1],
                sd,
                &[HIDDEN, 1, 1, 1],
            )?;
            ys.push(b.conv(xi, &wd)?);
        }
        ys.push(b.conv_fp16(&af, &fd, &[HIDDEN, nf, 1, 1])?);
        let mut y = ys[0].clone();
        for z in &ys[1..] {
            y = b.add(&y, Operand::Var(z), None)?;
        }
        let y = b.identity(&y, Some("y"));
        b.finish(&[&y, &s_scale, &u_scale])
    }
}

/// The GPU share of `mlp`: gate and up rows and down columns
/// `[0, gpu_columns)`, as contiguous Q4 copies. The down packed layout needs
/// `gpu_columns` to be a multiple of 64.
pub fn gpu_share_with_columns(mlp: &Mlp, gpu_columns: usize) -> Result<Mlp> {
    if gpu_columns >= INTERMEDIATE || gpu_columns % 64 != 0 {
        return Err(Error::Contract(
            "Clef GPU+ANE columns need a 64-multiple below 12288",
        ));
    }
    let rows = |q: &Q4| -> Result<Q4> {
        Ok(Q4 {
            weight: q.weight.slice_axis(0, 0, gpu_columns)?.contiguous()?,
            scales: q.scales.slice_axis(0, 0, gpu_columns)?.contiguous()?,
            biases: q.biases.slice_axis(0, 0, gpu_columns)?.contiguous()?,
        })
    };
    let d = &mlp.down;
    let share = Mlp {
        gate: rows(&mlp.gate)?,
        up: rows(&mlp.up)?,
        down: Q4 {
            weight: d.weight.slice_axis(1, 0, gpu_columns / 8)?.contiguous()?,
            scales: d.scales.slice_axis(1, 0, gpu_columns / 64)?.contiguous()?,
            biases: d.biases.slice_axis(1, 0, gpu_columns / 64)?.contiguous()?,
        },
    };
    let s = share.gate.weight.stream();
    s.eval(&[
        share.gate.weight.clone(),
        share.gate.scales.clone(),
        share.gate.biases.clone(),
        share.up.weight.clone(),
        share.up.scales.clone(),
        share.up.biases.clone(),
        share.down.weight.clone(),
        share.down.scales.clone(),
        share.down.biases.clone(),
    ])?;
    Ok(share)
}

/// The GPU share of `mlp` at the qualified [`GPU_COLUMNS`] depth.
pub fn gpu_share(mlp: &Mlp) -> Result<Mlp> {
    gpu_share_with_columns(mlp, GPU_COLUMNS)
}

/// Compiled-program cache: `APXINF_COREML_CACHE`, else
/// `~/Library/Caches/apxinf/coreml`. Programs compile once per macOS build.
pub fn cache_dir() -> std::path::PathBuf {
    std::env::var_os("APXINF_COREML_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join("Library/Caches/apxinf/coreml")
        })
}

/// The little-endian bytes of FP16 values, without a copy (Apple Silicon only).
fn f16_bytes(values: &[f16]) -> &[u8] {
    // SAFETY: f16 is a plain 2-byte value without padding, so its storage is valid as bytes; the
    // `coreml` feature targets little-endian aarch64 macOS only.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

thread_local! {
    /// Wall-clock stamp of the last traced column-split predict return on
    /// this thread: the Neural Engine service's idle-gap clock. Deliberately
    /// not a field of [`AneMlp`]: the gap that costs (G3: >=30 ms idle roughly
    /// doubles the next predict) is between any two predicts of the *service*,
    /// whichever program ran. A per-layer clock would only ever see the
    /// inter-request interval (it hides the cross-layer gap the wavefront
    /// exists to close). `Model` is `!Send`, so per thread is per owner.
    static LAST_PREDICT_END: std::cell::Cell<Option<std::time::Instant>> =
        const { std::cell::Cell::new(None) };
}

/// `APXINF_CLEF_ANE_TRACE` enables the per-predict trace line.
fn trace_enabled() -> bool {
    std::env::var_os("APXINF_CLEF_ANE_TRACE").is_some()
}

/// The ms since the previous traced predict returned on this thread (`None`
/// on the first call). Advances nothing; the caller stamps the return.
fn trace_gap(started: std::time::Instant) -> Option<f64> {
    LAST_PREDICT_END.with(|cell| {
        cell.get()
            .map(|last| started.saturating_duration_since(last).as_secs_f64() * 1000.0)
    })
}

/// Stamps a predict return on the service clock without a trace line (`warm`
/// has no layer index, but it is ANE activity and must advance the clock).
fn trace_stamp() {
    LAST_PREDICT_END.with(|cell| cell.set(Some(std::time::Instant::now())));
}

/// Retries for the transient `MLComputePlan` races documented on
/// [`AneMlp::package`]. Deliberately bounded: a genuine admission bug
/// (`not ANE-preferred`, bad ports) is not in the transient set and fails on
/// attempt 1, so a wrong candidate still fails loudly.
const PLACEMENT_RETRIES: usize = 3;

/// `load_with_placement` performs compile/cache-hit → canonicalise → plan
/// inspect → `Model::load` → admission, in that order. A transient failure
/// therefore always carries an inspection-step marker; an error before the
/// inspect (compile, path) or at a non-transient admission predicate is not
/// retried. The markers are also the observed `MLComputePlan` races under ANE
/// service load (`usage = None` is reported by admission as
/// `placement is unknown`).
fn is_transient_placement_error(error: &Error) -> bool {
    let message = error.to_string();
    message.contains("placement is unknown")
        || message.contains("plan wait timed out")
        || message.contains("cannot inspect the Core ML compute plan")
}

/// Attempts per layer (1 disables the retry). Env-gated so a demo or another
/// campaign can force the original single-shot behaviour.
fn placement_attempts() -> usize {
    match std::env::var("APXINF_CLEF_PLACEMENT_RETRIES").ok().as_deref() {
        None | Some("") => PLACEMENT_RETRIES,
        Some(value) => value.parse::<usize>().unwrap_or(PLACEMENT_RETRIES).clamp(1, 6),
    }
}

fn placement_retry_delay(attempt: usize) -> std::time::Duration {
    // 200ms then 400ms: enough for the planner's busy window, short enough
    // that a full 32-layer load is not noticeably delayed.
    std::time::Duration::from_millis(200 << attempt.min(2))
}

/// One layer's loaded Neural Engine program.
pub struct AneMlp {
    model: Model,
    tile: usize,
}

impl AneMlp {
    /// Loads the Neural Engine program for `mlp`'s columns
    /// `[gpu_columns, 12288)`, quantizing and compiling it on a cache miss.
    pub fn load_split(cache: &ProgramCache, mlp: &Mlp, gpu_columns: usize) -> Result<Self> {
        Self::load_package(cache, &program_with_columns(mlp, gpu_columns)?)
    }

    pub fn load(cache: &ProgramCache, mlp: &Mlp) -> Result<Self> {
        Self::load_package(cache, &program(mlp)?)
    }

    pub fn load_package(cache: &ProgramCache, package: &Package) -> Result<Self> {
        let input = package
            .model
            .description
            .as_ref()
            .and_then(|description| description.input.first())
            .and_then(|input| input.r#type.as_ref())
            .and_then(|kind| kind.multi_array_type.as_ref())
            .ok_or(Error::Contract("Clef Core ML input profile is absent"))?;
        let tile = if input.shape == [1, HIDDEN as i64, 1, TILE as i64] {
            TILE
        } else if input.shape == [1, HIDDEN as i64, 1, NEURON_TILE as i64] {
            NEURON_TILE
        } else {
            return Err(Error::Contract(
                "Clef Core ML input needs its fixed column or neuron profile",
            ));
        };
        let admission = super::ane_admission::Admission::new(package, [1, HIDDEN, 1, tile])?;
        let units = ComputeUnits::CpuAndNeuralEngine;
        let timeout = std::time::Duration::from_secs(60);
        // A placement-inspect race under ANE service load is transient (the
        // same compiled directory loads cleanly on a retry); a bad
        // placement/port/admission is not and must fail on the first attempt.
        let attempts = placement_attempts();
        let mut result = Err(Error::Other(
            "Core ML placement retry did not run".into(),
        ));
        for attempt in 0..attempts {
            let checked = cache.load_with_placement(package, units, timeout, |facts| {
                admission.check(facts)
            });
            match checked {
                Ok(model) => { result = Ok(model); break; }
                Err(error) => {
                    let transient = attempt + 1 < attempts
                        && is_transient_placement_error(&error);
                    result = Err(error);
                    if !transient { break; }
                }
            }
            std::thread::sleep(placement_retry_delay(attempt));
        }
        Ok(Self {
            model: result?,
            tile,
        })
    }

    pub fn warm(&self) -> Result<()> {
        let shape = [1, HIDDEN, 1, self.tile];
        let scale_shape = [1, 1, 1, self.tile];
        let outputs: [(&str, &[usize]); 3] = [
            ("y", &shape),
            ("gate_scale", &scale_shape),
            ("up_scale", &scale_shape),
        ];
        let input = vec![f16::ZERO; HIDDEN * self.tile];
        if self.tile == NEURON_TILE {
            let batch =
                vec![vec![input.as_slice()]; super::config::MAX_INPUT_TOKENS.div_ceil(self.tile)];
            drop(
                self.model
                    .predict_batch(&[("x", &shape)], &batch, &outputs)?,
            );
        } else {
            drop(self.model.predict(&[("x", &input, &shape)], &outputs)?);
        }
        if trace_enabled() {
            // Warm-up ANE activity closes the clock's first open interval; it
            // is not a trace row (no layer), but the first layer row after it
            // should not read as a multi-second gap.
            trace_stamp();
        }
        Ok(())
    }

    /// Neuron split: one Core ML batch prediction over the FP16 channel-first
    /// tiles `[tiles, 4096, NEURON_TILE]` (evaluated here). Returns the
    /// program's `y` as `[tiles, 4096, NEURON_TILE]` and both scales as
    /// `[tiles * NEURON_TILE]`, FP16 on `stream`, tile after tile.
    pub fn predict_tiles(&self, stream: &Stream, packed: &Array) -> Result<[Array; 3]> {
        let [tiles, HIDDEN, NEURON_TILE] = *packed.shape() else {
            return Err(Error::Contract(
                "Clef GPU+ANE tiles must be [tiles, 4096, 128]",
            ));
        };
        // Wait for the tiles only: the GPU share queued after them keeps running.
        let bytes = packed.to_bytes_ready()?;
        // SAFETY: every bit pattern is a valid f16; `align_to` returns only the aligned middle.
        let (head, aligned, tail) = unsafe { bytes.align_to::<f16>() };
        let copied: Vec<f16>;
        let values: &[f16] = if head.is_empty() && tail.is_empty() {
            aligned
        } else {
            copied = bytes
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();
            &copied
        };
        let batch: Vec<Vec<&[f16]>> = values
            .chunks_exact(HIDDEN * NEURON_TILE)
            .map(|tile| vec![tile])
            .collect();
        let results = self.model.predict_batch(
            &[("x", &[1, HIDDEN, 1, NEURON_TILE])],
            &batch,
            &[
                ("y", &[1, HIDDEN, 1, NEURON_TILE]),
                ("gate_scale", &[1, 1, 1, NEURON_TILE]),
                ("up_scale", &[1, 1, 1, NEURON_TILE]),
            ],
        )?;
        let stacked = |k: usize, shape: &[usize]| -> Result<Array> {
            let mut bytes = Vec::with_capacity(shape.iter().product::<usize>() * 2);
            for r in &results {
                bytes.extend_from_slice(f16_bytes(&r[k]));
            }
            Array::from_bytes(stream, shape, MlxDType::F16, &bytes)
        };
        Ok([
            stacked(0, &[tiles, HIDDEN, NEURON_TILE])?,
            stacked(1, &[tiles * NEURON_TILE])?,
            stacked(2, &[tiles * NEURON_TILE])?,
        ])
    }

    /// BF16 `[1, T, 4096]` lazily as the little-endian bytes of FP16
    /// channel-first tiles `[tiles, 4096, 512]`, zero-padded to whole tiles.
    /// The caller may queue other GPU work between this and [`host`](Self::host).
    pub fn pack(h: &Array) -> Result<Array> {
        let [1, tokens, channels] = *h.shape() else {
            return Err(Error::Contract("Clef GPU+ANE tiles take [1, T, 4096]"));
        };
        let tiles = tokens.div_ceil(TILE);
        let padded = if tiles * TILE > tokens {
            let zeros = Array::zeros(h.stream(), &[1, tiles * TILE - tokens, channels], h.dtype())?;
            Array::concat(&[h, &zeros], 1)?
        } else {
            h.clone()
        };
        padded
            .reshape(&[tiles, TILE, channels])?
            .transpose(&[0, 2, 1])?
            .cast(MlxDType::F16)?
            .contiguous()
    }

    /// The host bytes of [`packed`](Self::pack), row-contiguous as
    /// `to_bytes_ready` requires. Waits only for this array; work queued after
    /// the pack (for example the GPU MLP share) keeps running.
    pub fn host(packed: &Array) -> Result<Vec<u8>> {
        packed.to_bytes_ready()
    }

    /// Waits only for [`packed`](Self::pack) — work queued after it (the GPU
    /// share) keeps running — and returns its bytes.
    fn wait_tiles(packed: &Array) -> Result<Vec<u8>> {
        packed.to_bytes_ready()
    }

    /// Predicts every tile of [`packed`](Self::pack)'s bytes and restores BF16
    /// `[1, T, 4096]` on `stream`: rows times both scales in FP32. The host
    /// read of `packed` happens here, under the same timer as the ANE
    /// predictions, so the trace's `wait_ms` covers the GPU/ANE overlap
    /// window rather than a separate, untimed wait in the caller.
    ///
    /// With `APXINF_CLEF_ANE_TRACE=1` this prints one stderr line per call:
    /// `APXINF_CLEF_ANE layer=<L> tiles=<n> gap_ms=<ms since the previous
    /// predict's return, any program of the service> wait_ms=<ms to read the
    /// packed tiles from the GPU (the GPU/ANE overlap window)>
    /// predict_ms=<sum of the host-blocking Core ML predicts>`.
    /// `index` is the program's layer.
    pub fn predict(
        &self,
        stream: &Stream,
        packed: &Array,
        tokens: usize,
        index: usize,
    ) -> Result<Array> {
        let trace = trace_enabled();
        let chrome = super::trace_json::enabled();
        let started = std::time::Instant::now();
        // The gap is the ANE service's idle interval before this call, from
        // the previous predict's return on any program of the service (not
        // this layer's); first call reports 0.
        let since_last = if trace { trace_gap(started) } else { None };
        let chrome_wait = if chrome {
            super::trace_json::to_us(started)
        } else {
            None
        };
        let tiles = Self::wait_tiles(packed)?;
        let wait_ms = started.elapsed().as_secs_f64() * 1000.0;
        // SAFETY: every bit pattern is a valid f16; `align_to` returns only the aligned middle.
        let (head, aligned, tail) = unsafe { tiles.align_to::<f16>() };
        let copied: Vec<f16>;
        let values: &[f16] = if head.is_empty() && tail.is_empty() {
            aligned
        } else {
            copied = tiles
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect();
            &copied
        };
        if values.is_empty() || values.len() % (HIDDEN * TILE) != 0 {
            return Err(Error::Contract(
                "Clef GPU+ANE tiles must be whole [4096, 512] tiles",
            ));
        }
        let mut parts = Vec::with_capacity(values.len() / (HIDDEN * TILE));
        let mut predict_ms = 0.0;
        let chrome_predict_ts = if chrome {
            super::trace_json::to_us(std::time::Instant::now())
        } else {
            None
        };
        for (i, tile) in values.chunks_exact(HIDDEN * TILE).enumerate() {
            let rows = TILE.min(tokens - i * TILE);
            let predict_started = std::time::Instant::now();
            let out = self.model.predict(
                &[("x", tile, &[1, HIDDEN, 1, TILE])],
                &[
                    ("y", &[1, HIDDEN, 1, TILE]),
                    ("gate_scale", &[1, 1, 1, TILE]),
                    ("up_scale", &[1, 1, 1, TILE]),
                ],
            )?;
            predict_ms += predict_started.elapsed().as_secs_f64() * 1000.0;
            let scale = |v: &[f16]| -> Result<Array> {
                Array::from_bytes(stream, &[TILE, 1], MlxDType::F16, f16_bytes(v))?
                    .slice_axis(0, 0, rows)?
                    .cast(MlxDType::F32)
            };
            let y = Array::from_bytes(stream, &[HIDDEN, TILE], MlxDType::F16, f16_bytes(&out[0]))?
                .transpose(&[1, 0])?
                .slice_axis(0, 0, rows)?
                .cast(MlxDType::F32)?;
            parts.push(
                y.mul(&scale(&out[1])?)?
                    .mul(&scale(&out[2])?)?
                    .reshape(&[1, rows, HIDDEN])?
                    .cast(MlxDType::BF16)?,
            );
        }
        let result = Array::concat(&parts.iter().collect::<Vec<_>>(), 1)?;
        if let (Some(ts0), Some(ts_p)) = (chrome_wait, chrome_predict_ts) {
            // `wait0`-style pack-read overlap window (the GPU/ANE overlap).
            super::trace_json::record(
                "ane_wait_pack",
                "ane",
                &format!("L{index}"),
                ts0,
                wait_ms * 1000.0,
                serde_json::json!({"layer": index, "wait_ms": wait_ms}),
            );
            super::trace_json::record(
                "ane_predict",
                "ane",
                &format!("L{index}"),
                ts_p,
                predict_ms * 1000.0,
                // Metadata only — `tiles` is the raw packed weight buffer
                // (~8.8MB); serializing it into args produced ~13MB events
                // (1GB / 7.78GB traces, D31). The tile count and byte length
                // carry the same geometry for the timeline.
                serde_json::json!({
                    "layer": index,
                    "tiles": values.len() / (HIDDEN * TILE),
                    "tile_bytes": tiles.len(),
                    "predict_ms": predict_ms,
                }),
            );
            if let Some(gap) = since_last {
                super::trace_json::instant(
                    "ane_gap",
                    "ane",
                    "service",
                    ts0,
                    serde_json::json!({"layer": index, "gap_ms": gap}),
                );
            }
        }
        if trace {
            // The gap is measured from the previous predict's return on the
            // service clock; stamp only now, after this call has finished.
            trace_stamp();
            eprintln!(
                "APXINF_CLEF_ANE layer={index} tiles={} gap_ms={:.3} wait_ms={wait_ms:.3} predict_ms={predict_ms:.3}",
                values.len() / (HIDDEN * TILE),
                since_last.unwrap_or(0.0),
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ane_split_geometry_matches_the_campaign_depths() -> Result<()> {
        assert_eq!(ane_split_geometry(6656)?, (5632, 2, 2816));
        assert_eq!(ane_split_geometry(5632)?, (6656, 2, 3328));
        assert_eq!(ane_split_geometry(4608)?, (7680, 2, 3840));
        // Non-64-multiple columns would leave a packed down column partial.
        assert!(ane_split_geometry(6657).is_err());
        // GPU takes nothing: the Neural Engine runs all 12288 columns (three parts).
        assert_eq!(ane_split_geometry(0)?, (12288, 3, 4096));
        // The Neural Engine must keep at least one column.
        assert!(ane_split_geometry(INTERMEDIATE).is_err());
        assert!(ane_split_geometry(INTERMEDIATE + 64).is_err());
        Ok(())
    }

    /// Serializes the two `gpu_columns` probes: `APXINF_CLEF_ANE_COLUMNS` is
    /// process-global, and cargo runs tests in parallel within one binary.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn gpu_columns_defaults_to_6656_and_reads_the_campaign_depths() -> Result<()> {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var_os("APXINF_CLEF_ANE_COLUMNS");
        std::env::remove_var("APXINF_CLEF_ANE_COLUMNS");
        assert_eq!(gpu_columns()?, GPU_COLUMNS, "unset must default to 6656");
        for (value, expected) in [("6656", 6656), ("5632", 5632), ("4608", 4608)] {
            std::env::set_var("APXINF_CLEF_ANE_COLUMNS", value);
            assert_eq!(gpu_columns()?, expected, "value {value}");
        }
        match prior {
            Some(value) => std::env::set_var("APXINF_CLEF_ANE_COLUMNS", value),
            None => std::env::remove_var("APXINF_CLEF_ANE_COLUMNS"),
        }
        Ok(())
    }

    #[test]
    fn gpu_columns_rejects_unqualified_values() -> Result<()> {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var_os("APXINF_CLEF_ANE_COLUMNS");
        for value in ["0", "1", "64", "6657", "12288", "12352", "-1", "abc", "6656.0"] {
            std::env::set_var("APXINF_CLEF_ANE_COLUMNS", value);
            assert!(gpu_columns().is_err(), "accepted unqualified columns {value}");
        }
        match prior {
            Some(value) => std::env::set_var("APXINF_CLEF_ANE_COLUMNS", value),
            None => std::env::remove_var("APXINF_CLEF_ANE_COLUMNS"),
        }
        Ok(())
    }

    #[test]
    fn placement_retries_only_the_transient_inspect_races() {
        for message in [
            "invalid operator or storage input: Clef runtime placement is unknown",
            "Core ML plan wait timed out; framework work is not cancelled",
            "cannot inspect the Core ML compute plan: internal failure",
        ] {
            assert!(
                is_transient_placement_error(&Error::Other(message.into())),
                "expected transient: {message}",
            );
        }
        for message in [
            "Core ML path contains a NUL byte",
            "invalid operator or storage input: runtime operation is not ANE-preferred and supported",
            "duplicate compiled input port",
        ] {
            assert!(
                !is_transient_placement_error(&Error::Other(message.into())),
                "expected permanent: {message}",
            );
        }
    }

    #[test]
    fn placement_retry_attempts_default_to_three_and_clamp_invalid_env() -> Result<()> {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var_os("APXINF_CLEF_PLACEMENT_RETRIES");
        std::env::remove_var("APXINF_CLEF_PLACEMENT_RETRIES");
        assert_eq!(placement_attempts(), PLACEMENT_RETRIES);
        std::env::set_var("APXINF_CLEF_PLACEMENT_RETRIES", "");
        assert_eq!(placement_attempts(), PLACEMENT_RETRIES);
        std::env::set_var("APXINF_CLEF_PLACEMENT_RETRIES", "1");
        assert_eq!(placement_attempts(), 1);
        std::env::set_var("APXINF_CLEF_PLACEMENT_RETRIES", "abc");
        assert_eq!(placement_attempts(), PLACEMENT_RETRIES);
        std::env::set_var("APXINF_CLEF_PLACEMENT_RETRIES", "99");
        assert_eq!(placement_attempts(), 6);
        std::env::set_var("APXINF_CLEF_PLACEMENT_RETRIES", "0");
        assert_eq!(placement_attempts(), 1);
        match prior {
            Some(value) => std::env::set_var("APXINF_CLEF_PLACEMENT_RETRIES", value),
            None => std::env::remove_var("APXINF_CLEF_PLACEMENT_RETRIES"),
        }
        Ok(())
    }

    #[test]
    fn non_clef_package_is_rejected_before_compilation() -> Result<()> {
        let mut builder = ProgramBuilder::new();
        let input = builder.input("x", &[1, 8, 1, 16]);
        let output = builder.identity(&input, Some("y"));
        let package = builder.finish(&[&output])?;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../devlocal/clef-coreml-admission-tests")
            .join(format!("{}-{nonce}", std::process::id()));
        let cache = ProgramCache::new(&root);
        let result = AneMlp::load_package(&cache, &package);
        let rejected = result.is_err();
        drop(result);
        let compiled = root.exists();
        if compiled {
            std::fs::remove_dir_all(root)?;
        }
        assert!(rejected, "non-Clef input/output geometry was admitted");
        assert!(!compiled, "invalid profile reached Core ML compilation");
        Ok(())
    }

    #[test]
    fn int8_rows_match_the_python_rounding() {
        let w: Vec<f16> = [0.5f32, -1.27, 0.635, 0.0, 0.0, 0.0, 0.0, 0.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let (q, s) = int8_per_row(&w, 4);
        let scale = f16::from_f32(f16::from_f32(1.27).to_f32() / 127.0);
        assert_eq!(s, [scale, f16::ZERO]);
        let expect = |v: f32| (f16::from_f32(v).to_f32() / scale.to_f32()).round_ties_even() as i8;
        assert_eq!(&q[..4], &[expect(0.5), -127, expect(0.635), 0]);
        assert_eq!(&q[4..], &[0; 4]);
    }

    /// Every layer's program from the Q4 checkpoint matches a package that a
    /// coremltools 9.0 reference build compiled (in the directory
    /// `recorded_env` names): the same weight file bytes and the same decoded
    /// program (writer metadata and `buildInfo` excluded).
    fn layer_programs_match_recorded(
        recorded_env: &str,
        split: &str,
        build: impl Fn(&Mlp) -> Package,
    ) {
        use sha2::{Digest, Sha256};
        use std::collections::HashMap;
        let var = |n: &str| std::env::var(n).unwrap_or_else(|_| panic!("set {n}"));
        let checkpoint = std::path::PathBuf::from(var("APXINF_CLEF_CHECKPOINT"));
        let read = |n: &str| std::fs::read_to_string(checkpoint.join(n)).unwrap();
        let config =
            super::super::Config::from_json(&read("config.json"), &read("joint_head_config.json"))
                .unwrap();
        let stream = Stream::metal(0).unwrap();
        let backbone =
            super::super::weights::Backbone::load(&config, &stream, &checkpoint).unwrap();
        let mut recorded = HashMap::new();
        for entry in std::fs::read_dir(var(recorded_env)).unwrap() {
            let dir = entry.unwrap().path().join("Data/com.apple.CoreML");
            if let Ok(weights) = std::fs::read(dir.join("weights/weight.bin")) {
                recorded.insert(Sha256::digest(&weights).to_vec(), dir);
            }
        }
        let comparable = |mut m: apxinf_coreml::proto::Model| {
            m.description.as_mut().unwrap().metadata = None;
            m.ml_program.as_mut().unwrap().attributes.clear();
            m
        };
        let mut matched = 0;
        for (i, layer) in backbone.layers.iter().enumerate() {
            let package = build(&layer.mlp);
            let Some(dir) = recorded.get(&Sha256::digest(&package.weights).to_vec()) else {
                println!("layer {i}: no recorded package has this weight file");
                continue;
            };
            let theirs: apxinf_coreml::proto::Model =
                prost::Message::decode(&std::fs::read(dir.join("model.mlmodel")).unwrap()[..])
                    .unwrap();
            let same = comparable(theirs) == comparable(package.model.clone());
            println!(
                "layer {i}: weight file matches {}; program {}",
                dir.display(),
                if same { "matches" } else { "DIFFERS" }
            );
            matched += usize::from(same);
        }
        assert_eq!(
            matched,
            backbone.layers.len(),
            "layers matching a recorded {split} package"
        );
    }

    /// The column split's programs (`APXINF_CLEF_COLUMNS_RECORDED`).
    #[test]
    #[ignore = "requires the Clef checkpoint, recorded packages and a Metal device"]
    fn column_programs_match_the_recorded_packages() {
        layer_programs_match_recorded("APXINF_CLEF_COLUMNS_RECORDED", "column", |mlp| {
            program(mlp).unwrap()
        });
    }

    /// The neuron split's programs (`APXINF_CLEF_NEURONS_RECORDED`). A weight
    /// file that matches no package also means the neuron split differs.
    #[test]
    #[ignore = "requires the Clef checkpoint, recorded packages and a Metal device"]
    fn neuron_programs_match_the_recorded_packages() {
        layer_programs_match_recorded("APXINF_CLEF_NEURONS_RECORDED", "neuron", |mlp| {
            let dense = DenseMlp::new(mlp).unwrap();
            dense.program(&dense.split()).unwrap()
        });
    }

    #[test]
    fn columns_take_a_block_of_every_row() {
        let w: Vec<f16> = (0..12).map(|v| f16::from_f32(v as f32)).collect();
        let c: Vec<f32> = columns(&w, 4, 1, 2).iter().map(|v| v.to_f32()).collect();
        assert_eq!(c, [1., 2., 5., 6., 9., 10.]);
    }

    /// `pack`'s lazy graph: whole and partial tiles have the exact tile bytes
    /// `host` yields, in row-contiguous FP16 channel-first order.
    #[test]
    fn pack_tiles_are_whole_contiguous_and_w1_bit_equal() -> Result<()> {
        let stream = Stream::cpu()?;
        let cases = [
            (1, 4_194_304usize),
            (127, 4_194_304),
            (128, 4_194_304),
            (129, 4_194_304),
            (512, 4_194_304),
            (513, 8_388_608),
        ];
        for (tokens, bytes) in cases {
            let h = Array::from_f32(
                &stream,
                &[1, tokens, HIDDEN],
                &(0..tokens * HIDDEN)
                    .map(|index| index as f32)
                    .collect::<Vec<_>>(),
            )?
            .cast(MlxDType::BF16)?;
            let packed = AneMlp::pack(&h)?;
            // `cast` is elementwise: on a CPU stream the result keeps
            // `[1, tokens, 4096]` with the transpose realized only by the
            // following `contiguous`. Materializing bytes realizes it either
            // way; the byte layout and length are the contract.
            assert_eq!(packed.dtype(), MlxDType::F16);
            assert_eq!(packed.numel(), tokens.div_ceil(TILE) * HIDDEN * TILE);
            let eager = AneMlp::pack(&h)?.to_bytes()?;
            assert_eq!(AneMlp::host(&packed)?, eager, "tokens={tokens}");
            assert_eq!(AneMlp::host(&packed)?.len(), bytes, "tokens={tokens}");
            // W1 schedules `async_eval(pack); async_eval(gpu); host(pack)`:
            // the ready wait must not wait for a separately queued array.
            let other = AneMlp::pack(&h)?.mul(&AneMlp::pack(&h)?)?;
            stream.async_eval(std::slice::from_ref(&other))?;
            assert_eq!(AneMlp::host(&packed)?, eager, "after async, tokens={tokens}");
        }
        Ok(())
    }
}
