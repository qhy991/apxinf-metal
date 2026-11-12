//! aneops: which MIL op classes the compiler places on the Neural Engine, and
//! at what shape.
//!
//! Each case in a fixed table is authored with `ProgramBuilder`, compiled for
//! `CpuAndNeuralEngine`, and loaded. The compiler's own `MLComputePlan` then
//! gives every operation's preferred device, and the case is timed with its
//! declared inputs (median of 20 predictions after 5 warmups). A case the
//! builder cannot author, or the compiler refuses, is a result and is printed
//! like any other. A case with a check also reports how far its first
//! prediction is from an f64 reference.
//!
//!   cargo run -q -p apxinf-coreml --release --features native --example aneops [case ...]
//!
//! Naming cases runs only those, plus the cases they require.
use apxinf_core::{Error, Result};
use apxinf_coreml::program::GeluMode;
use apxinf_coreml::{ComputeDeviceKind, ComputeUnits, Operand, Package, ProgramBuilder, ProgramCache, Var};
use half::f16;
use std::time::{Duration, Instant};

const WARMUP: usize = 5;
const TIMED: usize = 20;

/// An authored case: the package and the values it is fed and read through.
struct Program {
    package: Package,
    inputs: Vec<Var>,
    outputs: Vec<Var>,
    /// Where the authored program differs from the case as specified.
    note: &'static str,
    check: Check,
    /// Per input, a factor the host folds into it before the program sees it.
    /// The check is given the values before the fold. Empty means none.
    input_scales: Vec<f32>,
}

/// What a case's first output is compared with, in f64.
enum Check {
    None,
    /// `f` of the first input, element by element, with the inputs spread over
    /// `[-span, span)` so the function leaves its linear range.
    Elementwise { span: f32, f: Box<dyn Fn(f64) -> f64> },
    /// `(flat output index, expected value)` at a sample of the first output,
    /// computed from the inputs, which are spread over `[-span, span)`.
    Sampled {
        span: f32,
        f: Box<dyn Fn(&[Vec<f16>]) -> Vec<(usize, f64)>>,
    },
}

struct Case {
    name: &'static str,
    shape: String,
    author: Box<dyn Fn() -> Result<Program>>,
    /// Cases that must place entirely on the engine before this one is worth building.
    requires: &'static [&'static str],
}

enum Outcome {
    /// A required case did not place entirely on the engine.
    Blocked(Vec<String>),
    /// The builder refused to author the case.
    Unsupported(String),
    /// Compile, load or prediction failed.
    Refused(String),
    Measured(Measurement),
}

struct Measurement {
    on_ane: Vec<String>,
    off_ane: Vec<String>,
    no_device: usize,
    ms: f64,
    ms_min: f64,
    ms_max: f64,
    finite: bool,
    nonzero: bool,
    note: &'static str,
    deviation: Option<Deviation>,
}

struct Deviation {
    rel_rmse: f64,
    max_abs: f64,
}

/// Over `(got, want)` pairs.
fn deviation(pairs: impl Iterator<Item = (f64, f64)>) -> Deviation {
    let (mut err2, mut ref2, mut max_abs) = (0.0, 0.0, 0.0f64);
    for (g, w) in pairs {
        let d = g - w;
        err2 += d * d;
        ref2 += w * w;
        max_abs = max_abs.max(d.abs());
    }
    Deviation {
        rel_rmse: (err2 / ref2).sqrt(),
        max_abs,
    }
}

fn values(n: usize, seed: usize) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32((((i + seed) * 40503) % 200) as f32 / 100.0 - 1.0))
        .collect()
}

fn weights(n: usize) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32((((i * 2654435761) % 2000) as f32 / 1000.0 - 1.0) * 0.02))
        .collect()
}

/// Uniform on `[-amplitude, amplitude)` from splitmix64. Unlike `weights`, which
/// repeats every 2000 values, these carry no structure for a long sum to cancel.
fn random_weights(n: usize, seed: u64, amplitude: f32) -> Vec<f16> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let unit = ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32;
            f16::from_f32((2.0 * unit - 1.0) * amplitude)
        })
        .collect()
}

fn finish(b: ProgramBuilder, inputs: Vec<Var>, outputs: &[&Var], note: &'static str) -> Result<Program> {
    Ok(Program {
        package: b.finish(outputs)?,
        inputs,
        outputs: outputs.iter().map(|&v| v.clone()).collect(),
        note,
        check: Check::None,
        input_scales: Vec::new(),
    })
}

fn gemm(tile: usize) -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 2048, 1, tile]);
    let y = b.conv_fp16(&x, &weights(6144 * 2048), &[6144, 2048, 1, 1])?;
    finish(b, vec![x], &[&y], "")
}

fn conv2d_3x3() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 128, 64, 64]);
    let y = b.conv_fp16(&x, &weights(128 * 128 * 9), &[128, 128, 3, 3])?;
    finish(
        b,
        vec![x],
        &[&y],
        "pad_type valid, output [1,128,62,62]: conv_fp16 emits only valid padding, so same is not authorable",
    )
}

fn rmsnorm() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 2048, 1, 1408]);
    let gamma = b.input("gamma", &[1, 2048, 1, 1]);
    let sq = b.mul(&x, Operand::Var(&x), None)?;
    let sum = b.reduce_sum(&sq, &[1], true)?;
    let r = b.rsqrt(&sum);
    let normed = b.mul(&x, Operand::Var(&r), None)?;
    let y = b.mul(&normed, Operand::Var(&gamma), None)?;
    finish(
        b,
        vec![x, gamma],
        &[&y],
        "gamma is an input [1,2048,1,1], not a constant: the builder has no public fp16 constant, and [2048,1,1,1] would broadcast x to [2048,2048,1,1408]; no eps, 1/C folds into gamma",
    )
}

/// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
fn tanh_gelu(b: &mut ProgramBuilder, x: &Var) -> Result<Var> {
    let s = |v: f32| Operand::Scalar(f16::from_f32(v));
    let x2 = b.mul(x, Operand::Var(x), None)?;
    let x3 = b.mul(&x2, Operand::Var(x), None)?;
    let cubic = b.mul(&x3, s(0.044715), None)?;
    let inner = b.add(x, Operand::Var(&cubic), None)?;
    let scaled = b.mul(&inner, s((2.0 / std::f32::consts::PI).sqrt()), None)?;
    let t = b.tanh(&scaled);
    let one_plus = b.add(&t, s(1.0), None)?;
    let half = b.mul(x, s(0.5), None)?;
    b.mul(&half, Operand::Var(&one_plus), None)
}

fn gelu() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 6144, 1, 1408]);
    let y = tanh_gelu(&mut b, &x)?;
    finish(b, vec![x], &[&y], "")
}

fn gemm_plus_gelu() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 2048, 1, 1408]);
    let h = b.conv_fp16(&x, &weights(6144 * 2048), &[6144, 2048, 1, 1])?;
    let y = tanh_gelu(&mut b, &h)?;
    finish(b, vec![x], &[&y], "")
}

fn softmax_op() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 1, 128, 1408]);
    let y = b.softmax(&x, 3)?;
    finish(b, vec![x], &[&y], "")
}

fn conv_dynamic_weight() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 256, 1, 1408]);
    let w = b.input("w", &[128, 256, 1, 1]);
    let y = b.conv(&x, &w)?;
    finish(b, vec![x, w], &[&y], "")
}

const HEADS: usize = 8;
const SEQ: usize = 1408;
const HEAD_DIM: usize = 128;

/// Per head: `S = conv(Q_h, K_h)` is `[1, keys, 1, queries]`, softmax over the
/// key (channel) axis, then `O_h = conv(P, V_h^T)` is `[1, head_dim, 1, queries]`.
/// Gated on `softmax_in_block`, the softmax at this block's score size; the
/// `[1,1,128,1408]` softmax it used to wait on is below the placement threshold.
fn attention_core_conv() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let q = b.input("q", &[1, HEADS * HEAD_DIM, 1, SEQ]);
    let k = b.input("k", &[HEADS * SEQ, HEAD_DIM, 1, 1]);
    let v = b.input("v", &[HEADS * HEAD_DIM, SEQ, 1, 1]);
    let qs = b.split(&q, &[HEAD_DIM; HEADS], 1)?;
    let ks = b.split(&k, &[SEQ; HEADS], 0)?;
    let vs = b.split(&v, &[HEAD_DIM; HEADS], 0)?;
    let mut heads = Vec::with_capacity(HEADS);
    for h in 0..HEADS {
        let scores = b.conv(&qs[h], &ks[h])?;
        let p = b.softmax(&scores, 1)?;
        heads.push(b.conv(&p, &vs[h])?);
    }
    let y = b.concat(&heads.iter().collect::<Vec<_>>(), 1, false)?;
    finish(
        b,
        vec![q, k, v],
        &[&y],
        "one conv pair per head: the builder's conv has groups=1, so 8 heads cannot share one grouped conv; q/k/v arrive in conv layout and are split per head, output [1,1024,1,1408]; no 1/sqrt(d) scale",
    )
}

fn reshape_matmul() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 8, 1408, 256]);
    let value = b.input("value", &[8, 256, 1408]);
    let r = b.reshape(&x, &[8, 1408, 256])?;
    let y = b.matmul(&r, &value)?;
    finish(b, vec![x, value], &[&y], "")
}

fn softmax_last_axis(b: &mut ProgramBuilder, x: &Var) -> Result<Var> {
    let e = b.exp(x);
    let sum = b.reduce_sum(&e, &[x.shape().len() as i32 - 1], true)?;
    b.real_div(&e, Operand::Var(&sum), None)
}

fn softmax() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 1, 128, 1408]);
    let y = softmax_last_axis(&mut b, &x)?;
    finish(b, vec![x], &[&y], "")
}

/// `softmax_last_axis` of `x - max(x)`: the same value, but `exp` cannot
/// overflow FP16, which it does above 11.09.
fn stable_softmax_last_axis(b: &mut ProgramBuilder, x: &Var) -> Result<Var> {
    let max = b.reduce_max(x, &[x.shape().len() as i32 - 1], true)?;
    let neg = b.mul(&max, Operand::Scalar(f16::from_f32(-1.0)), None)?;
    let shifted = b.add(x, Operand::Var(&neg), None)?;
    softmax_last_axis(b, &shifted)
}

/// Unscaled attention with the softmax spelled as elementwise ops, so each one
/// is a separate pass over the `[1,8,1408,1408]` scores.
fn attention_core() -> Result<Program> {
    let (heads, rows, d) = (8, 1408, 256);
    let mut b = ProgramBuilder::new();
    let scores = b.input("scores", &[1, heads, rows, d]);
    let value = b.input("value", &[1, heads, d, rows]);
    let out_value = b.input("out_value", &[1, heads, rows, d]);
    let s = b.matmul(&scores, &value)?;
    let p = stable_softmax_last_axis(&mut b, &s)?;
    let y = b.matmul(&p, &out_value)?;
    Ok(Program {
        // Without the max subtraction, `exp` of these unscaled scores overflowed.
        check: attention_check(heads, rows, rows, d, 1.0, KeyLayout::Transposed),
        ..finish(
            b,
            vec![scores, value, out_value],
            &[&y],
            "no 1/sqrt(d) scale; softmax is reduce_max, mul -1, add, exp, reduce_sum, real_div",
        )?
    })
}

fn transpose_head() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, 8, 1408, 256]);
    let y = b.transpose(&x, &[0, 1, 3, 2])?;
    finish(b, vec![x], &[&y], "")
}

fn concat_heads() -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let heads: Vec<Var> = (0..4)
        .map(|i| b.input(&format!("h{i}"), &[1, 8, 1408, 64]))
        .collect();
    let y = b.concat(&heads.iter().collect::<Vec<_>>(), 3, false)?;
    finish(b, heads, &[&y], "")
}

/// `values` repeats every 200 elements, so a 128-wide row has 25 distinct scores.
/// At span 4 the top one wins with or without `1/sqrt(d)`, and a block with no
/// scale passed the check at 1.2e-4. At span 1 a missing scale moves the output
/// by 0.5 relative, and the scores are still peaked enough not to average out.
const ATTENTION_SPAN: f32 = 1.0;

/// Query rows of each head that the f64 oracle recomputes.
fn sampled_rows(rows: usize) -> impl Iterator<Item = usize> {
    (0..rows).step_by(rows.div_ceil(4))
}

/// How the `k` input of an attention case is laid out per head.
#[derive(Clone, Copy)]
enum KeyLayout {
    /// `[keys, d]`, row-major.
    Rows,
    /// `[d, keys]`, already `k^T`.
    Transposed,
}

/// `softmax(scale q k^T) v` at four query rows per head, for row-major
/// `q[1, heads, rows, d]` and `v[1, heads, keys, d]`, inputs on `[-ATTENTION_SPAN, ATTENTION_SPAN)`.
fn attention_check(heads: usize, rows: usize, keys: usize, d: usize, scale: f64, layout: KeyLayout) -> Check {
    Check::Sampled {
        span: ATTENTION_SPAN,
        f: Box::new(move |inputs: &[Vec<f16>]| {
            let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
            let key = |h: usize, j: usize, c: usize| match layout {
                KeyLayout::Rows => k[(h * keys + j) * d + c],
                KeyLayout::Transposed => k[(h * d + c) * keys + j],
            };
            let mut expected = Vec::new();
            for h in 0..heads {
                for i in sampled_rows(rows) {
                    let qi = &q[(h * rows + i) * d..][..d];
                    let scores: Vec<f64> = (0..keys)
                        .map(|j| {
                            qi.iter().enumerate().map(|(c, a)| a.to_f64() * key(h, j, c).to_f64()).sum::<f64>() * scale
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for c in 0..d {
                        let o: f64 = (0..keys).map(|j| weights[j] * v[(h * keys + j) * d + c].to_f64()).sum();
                        expected.push(((h * rows + i) * d + c, o / total));
                    }
                }
            }
            expected
        }),
    }
}

/// Where an attention block applies `1/sqrt(d)`.
#[derive(Clone, Copy)]
enum ScoreScale {
    /// A scalar `mul` over the scores, inside the block.
    Separate,
    /// Multiplied into `q` on the host, so the block has no `mul`.
    Folded,
    /// Nowhere. Numerically wrong; it only prices the `mul`.
    Omitted,
}

/// Decomposed attention as one program: `QK^T` and `AV` as batched matmuls and
/// the native `softmax` over the keys, with the scale where `scale` puts it.
/// The check is always the correctly scaled attention of `q` before any fold.
fn attention_block_matmul(heads: usize, rows: usize, keys: usize, d: usize, scale: ScoreScale) -> Result<Program> {
    let factor = 1.0 / (d as f32).sqrt();
    let mut b = ProgramBuilder::new();
    let q = b.input("q", &[1, heads, rows, d]);
    let k = b.input("k", &[1, heads, keys, d]);
    let v = b.input("v", &[1, heads, keys, d]);
    let kt = b.transpose(&k, &[0, 1, 3, 2])?;
    let mut scores = b.matmul(&q, &kt)?;
    if let ScoreScale::Separate = scale {
        scores = b.mul(&scores, Operand::Scalar(f16::from_f32(factor)), None)?;
    }
    let p = b.softmax(&scores, -1)?;
    let y = b.matmul(&p, &v)?;
    let (note, input_scales) = match scale {
        ScoreScale::Separate => ("k^T is a transpose op: the builder's matmul always emits transpose_y=false", vec![]),
        ScoreScale::Folded => ("q is multiplied by 1/sqrt(d) on the host and rounded to FP16 before the program", vec![factor]),
        ScoreScale::Omitted => ("NUMERICALLY WRONG, a cost floor only: no 1/sqrt(d) anywhere; same program as the folded case", vec![]),
    };
    Ok(Program {
        check: attention_check(heads, rows, keys, d, f64::from(factor), KeyLayout::Rows),
        input_scales,
        ..finish(b, vec![q, k, v], &[&y], note)?
    })
}

fn attention_sdpa_op(heads: usize, seq: usize, d: usize) -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let q = b.input("q", &[1, heads, seq, d]);
    let k = b.input("k", &[1, heads, seq, d]);
    let v = b.input("v", &[1, heads, seq, d]);
    let y = b.sdpa(&q, &k, &v, None)?;
    Ok(Program {
        check: attention_check(heads, seq, seq, d, 1.0 / (d as f64).sqrt(), KeyLayout::Rows),
        ..finish(
            b,
            vec![q, k, v],
            &[&y],
            "no scale or causal input exists in the CoreML8 opset; the op applies 1/sqrt(d) itself",
        )?
    })
}

fn softmax_in_block() -> Result<Program> {
    let (heads, rows, keys) = (HEADS, SEQ, SEQ);
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, heads, rows, keys]);
    let y = b.softmax(&x, -1)?;
    let check = Check::Sampled {
        span: ATTENTION_SPAN,
        f: Box::new(move |inputs: &[Vec<f16>]| {
            let mut expected = Vec::new();
            for h in 0..heads {
                for i in sampled_rows(rows) {
                    let at = (h * rows + i) * keys;
                    let row: Vec<f64> = inputs[0][at..][..keys].iter().map(|v| v.to_f64()).collect();
                    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let total: f64 = row.iter().map(|s| (s - max).exp()).sum();
                    expected.extend(row.iter().enumerate().map(|(j, s)| (at + j, (s - max).exp() / total)));
                }
            }
            expected
        }),
    };
    Ok(Program {
        check,
        ..finish(b, vec![x], &[&y], "")?
    })
}

const MLP_HIDDEN: usize = 2048;
const MLP_WIDTH: usize = 6144;
const MLP_TILE: usize = 1408;
const K_CHUNK: usize = 1024;

fn silu_ref(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

fn tanh_gelu_ref(v: f64) -> f64 {
    0.5 * v * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
}

/// One elementwise program body: a single MIL op, or one spelling of an activation.
#[derive(Clone, Copy)]
enum Elementwise {
    Identity,
    /// `x * 0.5`, the scalar form the SiLU chain uses.
    Mul,
    /// `x + 1`, the scalar form the SiLU chain uses.
    Add,
    /// `x + x`, which the compiler cannot fold into a neighbour.
    AddSelf,
    Tanh,
    Sigmoid,
    Silu,
    /// `clip(x, -1, 1)`.
    Clamp,
    GeluTanh,
    /// Jev's shipped `(0.5 g) (1 + tanh(0.5 g))`.
    SiluMulAddTanh,
    /// `g * sigmoid(g)`.
    SiluSigmoid,
    /// `h (1 + tanh(h))` of `h = g / 2`, which is `silu(g)` exactly. The halving
    /// is folded into the gate weights, exact in FP16 above the subnormal range.
    SiluHalfTanh,
}

impl Elementwise {
    /// What the body's input is, as a multiple of the gate it stands for.
    fn gate_scale(self) -> f32 {
        match self {
            Self::SiluHalfTanh => 0.5,
            _ => 1.0,
        }
    }

    /// The body's output as a function of its input.
    fn reference(self) -> fn(f64) -> f64 {
        match self {
            Self::Identity => |v| v,
            Self::Mul => |v| v * 0.5,
            Self::Add => |v| v + 1.0,
            Self::AddSelf => |v| 2.0 * v,
            Self::Tanh => f64::tanh,
            Self::Sigmoid => |v| 1.0 / (1.0 + (-v).exp()),
            Self::Clamp => |v| v.clamp(-1.0, 1.0),
            Self::GeluTanh => tanh_gelu_ref,
            Self::Silu | Self::SiluMulAddTanh | Self::SiluSigmoid => silu_ref,
            Self::SiluHalfTanh => |h| silu_ref(2.0 * h),
        }
    }

    fn apply(self, b: &mut ProgramBuilder, x: &Var) -> Result<Var> {
        let s = |v: f32| Operand::Scalar(f16::from_f32(v));
        Ok(match self {
            Self::Identity => b.identity(x, None),
            Self::Mul => b.mul(x, s(0.5), None)?,
            Self::Add => b.add(x, s(1.0), None)?,
            Self::AddSelf => b.add(x, Operand::Var(x), None)?,
            Self::Tanh => b.tanh(x),
            Self::Sigmoid => b.sigmoid(x),
            Self::Silu => b.silu(x),
            Self::Clamp => b.clamp(x, f16::from_f32(-1.0), f16::from_f32(1.0))?,
            Self::GeluTanh => b.gelu(x, GeluMode::TanhApproximation),
            Self::SiluMulAddTanh => {
                let scaled = b.mul(x, s(0.5), None)?;
                let half = b.mul(x, s(0.5), None)?;
                let t = b.tanh(&half);
                let one_plus = b.add(&t, s(1.0), None)?;
                b.mul(&scaled, Operand::Var(&one_plus), None)?
            }
            Self::SiluSigmoid => {
                let sig = b.sigmoid(x);
                b.mul(x, Operand::Var(&sig), None)?
            }
            Self::SiluHalfTanh => {
                let t = b.tanh(x);
                let one_plus = b.add(&t, s(1.0), None)?;
                b.mul(x, Operand::Var(&one_plus), None)?
            }
        })
    }
}

/// `body` at the MLP's gate shape, checked against its f64 reference. With
/// `anchored`, it reads `AddSelf` of the input instead, which keeps ops the
/// compiler runs on the CPU when alone on the engine; `op_add_self` is the floor.
fn elementwise(body: Elementwise, anchored: bool) -> Result<Program> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, MLP_WIDTH, 1, MLP_TILE]);
    let input = match anchored {
        true => Elementwise::AddSelf.apply(&mut b, &x)?,
        false => x.clone(),
    };
    let y = body.apply(&mut b, &input)?;
    let note = match body {
        Elementwise::SiluHalfTanh => "input is h = g/2, as the MLP folds the halving into Wg; checked against silu(2h)",
        _ => "",
    };
    let f = body.reference();
    let lead = if anchored { 2.0 } else { 1.0 };
    Ok(Program {
        check: Check::Elementwise { span: 8.0, f: Box::new(move |v| f(lead * v)) },
        ..finish(b, vec![x], &[&y], note)?
    })
}

/// `conv(x, w)` for a row-major `[n, k]` weight with K split into `K_CHUNK`
/// parts whose partial convs are summed in order, as Jev's `chunked_conv`.
fn split_k_conv(b: &mut ProgramBuilder, x: &Var, w: &[f16], n: usize, k: usize) -> Result<Var> {
    let parts = b.split(x, &vec![K_CHUNK; k / K_CHUNK], 1)?;
    let mut acc: Option<Var> = None;
    for (i, part) in parts.iter().enumerate() {
        let chunk: Vec<f16> = w
            .chunks_exact(k)
            .flat_map(|row| &row[i * K_CHUNK..(i + 1) * K_CHUNK])
            .copied()
            .collect();
        let y = b.conv_fp16(part, &chunk, &[n, K_CHUNK, 1, 1])?;
        acc = Some(match acc {
            None => y,
            Some(a) => b.add(&a, Operand::Var(&y), None)?,
        });
    }
    Ok(acc.expect("k holds at least one chunk"))
}

/// Jev's MLP at the wavefront tile: `conv(body(conv(x, Wg)) * conv(x, Wu), Wd)`,
/// every matrix split-K at `K_CHUNK`, returned through `identity` named `y`.
fn mlp(body: Elementwise) -> Result<Program> {
    let (h, n) = (MLP_HIDDEN, MLP_WIDTH);
    // With x on [-1, 1), an amplitude of 0.05 puts the gate's standard deviation
    // near 1.3, where SiLU is no longer close to linear.
    let scale = body.gate_scale();
    let wg: Vec<f16> = random_weights(n * h, 1, 0.05)
        .into_iter()
        .map(|v| f16::from_f32(v.to_f32() * scale))
        .collect();
    let wu = random_weights(n * h, 2, 0.05);
    let wd = random_weights(h * n, 3, 0.05);
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, h, 1, MLP_TILE]);
    let gate = split_k_conv(&mut b, &x, &wg, n, h)?;
    let up = split_k_conv(&mut b, &x, &wu, n, h)?;
    let act = body.apply(&mut b, &gate)?;
    let hidden = b.mul(&act, Operand::Var(&up), None)?;
    let down = split_k_conv(&mut b, &hidden, &wd, h, n)?;
    let y = b.identity(&down, Some("y"));
    let f = body.reference();
    let check = Check::Sampled { span: 1.0, f: Box::new(move |inputs: &[Vec<f16>]| {
        let x = &inputs[0];
        let dot = |row: &[f16], v: &[f64]| row.iter().zip(v).map(|(w, v)| w.to_f64() * v).sum::<f64>();
        (0..MLP_TILE)
            .step_by(88)
            .flat_map(|t| {
                let xt: Vec<f64> = (0..h).map(|c| x[c * MLP_TILE + t].to_f64()).collect();
                let hidden: Vec<f64> = wg
                    .chunks_exact(h)
                    .zip(wu.chunks_exact(h))
                    .map(|(g, u)| f(dot(g, &xt)) * dot(u, &xt))
                    .collect();
                wd.chunks_exact(n)
                    .enumerate()
                    .map(|(c, row)| (c * MLP_TILE + t, dot(row, &hidden)))
                    .collect::<Vec<_>>()
            })
            .collect()
    }) };
    let note = match body {
        Elementwise::SiluHalfTanh => "Wg pre-scaled by 0.5",
        Elementwise::GeluTanh => "gelu replaces silu; the oracle computes gelu",
        _ => "",
    };
    Ok(Program {
        check,
        ..finish(b, vec![x], &[&y], note)?
    })
}

fn cases() -> Vec<Case> {
    let case = |name: &'static str, shape: &str, author: fn() -> Result<Program>| Case {
        name,
        shape: shape.to_owned(),
        author: Box::new(author),
        requires: &[],
    };
    let mut cases = vec![Case {
        name: "gemm_baseline",
        shape: "x[1,2048,1,1408] w[6144,2048,1,1]".into(),
        author: Box::new(|| gemm(1408)),
        requires: &[],
    }];
    for tile in [1, 4, 8, 16, 32, 64, 128, 256, 512] {
        cases.push(Case {
            name: "gemm_small_m",
            shape: format!("x[1,2048,1,{tile}] w[6144,2048,1,1]"),
            author: Box::new(move || gemm(tile)),
            requires: &[],
        });
    }
    cases.extend([
        case("conv2d_3x3", "x[1,128,64,64] w[128,128,3,3]", conv2d_3x3),
        case("rmsnorm", "x[1,2048,1,1408]", rmsnorm),
        case("gelu", "x[1,6144,1,1408]", gelu),
        case("softmax_last_axis", "x[1,1,128,1408]", softmax),
        case(
            "attention_core",
            "scores[1,8,1408,256] value[1,8,256,1408] out_value[1,8,1408,256]",
            attention_core,
        ),
        case("transpose_head", "x[1,8,1408,256] perm[0,1,3,2]", transpose_head),
        case("concat_heads", "4 x [1,8,1408,64] axis 3", concat_heads),
        case("softmax_op", "x[1,1,128,1408] axis 3", softmax_op),
        case(
            "gemm_plus_gelu",
            "x[1,2048,1,1408] w[6144,2048,1,1] then tanh gelu",
            gemm_plus_gelu,
        ),
        case(
            "conv_dynamic_weight",
            "x[1,256,1,1408] w[128,256,1,1] (input)",
            conv_dynamic_weight,
        ),
        case(
            "reshape_matmul",
            "x[1,8,1408,256] -> [8,1408,256], matmul with value[8,256,1408]",
            reshape_matmul,
        ),
    ]);
    let program = |name: &'static str, shape: &str, author: Box<dyn Fn() -> Result<Program>>| Case {
        name,
        shape: shape.to_owned(),
        author,
        requires: &[],
    };
    use ScoreScale::{Folded, Omitted, Separate};
    for (name, shape, (heads, rows, keys), scale) in [
        ("attention_block_matmul", "q,k,v[1,8,1408,128]", (HEADS, SEQ, SEQ), Separate),
        ("attention_scale_folded", "q,k,v[1,8,1408,128]", (HEADS, SEQ, SEQ), Folded),
        ("attention_no_scale", "q,k,v[1,8,1408,128]", (HEADS, SEQ, SEQ), Omitted),
        ("attention_block_matmul_wide", "q,k,v[1,32,512,128]", (32, 512, 512), Separate),
        ("attention_scale_folded_wide", "q,k,v[1,32,512,128]", (32, 512, 512), Folded),
        ("attention_decode_shape", "q[1,8,1,128] k,v[1,8,2048,128], scores [1,8,1,2048]", (HEADS, 1, 2048), Separate),
        ("attention_fold_decode", "q[1,8,1,128] k,v[1,8,2048,128], scores [1,8,1,2048]", (HEADS, 1, 2048), Folded),
    ] {
        cases.push(program(
            name,
            shape,
            Box::new(move || attention_block_matmul(heads, rows, keys, HEAD_DIM, scale)),
        ));
    }
    cases.extend([
        program(
            "attention_sdpa_op",
            "q,k,v[1,8,1408,128]",
            Box::new(|| attention_sdpa_op(HEADS, SEQ, HEAD_DIM)),
        ),
        program(
            "attention_sdpa_op_64h",
            "q,k,v[1,64,496,64]",
            Box::new(|| attention_sdpa_op(64, 496, 64)),
        ),
        case("softmax_in_block", "x[1,8,1408,1408] axis -1", softmax_in_block),
        Case {
            requires: &["softmax_in_block", "conv_dynamic_weight"],
            ..case(
                "attention_core_conv",
                "8 heads x seq 1408 x head_dim 128: q[1,1024,1,1408] k[11264,128,1,1] v[1024,1408,1,1]",
                attention_core_conv,
            )
        },
    ]);
    use Elementwise::*;
    for (name, body) in [
        ("op_identity", Identity),
        ("op_mul", Mul),
        ("op_add", Add),
        ("op_add_self", AddSelf),
        ("op_tanh", Tanh),
        ("op_sigmoid", Sigmoid),
        ("op_silu", Silu),
        ("op_clamp", Clamp),
        ("op_gelu", GeluTanh),
        ("act_silu_mul_add_tanh", SiluMulAddTanh),
        ("act_silu_sigmoid", SiluSigmoid),
        ("act_silu_half_tanh", SiluHalfTanh),
    ] {
        cases.push(Case {
            name,
            shape: format!("x[1,{MLP_WIDTH},1,{MLP_TILE}]"),
            author: Box::new(move || elementwise(body, false)),
            requires: &[],
        });
    }
    for (name, body) in [
        ("anchored_tanh", Tanh),
        ("anchored_sigmoid", Sigmoid),
        ("anchored_silu", Silu),
        ("anchored_clamp", Clamp),
        ("anchored_gelu", GeluTanh),
    ] {
        cases.push(Case {
            name,
            shape: format!("x[1,{MLP_WIDTH},1,{MLP_TILE}], op of x + x"),
            author: Box::new(move || elementwise(body, true)),
            requires: &[],
        });
    }
    // The control runs first and again last, so drift across the MLP cases shows.
    for (name, body) in [
        ("mlp_silu_mul_add_tanh", SiluMulAddTanh),
        ("mlp_silu_native", Silu),
        ("mlp_silu_sigmoid", SiluSigmoid),
        ("mlp_silu_half_tanh", SiluHalfTanh),
        ("mlp_gelu_native", GeluTanh),
        ("mlp_control_repeat", SiluMulAddTanh),
    ] {
        cases.push(Case {
            name,
            shape: format!(
                "x[1,{MLP_HIDDEN},1,{MLP_TILE}] Wg,Wu[{MLP_WIDTH},{MLP_HIDDEN}] Wd[{MLP_HIDDEN},{MLP_WIDTH}] split-K {K_CHUNK}"
            ),
            author: Box::new(move || mlp(body)),
            requires: &[],
        });
    }
    cases
}

fn measure(program: &Program, cache: &ProgramCache) -> Result<Measurement> {
    let (mut on_ane, mut off_ane, mut no_device) = (Vec::new(), Vec::new(), 0);
    let model = cache.load_with_placement(
        &program.package,
        ComputeUnits::CpuAndNeuralEngine,
        Duration::from_secs(180),
        |facts| {
            for op in facts.functions.iter().filter(|f| f.name == "main").flat_map(|f| &f.operations) {
                match op.usage.as_ref().and_then(|u| u.preferred.as_ref()) {
                    Some(d) if d.kind == ComputeDeviceKind::NeuralEngine => on_ane.push(op.operator.clone()),
                    Some(d) => off_ane.push(format!("{}:{}", op.operator, d.class_name)),
                    None => no_device += 1,
                }
            }
            Ok(())
        },
    )?;
    let span = match &program.check {
        Check::Elementwise { span, .. } | Check::Sampled { span, .. } => *span,
        Check::None => 1.0,
    };
    let data: Vec<Vec<f16>> = program
        .inputs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            values(v.shape().iter().product(), i * 7919)
                .into_iter()
                .map(|x| f16::from_f32(x.to_f32() * span))
                .collect()
        })
        .collect();
    let fed: Vec<Vec<f16>> = data
        .iter()
        .enumerate()
        .map(|(i, d)| match program.input_scales.get(i) {
            Some(&s) => d.iter().map(|x| f16::from_f32(x.to_f32() * s)).collect(),
            None => d.clone(),
        })
        .collect();
    let inputs: Vec<(&str, &[f16], &[usize])> = program
        .inputs
        .iter()
        .zip(&fed)
        .map(|(v, d)| (v.name(), &d[..], v.shape()))
        .collect();
    let outputs: Vec<(&str, &[usize])> = program.outputs.iter().map(|v| (v.name(), v.shape())).collect();

    let mut got = model.predict(&inputs, &outputs)?;
    let finite = got.iter().flatten().all(|v| v.is_finite());
    let nonzero = got.iter().flatten().any(|v| v.to_f32() != 0.0);
    let first = got.swap_remove(0);
    for _ in 0..WARMUP {
        model.predict(&inputs, &outputs)?;
    }
    let mut ts: Vec<f64> = (0..TIMED)
        .map(|_| {
            let t0 = Instant::now();
            model.predict(&inputs, &outputs).map(|_| t0.elapsed().as_secs_f64() * 1e3)
        })
        .collect::<Result<_>>()?;
    ts.sort_by(f64::total_cmp);
    let deviation = match &program.check {
        Check::None => None,
        Check::Elementwise { f, .. } => Some(deviation(
            first.iter().zip(&data[0]).map(|(g, x)| (g.to_f64(), f(x.to_f64()))),
        )),
        Check::Sampled { f, .. } => Some(deviation(
            f(&data).into_iter().map(|(i, want)| (first[i].to_f64(), want)),
        )),
    };
    Ok(Measurement {
        on_ane,
        off_ane,
        no_device,
        ms: ts[TIMED / 2],
        ms_min: ts[0],
        ms_max: ts[TIMED - 1],
        finite,
        nonzero,
        note: program.note,
        deviation,
    })
}

fn esc(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            c if (c as u32) < 0x20 => vec![' '],
            c => vec![c],
        })
        .collect()
}

fn strings(v: &[String]) -> String {
    let items: Vec<String> = v.iter().map(|s| format!("\"{}\"", esc(s))).collect();
    format!("[{}]", items.join(","))
}

fn sysctl(name: &str) -> String {
    std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

/// The cases named on the command line and the cases they require, or every
/// case when none is named.
fn selected(all: Vec<Case>, names: &[String]) -> Result<Vec<Case>> {
    if let Some(unknown) = names.iter().find(|n| !all.iter().any(|c| c.name == n.as_str())) {
        return Err(Error::Other(format!("no case named {unknown}")));
    }
    let wanted: Vec<&str> = all
        .iter()
        .filter(|c| names.iter().any(|n| n == c.name))
        .flat_map(|c| c.requires.iter().copied().chain([c.name]))
        .collect();
    Ok(all
        .into_iter()
        .filter(|c| names.is_empty() || wanted.contains(&c.name))
        .collect())
}

fn main() -> Result<()> {
    let names: Vec<String> = std::env::args().skip(1).collect();
    let root = std::env::temp_dir().join(format!("aneops-{}", std::process::id()));
    let (mut measured, mut refused, mut unsupported, mut blocked) = (0, 0, 0, 0);
    let mut all_on_ane: Vec<String> = Vec::new();
    let mut placed: Vec<&str> = Vec::new();
    for (i, case) in selected(cases(), &names)?.iter().enumerate() {
        let head = format!("\"case\":\"{}\",\"shape\":\"{}\"", case.name, esc(&case.shape));
        let t0 = Instant::now();
        let missing: Vec<String> = case
            .requires
            .iter()
            .filter(|r| !placed.contains(r))
            .map(|r| r.to_string())
            .collect();
        let outcome = if !missing.is_empty() {
            Outcome::Blocked(missing)
        } else {
            match (case.author)() {
                Err(e) => Outcome::Unsupported(e.to_string()),
                Ok(program) => {
                    let dir = root.join(i.to_string());
                    let _ = std::fs::remove_dir_all(&dir);
                    let result = measure(&program, &ProgramCache::new(&dir));
                    let _ = std::fs::remove_dir_all(&dir);
                    match result {
                        Ok(m) => Outcome::Measured(m),
                        Err(e) => Outcome::Refused(e.to_string()),
                    }
                }
            }
        };
        let build_s = t0.elapsed().as_secs_f64();
        let line = match outcome {
            Outcome::Blocked(by) => {
                blocked += 1;
                format!("{{{head},\"status\":\"blocked\",\"blocked_by\":{}}}", strings(&by))
            }
            Outcome::Unsupported(reason) => {
                unsupported += 1;
                format!(
                    "{{{head},\"status\":\"unsupported\",\"reason\":\"{}\",\"build_s\":{build_s:.2}}}",
                    esc(&reason)
                )
            }
            Outcome::Refused(reason) => {
                refused += 1;
                format!(
                    "{{{head},\"status\":\"refused\",\"refused\":\"{}\",\"build_s\":{build_s:.2}}}",
                    esc(&reason)
                )
            }
            Outcome::Measured(m) => {
                measured += 1;
                let deviation = match &m.deviation {
                    Some(d) => format!("{{\"rel_rmse\":{:.3e},\"max_abs\":{:.3e}}}", d.rel_rmse, d.max_abs),
                    None => "null".into(),
                };
                if m.off_ane.is_empty() && !m.on_ane.is_empty() {
                    all_on_ane.push(format!("{} {}", case.name, case.shape));
                    placed.push(case.name);
                }
                format!(
                    "{{{head},\"status\":\"measured\",\"on_ane\":{},\"off_ane\":{},\"on_ane_ops\":{},\"no_device\":{},\"ms\":{:.4},\"ms_min\":{:.4},\"ms_max\":{:.4},\"build_s\":{build_s:.2},\"finite\":{},\"nonzero\":{},\"deviation\":{deviation},\"note\":\"{}\"}}",
                    m.on_ane.len(),
                    strings(&m.off_ane),
                    strings(&m.on_ane),
                    m.no_device,
                    m.ms,
                    m.ms_min,
                    m.ms_max,
                    m.finite,
                    m.nonzero,
                    esc(m.note)
                )
            }
        };
        println!("{line}");
    }
    let _ = std::fs::remove_dir_all(&root);
    println!(
        "{{\"summary\":\"aneops\",\"chip\":\"{}\",\"os_build\":\"{}\",\"compute_units\":\"CpuAndNeuralEngine\",\"warmup\":{WARMUP},\"timed\":{TIMED},\"cases\":{},\"measured\":{measured},\"refused\":{refused},\"unsupported\":{unsupported},\"blocked\":{blocked},\"all_on_ane\":{}}}",
        esc(&sysctl("machdep.cpu.brand_string")),
        esc(&sysctl("kern.osversion")),
        measured + refused + unsupported + blocked,
        strings(&all_on_ane)
    );
    if measured == 0 {
        return Err(Error::Other("aneops measured no case".into()));
    }
    Ok(())
}
