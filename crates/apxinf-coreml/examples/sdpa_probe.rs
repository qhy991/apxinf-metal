//! Does scaled dot-product attention run on the Neural Engine, and how fast?
//!
//! Phase C of the Qwen-Image attention offload (#242) only pays if the ANE can
//! carry `QK^T -> softmax -> softmax*V` near its conv rate. This probe builds
//! that core alone, as one Core ML program per head count, and reports the
//! measured TMAC/s, the CPU seconds the warm-up burned, and the device Core ML
//! says each operation will run on. Projections are plain convs whose rate is
//! already known, so they stay out.
//!
//! `cargo run -p apxinf-coreml --features native --example sdpa_probe`

use apxinf_coreml::{Var,
    ComputeDeviceKind, ComputeUnits, Operand, Package, ProgramBuilder, ProgramCache,
};
use apxinf_core::Result;
use half::f16;
use std::time::{Duration, Instant};

const TOKENS: usize = 1152;
const HEAD_DIM: usize = 128;
const DIM: usize = 4096;
const WARMUP: usize = 3;
const RUNS: usize = 20;

/// `softmax(q k^T / sqrt(d)) v` over `heads` heads, with q/k/v supplied in the
/// `[head, token, channel]` layout the ANE wants, so no transpose op is needed
/// to isolate what the probe is measuring.
fn sdpa(heads: usize) -> Result<Package> {
    let shape = [heads, TOKENS, HEAD_DIM];
    let mut b = ProgramBuilder::new();
    let q = b.input("q", &shape)?;
    let k = b.input("k", &shape)?;
    let v = b.input("v", &shape)?;
    let scores = b.matmul(&q, &k, true)?;
    let scale = f16::from_f64(1.0 / (HEAD_DIM as f64).sqrt());
    let scores = b.mul(&scores, Operand::Scalar(scale), None)?;
    let weights = b.softmax(&scores, 2)?;
    let out = b.matmul(&weights, &v, false)?;
    let out = b.identity(&out, Some("o"))?;
    b.finish(&[&out])
}

/// The other half of a whole-head offload: an FP16 projection of `cols`
/// output channels out of `DIM`, the shape a q/k/v or `to_out` share has. The
/// weights are constants, which is the case the Neural Engine is built for, so
/// this is the rate the non-sdpa 88% of attention would run at.
fn projection(cols: usize) -> Result<Package> {
    let weight = vec![f16::from_f32(0.01); cols * DIM];
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, DIM, 1, TOKENS])?;
    let y = b.conv_fp16(&x, &weight, &[cols, DIM, 1, 1])?;
    let y = b.identity(&y, Some("o"))?;
    b.finish(&[&y])
}

/// The shipped MLP split's shape, in FP16: gate and up over `rows` of the
/// 12288, a SwiGLU-shaped activation, and `out` contracting those rows back to
/// `DIM`. Same MAC count and same op chain as the program `predict` runs, so
/// the difference between this and the measured 28.19 ms/call is the host and
/// Core ML overhead that `dit.rs` could in principle remove.
fn mlp(rows: usize, tokens: usize, int8: bool) -> Result<Package> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, DIM, 1, tokens])?;
    let (g, u, y);
    if int8 {
        // The shipped recipe: INT8 gate/up and INT8 `out`, per-row FP16 scales.
        let q = vec![1i8; rows * DIM];
        let s = vec![f16::from_f32(0.01); rows];
        let wg = b.constexpr_blockwise_shift_scale(&q, &[rows, DIM, 1, 1], &s, &[rows, 1, 1, 1])?;
        let wu = b.constexpr_blockwise_shift_scale(&q, &[rows, DIM, 1, 1], &s, &[rows, 1, 1, 1])?;
        let qo = vec![1i8; DIM * rows];
        let so = vec![f16::from_f32(0.01); DIM];
        let wo = b.constexpr_blockwise_shift_scale(&qo, &[DIM, rows, 1, 1], &so, &[DIM, 1, 1, 1])?;
        g = b.conv(&x, &wg)?;
        u = b.conv(&x, &wu)?;
        let t = b.tanh(&g)?;
        let h = b.mul(&t, Operand::Var(&u), None)?;
        y = b.conv(&h, &wo)?;
    } else {
        let gate = vec![f16::from_f32(0.01); rows * DIM];
        let down = vec![f16::from_f32(0.01); DIM * rows];
        g = b.conv_fp16(&x, &gate, &[rows, DIM, 1, 1])?;
        u = b.conv_fp16(&x, &gate, &[rows, DIM, 1, 1])?;
        let t = b.tanh(&g)?;
        let h = b.mul(&t, Operand::Var(&u), None)?;
        y = b.conv_fp16(&h, &down, &[DIM, rows, 1, 1])?;
    }
    let y = b.identity(&y, Some("o"))?;
    b.finish(&[&y])
}

/// `ane.rs::program_int8` rebuilt operation for operation with synthetic
/// weights of the same shapes: INT8 gate and up, the `r`/`s` scalar inputs,
/// the full SwiGLU elementwise chain, and `out` split into an INT8 head of
/// `OUT_INT8` columns plus an FP16 tail. Only the weight *values* differ from
/// the shipped program, so the gap between this and the measured 28.19 ms/call
/// is host and Core ML overhead, not recipe.
fn shipped(rows: usize, tokens: usize) -> Result<Package> {
    shipped_scaled(rows, tokens, f16::from_f32(0.01))
}

fn shipped_scaled(rows: usize, tokens: usize, fill: f16) -> Result<Package> {
    const OUT_INT8: usize = 2560;
    let tail = rows - OUT_INT8;
    let q = vec![1i8; rows * DIM];
    let s_rows = vec![fill; rows];
    let qo = vec![1i8; DIM * OUT_INT8];
    let so = vec![fill; DIM];
    let high = vec![fill; DIM * tail];

    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, DIM, 1, tokens])?;
    let r = b.input("r", &[1, 1, 1, 1])?;
    let s = b.input("s", &[1, 1, 1, 1])?;
    let wg = b.constexpr_blockwise_shift_scale(&q, &[rows, DIM, 1, 1], &s_rows, &[rows, 1, 1, 1])?;
    let wu = b.constexpr_blockwise_shift_scale(&q, &[rows, DIM, 1, 1], &s_rows, &[rows, 1, 1, 1])?;
    let wl =
        b.constexpr_blockwise_shift_scale(&qo, &[DIM, OUT_INT8, 1, 1], &so, &[DIM, 1, 1, 1])?;
    let g = b.conv(&x, &wg)?;
    let g = b.mul(&g, Operand::Var(&r), None)?;
    let u = b.conv(&x, &wu)?;
    let u = b.mul(&u, Operand::Var(&r), None)?;
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let t = b.tanh(&half)?;
    let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
    let silu = b.mul(&half, Operand::Var(&t1), None)?;
    let m = b.mul(&silu, Operand::Var(&u), None)?;
    let m = b.mul(&m, Operand::Var(&s), None)?;
    let parts = b.split(&m, &[OUT_INT8, tail], 1)?;
    let low = b.conv(&parts[0], &wl)?;
    let high = b.conv_fp16(&parts[1], &high, &[DIM, tail, 1, 1])?;
    let y = b.add(&low, Operand::Var(&high), Some("y"))?;
    b.finish(&[&y])
}

/// The shipped program's shape, timed two ways: one prediction at a time, as
/// `dit.rs` calls it, and `predict_batch`. The difference between them is the
/// per-call cost a batched submission would remove.
fn probe_shipped(cache: &ProgramCache, rows: usize, tokens: usize) -> Result<()> {
    let macs = (3 * tokens * DIM * rows) as f64;
    let model = cache.load(&shipped(rows, tokens)?, ComputeUnits::All)?;
    let x = vec![f16::from_f32(0.01); DIM * tokens];
    let one = [f16::ONE];
    let shape = [1usize, DIM, 1, tokens];
    let scalar = [1usize, 1, 1, 1];
    let feed = [
        ("x", &x[..], &shape[..]),
        ("r", &one[..], &scalar[..]),
        ("s", &one[..], &scalar[..]),
    ];
    let want = [("y", &shape[..])];
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        let out = model.predict(&feed, &want)?;
        samples.push(run.elapsed().as_secs_f64());
        assert!(out[0][0].to_f32().is_finite(), "the shipped probe went non-finite");
    }
    samples.sort_by(f64::total_cmp);
    let single = samples[RUNS / 2];

    let specs = [("x", &shape[..]), ("r", &scalar[..]), ("s", &scalar[..])];
    let item: Vec<&[f16]> = vec![&x, &one, &one];
    let batched = |n: usize| -> Result<f64> {
        let batch = vec![item.clone(); n];
        model.predict_batch(&specs, &batch, &want)?;
        let run = Instant::now();
        model.predict_batch(&specs, &batch, &want)?;
        Ok(run.elapsed().as_secs_f64() / n as f64)
    };
    let (b4, b8) = (batched(4)?, batched(8)?);

    println!(
        "sdpa-probe: shipped rows={rows} tokens={tokens} macs={:.2}G single={:.3}ms \
         rate={:.2}TMAC/s batch4={:.3}ms batch8={:.3}ms",
        macs / 1e9,
        single * 1e3,
        macs / single / 1e12,
        b4 * 1e3,
        b8 * 1e3,
    );
    Ok(())
}

/// The one structural difference left between the probe and `dit.rs`: the
/// denoise loop cycles through 32 *distinct* block programs, so consecutive
/// predictions never share resident weights, while a probe repeating one model
/// keeps them warm. Round-robin over `n` distinct programs isolates that cost.
fn probe_rotation(cache: &ProgramCache, rows: usize, tokens: usize, n: usize) -> Result<()> {
    let models = (0..n)
        .map(|i| {
            // A different scale per program makes each package, and so each
            // compiled model, genuinely distinct.
            let package = shipped_scaled(rows, tokens, f16::from_f32(0.01 + i as f32 * 1e-4))?;
            cache.load(&package, ComputeUnits::All)
        })
        .collect::<Result<Vec<_>>>()?;

    let x = vec![f16::from_f32(0.01); DIM * tokens];
    let one = [f16::ONE];
    let shape = [1usize, DIM, 1, tokens];
    let scalar = [1usize, 1, 1, 1];
    let feed = [
        ("x", &x[..], &shape[..]),
        ("r", &one[..], &scalar[..]),
        ("s", &one[..], &scalar[..]),
    ];
    let want = [("y", &shape[..])];
    for model in &models {
        for _ in 0..WARMUP {
            model.predict(&feed, &want)?;
        }
    }
    let mut samples = Vec::with_capacity(RUNS);
    for i in 0..RUNS {
        let run = Instant::now();
        models[i % n].predict(&feed, &want)?;
        samples.push(run.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "sdpa-probe: rotation programs={n} rows={rows} median={:.3}ms",
        samples[RUNS / 2] * 1e3,
    );
    Ok(())
}

/// Phase C's convolution half at its real weight volume: q, k and v each
/// projecting `heads * HEAD_DIM` channels out of `DIM`, plus `to_out`
/// contracting them back. The sdpa core is timed separately, so the whole-head
/// predict time is this plus that, and the rate here is what the weight-volume
/// curve actually does to four attention projections rather than one.
fn attention_convs(heads: usize) -> Result<Package> {
    let cols = heads * HEAD_DIM;
    let wqkv = vec![f16::from_f32(0.01); cols * DIM];
    let wout = vec![f16::from_f32(0.01); DIM * cols];
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, DIM, 1, TOKENS])?;
    let q = b.conv_fp16(&x, &wqkv, &[cols, DIM, 1, 1])?;
    let k = b.conv_fp16(&x, &wqkv, &[cols, DIM, 1, 1])?;
    let v = b.conv_fp16(&x, &wqkv, &[cols, DIM, 1, 1])?;
    // Stands in for the sdpa core, which is timed on its own: the point here
    // is the four projections' rate at this weight volume.
    let qk = b.mul(&q, Operand::Var(&k), None)?;
    let o = b.mul(&qk, Operand::Var(&v), None)?;
    let y = b.conv_fp16(&o, &wout, &[DIM, cols, 1, 1])?;
    let y = b.identity(&y, Some("y"))?;
    b.finish(&[&y])
}

fn probe_attention_convs(cache: &ProgramCache, heads: usize) -> Result<()> {
    let cols = heads * HEAD_DIM;
    let macs = (4 * TOKENS * DIM * cols) as f64;
    let weights = (4 * DIM * cols * 2) as f64 / 1e6;
    let model = cache.load(&attention_convs(heads)?, ComputeUnits::All)?;
    let x = vec![f16::from_f32(0.01); DIM * TOKENS];
    let shape = [1usize, DIM, 1, TOKENS];
    let feed = [("x", &x[..], &shape[..])];
    let want = [("y", &shape[..])];
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        let out = model.predict(&feed, &want)?;
        samples.push(run.elapsed().as_secs_f64());
        assert!(out[0][0].to_f32().is_finite(), "the attention probe went non-finite");
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[RUNS / 2];
    println!(
        "sdpa-probe: attn-convs heads={heads} weights={weights:.0}MB macs={:.2}G \
         median={:.3}ms rate={:.2}TMAC/s",
        macs / 1e9,
        median * 1e3,
        macs / median / 1e12,
    );
    Ok(())
}

/// Phase C whole and entire: `heads` complete attention heads inside one Core
/// ML program. Takes the normed block input channels-first, projects q/k/v,
/// runs the sdpa core per head, and contracts those heads back through
/// `to_out`, returning the partial `[DIM, TOKENS]` projection for the GPU to
/// add to its own heads' share. The reshape and transpose operations between
/// the convolution layout and the matmul layout are the cost this measurement
/// exists to capture.
fn whole_head(heads: usize) -> Result<Package> {
    let cols = heads * HEAD_DIM;
    let wqkv = vec![f16::from_f32(0.01); cols * DIM];
    let wout = vec![f16::from_f32(0.01); DIM * cols];
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, DIM, 1, TOKENS])?;

    // [1, cols, 1, T] -> [heads, T, HEAD_DIM] for q and v, and
    // [heads, HEAD_DIM, T] for k, which is already QK^T's right operand.
    let mut head_major = |w: &[f16], to_tokens_major: bool| -> Result<Var> {
        let c = b.conv_fp16(&x, w, &[cols, DIM, 1, 1])?;
        let c = b.reshape(&c, &[heads, HEAD_DIM, TOKENS])?;
        match to_tokens_major {
            true => b.transpose(&c, &[0, 2, 1]),
            false => Ok(c),
        }
    };
    let q = head_major(&wqkv, true)?;
    let k = head_major(&wqkv, false)?;
    let v = head_major(&wqkv, true)?;

    let scores = b.matmul(&q, &k, false)?;
    let scale = f16::from_f64(1.0 / (HEAD_DIM as f64).sqrt());
    let scores = b.mul(&scores, Operand::Scalar(scale), None)?;
    let weights = b.softmax(&scores, 2)?;
    let o = b.matmul(&weights, &v, false)?;

    let o = b.transpose(&o, &[0, 2, 1])?;
    let o = b.reshape(&o, &[1, cols, 1, TOKENS])?;
    let y = b.conv_fp16(&o, &wout, &[DIM, cols, 1, 1])?;
    let y = b.identity(&y, Some("y"))?;
    b.finish(&[&y])
}

fn probe_whole_head(cache: &ProgramCache, heads: usize) -> Result<()> {
    let cols = heads * HEAD_DIM;
    let macs = (4 * TOKENS * DIM * cols + 2 * heads * TOKENS * TOKENS * HEAD_DIM) as f64;
    let mut devices = Vec::new();
    let model = cache.load_with_placement(
        &whole_head(heads)?,
        ComputeUnits::All,
        Duration::from_secs(120),
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
    let x = vec![f16::from_f32(0.01); DIM * TOKENS];
    let shape = [1usize, DIM, 1, TOKENS];
    let feed = [("x", &x[..], &shape[..])];
    let want = [("y", &shape[..])];
    let (wall, cpu) = (Instant::now(), cpu_seconds());
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let (warm, warm_cpu) = (wall.elapsed().as_secs_f64(), cpu_seconds() - cpu);
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        let out = model.predict(&feed, &want)?;
        samples.push(run.elapsed().as_secs_f64());
        assert!(out[0][0].to_f32().is_finite(), "the whole-head probe went non-finite");
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[RUNS / 2];
    println!(
        "sdpa-probe: whole-head heads={heads} macs={:.2}G median={:.3}ms rate={:.2}TMAC/s \
         warm_cpu={warm_cpu:.3}s cpu_share={:.0}%",
        macs / 1e9,
        median * 1e3,
        macs / median / 1e12,
        100.0 * warm_cpu / warm,
    );
    println!("sdpa-probe: whole-head heads={heads} placement={}", devices.join(" "));
    Ok(())
}

/// Text-prefix keys and values the Neural Engine's heads attend to, padded to
/// a fixed width because a Core ML program's shapes are fixed and the prompt's
/// length is not. A007's prompts tokenize to 69-83 rows, so 128 holds them
/// with the mask switching the rest off.
const TEXT_PAD: usize = 128;

/// The program Phase C actually needs, as against [`whole_head`]'s core.
///
/// `whole_head` measured `conv -> QK^T -> softmax -> AV -> conv` and nothing
/// else, which is not the block's attention. The block also RMSNorms q and k
/// per head, rotates both by RoPE, and attends over the text prefix
/// concatenated in front of the image tokens. Each of those costs ops and the
/// prefix widens the score matrix, so the honest rate has to be measured on
/// this shape rather than extrapolated from that one (#242).
///
/// RoPE is applied in the even/odd-permuted layout and left there. The score
/// is a dot product over the head dimension, so permuting q and k the same way
/// leaves it unchanged, and that saves interleaving the halves back together.
/// `v` is never rotated, so the output is in the ordinary layout already.
/// `[h, rows, 128]` with the even channels first, then the odd, which is the
/// layout RoPE's two halves want.
fn deinterleave(b: &mut ProgramBuilder, x: &Var, heads: usize, rows: usize) -> Result<Var> {
    let half = HEAD_DIM / 2;
    let pairs = b.reshape(x, &[heads, rows, half, 2])?;
    let split = b.transpose(&pairs, &[0, 1, 3, 2])?;
    b.reshape(&split, &[heads, rows, HEAD_DIM])
}

/// RoPE over the token axis, leaving the result in the permuted layout.
fn rope(b: &mut ProgramBuilder, x: &Var, heads: usize, cos: &Var, sin: &Var) -> Result<Var> {
    let half = HEAD_DIM / 2;
    let permuted = deinterleave(b, x, heads, TOKENS)?;
    let parts = b.split(&permuted, &[half, half], 2)?;
    let (even, odd) = (parts[0].clone(), parts[1].clone());
    let ec = b.mul(&even, Operand::Var(cos), None)?;
    let os = b.mul(&odd, Operand::Var(sin), None)?;
    let neg = b.mul(&os, Operand::Scalar(f16::from_f32(-1.0)), None)?;
    let lo = b.add(&ec, Operand::Var(&neg), None)?;
    let es = b.mul(&even, Operand::Var(sin), None)?;
    let oc = b.mul(&odd, Operand::Var(cos), None)?;
    let hi = b.add(&es, Operand::Var(&oc), None)?;
    b.concat(&[&lo, &hi], 2)
}

/// `[1, cols, 1, T]` from one projection, reshaped to `[head, token, channel]`.
fn project(b: &mut ProgramBuilder, x: &Var, w: &[f16], heads: usize) -> Result<Var> {
    let cols = heads * HEAD_DIM;
    let c = b.conv_fp16(x, w, &[cols, DIM, 1, 1])?;
    let c = b.reshape(&c, &[heads, HEAD_DIM, TOKENS])?;
    b.transpose(&c, &[0, 2, 1])
}

/// `x * rsqrt(mean(x^2) + eps) * weight`, the per-head RMSNorm. The weight is
/// a scalar stand-in: the real one is a `[128]` vector, which costs the same
/// one broadcast multiply.
fn rms(b: &mut ProgramBuilder, x: &Var) -> Result<Var> {
    let sq = b.mul(x, Operand::Var(x), None)?;
    let mean = b.reduce_mean(&sq, &[2], true)?;
    let inv = b.rsqrt(&mean, f16::from_f32(1e-6))?;
    let normed = b.mul(x, Operand::Var(&inv), None)?;
    b.mul(&normed, Operand::Scalar(f16::ONE), None)
}

fn phase_c(heads: usize) -> Result<Package> {
    let cols = heads * HEAD_DIM;
    let half = HEAD_DIM / 2;
    let keys = TEXT_PAD + TOKENS;
    let wqkv = vec![f16::from_f32(0.01); cols * DIM];
    let wout = vec![f16::from_f32(0.01); DIM * cols];
    let mut b = ProgramBuilder::new();

    let x = b.input("x", &[1, DIM, 1, TOKENS])?;
    let kt = b.input("kt", &[heads, TEXT_PAD, HEAD_DIM])?;
    let vt = b.input("vt", &[heads, TEXT_PAD, HEAD_DIM])?;
    let mask = b.input("mask", &[1, 1, keys])?;
    let cos = b.input("cos", &[1, TOKENS, half])?;
    let sin = b.input("sin", &[1, TOKENS, half])?;

    let q = project(&mut b, &x, &wqkv, heads)?;
    let k = project(&mut b, &x, &wqkv, heads)?;
    let v = project(&mut b, &x, &wqkv, heads)?;

    let q = rms(&mut b, &q)?;
    let k = rms(&mut b, &k)?;

    let q = rope(&mut b, &q, heads, &cos, &sin)?;
    let k = rope(&mut b, &k, heads, &cos, &sin)?;

    // The text keys arrive already normed and rotated from the GPU's cache, so
    // they need only the same permutation to line up with `k`.
    let kt = deinterleave(&mut b, &kt, heads, TEXT_PAD)?;
    let k_all = b.concat(&[&kt, &k], 1)?;
    let v_all = b.concat(&[&vt, &v], 1)?;

    let scores = b.matmul(&q, &k_all, true)?;
    let scale = f16::from_f64(1.0 / (HEAD_DIM as f64).sqrt());
    let scores = b.mul(&scores, Operand::Scalar(scale), None)?;
    let scores = b.add(&scores, Operand::Var(&mask), None)?;
    let weights = b.softmax(&scores, 2)?;
    let o = b.matmul(&weights, &v_all, false)?;

    let o = b.transpose(&o, &[0, 2, 1])?;
    let o = b.reshape(&o, &[1, cols, 1, TOKENS])?;
    let y = b.conv_fp16(&o, &wout, &[DIM, cols, 1, 1])?;
    let y = b.identity(&y, Some("y"))?;
    b.finish(&[&y])
}

fn probe_phase_c(cache: &ProgramCache, heads: usize) -> Result<()> {
    let cols = heads * HEAD_DIM;
    let half = HEAD_DIM / 2;
    let keys = TEXT_PAD + TOKENS;
    let macs = (4 * TOKENS * DIM * cols + 2 * heads * TOKENS * keys * HEAD_DIM) as f64;
    let mut devices = Vec::new();
    let model = cache.load_with_placement(
        &phase_c(heads)?,
        ComputeUnits::All,
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

    let x = vec![f16::from_f32(0.01); DIM * TOKENS];
    // Comparable in magnitude to a RoPE-rotated image key, so the softmax
    // actually weighs the prefix; a smaller value puts it 10 logits down and
    // the check below could not see it at FP16 resolution.
    let kt = vec![f16::from_f32(1.4); heads * TEXT_PAD * HEAD_DIM];
    let vt = vec![f16::from_f32(2.0); heads * TEXT_PAD * HEAD_DIM];
    let table = vec![f16::from_f32(0.7); TOKENS * half];
    // The prompt's real rows are admitted and the padding is switched off.
    let text_rows = 80;
    let open: Vec<f16> = (0..keys)
        .map(|i| match i < text_rows || i >= TEXT_PAD {
            true => f16::ZERO,
            false => f16::from_f32(-10000.0),
        })
        .collect();
    let shut: Vec<f16> = (0..keys)
        .map(|i| match i >= TEXT_PAD {
            true => f16::ZERO,
            false => f16::from_f32(-10000.0),
        })
        .collect();

    let xs = [1usize, DIM, 1, TOKENS];
    let kts = [heads, TEXT_PAD, HEAD_DIM];
    let ms = [1usize, 1, keys];
    let ts = [1usize, TOKENS, half];
    let mut feed = [
        ("x", &x[..], &xs[..]),
        ("kt", &kt[..], &kts[..]),
        ("vt", &vt[..], &kts[..]),
        ("mask", &open[..], &ms[..]),
        ("cos", &table[..], &ts[..]),
        ("sin", &table[..], &ts[..]),
    ];
    let want = [("y", &xs[..])];

    // The prefix has to change the answer, or the concat and the mask are not
    // doing anything and the rate below would be measuring a smaller program.
    let with_text = model.predict(&feed, &want)?.remove(0)[0].to_f32();
    feed[3].1 = &shut[..];
    let without = model.predict(&feed, &want)?.remove(0)[0].to_f32();
    feed[3].1 = &open[..];
    assert!(
        with_text.is_finite() && with_text != 0.0,
        "phase-c probe produced {with_text}"
    );
    assert!(
        (with_text - without).abs() > 1e-4 * with_text.abs(),
        "the text prefix changed nothing: {with_text} vs {without}"
    );

    let (wall, cpu) = (Instant::now(), cpu_seconds());
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let (warm, warm_cpu) = (wall.elapsed().as_secs_f64(), cpu_seconds() - cpu);
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        model.predict(&feed, &want)?;
        samples.push(run.elapsed().as_secs_f64());
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[RUNS / 2];
    // Four projections and `to_out`'s rows for these heads, FP16.
    let weight_mb = (4 * cols * DIM * 2) as f64 / (1024.0 * 1024.0);
    println!(
        "sdpa-probe: phase-c heads={heads} macs={:.2}G weights={weight_mb:.1}MiB \
         median={:.3}ms rate={:.2}TMAC/s warm_cpu={warm_cpu:.3}s cpu_share={:.0}% \
         text_delta={:.4}",
        macs / 1e9,
        median * 1e3,
        macs / median / 1e12,
        100.0 * warm_cpu / warm,
        with_text - without,
    );
    println!("sdpa-probe: phase-c heads={heads} placement={}", devices.join(" "));
    Ok(())
}

fn cpu_seconds() -> f64 {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return f64::INFINITY;
    }
    let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
    secs(usage.ru_utime) + secs(usage.ru_stime)
}

fn directory_bytes(path: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return std::fs::metadata(path).map_or(0, |m| m.len());
    };
    entries
        .flatten()
        .map(|e| directory_bytes(&e.path()))
        .sum::<u64>()
}

fn probe_mlp(cache: &ProgramCache, rows: usize, tokens: usize, int8: bool) -> Result<()> {
    let macs = (3 * tokens * DIM * rows) as f64;
    let package = mlp(rows, tokens, int8)?;
    let model = cache.load(&package, ComputeUnits::All)?;
    let x = vec![f16::from_f32(0.01); DIM * tokens];
    let feed = [("x", &x[..], &[1usize, DIM, 1, tokens][..])];
    let want = [("o", &[1usize, DIM, 1, tokens][..])];
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        let out = model.predict(&feed, &want)?;
        samples.push(run.elapsed().as_secs_f64());
        assert!(out[0][0].to_f32().is_finite(), "the MLP probe went non-finite");
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[RUNS / 2];
    println!(
        "sdpa-probe: mlp int8={int8} rows={rows} tokens={tokens} macs={:.2}G median={:.3}ms \
         rate={:.2}TMAC/s",
        macs / 1e9,
        median * 1e3,
        macs / median / 1e12,
    );
    Ok(())
}

fn probe_projection(cache: &ProgramCache, heads: usize) -> Result<()> {
    let cols = heads * HEAD_DIM;
    let macs = (TOKENS * DIM * cols) as f64;
    let package = projection(cols)?;
    let mut devices = Vec::new();
    let model = cache.load_with_placement(
        &package,
        ComputeUnits::All,
        Duration::from_secs(60),
        |facts| {
            devices.extend(facts.functions.iter().flat_map(|f| {
                f.operations
                    .iter()
                    .map(|op| format!("{}:{}", op.operator, device_of(op)))
            }));
            Ok(())
        },
    )?;

    let x = vec![f16::from_f32(0.01); DIM * TOKENS];
    let feed = [("x", &x[..], &[1usize, DIM, 1, TOKENS][..])];
    let want = [("o", &[1usize, cols, 1, TOKENS][..])];
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        let out = model.predict(&feed, &want)?;
        samples.push(run.elapsed().as_secs_f64());
        // Every weight and input is 0.01, so each output channel sums DIM of
        // them: 4096 * 1e-4 = 0.4096.
        let got = out[0][0].to_f32();
        assert!((got - 0.4096).abs() < 0.05, "the projection gave {got}, not 0.4096 +- FP16 drift");
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[RUNS / 2];
    println!(
        "sdpa-probe: projection cols={cols} macs={:.2}G median={:.3}ms rate={:.2}TMAC/s \
         placement={}",
        macs / 1e9,
        median * 1e3,
        macs / median / 1e12,
        devices.join(" "),
    );
    Ok(())
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

fn probe(cache: &ProgramCache, heads: usize) -> Result<()> {
    let macs = 2.0 * heads as f64 * (TOKENS * TOKENS * HEAD_DIM) as f64;
    let built = Instant::now();
    let package = sdpa(heads)?;
    let compiled = cache.compiled(&package)?;
    let compile = built.elapsed().as_secs_f64();

    let mut devices = Vec::new();
    let model = cache.load_with_placement(
        &package,
        ComputeUnits::All,
        Duration::from_secs(60),
        |facts| {
            for function in &facts.functions {
                for op in &function.operations {
                    let on = match op.usage.as_ref().and_then(|u| u.preferred.as_ref()) {
                        Some(device) => match device.kind {
                            ComputeDeviceKind::NeuralEngine => "ANE",
                            ComputeDeviceKind::Gpu => "GPU",
                            ComputeDeviceKind::Cpu => "CPU",
                            ComputeDeviceKind::Unknown => "?",
                        },
                        None => "-",
                    };
                    devices.push(format!("{}:{on}", op.operator));
                }
            }
            Ok(())
        },
    )?;

    let elements = heads * TOKENS * HEAD_DIM;
    let q = vec![f16::from_f32(0.02); elements];
    let k = vec![f16::from_f32(0.03); elements];
    let v = vec![f16::from_f32(1.0); elements];
    let shape = [heads, TOKENS, HEAD_DIM];
    let feed = [("q", &q[..], &shape[..]), ("k", &k[..], &shape[..]), ("v", &v[..], &shape[..])];
    let want = [("o", &shape[..])];

    let (wall, cpu) = (Instant::now(), cpu_seconds());
    for _ in 0..WARMUP {
        model.predict(&feed, &want)?;
    }
    let (warm, warm_cpu) = (wall.elapsed().as_secs_f64(), cpu_seconds() - cpu);

    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        let out = model.predict(&feed, &want)?;
        samples.push(run.elapsed().as_secs_f64());
        assert_eq!(out[0].len(), elements, "the probe output lost its shape");
        // Uniform q and k make every score equal, so softmax is uniform and the
        // result is the mean of v. Anything else means the program did not run
        // the attention this probe claims to time.
        let got = out[0][0].to_f32();
        assert!(
            (got - 1.0).abs() < 1e-2,
            "softmax(q k^T) v over uniform inputs gave {got}, not v's 1.0"
        );
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[RUNS / 2];
    let resident = directory_bytes(&compiled) as f64 / 1e6;

    println!(
        "sdpa-probe: heads={heads} macs={:.2}G median={:.3}ms rate={:.2}TMAC/s \
         warm={warm:.3}s warm_cpu={warm_cpu:.3}s cpu_share={:.0}% \
         compile={compile:.2}s program={resident:.1}MB fastest={:.3}ms slowest={:.3}ms",
        macs / 1e9,
        median * 1e3,
        macs / median / 1e12,
        100.0 * warm_cpu / warm,
        samples[0] * 1e3,
        samples[RUNS - 1] * 1e3,
    );
    println!("sdpa-probe: heads={heads} placement={}", devices.join(" "));
    Ok(())
}

/// Narrows a zero result down to the op that produced it: each stage is the
/// previous one plus the next operation, so the first zero names the culprit.
fn selftest(cache: &ProgramCache) -> Result<()> {
    let heads = 2usize;
    let shape = [heads, TOKENS, HEAD_DIM];
    let elements = heads * TOKENS * HEAD_DIM;
    // Core ML rejects a declared input no operation reads, so each stage
    // declares only what it uses.
    let stages: Vec<(&str, fn(&mut ProgramBuilder) -> Result<Var>)> = vec![
        ("passthrough_v", |b| {
            let v = b.input("v", &[2, TOKENS, HEAD_DIM])?;
            b.identity(&v, None)
        }),
        ("scores", |b| {
            let q = b.input("q", &[2, TOKENS, HEAD_DIM])?;
            let k = b.input("k", &[2, TOKENS, HEAD_DIM])?;
            b.matmul(&q, &k, true)
        }),
        ("weights", |b| {
            let q = b.input("q", &[2, TOKENS, HEAD_DIM])?;
            let k = b.input("k", &[2, TOKENS, HEAD_DIM])?;
            let s = b.matmul(&q, &k, true)?;
            let s = b.mul(&s, Operand::Scalar(f16::from_f64(1.0 / 11.3137)), None)?;
            b.softmax(&s, 2)
        }),
        ("full", |b| {
            let q = b.input("q", &[2, TOKENS, HEAD_DIM])?;
            let k = b.input("k", &[2, TOKENS, HEAD_DIM])?;
            let v = b.input("v", &[2, TOKENS, HEAD_DIM])?;
            let s = b.matmul(&q, &k, true)?;
            let s = b.mul(&s, Operand::Scalar(f16::from_f64(1.0 / 11.3137)), None)?;
            let w = b.softmax(&s, 2)?;
            b.matmul(&w, &v, false)
        }),
    ];
    for (name, stage) in stages {
        let mut b = ProgramBuilder::new();
        let y = stage(&mut b)?;
        let out_shape = y.shape().to_vec();
        let y = b.identity(&y, Some("o"))?;
        let model = cache.load(&b.finish(&[&y])?, ComputeUnits::All)?;
        let (q, k, v) = (
            vec![f16::from_f32(0.02); elements],
            vec![f16::from_f32(0.03); elements],
            vec![f16::from_f32(1.0); elements],
        );
        let declared: &[&str] = match name {
            "passthrough_v" => &["v"],
            "full" => &["q", "k", "v"],
            _ => &["q", "k"],
        };
        let feed: Vec<(&str, &[f16], &[usize])> = declared
            .iter()
            .map(|&n| {
                let buffer: &[f16] = match n {
                    "q" => &q,
                    "k" => &k,
                    _ => &v,
                };
                (n, buffer, &shape[..])
            })
            .collect();
        let out = model.predict(&feed, &[("o", &out_shape)])?;
        println!(
            "sdpa-probe: selftest {name} shape={out_shape:?} first={:.6} last={:.6}",
            out[0][0].to_f32(),
            out[0][out[0].len() - 1].to_f32(),
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let dir = std::env::var("APXINF_SDPA_PROBE_CACHE")
        .unwrap_or_else(|_| "/tmp/sdpa-probe-cache".to_owned());
    std::fs::create_dir_all(&dir).ok();
    let cache = ProgramCache::new(dir);
    // Column split against token split, at the Neural Engine's shipped INT8
    // recipe. The shipped program is rows=6656 over all 1152 tokens; a token
    // split of the same MAC share is rows=12288 over 624 of them.
    for heads in [8usize, 13, 16] {
        probe_phase_c(&cache, heads)?;
    }
    if std::env::var("APXINF_SDPA_PROBE_PHASE_C_ONLY").is_ok() {
        return Ok(());
    }
    for heads in [8usize, 12, 16] {
        probe_whole_head(&cache, heads)?;
    }
    for heads in [8usize, 16, 32] {
        probe_attention_convs(&cache, heads)?;
    }
    probe_shipped(&cache, 6656, TOKENS)?;
    for tokens in [1152usize, 2304, 4608] {
        probe_shipped(&cache, 6656, tokens)?;
    }
    for n in [1usize, 2, 4, 8] {
        probe_rotation(&cache, 6656, TOKENS, n)?;
    }
    if std::env::var("APXINF_SDPA_PROBE_SHIPPED_ONLY").is_ok() {
        return Ok(());
    }
    for int8 in [true, false] {
        for rows in [2816usize, 6656, 12288] {
            probe_mlp(&cache, rows, TOKENS, int8)?;
        }
        for tokens in [847usize, 624, 576] {
            probe_mlp(&cache, 12288, tokens, int8)?;
        }
    }
    if std::env::var("APXINF_SDPA_PROBE_MLP_ONLY").is_ok() {
        return Ok(());
    }
    selftest(&cache)?;
    for heads in [8usize, 16] {
        probe_projection(&cache, heads)?;
    }
    for heads in [8usize, 16] {
        if let Err(error) = probe(&cache, heads) {
            println!("sdpa-probe: heads={heads} FAILED {error}");
        }
    }
    Ok(())
}
