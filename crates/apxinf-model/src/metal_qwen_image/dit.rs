//! The single-stream DiT (mflux `Qwen21Transformer`): the text prefix runs
//! once per prompt and caches each block's post-RoPE K/V; every denoise step
//! then runs one compiled image-token forward against that cache.
use super::ops::Ops;
use super::weights::{Block, Transformer, DIM, HEADS, HEAD_DIM, Q4};
use apxinf_core::Result;
use apxinf_mlx::fusions::{InterleavedQkNormRope, MppBf16GemmNt};
use apxinf_mlx::{Array, Compiled, MlxDType};
use std::rc::Rc;

const EPS: f32 = 1e-6;
/// Smallest image-token count that uses the MPP GEMM. MLX quantized matmul
/// switches to another reduction for small output grids (observed on M4 at
/// 64 tokens), where MPP no longer reproduces it.
pub const MPP_MIN_TOKENS: usize = 1024;
/// mflux `StepCache.MIN_STEPS`: shorter runs skip nothing.
const STEP_CACHE_MIN_STEPS: usize = 10;
/// mflux `StepCache.PROTECTED_FRACTION`: the first and last 10% of a run are
/// always computed.
const STEP_CACHE_PROTECTED_FRACTION: f64 = 0.1;

/// Select the steps a TeaCache-style cache skips, from row-major per-step
/// signals `[steps, dim]`. This mirrors mflux `StepCache.select_skip_steps`:
/// the consecutive L2 distance (index 0 is `inf`), a protected first/last 10%
/// window, and `round(ratio * steps)` picks of the smallest distances with
/// Python's ties-to-even rounding. A non-finite or out-of-range ratio and runs
/// under `STEP_CACHE_MIN_STEPS` skip nothing.
fn select_skip_steps(signals: &[f32], dim: usize, ratio: f32) -> Vec<bool> {
    if dim == 0 {
        return Vec::new();
    }
    let total = signals.len() / dim;
    let mut skip = vec![false; total];
    if !ratio.is_finite() || !(0.0 < ratio && ratio < 1.0) || total < STEP_CACHE_MIN_STEPS {
        return skip;
    }
    let mut diffs = Vec::with_capacity(total);
    diffs.push(f32::INFINITY);
    for i in 1..total {
        let mut acc = 0f64;
        for j in 0..dim {
            let delta = signals[i * dim + j] - signals[(i - 1) * dim + j];
            acc += f64::from(delta) * f64::from(delta);
        }
        diffs.push((acc as f32).sqrt());
    }
    let protected = (f64::from(total as u32) * STEP_CACHE_PROTECTED_FRACTION) as usize;
    let start = protected.max(1);
    let end = total.saturating_sub(protected);
    if start >= end {
        return skip;
    }
    let count = ((f64::from(ratio) * f64::from(total as u32)).round_ties_even() as usize)
        .min(end - start);
    let mut eligible: Vec<usize> = (start..end).collect();
    eligible.sort_by(|&a, &b| {
        diffs[a]
            .partial_cmp(&diffs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in eligible.iter().take(count) {
        skip[i] = true;
    }
    skip
}

pub struct Dit {
    w: Transformer,
    qk: InterleavedQkNormRope,
    /// `Qwen21Timesteps._freqs`, `[1, 128]` FP32.
    freqs: Array,
    ops: Ops,
    /// The image-token GEMM, when it reproduces MLX quantized matmul here.
    gemm: Option<Rc<MppBf16GemmNt>>,
}

/// Per-block post-RoPE text K/V, `[1, 32, T, 128]` BF16 each.
pub struct TextCache {
    kv: Vec<(Array, Array)>,
}

impl Dit {
    pub fn new(w: Transformer, ops: &Ops, gemm: Option<Rc<MppBf16GemmNt>>) -> Result<Self> {
        let stream = ops.stream();
        let freqs = super::tables::timestep_frequencies(128);
        Ok(Self {
            w,
            qk: InterleavedQkNormRope::new(stream)?,
            freqs: Array::from_f32(stream, &[1, 128], &freqs)?,
            ops: ops.clone(),
            gemm,
        })
    }

    /// `nn.QuantizedLinear`. At least `MPP_MIN_TOKENS` image tokens in a
    /// multiple of 64 decode the weight per call and use the probe-selected
    /// MPP GEMM. Full-image parity requires separate qualification for the
    /// selected device and profile.
    fn project(&self, w: &Q4, x: &Array, image: bool) -> Result<Array> {
        let tokens = x.shape()[1];
        match &self.gemm {
            Some(gemm) if image && tokens.is_multiple_of(64) && tokens >= MPP_MIN_TOKENS => {
                gemm.call(x, &w.weight.dequantize(&w.scales, &w.biases, 64, 4)?)
            }
            _ => w.apply(x),
        }
    }

    /// `Qwen21TimeTextEmbed` for FP32 timestep rows `[n]`: BF16 `[n, 4096]`.
    fn time_embedding(&self, rows: &Array) -> Result<Array> {
        let count = rows.numel().max(1);
        let args = rows
            .reshape(&[count, 1])?
            .mul(&self.ops.f32(1000.0)?)?
            .mul(&self.freqs)?;
        let projected = Array::concat(&[&args.cos()?, &args.sin()?], 1)?.cast(MlxDType::BF16)?;
        let hidden = self.ops.silu(&self.w.time_1.apply(&projected)?)?;
        self.w.time_2.apply(&hidden)
    }

    /// One batched `time_text_embed` over every step's sigma, then the mflux
    /// skip selection on the host. `step_sigmas` is FP32 `[steps]`; the cost is
    /// one tiny evaluated matmul per request, independent of resolution.
    pub fn select_skip_steps(&self, step_sigmas: &Array, ratio: f32) -> Result<Vec<bool>> {
        let signals = self.time_embedding(step_sigmas)?;
        signals.eval()?;
        let dim = signals.shape()[1];
        Ok(select_skip_steps(&signals.to_f32_vec()?, dim, ratio))
    }

    /// Shared modulation split into `(mod1, mod2)` rows `[2, 8192]`.
    fn modulation(&self, temb: &Array) -> Result<(Array, Array)> {
        let m = self.w.modulation.apply(&self.ops.silu(temb)?)?;
        Ok((
            m.slice_axis(1, 0, 2 * DIM)?,
            m.slice_axis(1, 2 * DIM, 4 * DIM)?,
        ))
    }

    fn modulate(&self, h: &Array, scale: &Array) -> Result<Array> {
        let one = self.ops.scalar(1.0, scale.dtype())?;
        h.fast_layer_norm(None, None, EPS)?.mul(&one.add(scale)?)
    }

    fn txt_in(&self, prompt: &Array) -> Result<Array> {
        // Qwen21ZeroCenterRMSNorm: x * rrms * (weight + 1) in FP32.
        let x = prompt.cast(MlxDType::F32)?;
        let rrms = x
            .mul(&x)?
            .mean(&[-1], true)?
            .add(&self.ops.f32(EPS)?)?
            .rsqrt()?;
        let scale = self
            .w
            .txt_norm
            .cast(MlxDType::F32)?
            .add(&self.ops.f32(1.0)?)?;
        let normed = x.mul(&rrms)?.mul(&scale)?.cast(prompt.dtype())?;
        let hidden = self.ops.gelu_approx(&self.w.txt_in.apply(&normed)?)?;
        self.w.txt_out.apply(&hidden)
    }

    fn attention_inputs(
        &self,
        b: &Block,
        x: &Array,
        cos: &Array,
        sin: &Array,
        image: bool,
    ) -> Result<(Array, Array, Array)> {
        let tokens = x.shape()[1];
        let (q, k) = self.qk.call(
            &self.project(&b.to_q, x, image)?,
            &self.project(&b.to_k, x, image)?,
            &b.norm_q,
            &b.norm_k,
            cos,
            sin,
            HEADS,
        )?;
        let v = self
            .project(&b.to_v, x, image)?
            .reshape(&[1, tokens, HEADS, HEAD_DIM])?
            .transpose(&[0, 2, 1, 3])?;
        Ok((q, k, v))
    }

    fn attention_output(&self, b: &Block, o: &Array, image: bool) -> Result<Array> {
        let tokens = o.shape()[2];
        let o = o.transpose(&[0, 2, 1, 3])?.reshape(&[1, tokens, DIM])?;
        self.project(&b.to_out, &o, image)
    }

    fn mlp(&self, b: &Block, x: &Array, image: bool) -> Result<Array> {
        let gate = self.ops.silu(&self.project(&b.gate, x, image)?)?;
        let gated = gate.mul(&self.project(&b.proj, x, image)?)?;
        self.project(&b.out, &gated, image)
    }

    /// One block over the image tokens (`text_kv = Some`) or the causal text
    /// prefix (`None`, also returning its K/V).
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        b: &Block,
        h: &Array,
        text_kv: Option<(&Array, &Array)>,
        mod1: &Array,
        mod2: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Option<(Array, Array)>)> {
        let half = |m: &Array| -> Result<(Array, Array)> {
            Ok((m.slice_axis(2, 0, DIM)?, m.slice_axis(2, DIM, 2 * DIM)?))
        };
        let (scale1, gate1) = half(mod1)?;
        let (scale2, gate2) = half(mod2)?;
        let x = self.modulate(h, &scale1)?;
        let image = text_kv.is_some();
        let (q, k, v) = self.attention_inputs(b, &x, cos, sin, image)?;
        let scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let (o, cached) = match text_kv {
            Some((key_text, value_text)) => {
                let k = Array::concat(&[key_text, &k], 2)?;
                let v = Array::concat(&[value_text, &v], 2)?;
                (q.sdpa(&k, &v, scale, false, None)?, None)
            }
            None => {
                let o = q.sdpa(&k, &v, scale, true, None)?;
                (o, Some((k.contiguous()?, v.contiguous()?)))
            }
        };
        let o = self.attention_output(b, &o, image)?;
        let h = h.add(&gate1.tanh()?.mul(&o)?)?;
        let y = self.mlp(b, &self.modulate(&h, &scale2)?, image)?;
        Ok((h.add(&gate2.tanh()?.mul(&y)?)?, cached))
    }

    /// Row `row` of `[2, C]` modulation, broadcast over `tokens`.
    fn row(m: &Array, row: usize, tokens: usize) -> Result<Array> {
        let width = m.shape()[1];
        m.slice_axis(0, row, row + 1)?
            .reshape(&[1, 1, width])?
            .broadcast_to(&[1, tokens, width])
    }

    /// The text prefix with the t = 0 modulation row; FP32 RoPE tables `[T, 64]`.
    pub fn text_cache(&self, prompt: &Array, cos: &Array, sin: &Array) -> Result<TextCache> {
        let stream = self.ops.stream();
        let tokens = prompt.shape()[1];
        let temb = self.time_embedding(&Array::zeros(stream, &[2], MlxDType::F32)?)?;
        let (mod1, mod2) = self.modulation(&temb)?;
        let (mod1, mod2) = (Self::row(&mod1, 1, tokens)?, Self::row(&mod2, 1, tokens)?);
        let mut h = self.txt_in(prompt)?;
        let mut kv = Vec::with_capacity(self.w.blocks.len());
        for b in &self.w.blocks {
            let (next, cached) = self.block(b, &h, None, &mod1, &mod2, cos, sin)?;
            h = next;
            kv.push(cached.expect("the text prefix returns its K/V"));
        }
        let mut arrays = vec![h];
        arrays.extend(kv.iter().flat_map(|(k, v)| [k.clone(), v.clone()]));
        stream.eval(&arrays)?;
        Ok(TextCache { kv })
    }

    /// Image-token forward: latents `[1, N, 64]` BF16, timestep `[1]` FP32,
    /// FP32 RoPE tables `[N, 64]`, then each block's text K/V.
    fn image_forward(&self, inputs: &[Array]) -> Result<Array> {
        let stream = self.ops.stream();
        let (latents, timestep, cos, sin) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
        let tokens = latents.shape()[1];
        let rows = Array::concat(&[timestep, &Array::zeros(stream, &[1], MlxDType::F32)?], 0)?;
        let temb = self.time_embedding(&rows)?;
        let (mod1, mod2) = self.modulation(&temb)?;
        let (mod1, mod2) = (Self::row(&mod1, 0, tokens)?, Self::row(&mod2, 0, tokens)?);
        let mut h = self.w.img_in.apply(latents)?;
        for (b, kv) in self.w.blocks.iter().zip(inputs[4..].chunks(2)) {
            let text = Some((&kv[0], &kv[1]));
            h = self.block(b, &h, text, &mod1, &mod2, cos, sin)?.0;
        }
        let scale = self.w.norm_out.apply(&self.ops.silu(&temb)?)?;
        let scale = Self::row(&scale, 0, tokens)?;
        self.w.proj_out.apply(&self.modulate(&h, &scale)?)
    }
}

/// The compiled denoise step of one request; it owns the transformer.
pub struct Step {
    compiled: Compiled,
}

impl Step {
    pub fn prepare(dit: Dit, inputs: &[Array]) -> Result<(Self, Array)> {
        let stream = dit.ops.stream().clone();
        let compiled = Compiled::new(&stream, 1, move |inputs| {
            Ok(vec![dit.image_forward(inputs)?])
        })?;
        let first = compiled.prepare(inputs)?.remove(0);
        Ok((Self { compiled }, first))
    }

    /// Velocity prediction `[1, N, 64]` BF16 for one timestep.
    pub fn inputs(
        latents: &Array,
        timestep: &Array,
        cos: &Array,
        sin: &Array,
        cache: &TextCache,
    ) -> Vec<Array> {
        let mut inputs = vec![latents.clone(), timestep.clone(), cos.clone(), sin.clone()];
        inputs.extend(cache.kv.iter().flat_map(|(k, v)| [k.clone(), v.clone()]));
        inputs
    }

    pub fn call(&self, inputs: &[Array]) -> Result<Array> {
        Ok(self.compiled.call(inputs)?.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row-major scalar signals whose consecutive distances are
    /// `[inf, 5, .5, 4, .5, 3, .5, 2, .5, 6, .5, 7]`. With 12 steps the
    /// protected window is one step on each side, leaving indices 1..=10, and
    /// r=0.25 selects three — the ties resolve in eligible order.
    #[test]
    fn skip_selection_picks_smallest_changes_outside_the_protected_window() {
        let diffs = [5.0, 0.5, 4.0, 0.5, 3.0, 0.5, 2.0, 0.5, 6.0, 0.5, 7.0];
        let mut signals = Vec::with_capacity(diffs.len() + 1);
        signals.push(0.0);
        for delta in diffs {
            signals.push(signals.last().copied().unwrap() + delta);
        }
        let skip = select_skip_steps(&signals, 1, 0.25);
        assert_eq!(
            skip.iter().enumerate().filter(|(_, &s)| s).map(|(i, _)| i).collect::<Vec<_>>(),
            vec![2, 4, 6],
            "signal distances {diffs:?}"
        );
        // Indices 0, 1 and 11 sit in the protected first/last 10%.
        assert!(!skip[0] && !skip[1] && !skip[11]);
    }

    /// Ten steps at r=0.25 is 2.5, which Python rounds ties-to-even to 2 —
    /// not the 3 `f32::round` would give. The protected window is one step.
    #[test]
    fn skip_selection_count_uses_ties_to_even_rounding() {
        let signals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let skip = select_skip_steps(&signals, 1, 0.25);
        assert_eq!(skip.iter().filter(|&&s| s).count(), 2);
    }

    #[test]
    fn skip_selection_short_runs_and_invalid_ratios_skip_nothing() {
        let signals: Vec<f32> = (0..STEP_CACHE_MIN_STEPS - 1).map(|i| i as f32).collect();
        assert!(select_skip_steps(&signals, 1, 0.25).iter().all(|&s| !s));
        let signals: Vec<f32> = (0..40).map(|i| i as f32).collect();
        for ratio in [0.0, 1.0, -0.1, f32::NAN, f32::INFINITY] {
            assert!(
                select_skip_steps(&signals, 1, ratio).iter().all(|&s| !s),
                "ratio {ratio} skipped a step"
            );
        }
    }

    #[test]
    fn skip_selection_is_deterministic_and_resolution_independent() {
        // A schedule-only signal: the same (steps, ratio) yields one skip set
        // regardless of which request asks for it.
        let signals: Vec<f32> = (0..40).map(|i| (i as f32).sin()).collect();
        let first = select_skip_steps(&signals, 1, 0.25);
        let second = select_skip_steps(&signals, 1, 0.25);
        assert_eq!(first, second);
        assert_eq!(first.iter().filter(|&&s| s).count(), 10);
    }
}
