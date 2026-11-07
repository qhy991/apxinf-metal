//! Prompt encoding: the fixed Qwen-Image-2.1 template and the Qwen3-VL text
//! stack (mflux `Qwen21TextEncoder`), read from disk one layer at a time.
use super::ops::Ops;
use super::weights::{self, TextEncoderFiles, TextLayer, DIM, HEADS, HEAD_DIM, TEXT_KV_HEADS, TEXT_LAYERS};
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType};
use apxinf_tokenizer::Tokenizer;

const SYSTEM_PREFIX: &str =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
/// The reference tokenizer truncates the templated prompt to this length.
const MAX_TOKENS: usize = 2048;
const EPS: f32 = 1e-6;

pub struct PromptTokens {
    pub ids: Vec<u32>,
    /// Leading system-prefix tokens whose states the DiT does not read.
    pub drop: usize,
}

pub fn tokenize(tokenizer: &Tokenizer, prompt: &str) -> Result<PromptTokens> {
    let text =
        format!("{SYSTEM_PREFIX}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let mut ids = tokenizer.encode_with_special_tokens(&text, true)?;
    ids.truncate(MAX_TOKENS);
    let drop = tokenizer
        .encode_with_special_tokens(SYSTEM_PREFIX, false)?
        .len();
    if ids.len() <= drop {
        return Err(Error::Other(
            "the Qwen-Image prompt template left no prompt tokens".into(),
        ));
    }
    Ok(PromptTokens { ids, drop })
}

/// `Qwen3VLRMSNorm`: FP32 normalization, FP32 weight multiply, cast back.
fn rms_norm(x: &Array, weight: &Array, ops: &Ops) -> Result<Array> {
    let x32 = x.cast(MlxDType::F32)?;
    let variance = x32.mul(&x32)?.mean(&[-1], true)?;
    let normalized = x32.mul(&variance.add(&ops.f32(EPS)?)?.rsqrt()?)?;
    weight
        .cast(MlxDType::F32)?
        .mul(&normalized)?
        .cast(x.dtype())
}

fn rotate_half(x: &Array) -> Result<Array> {
    let half = HEAD_DIM / 2;
    let x1 = x.slice_axis(3, 0, half)?;
    let x2 = x.slice_axis(3, half, HEAD_DIM)?;
    Array::concat(&[&x2.neg()?, &x1], 3)
}

/// `nn.Linear` without bias: `x @ W.T`.
fn linear(x: &Array, weight: &Array) -> Result<Array> {
    x.matmul(&weight.transpose(&[1, 0])?)
}

fn layer(h: &Array, w: &TextLayer, cos: &Array, sin: &Array, ops: &Ops) -> Result<Array> {
    let length = h.shape()[1];
    let x = rms_norm(h, &w.input_norm, ops)?;
    let heads = |p: &Array, n: usize| p.reshape(&[1, length, n, HEAD_DIM]);
    let q = rms_norm(&heads(&linear(&x, &w.q)?, HEADS)?, &w.q_norm, ops)?;
    let k = rms_norm(&heads(&linear(&x, &w.k)?, TEXT_KV_HEADS)?, &w.k_norm, ops)?;
    let v = heads(&linear(&x, &w.v)?, TEXT_KV_HEADS)?;
    let (q, k, v) = (
        q.transpose(&[0, 2, 1, 3])?,
        k.transpose(&[0, 2, 1, 3])?,
        v.transpose(&[0, 2, 1, 3])?,
    );
    let q = q.mul(cos)?.add(&rotate_half(&q)?.mul(sin)?)?;
    let k = k.mul(cos)?.add(&rotate_half(&k)?.mul(sin)?)?;
    let repeat = |a: &Array| {
        let groups = HEADS / TEXT_KV_HEADS;
        a.reshape(&[1, TEXT_KV_HEADS, 1, length, HEAD_DIM])?
            .broadcast_to(&[1, TEXT_KV_HEADS, groups, length, HEAD_DIM])?
            .reshape(&[1, HEADS, length, HEAD_DIM])
    };
    let (k, v) = (repeat(&k)?, repeat(&v)?);
    let scale = (1.0 / (HEAD_DIM as f64).sqrt()) as f32;
    // Causal self-attention over the templated prompt, in FP32 like the
    // reference. The causal flag selects MLX's masked-tile-skipping kernel;
    // it computes the same kept-set softmax as an additive -inf mask.
    let attn = q
        .cast(MlxDType::F32)?
        .sdpa(
            &k.cast(MlxDType::F32)?,
            &v.cast(MlxDType::F32)?,
            scale,
            true,
            None,
        )?
        .cast(q.dtype())?
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[1, length, DIM])?;
    let h = h.add(&linear(&attn, &w.o)?)?;
    let x = rms_norm(&h, &w.post_norm, ops)?;
    let mlp = ops.silu(&linear(&x, &w.gate)?)?.mul(&linear(&x, &w.up)?)?;
    h.add(&linear(&mlp, &w.down)?)
}

/// Prompt states after the final norm, without the system prefix: `[1, T, 4096]` BF16.
pub fn encode(
    files: &TextEncoderFiles,
    tokens: &PromptTokens,
    ops: &Ops,
    image_tokens: usize,
) -> Result<Array> {
    let stream = ops.stream();
    let timing = super::timing_enabled();
    // Overlap the per-layer disk reads with the previous layer's device
    // compute: the worker warms the page cache only `window` layers ahead
    // and purges each consumed layer's pages again right after use, so the
    // cache holds a ~0.8 GB window instead of the full 15.1 GB and the
    // 16 GB host does not page at the largest tier. The memory gate
    // degrades to purge-only reads when the host is already tight.
    // Scheduling only; the bytes, kernels and evaluation order are
    // unchanged, so the encoded states stay bit-exact.
    let plan = weights::prefetch_plan(image_tokens);
    let active = match &plan {
        weights::PrefetchPlan::Active(plan) => Some(plan),
        weights::PrefetchPlan::Disabled(_) => None,
    };
    if let Some(plan) = active {
        if plan.warm {
            for k in 0..=plan.window.min(TEXT_LAYERS - 1) {
                plan.worker.warm(files.layer_ranges(k));
            }
        }
    }
    if timing {
        match &plan {
            weights::PrefetchPlan::Active(plan) => {
                let mib = |bytes: u64| format!("{}MiB", bytes >> 20);
                let (avail, swap) = match plan.gate {
                    weights::PrefetchGate::Admit(stats) => stats,
                    weights::PrefetchGate::Memory(stats) => Some(stats),
                    _ => None,
                }
                .map(|s| (mib(s.available_bytes), mib(s.swap_used_bytes)))
                .unwrap_or_else(|| ("?".into(), "?".into()));
                eprintln!(
                    "qimage-timing: text_prefetch enabled={} window={} purge={} avail={} swap_used={}",
                    usize::from(plan.warm),
                    plan.window,
                    usize::from(plan.purge),
                    avail,
                    swap,
                );
            }
            weights::PrefetchPlan::Disabled(_) => {
                eprintln!("qimage-timing: text_prefetch enabled=0 window=0 purge=0");
            }
        }
    }
    let mut next = active
        .filter(|plan| plan.warm)
        .map_or(usize::MAX, |plan| plan.window + 1);
    let length = tokens.ids.len();
    let ids: Vec<i32> = tokens.ids.iter().map(|&id| id as i32).collect();
    let embed_start = std::time::Instant::now();
    let mut h = {
        let embedding = files.tensor(stream, "embed_tokens.weight")?;
        let h = embedding.take(&Array::from_i32(stream, &[1, length], &ids)?, 0)?;
        h.eval()?;
        h
    };
    if let Some(plan) = active {
        if plan.purge {
            // The ~0.8 GB embedding table is read once; drop its pages at
            // once instead of keeping them cached for the rest of the run.
            plan.worker.purge(files.tensor_ranges("embed_tokens.weight"));
        }
    }
    let embed_elapsed = embed_start.elapsed();
    // Text positions repeat on all three MRoPE axes, so the interleave is the identity.
    let inv_freq = files.tensor(stream, "rotary_emb.inv_freq")?;
    let positions: Vec<f32> = (0..length).map(|p| p as f32).collect();
    let freqs = inv_freq
        .reshape(&[HEAD_DIM / 2, 1])?
        .matmul(&Array::from_f32(stream, &[1, length], &positions)?)?
        .transpose(&[1, 0])?;
    let emb = Array::concat(&[&freqs, &freqs], 1)?.reshape(&[1, 1, length, HEAD_DIM])?;
    let cos = emb.cos()?.cast(h.dtype())?;
    let sin = emb.sin()?.cast(h.dtype())?;
    let mut load_total = std::time::Duration::ZERO;
    let mut compute_total = std::time::Duration::ZERO;
    for i in 0..TEXT_LAYERS {
        let load_start = std::time::Instant::now();
        let weights = files.layer(stream, i)?;
        let load = load_start.elapsed();
        if let Some(plan) = active {
            if plan.warm && next < TEXT_LAYERS {
                plan.worker.warm(files.layer_ranges(next));
                next += 1;
            }
        }
        let compute_start = std::time::Instant::now();
        h = layer(&h, &weights, &cos, &sin, ops)?;
        // Release this layer's weights before reading the next one.
        h.eval()?;
        let compute = compute_start.elapsed();
        drop(weights);
        if let Some(plan) = active {
            if plan.purge {
                // This layer's bytes are on the device now and never read
                // again; the worker drops their pages so the cache keeps
                // only the warm window.
                plan.worker.purge(files.layer_ranges(i));
            }
        }
        load_total += load;
        compute_total += compute;
        if timing {
            eprintln!(
                "qimage-timing: text_layer {i} load={:.3}s compute={:.3}s",
                load.as_secs_f64(),
                compute.as_secs_f64(),
            );
        }
    }
    let join_start = std::time::Instant::now();
    drop(plan);
    let join_elapsed = join_start.elapsed();
    if timing {
        eprintln!(
            "qimage-timing: text_layers embed={:.3}s load={:.3}s compute={:.3}s join={:.3}s",
            embed_elapsed.as_secs_f64(),
            load_total.as_secs_f64(),
            compute_total.as_secs_f64(),
            join_elapsed.as_secs_f64(),
        );
    }
    let norm = files.tensor(stream, "norm.weight")?;
    let h = rms_norm(&h, &norm, ops)?.slice_axis(1, tokens.drop, length)?;
    h.eval()?;
    Ok(h)
}
