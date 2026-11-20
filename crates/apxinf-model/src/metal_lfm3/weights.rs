//! LFM3 checkpoint mapping and canonical weight geometry.
//!
//! Every transformation from the Hugging Face checkpoint happens once here:
//! linear weights become canonical `[in, out]` views of the immutable `[out, in]`
//! rows, and the depthwise conv table is flattened from `[C, 1, K]` to `[C, K]`
//! and widened to FP32, which is the dtype the reference convolution multiplies
//! in. The convolution weight layout is the one adaptation this family needs, so
//! it is recorded here and replayed against captured reference tensors.

use super::config::Lfm3Config;
use apxinf_core::{DType, Device, Error, Result, Tensor};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::HashMap;

/// MLX affine packing derived once from the canonical weight (`APXINF_LFM3_Q4_BITS`).
struct Quantized {
    packed: Array,
    scales: Array,
    biases: Array,
    bits: i32,
}

impl Quantized {
    fn new(hf_weight: &Array) -> Result<Self> {
        let bits: i32 = match std::env::var("APXINF_LFM3_Q4_BITS").ok().as_deref() {
            Some("8") => 8,
            Some("6") => 6,
            _ => 4,
        };
        let [packed, scales, biases] = hf_weight.quantize(64, bits)?;
        hf_weight
            .stream()
            .eval(&[packed.clone(), scales.clone(), biases.clone()])?;
        Ok(Self {
            packed,
            scales,
            biases,
            bits,
        })
    }
}

/// One HF linear projection, kept as a canonical `[in, out]` view when dense.
pub(crate) enum Linear {
    Dense(Array),
    Quantized(Quantized),
}

impl Linear {
    pub fn call(&self, x: &Array) -> Result<Array> {
        match self {
            Linear::Dense(canonical) => x.matmul(canonical),
            Linear::Quantized(q) => {
                x.quantized_matmul(&q.packed, &q.scales, &q.biases, true, 64, q.bits)
            }
        }
    }
    pub fn array(&self) -> Array {
        match self {
            Linear::Dense(canonical) => canonical.clone(),
            Linear::Quantized(_) => {
                unreachable!("quantized LFM3 projection has no dense array")
            }
        }
    }
}

/// The layer's non-residual operator: exactly one of the two branches exists.
pub(crate) enum Mixer {
    Conv {
        in_proj: Linear,
        out_proj: Linear,
        /// `[hidden, kernel]` FP32 depthwise tap table (checkpoint stores `[C, 1, K]`).
        taps: Array,
        /// The same table as BF16 for the fused decode kernel. The checkpoint
        /// stores BF16 taps and `taps` above is their exact F32 widening, so this
        /// cast restores the checkpoint's bytes: it quantizes nothing new.
        taps_bf16: Array,
    },
    Attention {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        /// Per-head RMSNorm weight over `head_dim`.
        q_norm: Array,
        k_norm: Array,
        /// `hidden -> num_attention_heads` per-head output gate.
        gate: Linear,
    },
}

pub(crate) struct LayerWeights {
    pub operator_norm: Array,
    pub ffn_norm: Array,
    pub post_operator_norm: Array,
    pub post_ffn_norm: Array,
    pub w1: Linear,
    pub w3: Linear,
    pub w2: Linear,
    pub mixer: Mixer,
}

pub(crate) struct Weights {
    /// Tied embedding/lm-head table: gather rows for input, transpose for logits.
    embedding: Array,
    head: Array,
    pub final_norm: Array,
    pub layers: Vec<LayerWeights>,
}

fn take(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
    stream: &Stream,
) -> Result<Array> {
    let tensor = map
        .remove(name)
        .ok_or_else(|| Error::Other(format!("LFM3 missing weight `{name}`")))?;
    if tensor.shape().dims() != shape {
        return Err(Error::ShapeMismatch {
            expected: format!("{name} {shape:?}"),
            got: format!("{:?}", tensor.shape()),
        });
    }
    if tensor.device() != Device::Cpu {
        return Err(Error::UnsupportedDevice(tensor.device()));
    }
    if tensor.dtype() != DType::BF16 {
        return Err(Error::UnsupportedDType {
            got: tensor.dtype(),
            allowed: "LFM3 checkpoint BF16",
        });
    }
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("weight has no CPU storage".into()))?;
    Array::from_bytes(stream, shape, MlxDType::BF16, bytes)
}

/// Canonical `[in, out]` view of the checkpoint's immutable `[out, in]` rows.
fn linear(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    input: usize,
    output: usize,
    stream: &Stream,
) -> Result<Linear> {
    let rows = take(map, name, &[output, input], stream)?;
    if std::env::var("APXINF_LFM3_Q4").ok().as_deref() == Some("1") {
        return Ok(Linear::Quantized(Quantized::new(&rows)?));
    }
    Ok(Linear::Dense(rows.transpose(&[1, 0])?))
}

/// Per-head RMSNorm weights stay BF16 so the family uses the same weighted
/// native RMSNorm as the verified reference path.
fn norm(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    size: usize,
    stream: &Stream,
) -> Result<Array> {
    take(map, name, &[size], stream)
}

impl Weights {
    pub fn load(
        config: &Lfm3Config,
        mut map: HashMap<String, Tensor>,
        stream: &Stream,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let kind = config.layer_kind(i)?;
            let p = format!("model.layers.{i}");
            let mut n = |suffix: &str, size: usize| {
                norm(&mut map, &format!("{p}.{suffix}.weight"), size, stream)
            };
            let operator_norm = n("operator_norm", h)?;
            let ffn_norm = n("ffn_norm", h)?;
            let post_operator_norm = n("post_operator_norm", h)?;
            let post_ffn_norm = n("post_ffn_norm", h)?;
            let w1 = linear(
                &mut map,
                &format!("{p}.feed_forward.w1.weight"),
                h,
                config.intermediate_size,
                stream,
            )?;
            let w3 = linear(
                &mut map,
                &format!("{p}.feed_forward.w3.weight"),
                h,
                config.intermediate_size,
                stream,
            )?;
            let w2 = linear(
                &mut map,
                &format!("{p}.feed_forward.w2.weight"),
                config.intermediate_size,
                h,
                stream,
            )?;
            let mixer = if kind.is_conv() {
                let taps = take(
                    &mut map,
                    &format!("{p}.conv.conv.weight"),
                    &[h, 1, config.conv_l_cache],
                    stream,
                )?
                .reshape(&[h, config.conv_l_cache])?
                .cast(MlxDType::F32)?;
                // BF16 -> F32 is exact, so this round trip reproduces the
                // checkpoint's tap values for the fused kernel's BF16 input.
                let taps_bf16 = taps.cast(MlxDType::BF16)?;
                Mixer::Conv {
                    in_proj: linear(
                        &mut map,
                        &format!("{p}.conv.in_proj.weight"),
                        h,
                        3 * h,
                        stream,
                    )?,
                    out_proj: linear(&mut map, &format!("{p}.conv.out_proj.weight"), h, h, stream)?,
                    taps,
                    taps_bf16,
                }
            } else {
                Mixer::Attention {
                    q: linear(
                        &mut map,
                        &format!("{p}.self_attn.q_proj.weight"),
                        h,
                        q,
                        stream,
                    )?,
                    k: linear(
                        &mut map,
                        &format!("{p}.self_attn.k_proj.weight"),
                        h,
                        kv,
                        stream,
                    )?,
                    v: linear(
                        &mut map,
                        &format!("{p}.self_attn.v_proj.weight"),
                        h,
                        kv,
                        stream,
                    )?,
                    o: linear(
                        &mut map,
                        &format!("{p}.self_attn.o_proj.weight"),
                        q,
                        h,
                        stream,
                    )?,
                    q_norm: norm(
                        &mut map,
                        &format!("{p}.self_attn.q_norm.weight"),
                        config.head_dim,
                        stream,
                    )?,
                    k_norm: norm(
                        &mut map,
                        &format!("{p}.self_attn.k_norm.weight"),
                        config.head_dim,
                        stream,
                    )?,
                    gate: linear(
                        &mut map,
                        &format!("{p}.self_attn.gate_proj.weight"),
                        h,
                        config.num_attention_heads,
                        stream,
                    )?,
                }
            };
            layers.push(LayerWeights {
                operator_norm,
                ffn_norm,
                post_operator_norm,
                post_ffn_norm,
                w1,
                w3,
                w2,
                mixer,
            });
        }
        // The checkpoint ships no `lm_head`: the token table is tied, and the
        // logits projection is its transpose. A present head must be identical.
        if let Some(head) = map.remove("lm_head.weight") {
            let embed = map
                .get("model.embed_tokens.weight")
                .ok_or_else(|| Error::Other("LFM3 missing tied embedding".into()))?;
            if head.shape() != embed.shape()
                || head.dtype() != embed.dtype()
                || head.storage().as_cpu() != embed.storage().as_cpu()
            {
                return Err(Error::Other(
                    "LFM3 tied lm_head differs from embedding checkpoint bytes".into(),
                ));
            }
        }
        let embedding = take(
            &mut map,
            "model.embed_tokens.weight",
            &[config.vocab_size, h],
            stream,
        )?;
        let final_norm = norm(&mut map, "model.embedding_norm.weight", h, stream)?;
        // Any leftover tensor means this is not the architecture analyzed here.
        if !map.is_empty() {
            let mut names: Vec<_> = map.keys().cloned().collect();
            names.sort();
            names.truncate(6);
            return Err(Error::Other(format!(
                "unconsumed LFM3 checkpoint tensors: {}",
                names.join(", ")
            )));
        }
        let head = embedding.transpose(&[1, 0])?;
        let mut arrays = vec![embedding.clone(), head.clone(), final_norm.clone()];
        for layer in &layers {
            arrays.extend([
                layer.operator_norm.clone(),
                layer.ffn_norm.clone(),
                layer.post_operator_norm.clone(),
                layer.post_ffn_norm.clone(),
                layer.w1.array(),
                layer.w2.array(),
                layer.w3.array(),
            ]);
            match &layer.mixer {
                Mixer::Conv {
                    in_proj,
                    out_proj,
                    taps,
                    taps_bf16,
                } => {
                    arrays.extend([
                        in_proj.array(),
                        out_proj.array(),
                        taps.clone(),
                        taps_bf16.clone(),
                    ]);
                }
                Mixer::Attention {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    gate,
                } => {
                    arrays.extend([
                        q.array(),
                        k.array(),
                        v.array(),
                        o.array(),
                        q_norm.clone(),
                        k_norm.clone(),
                        gate.array(),
                    ]);
                }
            }
        }
        let mut arrays = arrays;
        arrays.shrink_to_fit();
        stream.eval(&arrays)?;
        Ok(Self {
            embedding,
            head,
            final_norm,
            layers,
        })
    }

    pub fn embed(&self, ids: &Array) -> Result<Array> {
        self.embedding.take(ids, 0)
    }

    pub fn project(&self, x: &Array) -> Result<Array> {
        x.matmul(&self.head)
    }
}
