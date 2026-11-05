//! Compatibility adapter for the existing public Tensor/Backend and sampler APIs.
//! Metal models execute through native Array/Compiled operations; they do not
//! route their layers through a portable operator contract.
use crate::{Array, MlxDType, Stream};
use apxinf_core::{
    Backend, DType, Device, Error, Graph, KvCache, NextTokenLogits, NormalGenerator, Result,
    SamplingBackend, Shape, Storage, Tensor, TokenPenalties, TokenSample, TokenSampler,
    TokenSamplingInit, TokenSamplingSpec, TokenSelection,
};
use std::{any::Any, rc::Rc};

fn err(s: impl Into<String>) -> Error {
    Error::Other(s.into())
}
fn mlx_dtype(d: DType) -> Result<MlxDType> {
    match d {
        DType::F32 => Ok(MlxDType::F32),
        DType::F16 => Ok(MlxDType::F16),
        DType::BF16 => Ok(MlxDType::BF16),
        DType::I32 => Ok(MlxDType::I32),
        _ => Err(Error::UnsupportedDType {
            got: d,
            allowed: "f32, f16, bf16, i32",
        }),
    }
}
fn core_dtype(d: MlxDType) -> Result<DType> {
    match d {
        MlxDType::F32 => Ok(DType::F32),
        MlxDType::F16 => Ok(DType::F16),
        MlxDType::BF16 => Ok(DType::BF16),
        MlxDType::I32 => Ok(DType::I32),
        _ => Err(err(
            "internal MLX index/mask dtype is not a public Tensor dtype",
        )),
    }
}
/// A concrete, thread-confined Metal backend. CPU-stream arrays are an explicit
/// low-level diagnostic facility, never an implicit fallback for this backend.
fn checked_bytes(shape: &[usize], dtype: DType) -> Result<usize> {
    shape.iter().try_fold(dtype.size_in_bytes(), |n, &d| {
        if d == 0 {
            return Err(err("empty tensor dimensions are unsupported"));
        }
        n.checked_mul(d)
            .ok_or_else(|| err("tensor byte size overflow"))
    })
}

#[derive(Clone)]
pub struct MlxBackend {
    stream: Stream,
    index: usize,
}
impl MlxBackend {
    pub fn new(index: usize) -> Result<Self> {
        let stream = Stream::metal(index)?;
        if let Some(caps) = stream.metal_caps() {
            eprintln!(
                "Metal {}: {} ({})",
                index, caps.device_name, caps.architecture
            );
        }
        Ok(Self { stream, index })
    }
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
    pub fn from_array(&self, array: Array) -> Result<Tensor> {
        if !self.stream.same(array.stream()) {
            return Err(err("MLX result belongs to another backend stream"));
        }
        let dtype = core_dtype(array.dtype())?;
        let array = array.contiguous()?;
        let shape = Shape::new(array.shape().to_vec());
        let bytes = checked_bytes(array.shape(), dtype)?;
        Tensor::from_opaque_parts(shape, dtype, self.device(), bytes, Rc::new(array))
    }
    pub fn array(&self, t: &Tensor) -> Result<Array> {
        if t.device() != self.device() {
            return Err(Error::DeviceMismatch {
                expected: self.device(),
                got: t.device(),
            });
        }
        let bytes = checked_bytes(t.shape().dims(), t.dtype())?;
        if t.storage().len() != bytes {
            return Err(Error::DataLengthMismatch {
                expected: bytes,
                got: t.storage().len(),
            });
        }
        let Storage::Opaque { device, handle } = t.storage() else {
            return Err(err("Tensor is not backed by an MLX array"));
        };
        if *device != self.device() {
            return Err(err("opaque device metadata disagrees"));
        }
        let a = handle
            .owner()
            .downcast_ref::<Array>()
            .ok_or_else(|| err("opaque Tensor belongs to another backend"))?;
        if !self.stream.same(a.stream()) || a.stream().metal_index() != Some(self.index) {
            return Err(err("opaque array belongs to another MLX execution owner"));
        }
        if a.dtype() != mlx_dtype(t.dtype())? || a.numel() != t.numel() {
            return Err(err(
                "opaque array dtype/extent disagrees with Tensor metadata",
            ));
        }
        a.reshape(t.shape().dims())
    }
    fn float_array(&self, t: &Tensor) -> Result<Array> {
        mlx_dtype(t.dtype())?;
        self.array(t)
    }
    fn same_float(&self, a: &Tensor, b: &Tensor) -> Result<(Array, Array)> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                expected: a.dtype(),
                got: b.dtype(),
            });
        }
        Ok((self.float_array(a)?, self.float_array(b)?))
    }
    fn cached_attention(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer: usize,
        heads: usize,
        kv_heads: usize,
        dim: usize,
        kv_len: usize,
        max_len: usize,
    ) -> Result<Tensor> {
        let cache = kv
            .as_any_mut()
            .downcast_mut::<MlxKvCache>()
            .ok_or_else(|| err("foreign KV cache"))?;
        cache.valid()?;
        if !cache.backend.stream.same(&self.stream)
            || cache.heads != kv_heads
            || cache.dim != dim
            || cache.max_len != max_len
        {
            return Err(err("KV geometry or owner mismatch"));
        }
        let pair = cache
            .pending
            .get(layer)
            .and_then(Option::as_ref)
            .or_else(|| cache.layers.get(layer).and_then(Option::as_ref))
            .ok_or_else(|| err("uninitialized KV layer"))?;
        let rows = q
            .shape()
            .dims()
            .first()
            .copied()
            .ok_or_else(|| err("invalid query rank"))?;
        if q.shape().dims() != [rows, heads, dim]
            || kv_len < rows
            || kv_len > max_len
            || pair.0.shape()[1] != kv_len
        {
            return Err(err("cached attention shape/length mismatch"));
        }
        let qt = self.from_array(self.array(q)?.reshape(&[1, rows, heads, dim])?)?;
        let kt = self.from_array(pair.0.clone())?;
        let vt = self.from_array(pair.1.clone())?;
        let out = self.causal_attention(&qt, &kt, &vt, kv_len - rows)?;
        self.from_array(self.array(&out)?.reshape(&[rows, heads * dim])?)
    }
    fn causal_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        q_start: usize,
    ) -> Result<Tensor> {
        if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
            return Err(err("cached attention dtype mismatch"));
        }
        mlx_dtype(q.dtype())?;
        let qs = q.shape().dims();
        let ks = k.shape().dims();
        let (batch, qlen, heads, dim) = (qs[0], qs[1], qs[2], qs[3]);
        let (klen, kh) = (ks[1], ks[2]);
        if kh == 0 || heads == 0 || heads % kh != 0 || v.shape() != k.shape() {
            return Err(err(
                "cached attention head grouping or value shape mismatch",
            ));
        }
        let qa = self
            .array(q)?
            .cast(MlxDType::F32)?
            .transpose(&[0, 2, 1, 3])?;
        let expand = |t: &Tensor| -> Result<Array> {
            self.array(t)?
                .cast(MlxDType::F32)?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[batch, kh, 1, klen, dim])?
                .broadcast_to(&[batch, kh, heads / kh, klen, dim])?
                .reshape(&[batch, heads, klen, dim])
        };
        let ka = expand(k)?;
        let va = expand(v)?;
        let scores = qa
            .matmul(&ka.transpose(&[0, 1, 3, 2])?)?
            .mul(&Array::scalar(
                &self.stream,
                (dim as f32).sqrt().recip(),
                MlxDType::F32,
            )?)?;
        // MLX arange's float-valued ABI must represent positions exactly.
        if q_start.checked_add(qlen).is_none_or(|n| n > 16_777_216) || klen > 16_777_216 {
            return Err(err("causal positions exceed exact MLX arange domain"));
        }
        let qp = Array::arange(
            &self.stream,
            q_start as f32,
            (q_start + qlen) as f32,
            1.,
            MlxDType::I32,
        )?
        .reshape(&[qlen, 1])?;
        let kp =
            Array::arange(&self.stream, 0., klen as f32, 1., MlxDType::I32)?.reshape(&[1, klen])?;
        let scores = kp.less_equal(&qp)?.where_select(
            &scores,
            &Array::scalar(&self.stream, f32::NEG_INFINITY, MlxDType::F32)?,
        )?;
        let valid = scores.max(-1)?.isfinite()?;
        let zero = Array::scalar(&self.stream, 0., MlxDType::F32)?;
        let probabilities = valid.where_select(&scores, &zero)?.softmax(-1)?;
        let probabilities = valid.where_select(&probabilities, &zero)?;
        self.from_array(
            probabilities
                .matmul(&va)?
                .cast(mlx_dtype(q.dtype())?)?
                .transpose(&[0, 2, 1, 3])?,
        )
    }
}
impl Backend for MlxBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        if !eps.is_finite()
            || eps <= 0.
            || weight.shape().dims()
                != [input
                    .shape()
                    .dims()
                    .last()
                    .copied()
                    .ok_or_else(|| err("norm scalar unsupported"))?]
        {
            return Err(err("invalid RMSNorm contract"));
        }
        let (x, w) = self.same_float(input, weight)?;
        let x = x.cast(MlxDType::F32)?;
        let var = x.mul(&x)?.mean(&[-1], true)?;
        let norm = x.mul(
            &var.add(&Array::scalar(&self.stream, eps, MlxDType::F32)?)?
                .rsqrt()?,
        )?;
        self.from_array(
            norm.mul(&w.cast(MlxDType::F32)?)?
                .cast(mlx_dtype(input.dtype())?)?,
        )
    }
    fn silu(&self, t: &Tensor) -> Result<Tensor> {
        let a = self.float_array(t)?.cast(MlxDType::F32)?;
        self.from_array(a.mul(&a.sigmoid()?)?.cast(mlx_dtype(t.dtype())?)?)
    }
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(err("MLX adapter add requires equal shapes"));
        }
        let (x, y) = self.same_float(a, b)?;
        self.from_array(
            x.cast(MlxDType::F32)?
                .add(&y.cast(MlxDType::F32)?)?
                .cast(mlx_dtype(a.dtype())?)?,
        )
    }
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        if a.shape() != b.shape() {
            return Err(err("MLX adapter mul requires equal shapes"));
        }
        let (x, y) = self.same_float(a, b)?;
        self.from_array(
            x.cast(MlxDType::F32)?
                .mul(&y.cast(MlxDType::F32)?)?
                .cast(mlx_dtype(a.dtype())?)?,
        )
    }
    fn scale(&self, t: &Tensor, f: f32) -> Result<Tensor> {
        self.from_array(
            self.float_array(t)?
                .cast(MlxDType::F32)?
                .mul(&Array::scalar(&self.stream, f, MlxDType::F32)?)?
                .cast(mlx_dtype(t.dtype())?)?,
        )
    }
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let (x, y) = self.same_float(a, b)?;
        self.from_array(
            x.cast(MlxDType::F32)?
                .matmul(&y.cast(MlxDType::F32)?)?
                .cast(mlx_dtype(a.dtype())?)?,
        )
    }
    fn rope(&self, t: &Tensor, heads: usize, dim: usize, theta: f32, pos: u32) -> Result<Tensor> {
        let seq = t.shape().dims().first().copied().unwrap_or(0);
        if dim == 0
            || dim % 2 != 0
            || t.shape().dims() != [seq, heads, dim]
            || !theta.is_finite()
            || theta <= 0.
            || u64::from(pos) + seq as u64 > 16_777_216
        {
            return Err(err("invalid RoPE contract"));
        }
        let x = self.float_array(t)?.cast(MlxDType::F32)?;
        let frequency = Array::arange(&self.stream, 0., dim as f32, 2., MlxDType::F32)?
            .div(&Array::scalar(&self.stream, dim as f32, MlxDType::F32)?)?;
        let inv = Array::scalar(&self.stream, theta, MlxDType::F32)?
            .pow(&frequency)?
            .pow(&Array::scalar(&self.stream, -1., MlxDType::F32)?)?;
        let p = Array::arange(
            &self.stream,
            pos as f32,
            pos as f32 + seq as f32,
            1.,
            MlxDType::F32,
        )?
        .reshape(&[seq, 1, 1])?;
        let freq = p.mul(&inv.reshape(&[1, 1, dim / 2])?)?;
        let angles = Array::concat(&[&freq, &freq], 2)?;
        let rot = Array::concat(
            &[
                &x.slice_axis(2, dim / 2, dim)?.neg()?,
                &x.slice_axis(2, 0, dim / 2)?,
            ],
            2,
        )?;
        self.from_array(
            x.mul(&angles.cos()?)?
                .add(&rot.mul(&angles.sin()?)?)?
                .cast(mlx_dtype(t.dtype())?)?,
        )
    }
    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        if table.ndim() != 2
            || ids.is_empty()
            || ids
                .iter()
                .any(|&x| x as usize >= table.shape().dims()[0] || x > i32::MAX as u32)
        {
            return Err(err("invalid embedding ids/shape"));
        }
        let index = Array::from_i32(
            &self.stream,
            &[ids.len()],
            &ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        )?;
        self.from_array(self.array(table)?.take(&index, 0)?)
    }
    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        l: usize,
        h: usize,
        kh: usize,
        d: usize,
        n: usize,
        max: usize,
    ) -> Result<Tensor> {
        self.cached_attention(q, kv, l, h, kh, d, n, max)
    }
    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        l: usize,
        h: usize,
        kh: usize,
        d: usize,
        n: usize,
        max: usize,
    ) -> Result<Tensor> {
        self.cached_attention(q, kv, l, h, kh, d, n, max)
    }
    fn create_kv_cache(
        &self,
        layers: usize,
        heads: usize,
        dim: usize,
        max_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(MlxKvCache {
            backend: self.clone(),
            layers: vec![None; layers],
            pending: vec![None; layers],
            seq_len: 0,
            heads,
            dim,
            max_len,
            invalid: (layers == 0 || heads == 0 || dim == 0 || max_len == 0)
                .then(|| "invalid KV dimensions".into()),
        })
    }
    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        l: usize,
        k: &Tensor,
        v: &Tensor,
        n: usize,
    ) -> Result<()> {
        let cache = kv
            .as_any_mut()
            .downcast_mut::<MlxKvCache>()
            .ok_or_else(|| err("foreign KV cache"))?;
        if !cache.backend.stream.same(&self.stream) {
            return Err(err("KV belongs to another backend"));
        }
        cache.append(l, k, v, n)
    }
    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }
    fn begin_capture(&self) -> Result<()> {
        Err(Error::UnsupportedOp("CUDA-style graph capture on MLX"))
    }
    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Err(Error::UnsupportedOp("CUDA-style graph capture on MLX"))
    }
    fn device(&self) -> Device {
        Device::Metal(self.index)
    }
    fn to_device(&self, t: &Tensor) -> Result<Tensor> {
        if t.device() == self.device() {
            return self.from_array(self.array(t)?);
        }
        if t.device() != Device::Cpu {
            return Err(Error::UnsupportedDevice(t.device()));
        }
        let bytes = t
            .storage()
            .as_cpu()
            .ok_or_else(|| err("CPU tensor has no CPU bytes"))?;
        self.from_array(Array::from_bytes(
            &self.stream,
            t.shape().dims(),
            mlx_dtype(t.dtype())?,
            bytes,
        )?)
    }
    fn to_cpu(&self, t: &Tensor) -> Result<Tensor> {
        let a = self.array(t)?;
        Tensor::from_raw(t.shape().clone(), t.dtype(), Device::Cpu, a.to_bytes()?)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl MlxBackend {
    pub fn create_greedy_sampler(&self, spec: TokenSamplingSpec) -> Result<MlxGreedySampler> {
        spec.validate()?;
        Ok(MlxGreedySampler {
            backend: self.clone(),
            spec,
            remaining: None,
        })
    }
}
impl SamplingBackend for MlxBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        Ok(Box::new(self.create_greedy_sampler(spec)?))
    }
    fn create_normal_generator(&self, _output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        Err(Error::UnsupportedOp("MLX normal generator"))
    }
}
/// Greedy selection with exclusive ownership of the request capacity.
pub struct MlxGreedySampler {
    backend: MlxBackend,
    spec: TokenSamplingSpec,
    remaining: Option<usize>,
}
impl TokenSampler for MlxGreedySampler {
    fn spec(&self) -> TokenSamplingSpec {
        self.spec
    }
    fn begin(&mut self, init: TokenSamplingInit<'_>) -> Result<()> {
        self.remaining = None;
        init.params.validate(self.spec.vocab_size)?;
        if init.params.selection != TokenSelection::Greedy
            || init.params.penalties != TokenPenalties::default()
            || init.params.return_logprob
        {
            return Err(Error::UnsupportedOp("MLX sampling options beyond greedy"));
        }
        if init
            .prompt_token_ids
            .iter()
            .any(|&x| x as usize >= self.spec.vocab_size)
        {
            return Err(err("prompt token exceeds vocabulary"));
        }
        self.remaining = Some(
            self.spec
                .max_sequence_len
                .checked_sub(init.prompt_token_ids.len())
                .ok_or_else(|| err("prompt exceeds sampler capacity"))?,
        );
        Ok(())
    }
    fn sample(&mut self, logits: NextTokenLogits<'_>) -> Result<TokenSample> {
        let token = self.token_array(logits)?.to_u32_scalar()?;
        self.accept_token(token)
    }
}

impl MlxGreedySampler {
    fn token_array(&self, logits: NextTokenLogits<'_>) -> Result<Array> {
        let remaining = self
            .remaining
            .ok_or_else(|| err("sampler requires begin"))?;
        if remaining == 0 {
            return Err(err("sampler capacity exhausted"));
        }
        if logits.vocab_size() != self.spec.vocab_size {
            return Err(err("sampler vocabulary mismatch"));
        }
        let a = self
            .backend
            .array(logits.tensor())?
            .reshape(&[
                logits.tensor().numel() / self.spec.vocab_size,
                self.spec.vocab_size,
            ])?
            .slice_axis(0, logits.row_index(), logits.row_index() + 1)?
            .cast(MlxDType::F32)?;
        greedy_token_array(&a)
    }

    fn accept_token(&mut self, token: u32) -> Result<TokenSample> {
        if token == u32::MAX {
            return Err(err("all token logits are invalid"));
        }
        self.remaining = Some(self.remaining.expect("sample requires begin") - 1);
        Ok(TokenSample {
            token_id: token,
            logprob: None,
        })
    }

    /// Build selection without evaluating logits or consuming request capacity.
    pub fn plan(&mut self, logits: NextTokenLogits<'_>) -> Result<PlannedGreedy<'_>> {
        let token = self.token_array(logits)?.contiguous()?;
        Ok(PlannedGreedy {
            sampler: self,
            token,
        })
    }
}

/// A selection graph that exclusively borrows its sampler until completion.
pub struct PlannedGreedy<'a> {
    sampler: &'a mut MlxGreedySampler,
    token: Array,
}

/// A completed graph whose scalar result has not yet consumed sampler capacity.
pub struct EvaluatedGreedy<'a> {
    sampler: &'a mut MlxGreedySampler,
    token: Array,
}

impl<'a> PlannedGreedy<'a> {
    /// Evaluate selection and every required array in one synchronized call.
    pub fn evaluate(self, required: &[Array]) -> Result<EvaluatedGreedy<'a>> {
        let mut roots = Vec::with_capacity(required.len() + 1);
        roots.extend_from_slice(required);
        roots.push(self.token.clone());
        self.sampler.backend.stream.eval(&roots)?;
        Ok(EvaluatedGreedy {
            sampler: self.sampler,
            token: self.token,
        })
    }
}

impl EvaluatedGreedy<'_> {
    /// Read the completed scalar and consume capacity only for a valid token.
    pub fn finish(self) -> Result<TokenSample> {
        let bytes = self.token.to_bytes_ready()?;
        let token = u32::from_ne_bytes(
            bytes
                .try_into()
                .map_err(|_| err("invalid greedy scalar size"))?,
        );
        self.sampler.accept_token(token)
    }
}

fn greedy_token_array(a: &Array) -> Result<Array> {
    // Match core NaN/+inf/-inf normalization; argmax resolves ties to first.
    let adjusted = a.nan_to_num(f32::NEG_INFINITY, f32::MAX, f32::NEG_INFINITY)?;
    let valid = adjusted.max(-1)?.isfinite()?.reshape(&[1])?;
    let invalid = Array::from_i32(a.stream(), &[1], &[-1])?.cast(MlxDType::U32)?;
    valid.where_select(&adjusted.argmax(-1)?.reshape(&[1])?, &invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::{RngKey, TokenSamplingParams};

    fn cpu_sampler() -> Result<MlxGreedySampler> {
        MlxBackend {
            stream: Stream::cpu()?,
            index: 0,
        }
        .create_greedy_sampler(TokenSamplingSpec {
            vocab_size: 4,
            max_sequence_len: 3,
        })
    }

    fn begin(sampler: &mut MlxGreedySampler) -> Result<()> {
        sampler.begin(TokenSamplingInit {
            prompt_token_ids: &[0],
            params: &TokenSamplingParams::greedy(),
            rng: RngKey::default(),
        })
    }

    fn cpu_selection(
        sampler: &mut MlxGreedySampler,
        values: &[f32],
        joint: bool,
    ) -> Result<TokenSample> {
        let array = Array::from_f32(sampler.backend.stream(), &[1, 4], values)?;
        let token = greedy_token_array(&array)?;
        if joint {
            let required = array.add(&Array::scalar(array.stream(), 2., MlxDType::F32)?)?;
            let stream = array.stream().clone();
            let planned = PlannedGreedy {
                token: token.contiguous()?,
                sampler,
            };
            let before = stream.counters().eval_calls;
            let evaluated = planned.evaluate(&[array, required.clone()])?;
            assert_eq!(stream.counters().eval_calls, before + 1);
            let sample = evaluated.finish();
            assert_eq!(stream.counters().eval_calls, before + 2);
            let bytes = required.to_bytes_ready()?;
            assert_eq!(bytes.len(), 16);
            for (bytes, input) in bytes.chunks_exact(4).zip(values) {
                let actual = f32::from_ne_bytes(bytes.try_into().unwrap());
                let expected = input + 2.;
                assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
            }
            sample
        } else {
            sampler.accept_token(token.to_u32_scalar()?)
        }
    }

    #[test]
    fn greedy_joint_and_direct_share_nonfinite_tie_and_invalid_rules() -> Result<()> {
        for (values, expected) in [
            ([1., 4., 4., -2.], Some(1)),
            ([f32::NAN, 4., 4., f32::NEG_INFINITY], Some(1)),
            ([1., f32::INFINITY, f32::INFINITY, 2.], Some(1)),
            ([f32::NEG_INFINITY, -4., -4., f32::NAN], Some(1)),
            (
                [f32::NAN, f32::NEG_INFINITY, f32::NAN, f32::NEG_INFINITY],
                None,
            ),
        ] {
            for joint in [false, true] {
                let mut sampler = cpu_sampler()?;
                begin(&mut sampler)?;
                let sample = cpu_selection(&mut sampler, &values, joint);
                match expected {
                    Some(token) => {
                        assert_eq!(sample?.token_id, token);
                        assert_eq!(sampler.remaining, Some(1));
                    }
                    None => {
                        assert!(sample
                            .unwrap_err()
                            .to_string()
                            .contains("all token logits are invalid"));
                        assert_eq!(sampler.remaining, Some(2));
                        assert_eq!(
                            cpu_selection(&mut sampler, &[3., 1., 0., 0.], joint)?.token_id,
                            0
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn greedy_begin_capacity_and_reinitialization_are_shared() -> Result<()> {
        let mut sampler = cpu_sampler()?;
        let tensor = Tensor::from_f32(vec![1, 4], &[0.; 4])?;
        assert!(sampler.plan(NextTokenLogits::last(&tensor, 4)?).is_err());
        begin(&mut sampler)?;
        cpu_selection(&mut sampler, &[0., 1., 0., 0.], true)?;
        cpu_selection(&mut sampler, &[0., 0., 1., 0.], false)?;
        assert_eq!(sampler.remaining, Some(0));
        assert!(sampler
            .plan(NextTokenLogits::last(&tensor, 4)?)
            .err()
            .unwrap()
            .to_string()
            .contains("capacity exhausted"));
        assert!(sampler
            .sample(NextTokenLogits::last(&tensor, 4)?)
            .unwrap_err()
            .to_string()
            .contains("capacity exhausted"));
        for params in [
            TokenSamplingParams {
                return_logprob: true,
                ..TokenSamplingParams::greedy()
            },
            TokenSamplingParams {
                penalties: TokenPenalties {
                    repetition: 1.1,
                    ..TokenPenalties::default()
                },
                ..TokenSamplingParams::greedy()
            },
            TokenSamplingParams {
                selection: TokenSelection::Random {
                    temperature: 1.,
                    top_k: None,
                    top_p: 1.,
                },
                ..TokenSamplingParams::greedy()
            },
        ] {
            assert!(sampler
                .begin(TokenSamplingInit {
                    prompt_token_ids: &[0],
                    params: &params,
                    rng: RngKey::default()
                })
                .is_err());
            assert_eq!(sampler.remaining, None);
            begin(&mut sampler)?;
            assert_eq!(sampler.remaining, Some(2));
        }
        for prompt in [&[4][..], &[0, 0, 0, 0][..]] {
            assert!(sampler
                .begin(TokenSamplingInit {
                    prompt_token_ids: prompt,
                    params: &TokenSamplingParams::greedy(),
                    rng: RngKey::default()
                })
                .is_err());
            assert_eq!(sampler.remaining, None);
        }
        begin(&mut sampler)?;
        let smaller = Tensor::from_f32(vec![1, 3], &[0.; 3])?;
        assert!(sampler
            .plan(NextTokenLogits::last(&smaller, 3)?)
            .err()
            .unwrap()
            .to_string()
            .contains("vocabulary mismatch"));
        assert_eq!(sampler.remaining, Some(2));
        Ok(())
    }

    #[test]
    #[ignore = "requires exclusive Apple Silicon Metal test slot and pinned MLX SDK"]
    fn greedy_public_sampler_and_joint_plan_match_on_metal() -> Result<()> {
        let backend = MlxBackend::new(0)?;
        for values in [
            [1., 4., 4., -2.],
            [f32::NAN, 4., 4., f32::NEG_INFINITY],
            [1., f32::INFINITY, f32::INFINITY, 2.],
            [f32::NAN, f32::NEG_INFINITY, f32::NAN, f32::NEG_INFINITY],
        ] {
            let spec = TokenSamplingSpec {
                vocab_size: 4,
                max_sequence_len: 3,
            };
            let mut direct = backend.create_token_sampler(spec)?;
            let mut joint = backend.create_greedy_sampler(spec)?;
            let params = TokenSamplingParams::greedy();
            let init = TokenSamplingInit {
                prompt_token_ids: &[0],
                params: &params,
                rng: RngKey::default(),
            };
            direct.begin(init)?;
            joint.begin(init)?;
            let array = Array::from_f32(backend.stream(), &[1, 4], &values)?;
            let logits = backend.from_array(array.clone())?;
            let expected = direct
                .sample(NextTokenLogits::last(&logits, 4)?)
                .map(|x| x.token_id)
                .map_err(|e| e.to_string());
            let actual = joint
                .plan(NextTokenLogits::last(&logits, 4)?)?
                .evaluate(&[array])?
                .finish()
                .map(|x| x.token_id)
                .map_err(|e| e.to_string());
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn greedy_abandoned_or_failed_evaluation_does_not_consume_capacity() -> Result<()> {
        let mut sampler = cpu_sampler()?;
        begin(&mut sampler)?;
        let token = Array::from_i32(sampler.backend.stream(), &[1], &[1])?.cast(MlxDType::U32)?;
        drop(PlannedGreedy {
            token: token.clone(),
            sampler: &mut sampler,
        });
        assert_eq!(sampler.remaining, Some(2));
        let other = Stream::cpu()?;
        let foreign = Array::zeros(&other, &[1], MlxDType::F32)?;
        assert!(PlannedGreedy {
            token: token.clone(),
            sampler: &mut sampler
        }
        .evaluate(&[foreign])
        .is_err());
        assert_eq!(sampler.remaining, Some(2));
        drop(
            PlannedGreedy {
                token,
                sampler: &mut sampler,
            }
            .evaluate(&[])?,
        );
        assert_eq!(sampler.remaining, Some(2));
        Ok(())
    }
}

type Pair = (Array, Array);
struct MlxKvCache {
    backend: MlxBackend,
    layers: Vec<Option<Pair>>,
    pending: Vec<Option<Pair>>,
    seq_len: usize,
    heads: usize,
    dim: usize,
    max_len: usize,
    invalid: Option<String>,
}
impl MlxKvCache {
    fn valid(&self) -> Result<()> {
        if let Some(e) = &self.invalid {
            Err(err(e.clone()))
        } else {
            Ok(())
        }
    }
}
impl KvCache for MlxKvCache {
    fn append(&mut self, l: usize, k: &Tensor, v: &Tensor, n: usize) -> Result<()> {
        self.valid()?;
        if l >= self.layers.len()
            || n == 0
            || self.seq_len.checked_add(n).is_none_or(|v| v > self.max_len)
            || self.pending[l].is_some()
            || k.shape().dims() != [n, self.heads, self.dim]
            || v.shape() != k.shape()
            || v.dtype() != k.dtype()
        {
            return Err(err("invalid KV append"));
        }
        let k = self
            .backend
            .array(k)?
            .reshape(&[1, n, self.heads, self.dim])?;
        let v = self
            .backend
            .array(v)?
            .reshape(&[1, n, self.heads, self.dim])?;
        self.pending[l] = Some(if let Some((oldk, oldv)) = &self.layers[l] {
            (
                Array::concat(&[oldk, &k], 1)?,
                Array::concat(&[oldv, &v], 1)?,
            )
        } else {
            (k, v)
        });
        Ok(())
    }
    fn advance(&mut self, n: usize) {
        let result = (|| -> Result<()> {
            self.valid()?;
            let next = self
                .seq_len
                .checked_add(n)
                .filter(|&x| x <= self.max_len)
                .ok_or_else(|| err("KV advance exceeds capacity"))?;
            if self
                .pending
                .iter()
                .any(|x| x.as_ref().is_none_or(|p| p.0.shape()[1] != next))
            {
                return Err(err("all KV layers must append before advance"));
            }
            let all: Vec<_> = self
                .pending
                .iter()
                .filter_map(Option::as_ref)
                .flat_map(|(k, v)| [k.clone(), v.clone()])
                .collect();
            self.backend.stream.eval(&all)?;
            Ok(())
        })();
        match result {
            Ok(()) => {
                for (old, new) in self.layers.iter_mut().zip(&mut self.pending) {
                    *old = new.take()
                }
                self.seq_len += n
            }
            Err(e) => {
                self.invalid = Some(e.to_string());
                self.pending.fill(None)
            }
        }
    }
    fn seq_len(&self) -> usize {
        self.seq_len
    }
    fn clear(&mut self) -> Result<()> {
        self.layers.fill(None);
        self.pending.fill(None);
        self.seq_len = 0;
        self.invalid = None;
        Ok(())
    }
    fn n_layers(&self) -> usize {
        self.layers.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
