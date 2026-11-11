//! A typed builder for single-function FP16 ML Programs (specification 9,
//! opset `CoreML8`) over a fixed operation list. It names operations and
//! their constants as coremltools 9.0 `mb.program` does, so a program built
//! with the same calls decodes to the same message.
use crate::blob::{BlobDataType, BlobWriter};
use crate::proto::{self, binding, data_type, tensor_value, value};
use apxinf_core::{Error, Result};
use half::f16;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const OPSET: &str = "CoreML8";
const WEIGHTS: &str = "@model_path/weights/weight.bin";

/// A named program value: its MIL data type and static shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Var {
    name: String,
    dtype: i32,
    shape: Vec<usize>,
}

impl Var {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// The second operand of a binary operation.
pub enum Operand<'a> {
    Var(&'a Var),
    /// An FP16 scalar constant.
    Scalar(f16),
}

#[derive(Default)]
pub struct ProgramBuilder {
    inputs: Vec<Var>,
    operations: Vec<proto::Operation>,
    blobs: BlobWriter,
    counts: BTreeMap<&'static str, usize>,
    /// Names taken out of counter order (a fused operation's); counters skip them.
    reserved: BTreeSet<String>,
}

fn tensor_type(dtype: i32, shape: &[usize]) -> proto::ValueType {
    proto::ValueType {
        tensor_type: Some(proto::TensorType {
            data_type: dtype,
            rank: shape.len() as i64,
            dimensions: shape
                .iter()
                .map(|&size| proto::Dimension {
                    constant: Some(proto::ConstantDimension { size: size as u64 }),
                })
                .collect(),
        }),
    }
}

fn immediate(dtype: i32, shape: &[usize], tensor: tensor_value::Value) -> proto::Value {
    proto::Value {
        doc_string: String::new(),
        r#type: Some(tensor_type(dtype, shape)),
        value: Some(value::Value::ImmediateValue(proto::ImmediateValue {
            tensor: Some(proto::TensorValue {
                value: Some(tensor),
            }),
        })),
    }
}

fn string_value(s: &str) -> proto::Value {
    immediate(
        data_type::STRING,
        &[],
        tensor_value::Value::Strings(proto::RepeatedStrings {
            values: vec![s.to_owned()],
        }),
    )
}

fn by_name(var: &Var) -> proto::Argument {
    proto::Argument {
        arguments: vec![proto::Binding {
            binding: Some(binding::Binding::Name(var.name.clone())),
        }],
    }
}

fn contract(message: &'static str) -> Error {
    Error::Contract(message)
}

/// Numpy-style broadcast of two static shapes of equal rank.
fn broadcast(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    if b.is_empty() {
        return Ok(a.to_vec());
    }
    if a.len() != b.len() {
        return Err(contract(
            "ML Program binary operands need equal rank or a scalar",
        ));
    }
    a.iter()
        .zip(b)
        .map(|(&x, &y)| match (x, y) {
            (x, y) if x == y => Ok(x),
            (1, y) => Ok(y),
            (x, 1) => Ok(x),
            _ => Err(contract(
                "ML Program binary operand shapes do not broadcast",
            )),
        })
        .collect()
}

impl ProgramBuilder {
    pub fn new() -> Self {
        Self {
            blobs: BlobWriter::new(),
            ..Self::default()
        }
    }

    /// An FP16 function input.
    pub fn input(&mut self, name: &str, shape: &[usize]) -> Var {
        let var = Var {
            name: name.to_owned(),
            dtype: data_type::FLOAT16,
            shape: shape.to_vec(),
        };
        self.inputs.push(var.clone());
        var
    }

    fn op_name(&mut self, kind: &'static str, name: Option<&str>) -> String {
        match name {
            Some(name) => name.to_owned(),
            None => loop {
                let count = self.counts.entry(kind).or_default();
                *count += 1;
                let name = format!("{kind}_{}", *count - 1);
                if !self.reserved.contains(&name) {
                    break name;
                }
            },
        }
    }

    fn push(
        &mut self,
        kind: &str,
        name: &str,
        inputs: BTreeMap<String, proto::Argument>,
        outputs: &[Var],
    ) {
        self.operations.push(proto::Operation {
            r#type: kind.to_owned(),
            inputs,
            outputs: outputs
                .iter()
                .map(|v| proto::NamedValueType {
                    name: v.name.clone(),
                    r#type: Some(tensor_type(v.dtype, &v.shape)),
                })
                .collect(),
            attributes: [("name".to_owned(), string_value(name))].into(),
        });
    }

    /// A `const` operation named `{op}_{param}_0`.
    fn constant(
        &mut self,
        op: &str,
        param: &str,
        value: proto::Value,
        dtype: i32,
        shape: &[usize],
    ) -> Var {
        let name = format!("{op}_{param}_0");
        let var = Var {
            name: name.clone(),
            dtype,
            shape: shape.to_vec(),
        };
        self.operations.push(proto::Operation {
            r#type: "const".to_owned(),
            inputs: BTreeMap::new(),
            outputs: vec![proto::NamedValueType {
                name: name.clone(),
                r#type: Some(tensor_type(dtype, shape)),
            }],
            attributes: [
                ("name".to_owned(), string_value(&name)),
                ("val".to_owned(), value),
            ]
            .into(),
        });
        var
    }

    fn ints(&mut self, op: &str, param: &str, values: &[i32], scalar: bool) -> Var {
        let shape: &[usize] = if scalar { &[] } else { &[values.len()] };
        let value = immediate(
            data_type::INT32,
            shape,
            tensor_value::Value::Ints(proto::RepeatedInts {
                values: values.to_vec(),
            }),
        );
        self.constant(op, param, value, data_type::INT32, shape)
    }

    fn fp16_scalar(&mut self, op: &str, param: &str, value: f16) -> Var {
        let value = immediate(
            data_type::FLOAT16,
            &[],
            tensor_value::Value::Bytes(proto::RepeatedBytes {
                values: value.to_le_bytes().to_vec(),
            }),
        );
        self.constant(op, param, value, data_type::FLOAT16, &[])
    }

    fn flag(&mut self, op: &str, param: &str, value: bool) -> Var {
        let value = immediate(
            data_type::BOOL,
            &[],
            tensor_value::Value::Bools(proto::RepeatedBools {
                values: vec![value],
            }),
        );
        self.constant(op, param, value, data_type::BOOL, &[])
    }

    fn blob_value(
        &mut self,
        dtype: i32,
        shape: &[usize],
        kind: BlobDataType,
        bytes: &[u8],
    ) -> proto::Value {
        let offset = self.blobs.append(kind, bytes);
        proto::Value {
            doc_string: String::new(),
            r#type: Some(tensor_type(dtype, shape)),
            value: Some(value::Value::BlobFileValue(proto::BlobFileValue {
                file_name: WEIGHTS.to_owned(),
                offset,
            })),
        }
    }

    /// `constexpr_blockwise_shift_scale(data, scale)`: INT8 `data` times FP16
    /// `scale` broadcast over its unit dimensions, materialized at load.
    pub fn constexpr_blockwise_shift_scale(
        &mut self,
        data: &[i8],
        shape: &[usize],
        scale: &[f16],
        scale_shape: &[usize],
    ) -> Result<Var> {
        if data.len() != shape.iter().product::<usize>()
            || scale.len() != scale_shape.iter().product::<usize>()
            || broadcast(shape, scale_shape)? != shape
        {
            return Err(contract(
                "blockwise shift/scale data and scale must match their shapes and broadcast",
            ));
        }
        let name = self.op_name("constexpr_blockwise_shift_scale", None);
        let bytes: Vec<u8> = data.iter().map(|&v| v as u8).collect();
        let data = self.blob_value(data_type::INT8, shape, BlobDataType::Int8, &bytes);
        let bytes: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        let scale = self.blob_value(
            data_type::FLOAT16,
            scale_shape,
            BlobDataType::Float16,
            &bytes,
        );
        let arg = |v: proto::Value| proto::Argument {
            arguments: vec![proto::Binding {
                binding: Some(binding::Binding::Value(v)),
            }],
        };
        let out = Var {
            name: name.clone(),
            dtype: data_type::FLOAT16,
            shape: shape.to_vec(),
        };
        let inputs = [
            ("data".to_owned(), arg(data)),
            ("scale".to_owned(), arg(scale)),
        ]
        .into();
        self.push(
            "constexpr_blockwise_shift_scale",
            &name,
            inputs,
            std::slice::from_ref(&out),
        );
        Ok(out)
    }

    /// `constexpr_lut_to_dense(indices, lut)`: `bits`-wide palette indices into a
    /// per-group FP16 lookup table, materialized at load. `lut` holds one table of
    /// `1 << bits` entries per group of `shape[0] / groups` rows; `indices` is packed
    /// least-significant-first with no padding between rows.
    ///
    /// Core ML keeps per-block INT8/INT4 (`constexpr_blockwise_shift_scale` with a
    /// sub-row block) off the Neural Engine, but places this one on it, which is what
    /// makes sub-8-bit ANE weights reachable at all.
    pub fn constexpr_lut_to_dense(
        &mut self,
        indices: &[u8],
        shape: &[usize],
        lut: &[f16],
        groups: usize,
        bits: u32,
    ) -> Result<Var> {
        let count: usize = shape.iter().product();
        let entries = 1usize << bits;
        if !matches!(bits, 4 | 6) {
            return Err(contract("lut_to_dense supports 4- and 6-bit palettes"));
        }
        if indices.len() != count || lut.len() != groups * entries {
            return Err(contract("lut_to_dense needs one index per weight and one table per group"));
        }
        if groups == 0 || shape.is_empty() || shape[0] % groups != 0 {
            return Err(contract("lut_to_dense groups must divide the first dimension"));
        }
        if indices.iter().any(|&i| usize::from(i) >= entries) {
            return Err(contract("lut_to_dense index outside the palette"));
        }
        let name = self.op_name("constexpr_lut_to_dense", None);
        let (mil_dtype, blob_dtype) = match bits {
            4 => (data_type::UINT4, BlobDataType::UInt4),
            _ => (data_type::UINT6, BlobDataType::UInt6),
        };
        let packed = pack_indices(indices, bits);
        let indices = self.blob_value(mil_dtype, shape, blob_dtype, &packed);
        // Core ML's layout: one table per group, then the palette, then the vector axis.
        let mut lut_shape = vec![groups];
        lut_shape.extend(std::iter::repeat(1).take(shape.len() - 1));
        lut_shape.push(entries);
        lut_shape.push(1);
        let bytes: Vec<u8> = lut.iter().flat_map(|v| v.to_le_bytes()).collect();
        let lut = self.blob_value(data_type::FLOAT16, &lut_shape, BlobDataType::Float16, &bytes);
        let arg = |v: proto::Value| proto::Argument {
            arguments: vec![proto::Binding {
                binding: Some(binding::Binding::Value(v)),
            }],
        };
        let out = Var {
            name: name.clone(),
            dtype: data_type::FLOAT16,
            shape: shape.to_vec(),
        };
        let inputs = [
            ("indices".to_owned(), arg(indices)),
            ("lut".to_owned(), arg(lut)),
        ]
        .into();
        self.push(
            "constexpr_lut_to_dense",
            &name,
            inputs,
            std::slice::from_ref(&out),
        );
        Ok(out)
    }

    /// `matmul(x, y)` with both operands already in place, no transpose.
    ///
    /// This exists so a schedule search can ask the compiler the one question the
    /// conv path cannot: whether the same arithmetic spelled as a matmul is still
    /// placed on the Neural Engine. At batch 1 it measured entirely on the CPU
    /// while an equivalent 1x1 convolution reached the engine, so the spelling is
    /// a placement decision, not a style choice.
    pub fn matmul(&mut self, x: &Var, y: &Var) -> Result<Var> {
        let (&[xm, xk], &[yk, yn]) = (&x.shape[..], &y.shape[..]) else {
            return Err(contract("matmul needs two rank-2 operands"));
        };
        if xk != yk {
            return Err(contract("matmul inner dimensions must match"));
        }
        let name = self.op_name("matmul", None);
        let flag = |b: &mut Self, param: &'static str| -> Var {
            let v = immediate(
                data_type::BOOL,
                &[],
                tensor_value::Value::Bools(proto::RepeatedBools {
                    values: vec![false],
                }),
            );
            b.constant(&name, param, v, data_type::BOOL, &[])
        };
        let tx = flag(self, "transpose_x");
        let ty = flag(self, "transpose_y");
        let out = Var {
            name: name.clone(),
            dtype: data_type::FLOAT16,
            shape: vec![xm, yn],
        };
        let inputs = [("x", x), ("y", y), ("transpose_x", &tx), ("transpose_y", &ty)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("matmul", &name, inputs, std::slice::from_ref(&out));
        Ok(out)
    }

    /// A `const` FP16 tensor stored in the weight file, named `{op}_{param}_0`.
    fn fp16_blob(&mut self, op: &str, param: &str, values: &[f16], shape: &[usize]) -> Result<Var> {
        if values.len() != shape.iter().product::<usize>() {
            return Err(contract("FP16 constant values must match their shape"));
        }
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let value = self.blob_value(data_type::FLOAT16, shape, BlobDataType::Float16, &bytes);
        Ok(self.constant(op, param, value, data_type::FLOAT16, shape))
    }

    /// The output shape of a conv of `x` with a `weight_shape` kernel.
    fn conv_shape(x: &Var, weight_shape: &[usize]) -> Result<Vec<usize>> {
        let (&[n, cin, h, w], &[cout, wcin, kh, kw]) = (&x.shape[..], weight_shape) else {
            return Err(contract("conv needs rank-4 input and weight"));
        };
        if cin != wcin || kh > h || kw > w {
            return Err(contract("conv weight does not fit the input"));
        }
        Ok(vec![n, cout, h - kh + 1, w - kw + 1])
    }

    /// The stride, padding, dilation and group constants of conv `conv`.
    fn conv_params(&mut self, conv: &str) -> [(&'static str, Var); 5] {
        let strides = self.ints(conv, "strides", &[1, 1], false);
        let pad_type = immediate(
            data_type::STRING,
            &[],
            tensor_value::Value::Strings(proto::RepeatedStrings {
                values: vec!["valid".to_owned()],
            }),
        );
        let pad_type = self.constant(conv, "pad_type", pad_type, data_type::STRING, &[]);
        let pad = self.ints(conv, "pad", &[0, 0, 0, 0], false);
        let dilations = self.ints(conv, "dilations", &[1, 1], false);
        let groups = self.ints(conv, "groups", &[1], true);
        [
            ("strides", strides),
            ("pad_type", pad_type),
            ("pad", pad),
            ("dilations", dilations),
            ("groups", groups),
        ]
    }

    fn conv_op(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        tensors: &[(&str, &Var)],
        params: &[(&'static str, Var)],
    ) -> Var {
        let out = Var {
            name: name.to_owned(),
            dtype: data_type::FLOAT16,
            shape,
        };
        let inputs = tensors
            .iter()
            .map(|&(k, v)| (k, v))
            .chain(params.iter().map(|(k, v)| (*k, v)))
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("conv", name, inputs, std::slice::from_ref(&out));
        out
    }

    /// 2-D convolution, stride 1, `valid` padding, dilation 1, one group.
    pub fn conv(&mut self, x: &Var, weight: &Var) -> Result<Var> {
        let shape = Self::conv_shape(x, &weight.shape)?;
        let name = self.op_name("conv", None);
        let params = self.conv_params(&name);
        Ok(self.conv_op(&name, shape, &[("x", x), ("weight", weight)], &params))
    }

    /// [`conv`](Self::conv) with FP16 weights in the weight file, as
    /// coremltools stores a NumPy FP16 weight: `{conv}_weight_0` comes first.
    pub fn conv_fp16(&mut self, x: &Var, weight: &[f16], weight_shape: &[usize]) -> Result<Var> {
        let shape = Self::conv_shape(x, weight_shape)?;
        let name = self.op_name("conv", None);
        let w = self.fp16_blob(&name, "weight", weight, weight_shape)?;
        let params = self.conv_params(&name);
        Ok(self.conv_op(&name, shape, &[("x", x), ("weight", &w)], &params))
    }

    /// `mul(conv_fp16(x, weight), scale)` after coremltools 9.0's
    /// `fuse_conv_scale` pass: one conv named `name` (the absorbed `mul`'s
    /// name, reserved so counters skip it) with weight `weight * scale` and a
    /// zero bias, both FP16. The conv's parameter constants keep the conv
    /// counter's name.
    pub fn conv_fp16_scaled(
        &mut self,
        x: &Var,
        weight: &[f16],
        weight_shape: &[usize],
        scale: f16,
        name: &str,
    ) -> Result<Var> {
        let shape = Self::conv_shape(x, weight_shape)?;
        self.reserved.insert(name.to_owned());
        let conv = self.op_name("conv", None);
        let params = self.conv_params(&conv);
        let scaled: Vec<f16> = weight
            .iter()
            .map(|w| f16::from_f32(w.to_f32() * scale.to_f32()))
            .collect();
        let w = self.fp16_blob(name, "weight", &scaled, weight_shape)?;
        let b = self.fp16_blob(
            name,
            "bias",
            &vec![f16::ZERO; weight_shape[0]],
            &weight_shape[..1],
        )?;
        Ok(self.conv_op(
            name,
            shape,
            &[("x", x), ("weight", &w), ("bias", &b)],
            &params,
        ))
    }

    fn binary(
        &mut self,
        kind: &'static str,
        x: &Var,
        y: Operand<'_>,
        name: Option<&str>,
    ) -> Result<Var> {
        let name = self.op_name(kind, name);
        let y = match y {
            Operand::Var(v) => v.clone(),
            Operand::Scalar(s) => self.fp16_scalar(&name, "y", s),
        };
        let out = Var {
            name: name.clone(),
            dtype: data_type::FLOAT16,
            shape: broadcast(&x.shape, &y.shape)?,
        };
        let inputs = [("x".to_owned(), by_name(x)), ("y".to_owned(), by_name(&y))].into();
        self.push(kind, &name, inputs, std::slice::from_ref(&out));
        Ok(out)
    }

    pub fn mul(&mut self, x: &Var, y: Operand<'_>, name: Option<&str>) -> Result<Var> {
        self.binary("mul", x, y, name)
    }
    pub fn add(&mut self, x: &Var, y: Operand<'_>, name: Option<&str>) -> Result<Var> {
        self.binary("add", x, y, name)
    }
    pub fn real_div(&mut self, x: &Var, y: Operand<'_>, name: Option<&str>) -> Result<Var> {
        self.binary("real_div", x, y, name)
    }
    pub fn maximum(&mut self, x: &Var, y: Operand<'_>, name: Option<&str>) -> Result<Var> {
        self.binary("maximum", x, y, name)
    }

    fn unary(&mut self, kind: &'static str, x: &Var, name: Option<&str>) -> Var {
        let name = self.op_name(kind, name);
        let out = Var {
            name: name.clone(),
            ..x.clone()
        };
        self.push(
            kind,
            &name,
            [("x".to_owned(), by_name(x))].into(),
            std::slice::from_ref(&out),
        );
        out
    }

    pub fn tanh(&mut self, x: &Var) -> Var {
        self.unary("tanh", x, None)
    }
    pub fn abs(&mut self, x: &Var) -> Var {
        self.unary("abs", x, None)
    }
    pub fn exp(&mut self, x: &Var) -> Var {
        self.unary("exp", x, None)
    }
    /// `rsqrt(x)` with MIL's default `epsilon` of 1e-12, which is zero in FP16.
    /// The ML Program parser refuses an `rsqrt` that omits it.
    pub fn rsqrt(&mut self, x: &Var) -> Var {
        let name = self.op_name("rsqrt", None);
        let epsilon = self.fp16_scalar(&name, "epsilon", f16::from_f32(1e-12));
        let out = Var {
            name: name.clone(),
            ..x.clone()
        };
        let inputs = [("x", x), ("epsilon", &epsilon)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("rsqrt", &name, inputs, std::slice::from_ref(&out));
        out
    }
    pub fn identity(&mut self, x: &Var, name: Option<&str>) -> Var {
        self.unary("identity", x, name)
    }

    /// `reduce_max` over `axes` (non-negative).
    pub fn reduce_max(&mut self, x: &Var, axes: &[i32], keep_dims: bool) -> Result<Var> {
        self.reduce("reduce_max", x, axes, keep_dims)
    }
    /// `reduce_sum` over `axes` (non-negative).
    pub fn reduce_sum(&mut self, x: &Var, axes: &[i32], keep_dims: bool) -> Result<Var> {
        self.reduce("reduce_sum", x, axes, keep_dims)
    }

    fn reduce(
        &mut self,
        kind: &'static str,
        x: &Var,
        axes: &[i32],
        keep_dims: bool,
    ) -> Result<Var> {
        if axes
            .iter()
            .any(|&a| usize::try_from(a).map_or(true, |a| a >= x.shape.len()))
        {
            return Err(contract("reduce axes must index the input"));
        }
        let name = self.op_name(kind, None);
        let axes_var = self.ints(&name, "axes", axes, false);
        let keep = self.flag(&name, "keep_dims", keep_dims);
        let shape = x
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| match axes.contains(&(i as i32)) {
                true if keep_dims => Some(1),
                true => None,
                false => Some(d),
            })
            .collect();
        let out = Var {
            name: name.clone(),
            dtype: x.dtype,
            shape,
        };
        let inputs = [("x", x), ("axes", &axes_var), ("keep_dims", &keep)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push(kind, &name, inputs, std::slice::from_ref(&out));
        Ok(out)
    }

    /// `transpose` of `x`: output axis `i` is input axis `perm[i]`.
    pub fn transpose(&mut self, x: &Var, perm: &[i32]) -> Result<Var> {
        let Some(axes) = perm
            .iter()
            .map(|&p| usize::try_from(p).ok().filter(|&p| p < x.shape.len()))
            .collect::<Option<Vec<usize>>>()
        else {
            return Err(contract("transpose perm must index the input"));
        };
        if axes.len() != x.shape.len() || axes.iter().collect::<BTreeSet<_>>().len() != axes.len()
        {
            return Err(contract("transpose perm must be a permutation of the input axes"));
        }
        let name = self.op_name("transpose", None);
        let perm_var = self.ints(&name, "perm", perm, false);
        let out = Var {
            name: name.clone(),
            dtype: x.dtype,
            shape: axes.iter().map(|&a| x.shape[a]).collect(),
        };
        let inputs = [("x", x), ("perm", &perm_var)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("transpose", &name, inputs, std::slice::from_ref(&out));
        Ok(out)
    }

    /// `concat` of `values` along a non-negative `axis`.
    pub fn concat(&mut self, values: &[&Var], axis: usize, interleave: bool) -> Result<Var> {
        let Some(first) = values.first() else {
            return Err(contract("concat needs at least one value"));
        };
        if axis >= first.shape.len() {
            return Err(contract("concat axis must index the values"));
        }
        if values.iter().any(|v| v.dtype != first.dtype) {
            return Err(contract("concat values must share a dtype"));
        }
        let agrees = |v: &Var| {
            v.shape.len() == first.shape.len()
                && v.shape
                    .iter()
                    .zip(&first.shape)
                    .enumerate()
                    .all(|(i, (a, b))| i == axis || a == b)
        };
        if !values.iter().all(|v| agrees(v)) {
            return Err(contract(
                "concat values must agree on every axis except the concat axis",
            ));
        }
        let name = self.op_name("concat", None);
        let axis_var = self.ints(&name, "axis", &[axis as i32], true);
        let interleave = self.flag(&name, "interleave", interleave);
        let mut shape = first.shape.clone();
        shape[axis] = values.iter().map(|v| v.shape[axis]).sum();
        let out = Var {
            name: name.clone(),
            dtype: first.dtype,
            shape,
        };
        let list = proto::Argument {
            arguments: values.iter().flat_map(|v| by_name(v).arguments).collect(),
        };
        let inputs = [
            ("values".to_owned(), list),
            ("axis".to_owned(), by_name(&axis_var)),
            ("interleave".to_owned(), by_name(&interleave)),
        ]
        .into();
        self.push("concat", &name, inputs, std::slice::from_ref(&out));
        Ok(out)
    }

    /// MIL `select`, numpy's `where`: `a` where the bool `cond` holds, else `b`.
    pub fn select(&mut self, cond: &Var, a: &Var, b: &Var) -> Result<Var> {
        if cond.dtype != data_type::BOOL {
            return Err(contract("select cond must be a bool value"));
        }
        if a.dtype != b.dtype {
            return Err(contract("select branches must share a dtype"));
        }
        let shape = broadcast(&broadcast(&cond.shape, &a.shape)?, &b.shape)?;
        let name = self.op_name("select", None);
        let out = Var {
            name: name.clone(),
            dtype: a.dtype,
            shape,
        };
        let inputs = [("cond", cond), ("a", a), ("b", b)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("select", &name, inputs, std::slice::from_ref(&out));
        Ok(out)
    }

    /// `split` along a non-negative `axis` into parts of `sizes`.
    pub fn split(&mut self, x: &Var, sizes: &[usize], axis: usize) -> Result<Vec<Var>> {
        if axis >= x.shape.len() || sizes.iter().sum::<usize>() != x.shape[axis] || sizes.is_empty()
        {
            return Err(contract("split sizes must add up to the axis"));
        }
        let name = self.op_name("split", None);
        let sizes_i32: Vec<i32> = sizes.iter().map(|&s| s as i32).collect();
        let sizes_var = self.ints(&name, "split_sizes", &sizes_i32, false);
        let axis_var = self.ints(&name, "axis", &[axis as i32], true);
        let outs: Vec<Var> = sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let mut shape = x.shape.clone();
                shape[axis] = s;
                Var {
                    name: format!("{name}_{i}"),
                    dtype: x.dtype,
                    shape,
                }
            })
            .collect();
        let inputs = [("x", x), ("split_sizes", &sizes_var), ("axis", &axis_var)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), by_name(v)))
            .collect();
        self.push("split", &name, inputs, &outs);
        Ok(outs)
    }

    /// The package: one `main` function returning `outputs`, all FP16.
    pub fn finish(self, outputs: &[&Var]) -> Result<Package> {
        if outputs.is_empty() || outputs.iter().any(|v| v.dtype != data_type::FLOAT16) {
            return Err(contract("ML Program outputs must be FP16 values"));
        }
        let feature = |v: &Var| proto::FeatureDescription {
            name: v.name.clone(),
            short_description: String::new(),
            r#type: Some(proto::FeatureType {
                multi_array_type: Some(proto::ArrayFeatureType {
                    shape: v.shape.iter().map(|&d| d as i64).collect(),
                    data_type: proto::ARRAY_FLOAT16,
                }),
                is_optional: false,
            }),
        };
        let block = proto::Block {
            inputs: Vec::new(),
            outputs: outputs.iter().map(|v| v.name.clone()).collect(),
            operations: self.operations,
            attributes: BTreeMap::new(),
        };
        let function = proto::Function {
            inputs: self
                .inputs
                .iter()
                .map(|v| proto::NamedValueType {
                    name: v.name.clone(),
                    r#type: Some(tensor_type(v.dtype, &v.shape)),
                })
                .collect(),
            opset: OPSET.to_owned(),
            block_specializations: [(OPSET.to_owned(), block)].into(),
            attributes: BTreeMap::new(),
        };
        let model = proto::Model {
            specification_version: 9,
            description: Some(proto::ModelDescription {
                input: self.inputs.iter().map(feature).collect(),
                output: outputs.iter().map(|v| feature(v)).collect(),
                metadata: Some(proto::Metadata {
                    short_description: String::new(),
                    user_defined: [(
                        "com.github.apxinf.writer".to_owned(),
                        concat!("apxinf-coreml ", env!("CARGO_PKG_VERSION")).to_owned(),
                    )]
                    .into(),
                }),
            }),
            is_updatable: false,
            ml_program: Some(proto::Program {
                version: 1,
                functions: [("main".to_owned(), function)].into(),
                doc_string: String::new(),
                attributes: BTreeMap::new(),
            }),
        };
        Ok(Package {
            model,
            weights: self.blobs.finish(),
        })
    }
}

/// An ML Program package: the specification and its weight file.
pub struct Package {
    pub model: proto::Model,
    pub weights: Vec<u8>,
}

const MANIFEST: &str = r#"{
    "fileFormatVersion": "1.0.0",
    "itemInfoEntries": {
        "6C1A3E50-63A5-4E0F-9B53-0D5B4E1E7A01": {
            "author": "com.apple.CoreML",
            "description": "CoreML Model Specification",
            "name": "model.mlmodel",
            "path": "com.apple.CoreML/model.mlmodel"
        },
        "6C1A3E50-63A5-4E0F-9B53-0D5B4E1E7A02": {
            "author": "com.apple.CoreML",
            "description": "CoreML Model Weights",
            "name": "weights",
            "path": "com.apple.CoreML/weights"
        }
    },
    "rootModelIdentifier": "6C1A3E50-63A5-4E0F-9B53-0D5B4E1E7A01"
}
"#;

impl Package {
    pub fn model_bytes(&self) -> Vec<u8> {
        prost::Message::encode_to_vec(&self.model)
    }

    /// Writes `dir` as an `.mlpackage` directory. `dir` must not exist.
    pub fn write(&self, dir: &Path) -> Result<()> {
        let io = |e: std::io::Error| Error::Other(format!("cannot write ML Program package: {e}"));
        let data = dir.join("Data/com.apple.CoreML");
        std::fs::create_dir_all(data.join("weights")).map_err(io)?;
        std::fs::write(dir.join("Manifest.json"), MANIFEST).map_err(io)?;
        std::fs::write(data.join("model.mlmodel"), self.model_bytes()).map_err(io)?;
        std::fs::write(data.join("weights/weight.bin"), &self.weights).map_err(io)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// The spelling axis, at a production shape: does the same arithmetic written
    /// as `matmul` still reach the Neural Engine?
    ///
    /// At batch 1 a 4096x4096 GEMV measured ANE 0/3 as `matmul` and `linear` but
    /// ANE 3/3 as a 1x1 convolution, which is why every ANE-friendly layout
    /// converts projections to convolutions. This asserts the same question at a
    /// tile the engine is actually given, so a schedule search can trust the axis.
    ///
    ///   cargo test -p apxinf-coreml --release --features native \
    ///     spelling_placement -- --ignored --nocapture
    #[test]
    #[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
    #[ignore = "needs the native compiler and the host Metal lock"]
    fn spelling_placement() {
        use crate::{ComputeDeviceKind, ComputeUnits, ProgramCache};
        const N: usize = 1024;
        const K: usize = 1024;
        let w: Vec<f16> = (0..N * K)
            .map(|i| f16::from_f32((((i * 2654435761) % 1000) as f32 / 500.0 - 1.0) * 0.02))
            .collect();

        for tile in [1usize, 128, 512] {
            for spelling in ["conv1x1", "matmul"] {
                let mut b = ProgramBuilder::new();
                let y = if spelling == "matmul" {
                    let x = b.input("x", &[tile, K]);
                    let mut t = vec![f16::ZERO; N * K];
                    for r in 0..N {
                        for c in 0..K {
                            t[c * N + r] = w[r * K + c];
                        }
                    }
                    let wv = b.fp16_blob("w", "value", &t, &[K, N]).unwrap();
                    let m = b.matmul(&x, &wv).unwrap();
                    b.identity(&m, Some("y"))
                } else {
                    let x = b.input("x", &[1, K, 1, tile]);
                    let c = b.conv_fp16(&x, &w, &[N, K, 1, 1]).unwrap();
                    b.identity(&c, Some("y"))
                };
                let pkg = b.finish(&[&y]).unwrap();
                let dir = std::env::temp_dir().join(format!("spelling-{spelling}-{tile}"));
                let _ = std::fs::remove_dir_all(&dir);
                let cache = ProgramCache::new(&dir);
                let mut devices: Vec<String> = Vec::new();
                let loaded = cache.load_with_placement(
                    &pkg,
                    ComputeUnits::CpuAndNeuralEngine,
                    std::time::Duration::from_secs(120),
                    |facts| {
                        for f in &facts.functions {
                            for op in &f.operations {
                                if let Some(d) = op.usage.as_ref().and_then(|u| u.preferred.as_ref())
                                {
                                    devices.push(format!("{}:{:?}", op.operator, d.kind));
                                }
                            }
                        }
                        Ok(())
                    },
                );
                let ane = devices
                    .iter()
                    .filter(|d| d.ends_with("NeuralEngine"))
                    .count();
                println!(
                    "  tile={tile:<4} {spelling:<8} -> {}  ANE {ane}/{}  {devices:?}",
                    if loaded.is_ok() { "loaded" } else { "FAILED" },
                    devices.len()
                );
                let _ = std::fs::remove_dir_all(&dir);
            }
        }
    }

    use super::*;

    #[test]
    fn names_follow_coremltools_counters_and_named_ops() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, 4, 1, 8]);
        let w = b
            .constexpr_blockwise_shift_scale(&[1; 8], &[2, 4, 1, 1], &[f16::ONE; 2], &[2, 1, 1, 1])
            .unwrap();
        let c = b.conv(&x, &w).unwrap();
        assert_eq!((c.name(), c.shape()), ("conv_0", &[1, 2, 1, 8][..]));
        let m = b
            .mul(&c, Operand::Scalar(f16::from_f32(0.5)), None)
            .unwrap();
        let s = b
            .maximum(&m, Operand::Scalar(f16::ONE), Some("gate_scale"))
            .unwrap();
        let m2 = b.mul(&m, Operand::Var(&s), None).unwrap();
        let r = b.reduce_max(&m2, &[1], true).unwrap();
        let parts = b.split(&m2, &[1, 1], 1).unwrap();
        let y = b.identity(&parts[0], Some("y"));
        assert_eq!(
            [
                m.name(),
                s.name(),
                m2.name(),
                r.name(),
                parts[1].name(),
                y.name()
            ],
            [
                "mul_0",
                "gate_scale",
                "mul_1",
                "reduce_max_0",
                "split_0_1",
                "y"
            ]
        );
        assert_eq!(r.shape(), &[1, 1, 1, 8]);
        let names: Vec<String> = b
            .operations
            .iter()
            .map(|op| op.outputs[0].name.clone())
            .collect();
        assert_eq!(
            &names[..8],
            [
                "constexpr_blockwise_shift_scale_0",
                "conv_0_strides_0",
                "conv_0_pad_type_0",
                "conv_0_pad_0",
                "conv_0_dilations_0",
                "conv_0_groups_0",
                "conv_0",
                "mul_0_y_0"
            ]
        );
        let package = b.finish(&[&y]).unwrap();
        let decoded: proto::Model = prost::Message::decode(&package.model_bytes()[..]).unwrap();
        assert_eq!(decoded, package.model);
    }

    /// The order and names of coremltools 9.0's FP16 conv programs: a NumPy FP16
    /// conv stores its weight first; a fused `conv * 0.5` keeps the conv's
    /// parameter names, takes the mul's name and adds a zero bias.
    #[test]
    fn fp16_convs_follow_coremltools_storage_and_fusion() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, 4, 1, 8]);
        let w = [f16::from_f32(3.0); 8];
        let fused = b
            .conv_fp16_scaled(&x, &w, &[2, 4, 1, 1], f16::from_f32(0.5), "mul_0")
            .unwrap();
        let plain = b.conv_fp16(&x, &w, &[2, 4, 1, 1]).unwrap();
        let m = b.mul(&fused, Operand::Var(&plain), None).unwrap();
        assert_eq!(
            (fused.name(), plain.name(), m.name()),
            ("mul_0", "conv_1", "mul_1")
        );
        let names: Vec<&str> = b
            .operations
            .iter()
            .map(|op| op.outputs[0].name.as_str())
            .collect();
        assert_eq!(
            &names[..14],
            [
                "conv_0_strides_0",
                "conv_0_pad_type_0",
                "conv_0_pad_0",
                "conv_0_dilations_0",
                "conv_0_groups_0",
                "mul_0_weight_0",
                "mul_0_bias_0",
                "mul_0",
                "conv_1_weight_0",
                "conv_1_strides_0",
                "conv_1_pad_type_0",
                "conv_1_pad_0",
                "conv_1_dilations_0",
                "conv_1_groups_0",
            ]
        );
        assert!(b.operations[7].inputs.contains_key("bias"));
        let offsets: Vec<u64> = [5, 6, 8]
            .iter()
            .map(|&i| match &b.operations[i].attributes["val"].value {
                Some(value::Value::BlobFileValue(blob)) => blob.offset,
                _ => panic!("{} is not stored in the weight file", names[i]),
            })
            .collect();
        let package = b.finish(&[&m]).unwrap();
        // In operation order: the scaled weight (1.5), the zero bias, the plain weight (3.0).
        let halves = |offset: u64| -> Vec<f32> {
            let (dtype, bytes) = crate::blob::read(&package.weights, offset).unwrap();
            assert_eq!(dtype, BlobDataType::Float16 as u32);
            bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        };
        assert!(offsets[0] < offsets[1] && offsets[1] < offsets[2]);
        assert_eq!(halves(offsets[0]), [1.5; 8]);
        assert_eq!(halves(offsets[1]), [0.0; 2]);
        assert_eq!(halves(offsets[2]), [3.0; 8]);
    }

    fn op<'a>(b: &'a ProgramBuilder, out: &Var) -> &'a proto::Operation {
        b.operations
            .iter()
            .find(|op| op.outputs.iter().any(|o| o.name == out.name))
            .unwrap()
    }

    fn bound(op: &proto::Operation, param: &str) -> Vec<String> {
        op.inputs[param]
            .arguments
            .iter()
            .map(|a| match &a.binding {
                Some(binding::Binding::Name(n)) => n.clone(),
                _ => panic!("{param} is not bound by name"),
            })
            .collect()
    }

    fn bool_var(shape: &[usize]) -> Var {
        Var {
            name: "cond".to_owned(),
            dtype: data_type::BOOL,
            shape: shape.to_vec(),
        }
    }

    #[test]
    fn math_and_layout_ops_emit_their_mil_ops_and_shapes() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, 8, 16, 4]);
        let e = b.exp(&x);
        let r = b.rsqrt(&e);
        let kept = b.reduce_sum(&r, &[1, 3], true).unwrap();
        let dropped = b.reduce_sum(&r, &[3], false).unwrap();
        let t = b.transpose(&x, &[2, 0, 3, 1]).unwrap();
        let y = b.input("y", &[1, 8, 16, 2]);
        let c = b.concat(&[&x, &y, &e], 3, false).unwrap();
        let cond = bool_var(&[1, 8, 16, 1]);
        let s = b.select(&cond, &x, &e).unwrap();

        let emitted = [&e, &r, &kept, &dropped, &t, &c, &s].map(|v| op(&b, v).r#type.as_str());
        assert_eq!(
            emitted,
            ["exp", "rsqrt", "reduce_sum", "reduce_sum", "transpose", "concat", "select"]
        );
        assert_eq!((e.shape(), r.shape()), (&[1, 8, 16, 4][..], &[1, 8, 16, 4][..]));
        assert_eq!(kept.shape(), &[1, 1, 16, 1]);
        assert_eq!(dropped.shape(), &[1, 8, 16]);
        // perm [2, 0, 3, 1] applied, not its inverse (which would give [8, 4, 1, 16]).
        assert_eq!(t.shape(), &[16, 1, 4, 8]);
        assert_eq!(c.shape(), &[1, 8, 16, 10]);
        assert_eq!(s.shape(), &[1, 8, 16, 4]);

        let perm = bound(op(&b, &t), "perm");
        let perm_const = b
            .operations
            .iter()
            .find(|op| op.outputs[0].name == perm[0])
            .unwrap();
        match &perm_const.attributes["val"].value {
            Some(value::Value::ImmediateValue(v)) => assert_eq!(
                v.tensor.as_ref().unwrap().value,
                Some(tensor_value::Value::Ints(proto::RepeatedInts {
                    values: vec![2, 0, 3, 1]
                }))
            ),
            _ => panic!("perm is not an immediate"),
        }
        assert_eq!(bound(op(&b, &c), "values"), ["x", "y", e.name()]);
        assert_eq!(bound(op(&b, &r), "epsilon"), ["rsqrt_0_epsilon_0"]);
        let select = op(&b, &s);
        assert_eq!(
            ["cond", "a", "b"].map(|p| bound(select, p)),
            [["cond"], ["x"], [e.name()]].map(|n| n.map(str::to_owned).to_vec())
        );
    }

    #[test]
    fn math_and_layout_ops_refuse_invalid_arguments() {
        fn refused(r: Result<Var>, why: &str) {
            match r {
                Err(Error::Contract(m)) => assert!(m.contains(why), "{m:?} is not about {why:?}"),
                other => panic!("expected a refusal about {why:?}, got {other:?}"),
            }
        }
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, 8, 16, 4]);
        let before = b.operations.len();

        refused(b.reduce_sum(&x, &[4], true), "axes must index");
        refused(b.reduce_sum(&x, &[-1], true), "axes must index");

        refused(b.transpose(&x, &[0, 1, 2, 4]), "perm must index");
        refused(b.transpose(&x, &[0, 1, -2, 3]), "perm must index");
        refused(b.transpose(&x, &[0, 1, 2]), "permutation");
        refused(b.transpose(&x, &[0, 1, 1, 3]), "permutation");

        refused(b.concat(&[], 0, false), "at least one value");
        refused(b.concat(&[&x, &x], 4, false), "axis must index");
        let off_axis = b.input("w", &[1, 8, 8, 4]);
        refused(b.concat(&[&x, &off_axis], 3, false), "every axis except");
        let low_rank = b.input("v", &[1, 8, 16]);
        refused(b.concat(&[&x, &low_rank], 1, false), "every axis except");
        let flags = bool_var(&[1, 8, 16, 4]);
        refused(b.concat(&[&x, &flags], 3, false), "share a dtype");

        refused(b.select(&x, &x, &x), "bool");
        refused(b.select(&flags, &x, &flags), "share a dtype");
        let wide = b.input("z", &[1, 3, 16, 4]);
        refused(b.select(&flags, &x, &wide), "broadcast");

        assert_eq!(b.operations.len(), before, "a refused call emitted an operation");
    }

    #[test]
    fn invalid_programs_are_rejected() {
        let mut b = ProgramBuilder::new();
        let x = b.input("x", &[1, 4, 1, 8]);
        assert!(b
            .constexpr_blockwise_shift_scale(&[1; 7], &[2, 4, 1, 1], &[f16::ONE; 2], &[2, 1, 1, 1])
            .is_err());
        let w = b
            .constexpr_blockwise_shift_scale(&[1; 6], &[2, 3, 1, 1], &[f16::ONE; 2], &[2, 1, 1, 1])
            .unwrap();
        assert!(b.conv(&x, &w).is_err());
        assert!(b.split(&x, &[3], 1).is_err());
        assert!(b.reduce_max(&x, &[4], true).is_err());
        let wide = b.input("z", &[1, 3, 1, 8]);
        assert!(b.mul(&x, Operand::Var(&wide), None).is_err());
    }
}

/// Packs `bits`-wide indices least-significant-first into whole bytes, the layout
/// Core ML's blob reader expects for `UInt4` / `UInt6` (no padding between rows).
fn pack_indices(indices: &[u8], bits: u32) -> Vec<u8> {
    let total_bits = indices.len() * bits as usize;
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    for (i, &v) in indices.iter().enumerate() {
        let start = i * bits as usize;
        for b in 0..bits as usize {
            if v >> b & 1 == 1 {
                let bit = start + b;
                out[bit / 8] |= 1 << (bit % 8);
            }
        }
    }
    out
}
