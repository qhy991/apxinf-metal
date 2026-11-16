//! The whole MiniCPM5-2B MLP on the Apple Neural Engine: one FP16 Core ML
//! program per layer, run over 256-token prefill tiles while the rest of the
//! layer stays on the GPU. Portions adapted from EngineTailor (MIT); see
//! NOTICE. EngineTailor reference: fp16 ANE MLP prefill,
//! `enginetailor/models/minicpm/wavefront.py` @ 9ae7feda.
//!
//! The program: gate and up as FP16 1x1 convolutions, SiLU as
//! `0.5 v (1 + tanh(v / 2))`, gate and up normalized by their per-tile row
//! maxima before the product, down as two K-split FP16 convolutions whose
//! results are summed, and both scale factors as outputs so the host applies
//! them in FP32 after the tile. The normalization factor is capped from the
//! largest down-row absolute-sum, so a tile's down dot products stay inside
//! the FP16 range however large the activations grow.
use crate::metal_clef::ane_admission::Admission;
use apxinf_core::{Error, Result};
use apxinf_coreml::{ComputeUnits, Model, Operand, Package, ProgramBuilder, ProgramCache};
use apxinf_mlx::{fusions::AneTileHandoff, Array, MlxDType, Stream};
use half::f16;

/// Hidden width and MLP width of the only geometry this program was built for.
pub const HIDDEN: usize = 2048;
pub const INTERMEDIATE: usize = 6144;
/// Tokens per tile: the program's static batch surface.
pub const TILE: usize = 256;

/// Columns per K-split down convolution: `INTERMEDIATE` splits into two equal
/// parts of 3072, matching the Clef split.
const DOWN_PART: usize = 3072;
/// Largest exponent the normalization factor may reach; beyond it the row
/// scaling stops improving and only adds error.
const FACTOR_SCALE: f64 = 16.0;
/// Largest absolute down dot product a tile may reach: a normalized SwiGLU
/// operand is at most `factor_cap^2` in magnitude, so `factor_cap^2 *
/// sum_j |down[i][j]|` must stay under this FP16-safe bound.
const DOWN_ABS_SUM_LIMIT: f64 = 8192.0;
/// Env override of the compiled-plan timeout, in whole seconds.
const PLAN_TIMEOUT_ENV: &str = "APXINF_MINICPM5_ANE_PLAN_TIMEOUT_S";
const PLAN_TIMEOUT_SECS: u64 = 60;

/// Columns `[part, part + width)` of a row-major `[HIDDEN, INTERMEDIATE]`
/// down weight, where `width = INTERMEDIATE / parts`.
fn down_part(down: &[f16], part: usize, width: usize) -> Vec<f16> {
    let start = part * width;
    down.chunks_exact(INTERMEDIATE)
        .flat_map(|row| row[start..start + width].iter().copied())
        .collect()
}

/// The normalization-factor cap: the largest power of two that changes the
/// peak by a whole exponent and keeps every down dot product under
/// `DOWN_ABS_SUM_LIMIT`. An FP16 down weight is scaled by nothing, so only its
/// own absolute-sum bounds the product.
fn factor_cap(down: &[f16]) -> Result<f64> {
    let max_row_sum = down
        .chunks_exact(INTERMEDIATE)
        .map(|row| row.iter().map(|v| f64::from(v.to_f32().abs())).sum::<f64>())
        .fold(1f64, f64::max);
    let required = (DOWN_ABS_SUM_LIMIT / max_row_sum).sqrt();
    let cap = 2f64.powf(FACTOR_SCALE.min(required).log2().floor());
    if !(cap >= 1.0) {
        return Err(Error::Contract(
            "MiniCPM5 ANE down weights need factor scales outside the FP16 range",
        ));
    }
    Ok(cap)
}

/// One layer's Core ML program for the whole MLP.
///
/// `gate` and `up` are row-major `[INTERMEDIATE, HIDDEN]` and `down` is
/// row-major `[HIDDEN, INTERMEDIATE]`; they have the same element count, so a
/// length check cannot tell them apart and the shapes are fixed here.
pub fn program(gate: &[f16], up: &[f16], down: &[f16]) -> Result<Package> {
    if gate.len() != INTERMEDIATE * HIDDEN || up.len() != INTERMEDIATE * HIDDEN {
        return Err(Error::Contract(
            "MiniCPM5 ANE gate and up must be [6144, 2048]",
        ));
    }
    if down.len() != HIDDEN * INTERMEDIATE {
        return Err(Error::Contract("MiniCPM5 ANE down must be [2048, 6144]"));
    }
    let cap = factor_cap(down)?;
    let parts = INTERMEDIATE.div_ceil(DOWN_PART);
    let width = INTERMEDIATE / parts;
    let split_down: Vec<Vec<f16>> = (0..parts).map(|p| down_part(down, p, width)).collect();

    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, HIDDEN, 1, TILE])?;
    let g = b.conv_fp16(&x, gate, &[INTERMEDIATE, HIDDEN, 1, 1])?;
    let u = b.conv_fp16(&x, up, &[INTERMEDIATE, HIDDEN, 1, 1])?;
    // SiLU as `0.5 v (1 + tanh(v / 2))`: tanh is the only transcendental the
    // Neural Engine maps directly.
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let t = b.tanh(&half)?;
    let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
    let s = b.mul(&half, Operand::Var(&t1), None)?;
    // Per-tile row maxima, capped so the product below cannot overflow FP16.
    let sa = b.abs(&s)?;
    let s_max = b.reduce_max(&sa, &[1], true)?;
    let ua = b.abs(&u)?;
    let u_max = b.reduce_max(&ua, &[1], true)?;
    let inverse_cap = f16::from_f64(1.0 / cap);
    let floor = f16::from_f64(1.0 / FACTOR_SCALE);
    let sm = b.mul(&s_max, Operand::Scalar(inverse_cap), None)?;
    let s_scale = b.maximum(&sm, Operand::Scalar(floor), Some("gate_scale"))?;
    let um = b.mul(&u_max, Operand::Scalar(inverse_cap), None)?;
    let u_scale = b.maximum(&um, Operand::Scalar(floor), Some("up_scale"))?;
    // Normalize before the product so a large SwiGLU value cannot overflow first.
    let sn = b.real_div(&s, Operand::Var(&s_scale), None)?;
    let un = b.real_div(&u, Operand::Var(&u_scale), None)?;
    let a = b.mul(&sn, Operand::Var(&un), None)?;
    let xs = b.split(&a, &vec![width; parts], 1)?;
    let Some((first, rest)) = xs.split_first() else {
        return Err(Error::Contract(
            "MiniCPM5 ANE K-split down needs at least one part",
        ));
    };
    let mut y = b.conv_fp16(first, &split_down[0], &[HIDDEN, width, 1, 1])?;
    for (xi, wd) in rest.iter().zip(&split_down[1..]) {
        let yi = b.conv_fp16(xi, wd, &[HIDDEN, width, 1, 1])?;
        y = b.add(&y, Operand::Var(&yi), None)?;
    }
    let y = b.identity(&y, Some("y"))?;
    b.finish(&[&y, &s_scale, &u_scale])
}

/// The compiled-plan timeout, with an env override.
fn plan_timeout() -> std::time::Duration {
    let seconds = std::env::var(PLAN_TIMEOUT_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|&seconds| seconds > 0)
        .unwrap_or(PLAN_TIMEOUT_SECS);
    std::time::Duration::from_secs(seconds)
}

/// One layer's loaded Neural Engine program.
pub struct AneMlp {
    model: Model,
    tile: usize,
}

impl AneMlp {
    pub fn load_package(cache: &ProgramCache, package: &Package) -> Result<Self> {
        let input = package
            .model
            .description
            .as_ref()
            .and_then(|description| description.input.first())
            .and_then(|input| input.r#type.as_ref())
            .and_then(|kind| kind.multi_array_type.as_ref())
            .ok_or(Error::Contract(
                "MiniCPM5 ANE Core ML input profile is absent",
            ))?;
        if input.shape != [1, HIDDEN as i64, 1, TILE as i64] {
            return Err(Error::Contract(
                "MiniCPM5 ANE Core ML input needs its [1, 2048, 1, 256] profile",
            ));
        }
        let admission = Admission::new(package, [1, HIDDEN, 1, TILE])?;
        Ok(Self {
            model: cache.load_with_placement(
                package,
                ComputeUnits::CpuAndNeuralEngine,
                plan_timeout(),
                |facts| admission.check(facts),
            )?,
            tile: TILE,
        })
    }

    /// One charging prediction on zeros.
    pub fn warm(&self) -> Result<()> {
        let shape = [1, HIDDEN, 1, self.tile];
        let scale_shape = [1, 1, 1, self.tile];
        let input = vec![f16::ZERO; HIDDEN * self.tile];
        drop(self.model.predict(
            &[("x", &input, &shape)],
            &[
                ("y", &shape),
                ("gate_scale", &scale_shape),
                ("up_scale", &scale_shape),
            ],
        )?);
        Ok(())
    }

    /// BF16 `[1, T, HIDDEN]` as FP16 channel-first tiles `[tiles, HIDDEN, 256]`
    /// (zero-padded past `T`), in one pass. The result is row-contiguous, so
    /// [`predict`](Self::predict) may read it back without a copy.
    pub fn tiles(h: &Array) -> Result<Array> {
        let [1, tokens, channels] = *h.shape() else {
            return Err(Error::Contract("MiniCPM5 ANE tiles take BF16 [1, T, 2048]"));
        };
        if channels != HIDDEN || tokens == 0 {
            return Err(Error::Contract("MiniCPM5 ANE tiles take BF16 [1, T, 2048]"));
        }
        let tiles = tokens.div_ceil(TILE);
        let (_, packed) = AneTileHandoff::new(h.stream())?.call(h, TILE)?;
        let [packed_tiles, packed_channels, packed_tile] = *packed.shape() else {
            return Err(Error::Contract(
                "MiniCPM5 ANE handoff did not produce channel-first tiles",
            ));
        };
        if packed_tiles != tiles || packed_channels != HIDDEN || packed_tile != TILE {
            return Err(Error::Contract(
                "MiniCPM5 ANE handoff did not cover the token geometry",
            ));
        }
        if tokens > tiles * TILE {
            return Err(Error::Contract(
                "MiniCPM5 ANE tiles do not cover every token",
            ));
        }
        Ok(packed)
    }

    /// Runs one batch prediction over every tile of `packed` and restores BF16
    /// `[1, tokens, HIDDEN]` on `stream`: the program's `y` times both scales
    /// in FP32, rounded once. Waits for the tiles only, so GPU work queued
    /// after them keeps running.
    pub fn predict(&self, stream: &Stream, packed: &Array, tokens: usize) -> Result<Array> {
        let [tiles, channels, tile] = *packed.shape() else {
            return Err(Error::Contract(
                "MiniCPM5 ANE tiles must be [tiles, 2048, 256]",
            ));
        };
        if channels != HIDDEN || tile != self.tile || tokens == 0 {
            return Err(Error::Contract(
                "MiniCPM5 ANE tiles must be [tiles, 2048, 256]",
            ));
        }
        if tiles != tokens.div_ceil(self.tile) {
            return Err(Error::Contract(
                "MiniCPM5 ANE tiles must be exactly the tokens' tiles",
            ));
        }
        if packed.dtype() != MlxDType::F16 {
            return Err(Error::Contract("MiniCPM5 ANE packed tiles must be FP16"));
        }
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
            .chunks_exact(HIDDEN * self.tile)
            .map(|tile| vec![tile])
            .collect();
        if batch.len() != tiles {
            return Err(Error::Contract(
                "MiniCPM5 ANE packed tiles have the wrong extent",
            ));
        }
        let results = self.model.predict_batch(
            &[("x", &[1, HIDDEN, 1, self.tile])],
            &batch,
            &[
                ("y", &[1, HIDDEN, 1, self.tile]),
                ("gate_scale", &[1, 1, 1, self.tile]),
                ("up_scale", &[1, 1, 1, self.tile]),
            ],
        )?;
        let stacked = |index: usize, shape: &[usize]| -> Result<Array> {
            let mut bytes = Vec::with_capacity(shape.iter().product::<usize>() * 2);
            for result in &results {
                bytes.extend(result[index].iter().flat_map(|v| v.to_le_bytes()));
            }
            Array::from_bytes(stream, shape, MlxDType::F16, &bytes)
        };
        let rows = tiles * self.tile;
        let y = stacked(0, &[tiles, HIDDEN, self.tile])?
            .transpose(&[0, 2, 1])?
            .reshape(&[rows, HIDDEN])?
            .cast(MlxDType::F32)?;
        let gate_scale = stacked(1, &[rows])?
            .reshape(&[rows, 1])?
            .cast(MlxDType::F32)?;
        let up_scale = stacked(2, &[rows])?
            .reshape(&[rows, 1])?
            .cast(MlxDType::F32)?;
        let scaled = y.mul(&gate_scale)?.mul(&up_scale)?;
        let scaled = if tokens < rows {
            scaled.slice_axis(0, 0, tokens)?
        } else {
            scaled
        };
        scaled.reshape(&[1, tokens, HIDDEN])?.cast(MlxDType::BF16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature_shape(feature: &apxinf_coreml::proto::FeatureDescription) -> Result<[i64; 4]> {
        let array = feature
            .r#type
            .as_ref()
            .and_then(|kind| kind.multi_array_type.as_ref())
            .ok_or(Error::Contract("test feature has no multi-array type"))?;
        assert_eq!(array.data_type, apxinf_coreml::proto::ARRAY_FLOAT16);
        array
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::Contract("test feature is not rank 4"))
    }

    #[test]
    fn program_rejects_wrong_weight_lengths() {
        // gate and up are [6144, 2048] and down is [2048, 6144]: the same
        // element count, so only a genuinely wrong length is rejected.
        let valid = vec![f16::ONE; HIDDEN * INTERMEDIATE];
        let short = vec![f16::ONE; HIDDEN];
        let empty: [f16; 0] = [];
        let rejects = |gate: &[f16], up: &[f16], down: &[f16]| {
            matches!(program(gate, up, down), Err(Error::Contract(_)))
        };
        assert!(rejects(&short, &valid, &valid), "short gate");
        assert!(rejects(&valid, &short, &valid), "short up");
        assert!(rejects(&valid, &valid, &short), "short down");
        assert!(rejects(&empty, &empty, &empty), "empty weights");
    }

    #[test]
    fn program_builds_the_expected_signature() -> Result<()> {
        // One weight block serves all three matrices: gate, up and down have
        // the same element count and the signature does not depend on values.
        let weights = vec![f16::ONE; HIDDEN * INTERMEDIATE];
        let package = program(&weights, &weights, &weights)?;
        let description = package.model.description.as_ref().ok_or(Error::Contract(
            "MiniCPM5 ANE Core ML description is absent",
        ))?;
        assert_eq!(description.input.len(), 1);
        assert_eq!(description.input[0].name, "x");
        assert_eq!(
            feature_shape(&description.input[0])?,
            [1, HIDDEN as i64, 1, TILE as i64]
        );
        let outputs: [(&str, [i64; 4]); 3] = [
            ("y", [1, HIDDEN as i64, 1, TILE as i64]),
            ("gate_scale", [1, 1, 1, TILE as i64]),
            ("up_scale", [1, 1, 1, TILE as i64]),
        ];
        assert_eq!(description.output.len(), outputs.len());
        for (feature, (name, shape)) in description.output.iter().zip(outputs) {
            assert_eq!(feature.name, name);
            assert_eq!(feature_shape(feature)?, shape);
        }
        // Every weight the program needs is in its blob file.
        assert!(package.weights.len() > 3 * HIDDEN * INTERMEDIATE);
        Ok(())
    }
}
