//! LFM3 MLP GPU reference for the ANE INT8 parity probe (issue #150,
//! preregister `doc/lfm3-ane-int8-parity.md`).
//!
//! Runs the deployed LFM3 MLP chain — `gate/up matmul -> silu(g) * u -> down
//! matmul` — on the same real checkpoint weights and the same real post-
//! `ffn_norm` activation the ANE probe consumes:
//!
//! * BF16 weights and activation (the certified #121 engine precision), with
//!   the 5632-column share's contribution also written for the deployment
//!   split;
//! * the same weights in FP32 (the noise-floor reference).
//!
//! Inputs are the raw f32 files exported by
//! `devlocal/lfm3-ane-int8-parity/scripts/export_inputs.py`; BF16 is recovered
//! exactly because every value was exported from a BF16 checkpoint.
//!
//! ```sh
//! DYLD_LIBRARY_PATH=$MLX_ROOT/lib cargo run -p apxinf-mlx --features native \
//!   --example lfm3_mlp_parity_reference --release -- <inputs-dir> <outputs-dir>
//! ```

use apxinf_core::Result;
use apxinf_mlx::{Array, MlxDType, Stream};
use std::path::{Path, PathBuf};

const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 7680;
const TOKENS: usize = 512;
const LAYERS: [usize; 4] = [0, 4, 11, 22];
const SHARE: usize = 5632;
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

fn write_array(out: &Path, stem: &str, array: &Array) -> Result<()> {
    let f32_array = if array.dtype() == MlxDType::F32 {
        array.clone()
    } else {
        array.cast(MlxDType::F32)?
    };
    let bytes = f32_array.to_bytes()?;
    std::fs::write(out.join(format!("{stem}.f32")), bytes)
        .map_err(|error| apxinf_core::Error::Other(format!("write {stem}: {error}")))?;
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let inputs = PathBuf::from(args.next().expect("usage: <inputs-dir> <outputs-dir>"));
    let out = PathBuf::from(args.next().expect("usage: <inputs-dir> <outputs-dir>"));
    std::fs::create_dir_all(&out).ok();
    let stream = Stream::metal(0)?;
    let mut rows = Vec::new();

    for layer in LAYERS {
        let x_data = read_f32(&inputs.join(format!("layer{layer}_x.f32")), TOKENS * HIDDEN)?;
        let w1_data = read_f32(
            &inputs.join(format!("layer{layer}_w1.f32")),
            INTERMEDIATE * HIDDEN,
        )?;
        let w3_data = read_f32(
            &inputs.join(format!("layer{layer}_w3.f32")),
            INTERMEDIATE * HIDDEN,
        )?;
        let w2_data = read_f32(
            &inputs.join(format!("layer{layer}_w2.f32")),
            HIDDEN * INTERMEDIATE,
        )?;

        // ---- BF16 arm: the deployed engine precision. ----
        let x16 = Array::from_f32(&stream, &[1, TOKENS, HIDDEN], &x_data)?.cast(MlxDType::BF16)?;
        let w1_16 =
            Array::from_f32(&stream, &[INTERMEDIATE, HIDDEN], &w1_data)?.cast(MlxDType::BF16)?;
        let w3_16 =
            Array::from_f32(&stream, &[INTERMEDIATE, HIDDEN], &w3_data)?.cast(MlxDType::BF16)?;
        let w2_16 =
            Array::from_f32(&stream, &[HIDDEN, INTERMEDIATE], &w2_data)?.cast(MlxDType::BF16)?;
        let g = x16.matmul(&w1_16.transpose(&[1, 0])?)?;
        let u = x16.matmul(&w3_16.transpose(&[1, 0])?)?;
        let h = g.mul(&g.sigmoid()?)?.mul(&u)?;
        let y = h.matmul(&w2_16.transpose(&[1, 0])?)?;
        // The deployment share [2048, 7680) as its own BF16 output.
        let h_share = h.slice_axis(2, SHARE_START, INTERMEDIATE)?;
        let w2_share = w2_16.slice_axis(1, SHARE_START, INTERMEDIATE)?;
        let y_share = h_share.matmul(&w2_share.transpose(&[1, 0])?)?;
        stream.eval(&[y.clone(), h.clone(), y_share.clone()])?;
        write_array(&out, &format!("layer{layer}_bf16_y"), &y)?;
        write_array(&out, &format!("layer{layer}_bf16_h"), &h)?;
        write_array(&out, &format!("layer{layer}_bf16_share5632_y"), &y_share)?;
        println!(
            "lfm3-mlp-parity-reference: layer={layer} bf16 y={:?} h={:?}",
            y.shape(),
            h.shape()
        );
        drop((
            x16, w1_16, w3_16, w2_16, g, u, h, y, h_share, w2_share, y_share,
        ));

        // ---- FP32 arm: same weights, the numerical-floor reference. ----
        let x32 = Array::from_f32(&stream, &[1, TOKENS, HIDDEN], &x_data)?;
        let w1_32 = Array::from_f32(&stream, &[INTERMEDIATE, HIDDEN], &w1_data)?;
        let w3_32 = Array::from_f32(&stream, &[INTERMEDIATE, HIDDEN], &w3_data)?;
        let w2_32 = Array::from_f32(&stream, &[HIDDEN, INTERMEDIATE], &w2_data)?;
        let g32 = x32.matmul(&w1_32.transpose(&[1, 0])?)?;
        let u32 = x32.matmul(&w3_32.transpose(&[1, 0])?)?;
        let h32 = g32.mul(&g32.sigmoid()?)?.mul(&u32)?;
        let y32 = h32.matmul(&w2_32.transpose(&[1, 0])?)?;
        stream.eval(&[y32.clone(), h32.clone()])?;
        write_array(&out, &format!("layer{layer}_f32_y"), &y32)?;
        write_array(&out, &format!("layer{layer}_f32_h"), &h32)?;

        println!(
            "lfm3-mlp-parity-reference: layer={layer} f32 y={:?}",
            y32.shape()
        );
        rows.push(layer);
    }

    let json = format!(
        "{{\"kind\":\"lfm3-mlp-parity-gpu\",\"layers\":{:?},\"tokens\":{TOKENS},\
         \"share_start\":{SHARE_START},\"share\":{SHARE}}}",
        rows
    );
    std::fs::write(out.join("gpu.json"), json).expect("write gpu.json");
    println!("wrote {}", out.join("gpu.json").display());
    Ok(())
}
