//! Bit-level check of the fused residual-add + RMS-norm kernel against the
//! two-op path it replaces.
use apxinf_mlx::{Array, MlxBackend, MlxDType, RmsNormAdd};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = MlxBackend::new(0)?;
    let stream = backend.stream().clone();
    let hidden = 1024usize;
    let rows = 3usize;
    let eps = 1e-6f32;
    let count = rows * hidden;
    let x: Vec<f32> = (0..count).map(|i| ((i % 29) as f32 - 14.0) / 17.0).collect();
    let y: Vec<f32> = (0..count).map(|i| ((i % 31) as f32 - 15.0) / 19.0).collect();
    let w: Vec<f32> = (0..hidden).map(|i| 0.5 + (i % 7) as f32 / 10.0).collect();
    let xa = Array::from_f32(&stream, &[1, rows, hidden], &x)?.cast(MlxDType::BF16)?;
    let ya = Array::from_f32(&stream, &[1, rows, hidden], &y)?.cast(MlxDType::BF16)?;
    let wa = Array::from_f32(&stream, &[hidden], &w)?;

    // Reference: exactly what the model does today.
    let hidden_ref = xa.add(&ya)?;
    let norm_ref = hidden_ref.fast_rms_norm(&wa, eps)?;
    stream.eval(&[hidden_ref.clone(), norm_ref.clone()])?;

    let kernel = RmsNormAdd::new(&stream, hidden)?;
    let (hidden_new, norm_new) = kernel.call(&xa, &ya, &wa, eps)?;
    stream.eval(&[hidden_new.clone(), norm_new.clone()])?;

    let compare = |name: &str, a: &Array, b: &Array| -> Result<(), Box<dyn std::error::Error>> {
        let va = a.to_f32_vec()?;
        let vb = b.to_f32_vec()?;
        let mut differing = 0usize;
        let mut worst = 0f32;
        for (l, r) in va.iter().zip(&vb) {
            if l.to_bits() != r.to_bits() {
                differing += 1;
                worst = worst.max((l - r).abs());
            }
        }
        println!(
            "{name}: {differing}/{} elements differ, max |delta| {worst:.3e}",
            va.len()
        );
        Ok(())
    };
    compare("hidden", &hidden_ref, &hidden_new)?;
    compare("normed", &norm_ref, &norm_new)?;
    Ok(())
}
