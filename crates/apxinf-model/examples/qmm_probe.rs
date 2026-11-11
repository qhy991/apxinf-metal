//! Does the quantized decode lane beat BF16 on the TTS predictor's shapes?
use apxinf_mlx::{Array, MlxBackend, MlxDType};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = MlxBackend::new(0)?;
    let stream = backend.stream().clone();
    // (out, in) of the five predictor projections at hidden 1024 / mlp 3072.
    let shapes = [
        (1024usize, 1024usize),
        (1024, 1024),
        (3072, 1024),
        (3072, 1024),
        (1024, 3072),
    ];
    for (out, input) in shapes {
        let weight = Array::zeros(&stream, &[out, input], MlxDType::BF16)?
            .add(&Array::scalar(&stream, 0.001, MlxDType::BF16)?)?;
        let weight_t = weight.transpose(&[1, 0])?;
        let x = Array::zeros(&stream, &[1, 1, input], MlxDType::BF16)?;
        let [packed, scales, biases] = weight.quantize(64, 8)?;
        stream.eval(&[
            packed.clone(),
            scales.clone(),
            biases.clone(),
            weight_t.clone(),
            x.clone(),
        ])?;
        let iterations = 200;
        for _ in 0..50 {
            stream.eval(&[x.matmul(&weight_t)?])?;
        }
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            outputs.push(x.matmul(&weight_t)?);
        }
        stream.eval(&outputs)?;
        let bf16 = started.elapsed().as_secs_f64() / iterations as f64;

        for _ in 0..50 {
            stream.eval(&[x.quantized_matmul(&packed, &scales, &biases, true, 64, 8)?])?;
        }
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            outputs.push(x.quantized_matmul(&packed, &scales, &biases, true, 64, 8)?);
        }
        stream.eval(&outputs)?;
        let q8 = started.elapsed().as_secs_f64() / iterations as f64;

        let bytes = out as f64 * input as f64 * 2.0;
        println!(
            "[{out:5} x {input:5}] bf16 {:.3} ms ({:.0} GB/s) | w8 {:.3} ms ({:.0} GB/s) | speedup {:.2}x",
            bf16 * 1e3,
            bytes / bf16 / 1e9,
            q8 * 1e3,
            bytes / 2.0 / q8 / 1e9,
            bf16 / q8
        );
    }
    Ok(())
}
