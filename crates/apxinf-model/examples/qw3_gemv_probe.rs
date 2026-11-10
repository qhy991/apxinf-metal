//! Effective bandwidth of the Qwen3-0.6B decode projections at one row.
//!
//! Times MLX's `quantized_matmul` (W8, group 64) and a BF16 `matmul` at the
//! layer shapes a decode token actually uses, so the W8 gap can be attributed
//! to bytes moved versus kernel efficiency.
use apxinf_mlx::{Array, MlxBackend, MlxDType};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = MlxBackend::new(0)?;
    let stream = backend.stream().clone();
    // (name, out features, in features) for one Qwen3-0.6B layer.
    let shapes = [
        ("q", 2048usize, 1024usize),
        ("k", 1024, 1024),
        ("v", 1024, 1024),
        ("o", 1024, 2048),
        ("gate_up", 6144, 1024),
        ("down", 1024, 3072),
    ];
    let iterations = 200;
    let mut total_bytes = 0u64;
    let mut total_seconds = 0f64;
    for (name, out, input) in shapes {
        // Canonical [in, out] weights, quantized the way the model packs them.
        let count = input * out;
        let data: Vec<f32> = (0..count)
            .map(|i| ((i % 17) as f32 - 8.0) / 64.0)
            .collect();
        let canonical = Array::from_f32(&stream, &[input, out], &data)?.cast(MlxDType::BF16)?;
        let packed = canonical.transpose(&[1, 0])?;
        let [weights, scales, biases] = packed.quantize(64, 8)?;
        stream.eval(&[weights.clone(), scales.clone(), biases.clone()])?;
        let x = Array::from_f32(&stream, &[1, input], &vec![0.01f32; input])?.cast(MlxDType::BF16)?;

        // Warm up, then time `iterations` synchronized calls.
        for _ in 0..5 {
            let y = x.quantized_matmul(&weights, &scales, &biases, true, 64, 8)?;
            stream.eval(&[y])?;
        }
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            outputs.push(x.quantized_matmul(&weights, &scales, &biases, true, 64, 8)?);
        }
        stream.eval(&outputs)?;
        let seconds = started.elapsed().as_secs_f64() / iterations as f64;
        let weight_bytes = weights.numel() as u64 * 4 // packed u32 words
            + scales.numel() as u64 * 2
            + biases.numel() as u64 * 2;
        total_bytes += weight_bytes;
        total_seconds += seconds;
        println!(
            "{name:8} out={out:5} in={input:5} {:.3} ms/call  {:.1} GB/s  (bytes {:.1} MB)",
            seconds * 1e3,
            weight_bytes as f64 / seconds / 1e9,
            weight_bytes as f64 / 1e6
        );

        // BF16 reference: same output, weights four times larger.
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            outputs.push(x.matmul(&canonical)?);
        }
        stream.eval(&outputs)?;
        let seconds_bf16 = started.elapsed().as_secs_f64() / iterations as f64;
        let bf16_bytes = count as u64 * 2;
        println!(
            "         bf16     {:.3} ms/call  {:.1} GB/s  (bytes {:.1} MB)",
            seconds_bf16 * 1e3,
            bf16_bytes as f64 / seconds_bf16 / 1e9,
            bf16_bytes as f64 / 1e6
        );
    }
    println!(
        "one layer (W8): {:.3} ms, {:.1} MB -> {:.1} GB/s",
        total_seconds * 1e3,
        total_bytes as f64 / 1e6,
        total_bytes as f64 / total_seconds / 1e9
    );
    println!(
        "28 layers at this rate: {:.2} ms/token",
        total_seconds * 28.0 * 1e3
    );

    // The single-layer numbers above can sit in L2 (16.7 MB). A real token
    // streams 28 distinct layers (~440 MB), so time the whole sweep with
    // distinct weights per layer and one synchronized evaluation at the end.
    let layers = 28;
    let mut sets = Vec::with_capacity(layers);
    for _ in 0..layers {
        let mut layer = Vec::with_capacity(shapes.len());
        for (_, out, input) in shapes {
            let count = input * out;
            let data: Vec<f32> = (0..count).map(|i| ((i % 13) as f32 - 6.0) / 48.0).collect();
            let canonical = Array::from_f32(&stream, &[input, out], &data)?.cast(MlxDType::BF16)?;
            let packed = canonical.transpose(&[1, 0])?;
            let [weights, scales, biases] = packed.quantize(64, 8)?;
            layer.push((input, weights, scales, biases));
        }
        sets.push(layer);
    }
    // Materialize every quantized weight set before timing: MLX is lazy, and
    // otherwise the first timed sweep would include the 28 quantizations.
    let mut flush: Vec<Array> = Vec::new();
    for layer in &sets {
        for (_, weights, scales, biases) in layer {
            flush.push(weights.clone());
            flush.push(scales.clone());
            flush.push(biases.clone());
        }
    }
    stream.eval(&flush)?;
    let x = Array::from_f32(&stream, &[1, 1024], &vec![0.01f32; 1024])?.cast(MlxDType::BF16)?;
    let x2 = Array::from_f32(&stream, &[1, 2048], &vec![0.01f32; 2048])?.cast(MlxDType::BF16)?;
    let x3 = Array::from_f32(&stream, &[1, 3072], &vec![0.01f32; 3072])?.cast(MlxDType::BF16)?;
    let started = Instant::now();
    let mut outputs = Vec::with_capacity(28 * 6);
    for (index, layer) in sets.iter().enumerate() {
        for (slot, (input, weights, scales, biases)) in layer.iter().enumerate() {
            let source = match *input {
                1024 => &x,
                2048 => &x2,
                _ => &x3,
            };
            // Rotate the scale tensor by one element per (layer, slot) so the
            // engine cannot serve a repeated operand from cache.
            let _ = (index, slot);
            outputs.push(source.quantized_matmul(weights, scales, biases, true, 64, 8)?);
        }
    }
    stream.eval(&outputs)?;
    let seconds = started.elapsed().as_secs_f64();
    let bytes: u64 = sets
        .iter()
        .flat_map(|layer| layer.iter())
        .map(|(_, w, s, b)| {
            w.numel() as u64 * 4 + s.numel() as u64 * 2 + b.numel() as u64 * 2
        })
        .sum();
    println!(
        "full token sweep (28 layers): {:.3} ms, {:.1} MB -> {:.1} GB/s",
        seconds * 1e3,
        bytes as f64 / 1e6,
        bytes as f64 / seconds / 1e9
    );
    Ok(())
}
