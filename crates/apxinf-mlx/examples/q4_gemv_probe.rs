//! Op-level probe for the fused affine-Q4 GEMV (issue #142 follow-up):
//! numerical closeness + timing against stock `quantized_matmul` at the real
//! LFM3 decode shapes (M=1). Run under both Metal locks:
//!
//! ```sh
//! DYLD_LIBRARY_PATH=$HOME/mlx-sdk/lib ./q4_gemv_probe [out.json]
//! ```
use apxinf_mlx::q4_gemv::Q4GemvBf16;
use apxinf_mlx::{Array, MlxDType, Stream};
use std::time::Instant;

fn normal(stream: &Stream, shape: &[usize], seed: u64, scale: f32) -> apxinf_core::Result<Array> {
    Array::random_normal(shape, MlxDType::F32, &Array::random_key(stream, seed)?)?
        .mul(&Array::scalar(stream, scale, MlxDType::F32)?)?
        .cast(MlxDType::BF16)
}

fn time_it(
    stream: &Stream,
    iters: usize,
    mut f: impl FnMut() -> apxinf_core::Result<Array>,
) -> apxinf_core::Result<(f64, f64)> {
    for _ in 0..3 {
        stream.eval(&[f()?])?;
    }
    let mut rows = Vec::with_capacity(iters);
    for _ in 0..iters {
        let started = Instant::now();
        stream.eval(&[f()?])?;
        rows.push(started.elapsed().as_secs_f64() * 1e3);
    }
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((rows[rows.len() / 2], rows[0]))
}

fn main() -> apxinf_core::Result<()> {
    let stream = Stream::metal(0)?;
    let gemv = Q4GemvBf16::new(&stream)?;
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let out_path = std::env::args().nth(1);
    let mut json = String::from("{\"kind\":\"q4-gemv-probe\",\"rows\":[");
    let mut first = true;

    // (label, K, N) — every distinct projection geometry of LFM3-1.7B decode.
    for (label, k, n) in [
        ("conv.in_proj", 2048, 6144),
        ("conv.out_proj", 2048, 2048),
        ("attn.q", 2048, 2048),
        ("attn.kv", 2048, 256),
        ("attn.gate", 2048, 16),
        ("ffn.w1w3", 2048, 7680),
        ("ffn.w2", 7680, 2048),
        ("embed.head", 2048, 128000),
    ] {
        let dense = normal(&stream, &[n, k], 7, 0.02)?;
        let [weight, scales, biases] = dense.quantize(64, 4)?;
        let scales = scales.cast(MlxDType::BF16)?;
        let biases = biases.cast(MlxDType::BF16)?;
        let x = normal(&stream, &[1, 1, k], 11, 1.0)?;
        stream.eval(&[weight.clone(), scales.clone(), biases.clone(), x.clone()])?;

        let reference = x.quantized_matmul(&weight, &scales, &biases, true, 64, 4)?;
        stream.eval(&[reference.clone()])?;
        let out = gemv.call(&x, &weight, &scales, &biases)?;
        stream.eval(&[out.clone()])?;

        // Numerical comparison in fp32 on the host.
        let r = reference.reshape(&[n])?.cast(MlxDType::F32)?.to_bytes()?;
        let o = out.reshape(&[n])?.cast(MlxDType::F32)?.to_bytes()?;
        let rf = bytemuck_cast(&r);
        let of = bytemuck_cast(&o);
        let max_abs = rf
            .iter()
            .zip(of.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = rf
            .iter()
            .zip(of.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / rf.len() as f32;

        let (qmm_med, qmm_min) = time_it(&stream, iters, || {
            x.quantized_matmul(&weight, &scales, &biases, true, 64, 4)
        })?;
        let (gemv_med, gemv_min) =
            time_it(&stream, iters, || gemv.call(&x, &weight, &scales, &biases))?;
        let packed_bytes = n * k / 2;
        let bw = |ms: f64| packed_bytes as f64 / (ms * 1e-3) / 1e9;
        let delta = (qmm_med / gemv_med - 1.0) * 100.0;
        let delta_min = (qmm_min / gemv_min - 1.0) * 100.0;
        println!(
            "{label:14} K={k:<6} N={n:<7} max|d|={max_abs:8.5} mean|d|={mean_abs:8.6} \
             qmm {qmm_med:8.4}/{qmm_min:8.4} ms ({:6.1} GB/s)  \
             gemv {gemv_med:8.4}/{gemv_min:8.4} ms ({:6.1} GB/s)  med {delta:+.1}% min {delta_min:+.1}%",
            bw(qmm_med),
            bw(gemv_med),
        );
        if !first {
            json.push(',');
        }
        first = false;
        json.push_str(&format!(
            "{{\"label\":\"{label}\",\"k\":{k},\"n\":{n},\"max_abs\":{max_abs:.6},\"mean_abs\":{mean_abs:.6},\
             \"qmm_ms\":{qmm_med:.5},\"gemv_ms\":{gemv_med:.5},\
             \"qmm_gbps\":{:.2},\"gemv_gbps\":{:.2},\"delta_pct\":{delta:.3}}}",
            bw(qmm_med),
            bw(gemv_med),
        ));
    }
    json.push_str("]}");
    if let Some(path) = out_path {
        std::fs::write(&path, &json).expect("write probe json");
        println!("wrote {}", path);
    }
    Ok(())
}

fn bytemuck_cast(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
