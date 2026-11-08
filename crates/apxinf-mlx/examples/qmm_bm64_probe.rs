//! Round-24 (#237) op-level probe for the BM64 retiled qmm: byte-compare
//! against stock `quantized_matmul` and time both at the real fallback-tier
//! shapes. Run under both Metal locks:
//!
//! ```sh
//! DYLD_LIBRARY_PATH=$HOME/mlx-sdk/lib ./qmm_bm64_probe [out.json]
//! ```
use apxinf_mlx::qmm_bm64::QmmBm64;
use apxinf_mlx::{Array, MlxDType, Stream};
use std::time::Instant;

const DIM: usize = 4096;
const MLP: usize = 12288;

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
    let bm64 = QmmBm64::new(&stream)?;
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let out_path = std::env::args().nth(1);
    let mut json = String::from("{\"kind\":\"qmm-bm64-probe\",\"rows\":[");
    let mut first = true;

    for (label, m, k, n) in [
        ("attn.qkvo", 6889, DIM, DIM),
        ("mlp.gate_up", 6889, DIM, MLP * 2),
        ("mlp.down", 6889, MLP, DIM),
        ("attn.qkvo", 27556, DIM, DIM),
        ("mlp.gate_up", 27556, DIM, MLP),
        ("mlp.down", 27556, MLP, DIM),
        ("attn.qkvo", 1024, DIM, DIM),
        ("attn.tail4", 1092, DIM, DIM),
    ] {
        let dense = normal(&stream, &[n, k], 7, 0.02)?;
        let [weight, scales, biases] = dense.quantize(64, 4)?;
        let scales = scales.cast(MlxDType::BF16)?;
        let biases = biases.cast(MlxDType::BF16)?;
        let x = normal(&stream, &[1, m, k], 11, 1.0)?;
        stream.eval(&[weight.clone(), scales.clone(), biases.clone(), x.clone()])?;

        let reference = x.quantized_matmul(&weight, &scales, &biases, true, 64, 4)?;
        stream.eval(&[reference.clone()])?;
        let out = bm64.call(&x, &weight, &scales, &biases)?;
        stream.eval(&[out.clone()])?;
        let exact = out.to_bytes()? == reference.to_bytes()?;

        let (stock_med, _) = time_it(&stream, iters, || {
            x.quantized_matmul(&weight, &scales, &biases, true, 64, 4)
        })?;
        let (bm64_med, _) = time_it(&stream, iters, || {
            bm64.call(&x, &weight, &scales, &biases)
        })?;
        let tflops = |ms: f64| 2.0 * m as f64 * k as f64 * n as f64 / ms / 1e9;
        let delta = (stock_med / bm64_med - 1.0) * 100.0;
        println!(
            "{label:12} M={m:<6} K={k:<6} N={n:<6} exact={exact} \
             stock {stock_med:9.3} ms ({:.3} TF)  bm64 {bm64_med:9.3} ms ({:.3} TF)  \
             {delta:+.2}%",
            tflops(stock_med),
            tflops(bm64_med),
        );
        if !first {
            json.push(',');
        }
        first = false;
        json.push_str(&format!(
            "{{\"label\":\"{label}\",\"m\":{m},\"k\":{k},\"n\":{n},\"exact\":{exact},\
             \"stock_ms\":{stock_med:.4},\"bm64_ms\":{bm64_med:.4},\
             \"stock_tflops\":{:.4},\"bm64_tflops\":{:.4},\"delta_pct\":{delta:.4}}}",
            tflops(stock_med),
            tflops(bm64_med),
        ));
    }
    json.push_str("]}");
    if let Some(path) = out_path {
        std::fs::write(&path, &json).expect("write probe json");
        println!("wrote {}", path);
    }
    if !json.contains("\"exact\":false") {
        println!("ALL_BIT_EXACT");
        Ok(())
    } else {
        eprintln!("BIT_EXACTNESS FAILURE");
        std::process::exit(1);
    }
}
