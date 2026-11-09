//! How much of the Qwen-Image GPU leg is compute, how much is memory traffic,
//! and how much is dispatch overhead (#242, megakernel diagnostic).
//!
//! Times one 512² DiT block's GPU-side pieces at their real shapes, separately,
//! on an idle device: the Q4 weight dequantizations, the dense GEMMs that
//! consume them, the quantized-matmul path that skips the dequantization, the
//! attention core, and a dispatch-latency floor. Nothing here touches the
//! model or the shipped path; it rebuilds the shapes from constants.
//!
//! `cargo run --release --features mlx --example qimage_gpu_roles`
#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
mod native {
    use apxinf_mlx::{Array, MlxDType, Stream};
    use std::time::Instant;

    const DIM: usize = 4096;
    const MLP: usize = 12288;
    const TOKENS: usize = 1152;
    const HEADS: usize = 32;
    const HEAD_DIM: usize = 128;
    const REPEATS: usize = 20;

    /// Median wall time of `f`, with the stream drained inside the timed
    /// region so queued work cannot drift into the next sample.
    fn timed(
        stream: &Stream,
        repeats: usize,
        mut f: impl FnMut() -> apxinf_core::Result<Vec<Array>>,
    ) -> apxinf_core::Result<f64> {
        for _ in 0..3 {
            let out = f()?;
            stream.eval(&out)?;
        }
        let mut samples = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            let out = f()?;
            stream.eval(&out)?;
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(f64::total_cmp);
        Ok(samples[repeats / 2])
    }

    fn q4(stream: &Stream, rows: usize, cols: usize, key: &Array) -> apxinf_core::Result<[Array; 3]> {
        let w = Array::random_normal(&[rows, cols], MlxDType::BF16, key)?;
        stream.eval(&[w.clone()])?;
        w.quantize(64, 4)
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let stream = Stream::metal(0)?;
        let key = Array::random_key(&stream, 42)?;
        let x = Array::random_normal(&[1, TOKENS, DIM], MlxDType::BF16, &key)?;
        stream.eval(&[x.clone()])?;

        let [qw, qs, qb] = q4(&stream, DIM, DIM, &key)?;
        let [mw, ms, mb] = q4(&stream, MLP, DIM, &key)?;
        stream.eval(&[qw.clone(), qs.clone(), qb.clone(), mw.clone(), ms.clone(), mb.clone()])?;

        // 1. One Q4 -> BF16 weight dequantization, as `dense()` runs per
        //    projection per block.
        let dq = timed(&stream, REPEATS, || {
            Ok(vec![qw.dequantize(&qs, &qb, 64, 4)?])
        })?;

        // 2. The dense GEMM that consumes it, excluding the dequantization.
        let dense = qw.dequantize(&qs, &qb, 64, 4)?;
        stream.eval(&[dense.clone()])?;
        let gemm = timed(&stream, REPEATS, || Ok(vec![x.matmul(&dense)?]))?;

        // 3. The same projection through quantized_matmul, which never
        //    materializes the dense weight.
        let qmm = timed(&stream, REPEATS, || {
            Ok(vec![x.quantized_matmul(&qw, &qs, &qb, true, 64, 4)?])
        })?;

        // 4. The MLP's wider projection, both ways.
        let dense_mlp = mw.dequantize(&ms, &mb, 64, 4)?;
        stream.eval(&[dense_mlp.clone()])?;
        let dq_mlp = timed(&stream, REPEATS, || {
            Ok(vec![mw.dequantize(&ms, &mb, 64, 4)?])
        })?;
        let qmm_mlp = timed(&stream, REPEATS, || {
            Ok(vec![x.quantized_matmul(&mw, &ms, &mb, true, 64, 4)?])
        })?;

        // 5. Attention core at the block's real head layout.
        let heads = [1, HEADS, TOKENS, HEAD_DIM];
        let q = Array::random_normal(&heads, MlxDType::BF16, &key)?;
        let k = Array::random_normal(&heads, MlxDType::BF16, &key)?;
        let v = Array::random_normal(&heads, MlxDType::BF16, &key)?;
        stream.eval(&[q.clone(), k.clone(), v.clone()])?;
        let scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let sdpa = timed(&stream, REPEATS, || {
            Ok(vec![q.sdpa(&k, &v, scale, false, None)?])
        })?;

        // 6. Dispatch floor: a chain of trivial dependent ops, so each one
        //    must be launched after the last retires.
        const CHAIN: usize = 200;
        let tiny = Array::zeros(&stream, &[1024], MlxDType::F32)?;
        stream.eval(&[tiny.clone()])?;
        let chain = timed(&stream, 10, || {
            let mut a = tiny.clone();
            for _ in 0..CHAIN {
                a = a.abs()?;
            }
            Ok(vec![a])
        })?;

        // 7. One memory-bound elementwise pass over the activation, the shape
        //    every norm / residual / cast in the block touches.
        let bulk = timed(&stream, REPEATS, || Ok(vec![x.abs()?]))?;

        // 8. Best sustained GEMM this machine reaches, to place the block's
        //    rate against an attainable peak rather than a spec sheet.
        let big_a = Array::random_normal(&[4096, 4096], MlxDType::BF16, &key)?;
        let big_b = Array::random_normal(&[4096, 4096], MlxDType::BF16, &key)?;
        stream.eval(&[big_a.clone(), big_b.clone()])?;
        let big = timed(&stream, REPEATS, || Ok(vec![big_a.matmul(&big_b)?]))?;
        let af16 = big_a.cast(MlxDType::F16)?;
        let bf16 = big_b.cast(MlxDType::F16)?;
        stream.eval(&[af16.clone(), bf16.clone()])?;
        let big_f16 = timed(&stream, REPEATS, || Ok(vec![af16.matmul(&bf16)?]))?;

        // 9. What op fusion is worth: one elementwise pass over the activation
        //    against a chain of five, which a megakernel collapses into one.
        let chain5 = timed(&stream, REPEATS, || {
            let mut a = x.abs()?;
            for _ in 0..4 {
                a = a.abs()?;
            }
            Ok(vec![a])
        })?;

        let g = |m: f64| m / 1e9;
        let qmm_macs = g((TOKENS * DIM * DIM) as f64);
        let mlp_macs = g((TOKENS * DIM * MLP) as f64);
        let sdpa_macs = g((2 * HEADS * TOKENS * TOKENS * HEAD_DIM) as f64);
        println!("qimage-roles: dequantize dim={:.3}ms mlp={:.3}ms", dq * 1e3, dq_mlp * 1e3);
        println!(
            "qimage-roles: projection dense_gemm={:.3}ms qmm={:.3}ms macs={qmm_macs:.2}G \
             qmm_rate={:.2}TMAC/s dense_rate={:.2}TMAC/s",
            gemm * 1e3,
            qmm * 1e3,
            qmm_macs / qmm / 1e3,
            qmm_macs / (gemm + dq) / 1e3,
        );
        println!(
            "qimage-roles: mlp_projection qmm={:.3}ms macs={mlp_macs:.2}G qmm_rate={:.2}TMAC/s",
            qmm_mlp * 1e3,
            mlp_macs / qmm_mlp / 1e3,
        );
        println!(
            "qimage-roles: sdpa={:.3}ms macs={sdpa_macs:.2}G rate={:.2}TMAC/s",
            sdpa * 1e3,
            sdpa_macs / sdpa / 1e3,
        );
        println!(
            "qimage-roles: dispatch_floor={:.1}us/op over {CHAIN} dependent ops",
            chain / CHAIN as f64 * 1e6,
        );
        println!(
            "qimage-roles: elementwise_pass={:.3}ms over {:.1}MB",
            bulk * 1e3,
            (TOKENS * DIM * 2 * 2) as f64 / 1e6,
        );
        let big_macs = g((4096usize * 4096 * 4096) as f64);
        println!(
            "qimage-roles: peak_gemm 4096^3 bf16={:.3}ms={:.2}TMAC/s f16={:.3}ms={:.2}TMAC/s",
            big * 1e3,
            big_macs / big / 1e3,
            big_f16 * 1e3,
            big_macs / big_f16 / 1e3,
        );
        println!(
            "qimage-roles: fusion 1pass={:.3}ms 5pass={:.3}ms marginal={:.3}ms/pass",
            bulk * 1e3,
            chain5 * 1e3,
            (chain5 - bulk) / 4.0 * 1e3,
        );
        Ok(())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "mlx"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::run()
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "mlx")))]
fn main() {
    eprintln!("qimage_gpu_roles needs macOS aarch64 with the mlx feature");
}
