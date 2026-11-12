//! Does `constexpr_lut_to_dense` compile, land on the Neural Engine, and compute
//! the right thing? The whole lut6 residency plan rests on this op being placed on
//! the ANE, where per-block INT8/INT4 is not. Run:
//!   cargo run -p apxinf-coreml --release --features native --example lut_probe
use apxinf_core::Result;
use apxinf_coreml::{ComputeDeviceKind, ComputeUnits, ProgramBuilder, ProgramCache};
use half::f16;

const N: usize = 4096;   // HIDDEN
const K: usize = 5632;   // INTERMEDIATE - GPU_COLUMNS
const T: usize = 512;    // TILE
const CHECK_ROWS: usize = 8;  // f64 oracle on a slice; the full product is 11.8 GFLOP

/// Per-row symmetric uniform palette: `bits`-wide signed levels times a row scale.
fn encode(w: &[f16], cols: usize, bits: u32) -> (Vec<u8>, Vec<f16>) {
    let lv = ((1i32 << (bits - 1)) - 1) as f32;
    let entries = 1usize << bits;
    let (mut idx, mut lut) = (Vec::with_capacity(w.len()), Vec::with_capacity(entries * (w.len() / cols)));
    for row in w.chunks_exact(cols) {
        let m = row.iter().fold(0f32, |m, v| m.max(v.to_f32().abs())).max(1e-12);
        let scale = m / lv;
        // level j of the palette is (j - lv) * scale, so index lv is exactly zero
        for j in 0..entries {
            lut.push(f16::from_f32((j as f32 - lv) * scale));
        }
        for v in row {
            let q = (v.to_f32() / scale).round().clamp(-lv, lv);
            idx.push((q + lv) as u8);
        }
    }
    (idx, lut)
}

fn main() -> Result<()> {
    let w: Vec<f16> = (0..N * K)
        .map(|i| f16::from_f32((((i * 2654435761) % 1000) as f32 / 500.0 - 1.0) * 0.02))
        .collect();
    let x: Vec<f16> = (0..K * T)
        .map(|i| f16::from_f32(((i * 40503) % 100) as f32 / 50.0 - 1.0))
        .collect();

    for bits in [6u32, 4] {
        let (idx, lut) = encode(&w, K, bits);
        let mut b = ProgramBuilder::new();
        let inp = b.input("x", &[1, K, 1, T]);
        let weight = b.constexpr_lut_to_dense(&idx, &[N, K, 1, 1], &lut, N, bits)?;
        let y = b.conv(&inp, &weight)?;
        let y = b.identity(&y, Some("y"));
        let pkg = b.finish(&[&y])?;

        let dir = std::env::temp_dir().join(format!("lut{bits}-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).ok();
        let cache = ProgramCache::new(&dir);
        let model = cache.load_with_placement(
            &pkg,
            ComputeUnits::CpuAndNeuralEngine,
            std::time::Duration::from_secs(60),
            |facts| {
                for f in &facts.functions {
                    for op in &f.operations {
                        let d = op
                            .usage
                            .as_ref()
                            .and_then(|u| u.preferred.as_ref()).map(|d| format!("{:?}", d.kind))
                            .unwrap_or_else(|| "-".to_owned());
                        println!("    {:<28} -> {d}", op.operator);
                    }
                }
                let off: Vec<&str> = facts
                    .functions
                    .iter()
                    .flat_map(|f| &f.operations)
                    .filter(|o| {
                        o.usage
                            .as_ref()
                            .and_then(|u| u.preferred.as_ref())
                            .is_some_and(|d| d.kind != ComputeDeviceKind::NeuralEngine)
                    })
                    .map(|o| o.operator.as_str())
                    .collect();
                println!(
                    "    PLACEMENT: {}",
                    if off.is_empty() {
                        "every compute op on the Neural Engine".to_owned()
                    } else {
                        format!("OFF-ANE ops: {off:?}")
                    }
                );
                Ok(())
            },
        )?;

        let mut got = model.predict(&[("x", &x[..], &[1, K, 1, T][..])], &[("y", &[1, N, 1, T][..])])?;
        let out = got.remove(0);

        // reference: dequantize the palette and multiply on the host in f64
        let mut want = vec![0f64; CHECK_ROWS * T];
        let entries = 1usize << bits;
        for n in 0..CHECK_ROWS {
            for t in 0..T {
                let mut acc = 0f64;
                for k in 0..K {
                    acc += f64::from(lut[n * entries + idx[n * K + k] as usize].to_f32())
                        * f64::from(x[k * T + t].to_f32());
                }
                want[n * T + t] = acc;
            }
        }
        let (mut num, mut den) = (0f64, 0f64);
        for (o, w) in out.iter().take(CHECK_ROWS * T).zip(&want) {
            let d = f64::from(o.to_f32()) - w;
            num += d * d;
            den += w * w;
        }
        println!(
            "lut{bits}: COMPILED + RAN on CpuAndNeuralEngine, rel.rmse vs f64 palette oracle = {:.3e}",
            (num / den).sqrt()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
    Ok(())
}
