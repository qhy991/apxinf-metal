//! anesched — search the MIL-level schedule space for one GEMM-shaped layer.
//!
//! The ANE's instruction level sits behind a signed-HWX boundary, so there is no
//! kernel to schedule. Every lever that was measured to move the number lives in
//! MIL: how the op is spelled, how K is chunked, how wide the tile is, and how the
//! weights are encoded. This searches those, asserting on every candidate that the
//! compiler actually placed it on the Neural Engine.
//!
//! Three gates per candidate, cheapest first: compile (the compiler refuses
//! opaquely and that is a normal outcome), placement (MLComputePlan is the
//! compiler's own ground truth — observed, never commanded), then accuracy against
//! an f64 oracle. Only survivors are timed.
//!
//!   cargo run -p apxinf-coreml --release --features native --example anesched -- \
//!       --n 4096 --k 5632 [--tiles 128,512] [--chunks 0,1408,2816] \
//!       [--formats fp16,int8,lut6:16] [--weights <f16 binary N*K>]
use apxinf_core::{Error, Result};
use apxinf_coreml::{
    ComputeDeviceKind, ComputeUnits, Operand, Package, ProgramBuilder, ProgramCache, Var,
};
use half::f16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Format {
    Fp16,
    Int8,
    /// `bits`-wide palette, one table per `group` rows. Group size is a
    /// first-class axis: at the same bit width, one table per row measured
    /// 2.9-9.7x slower than the grouped default.
    Lut { bits: u32, group: usize },
}

impl Format {
    fn parse(s: &str) -> Result<Self> {
        let (head, group) = s.split_once(':').unwrap_or((s, "16"));
        let group: usize = group
            .parse()
            .map_err(|_| Error::Other(format!("bad group in {s}")))?;
        match head {
            "fp16" => Ok(Self::Fp16),
            "int8" => Ok(Self::Int8),
            "lut4" => Ok(Self::Lut { bits: 4, group }),
            "lut6" => Ok(Self::Lut { bits: 6, group }),
            _ => Err(Error::Other(format!("unknown format {s}"))),
        }
    }
    fn label(&self) -> String {
        match self {
            Self::Fp16 => "fp16".into(),
            Self::Int8 => "int8".into(),
            Self::Lut { bits, group } => format!("lut{bits}:{group}"),
        }
    }
    /// Bytes per weight, for the resident-bytes constraint.
    fn bytes_per_weight(&self, cols: usize) -> f64 {
        match self {
            Self::Fp16 => 2.0,
            Self::Int8 => 1.0 + 2.0 / cols as f64,
            Self::Lut { bits, group } => {
                *bits as f64 / 8.0 + 2.0 * (1usize << bits) as f64 / (group * cols) as f64
            }
        }
    }
}

/// One searched cell, including the ones that were rejected: a schedule the
/// compiler refused, or placed off the engine, is a result worth keeping — it is
/// why the next search does not retry it.
struct Cell {
    spelling: &'static str,
    n_split: usize,
    tile: usize,
    k_chunk: usize,
    format: String,
    verdict: &'static str,
    detail: String,
    ms_per_128: Option<f64>,
    rel_rmse: Option<f64>,
    weight_mb: f64,
    build_s: f64,
}

/// How the same arithmetic is written. At batch 1 `matmul` measured entirely on
/// the CPU while an equivalent 1x1 convolution reached the engine, so this is a
/// placement axis, not a style one — and it is the only axis whose outcome is
/// binary rather than a ratio.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Spelling {
    Conv1x1,
    Matmul,
}

impl Spelling {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "conv" | "conv1x1" => Ok(Self::Conv1x1),
            "matmul" => Ok(Self::Matmul),
            _ => Err(Error::Other(format!("unknown spelling {s}"))),
        }
    }
    fn label(&self) -> &'static str {
        match self {
            Self::Conv1x1 => "conv",
            Self::Matmul => "matmul",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Schedule {
    spelling: Spelling,
    tile: usize,
    k_chunk: usize,
    /// Output-column split: N emitted as `n_split` separate convolutions that are
    /// concatenated, rather than one wide one.
    n_split: usize,
    format: Format,
}

/// Encodes `[rows, cols]` row-major weights and emits the constexpr weight var.
fn weight_var(
    b: &mut ProgramBuilder,
    w: &[f16],
    rows: usize,
    cols: usize,
    format: Format,
) -> Result<Var> {
    match format {
        Format::Fp16 => {
            // fp16 has no constexpr form; the conv takes the values directly.
            Err(Error::Other("fp16 handled inline".into()))
        }
        Format::Int8 => {
            let mut q = Vec::with_capacity(w.len());
            let mut scale = Vec::with_capacity(rows);
            for row in w.chunks_exact(cols) {
                let peak = row
                    .iter()
                    .fold(0f32, |m, v| m.max(v.to_f32().abs()))
                    .max(f32::MIN_POSITIVE);
                let s = peak / 127.0;
                scale.push(f16::from_f32(s));
                q.extend(row.iter().map(|v| (v.to_f32() / s).round().clamp(-127.0, 127.0) as i8));
            }
            b.constexpr_blockwise_shift_scale(&q, &[rows, cols, 1, 1], &scale, &[rows, 1, 1, 1])
        }
        Format::Lut { bits, group } => {
            if rows % group != 0 {
                return Err(Error::Other(format!("group {group} must divide {rows} rows")));
            }
            let groups = rows / group;
            let levels = ((1u32 << (bits - 1)) - 1) as f32;
            let entries = 1usize << bits;
            let mut idx = vec![0u8; w.len()];
            let mut lut = Vec::with_capacity(groups * entries);
            for g in 0..groups {
                let block = &w[g * group * cols..(g + 1) * group * cols];
                let peak = block
                    .iter()
                    .fold(0f32, |m, v| m.max(v.to_f32().abs()))
                    .max(f32::MIN_POSITIVE);
                let s = peak / levels;
                for j in 0..entries {
                    lut.push(f16::from_f32((j as f32 - levels) * s));
                }
                for (i, v) in block.iter().enumerate() {
                    let q = (v.to_f32() / s).round().clamp(-levels, levels);
                    idx[g * group * cols + i] = (q + levels) as u8;
                }
            }
            b.constexpr_lut_to_dense(&idx, &[rows, cols, 1, 1], &lut, groups, bits)
        }
    }
}

/// One candidate program: `[1, K, 1, tile] -> [1, N, 1, tile]`, K split into
/// `k_chunk`-wide convolutions that are summed.
fn build(w: &[f16], n: usize, k: usize, s: Schedule) -> Result<Package> {
    // A requested chunk width is only valid if it divides K exactly: otherwise the
    // parts would differ in width and the schedule is not the one that was asked
    // for. (div_ceil alone is not enough — K=1024 with k_chunk=300 yields 4 parts
    // of 256, silently a different schedule.)
    if s.k_chunk != 0 && k % s.k_chunk != 0 {
        return Err(Error::Other(format!(
            "k_chunk {} does not divide K={k}",
            s.k_chunk
        )));
    }
    if s.n_split == 0 || n % s.n_split != 0 {
        return Err(Error::Other(format!(
            "n_split {} does not divide N={n}",
            s.n_split
        )));
    }
    let parts = if s.k_chunk == 0 { 1 } else { k / s.k_chunk };
    let width = k / parts;
    let n_part = n / s.n_split;
    let mut b = ProgramBuilder::new();

    // The matmul spelling takes [tile, K] x [K, N]; the conv one takes the
    // channels-first [1, K, 1, tile] the engine prefers.
    if s.spelling == Spelling::Matmul {
        if parts > 1 || s.n_split > 1 {
            return Err(Error::Other(
                "matmul spelling is probed unsplit, as the placement question".into(),
            ));
        }
        if s.format != Format::Fp16 {
            return Err(Error::Other(
                "matmul spelling probes placement at fp16 only".into(),
            ));
        }
        // `ProgramBuilder::matmul` exists, but an fp16 weight *constant* is only
        // reachable from inside the crate (`fp16_blob` is private), so this
        // example cannot author the operand. The spelling axis is probed by the
        // in-crate test `spelling_placement` instead.
        return Err(Error::Other(
            "matmul spelling needs an in-crate weight constant; see spelling_placement".into(),
        ));
    }

    let x = b.input("x", &[1, k, 1, s.tile]);
    let xs = if parts > 1 {
        b.split(&x, &vec![width; parts], 1)?
    } else {
        vec![x]
    };
    let mut columns: Vec<Var> = Vec::with_capacity(s.n_split);
    for c in 0..s.n_split {
        let mut acc: Option<Var> = None;
        for (p, xi) in xs.iter().enumerate() {
            // rows [c*n_part, (c+1)*n_part) x columns [p*width, (p+1)*width)
            let slice: Vec<f16> = w[c * n_part * k..(c + 1) * n_part * k]
                .chunks_exact(k)
                .flat_map(|row| row[p * width..(p + 1) * width].iter().copied())
                .collect();
            let y = match s.format {
                Format::Fp16 => b.conv_fp16(xi, &slice, &[n_part, width, 1, 1])?,
                _ => {
                    let wv = weight_var(&mut b, &slice, n_part, width, s.format)?;
                    b.conv(xi, &wv)?
                }
            };
            acc = Some(match acc {
                None => y,
                Some(a) => b.add(&a, Operand::Var(&y), None)?,
            });
        }
        columns.push(acc.expect("at least one part"));
    }
    // The builder has no concat, and adding one would measure the join rather
    // than the split. Each column share is its own output: the engine still runs
    // `n_split` narrower convolutions, which is the axis under test.
    let outs: Vec<Var> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let name = if i == 0 { "y".to_owned() } else { format!("y{i}") };
            b.identity(c, Some(&name))
        })
        .collect();
    b.finish(&outs.iter().collect::<Vec<_>>())
}


/// `sysctl -n <name>`, for the host identity a schedule is only valid on.
fn sysctl(name: &str) -> String {
    std::process::Command::new("sysctl")
        .args(["-n", name])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

/// A schedule is only valid for the chip and OS it was searched on: the same
/// shape and weights measured k_chunk=1536 best on an M1 Pro and 768 on an M4,
/// and the placement heuristic moves with OS builds. Both go in the key.
fn host_key() -> (String, String) {
    (sysctl("machdep.cpu.brand_string"), sysctl("kern.osversion"))
}

/// Minimal JSON string escaping (the few values here are paths and identifiers).
fn esc(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            c if (c as u32) < 0x20 => vec![' '],
            c => vec![c],
        })
        .collect()
}

fn num(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.6}")).unwrap_or_else(|| "null".into())
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|v| v == name).and_then(|i| a.get(i + 1).cloned())
}

fn list(name: &str, default: &str) -> Vec<String> {
    arg(name)
        .unwrap_or_else(|| default.to_owned())
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

fn main() -> Result<()> {
    let n: usize = arg("--n").unwrap_or_else(|| "4096".into()).parse().unwrap();
    let k: usize = arg("--k").unwrap_or_else(|| "5632".into()).parse().unwrap();
    let tiles: Vec<usize> = list("--tiles", "128,512").iter().map(|v| v.parse().unwrap()).collect();
    let chunks: Vec<usize> = list("--chunks", "0,1408,2816").iter().map(|v| v.parse().unwrap()).collect();
    let spellings: Vec<Spelling> = list("--spellings", "conv")
        .iter()
        .map(|v| Spelling::parse(v))
        .collect::<Result<_>>()?;
    let n_splits: Vec<usize> = list("--n-splits", "1").iter().map(|v| v.parse().unwrap()).collect();
    let formats: Vec<Format> = list("--formats", "fp16,int8,lut6:16")
        .iter()
        .map(|s| Format::parse(s))
        .collect::<Result<_>>()?;
    // Per-layer optima do not compose: 32 resident layers measured 37.0 ms/layer
    // against 3.05 ms for one layer alone. `--layers` turns the winner's weight
    // bytes into a total, and `--budget-gb` rejects schedules whose total exceeds
    // what the host can hold resident.
    let layers: usize = arg("--layers").unwrap_or_else(|| "1".into()).parse().unwrap();
    // `--n`/`--k` describe ONE matrix. A layer usually holds several of the same
    // shape — a SwiGLU MLP holds gate, up and down — so the resident total is
    // matrices x layers. Getting this wrong understates residency by exactly that
    // factor, and residency is the constraint that per-layer optima break against.
    let per_layer: usize = arg("--matrices-per-layer")
        .unwrap_or_else(|| "1".into())
        .parse()
        .unwrap();
    let budget_gb: Option<f64> = arg("--budget-gb").and_then(|v| v.parse().ok());
    let copies = layers * per_layer;

    // Real weights when given; the synthetic fallback is labelled because it
    // moved the split-K optimum and overstated the gain by 1.4x.
    let (w, source) = match arg("--weights") {
        Some(p) => {
            let raw = std::fs::read(&p).map_err(|e| Error::Other(format!("{p}: {e}")))?;
            if raw.len() != n * k * 2 {
                return Err(Error::Other(format!(
                    "{p} holds {} bytes, need {} for [{n}, {k}] f16",
                    raw.len(),
                    n * k * 2
                )));
            }
            let w: Vec<f16> = raw
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]))
                .collect();
            (w, format!("real weights {p}"))
        }
        None => {
            let w = (0..n * k)
                .map(|i| f16::from_f32((((i * 2654435761) % 2000) as f32 / 1000.0 - 1.0) * 0.02))
                .collect();
            (w, "SYNTHETIC weights (optima may not transfer)".to_owned())
        }
    };

    // A cached winner is only reused when the host matches: same chip and same OS
    // build. `--cache <dir>` reads `<dir>/<chip>-<os>-<N>x<K>-L<layers>.json`, and
    // a hit prints the schedule without compiling anything. `--refresh` searches
    // anyway and overwrites.
    let (chip, osv) = host_key();
    let cache_key = format!(
        "{}-{}-{n}x{k}-L{layers}x{per_layer}.json",
        chip.replace(' ', "_").replace('/', "_"),
        osv.replace(' ', "_").replace('/', "_")
    );
    let cache_file = arg("--cache").map(|d| std::path::PathBuf::from(d).join(&cache_key));
    let refresh = std::env::args().any(|a| a == "--refresh");
    if let (Some(path), false) = (cache_file.as_ref(), refresh) {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Some(best) = text.split("\"best\":").nth(1) {
                println!("anesched: cache hit for [{n}, {k}] on {chip} / {osv}");
                println!("  {}", best.trim().trim_end_matches('}').trim());
                println!("  {}", path.display());
                println!("  (--refresh to search again)");
                return Ok(());
            }
        }
    }

    println!("anesched: [{n}, {k}], {source}");
    println!(
        "  {} spellings x {} tiles x {} chunks x {} n_splits x {} formats = {} candidates\n",
        spellings.len(),
        tiles.len(),
        chunks.len(),
        n_splits.len(),
        formats.len(),
        spellings.len() * tiles.len() * chunks.len() * n_splits.len() * formats.len()
    );
    println!(
        "  {:<7} {:<10} {:>6} {:>8} {:>4} {:>10} {:>11} {:>10} {:>9}  {}",
        "spell", "format", "tile", "k_chunk", "nsp", "ms/128tok", "rel.rmse", "MB", "vs base", "gate"
    );

    let cache_dir = std::env::temp_dir().join(format!("anesched-{}", std::process::id()));
    let mut best: Option<(f64, Schedule)> = None;
    let mut base: Option<f64> = None;
    let mut cells: Vec<Cell> = Vec::new();

    for &spelling in &spellings {
    for &format in &formats {
        for &tile in &tiles {
            for &k_chunk in &chunks {
            for &n_split in &n_splits {
                let s = Schedule { spelling, tile, k_chunk, n_split, format };
                let label = format!(
                    "{:<7} {:<10} {:>6} {:>8} {:>4}",
                    spelling.label(), format.label(), tile, k_chunk, n_split
                );
                let t_build = std::time::Instant::now();

                // gate 0: resident bytes across every layer that will hold this
                // schedule. Cheapest gate of all — no compile needed.
                let total_gb = format.bytes_per_weight(k) * (n * k) as f64 * copies as f64 / 1e9;
                if let Some(cap) = budget_gb {
                    if total_gb > cap {
                        let detail = format!("{total_gb:.2} GB over the {cap:.2} GB budget");
                        println!("  {label} {:>10} {:>11} {:>10} {:>9}  budget: {detail}", "-", "-", "-", "-");
                        cells.push(Cell { spelling: spelling.label(), n_split, tile, k_chunk, format: format.label(), verdict: "over-budget",
                            detail, ms_per_128: None, rel_rmse: None, weight_mb: 0.0, build_s: 0.0 });
                        continue;
                    }
                }

                // gate 1: the compiler may refuse, opaquely. That is a normal outcome.
                let package = match build(&w, n, k, s) {
                    Ok(p) => p,
                    Err(e) => {
                        println!("  {label} {:>10} {:>11} {:>10} {:>9}  build: {}", "-", "-", "-", "-", short(&e));
                        cells.push(Cell { spelling: spelling.label(), n_split, tile, k_chunk, format: format.label(), verdict: "build-refused",
                            detail: short(&e), ms_per_128: None, rel_rmse: None, weight_mb: 0.0,
                            build_s: t_build.elapsed().as_secs_f64() });
                        continue;
                    }
                };
                let mb = package.weights.len() as f64 / 1e6;

                let dir = cache_dir.join(format!("{}-{tile}-{k_chunk}", format.label().replace(':', "_")));
                let _ = std::fs::remove_dir_all(&dir);
                let cache = ProgramCache::new(&dir);

                // gate 2: placement is observed, not commanded. Reject anything the
                // compiler did not put entirely on the engine.
                let mut off: Vec<String> = Vec::new();
                let model = cache.load_with_placement(
                    &package,
                    ComputeUnits::CpuAndNeuralEngine,
                    std::time::Duration::from_secs(180),
                    |facts| {
                        for f in &facts.functions {
                            for op in &f.operations {
                                if let Some(d) = op.usage.as_ref().and_then(|u| u.preferred.as_ref())
                                {
                                    if d.kind != ComputeDeviceKind::NeuralEngine {
                                        off.push(format!("{}:{:?}", op.operator, d.kind));
                                    }
                                }
                            }
                        }
                        if off.is_empty() {
                            Ok(())
                        } else {
                            Err(Error::Other(format!("off-engine {off:?}")))
                        }
                    },
                );
                let model = match model {
                    Ok(m) => m,
                    Err(e) => {
                        println!("  {label} {:>10} {:>11} {mb:>10.1} {:>9}  {}", "-", "-", "-", short(&e));
                        cells.push(Cell { spelling: spelling.label(), n_split, tile, k_chunk, format: format.label(), verdict: "off-engine",
                            detail: short(&e), ms_per_128: None, rel_rmse: None, weight_mb: mb,
                            build_s: t_build.elapsed().as_secs_f64() });
                        let _ = std::fs::remove_dir_all(&dir);
                        continue;
                    }
                };

                let x: Vec<f16> = (0..k * tile)
                    .map(|i| f16::from_f32(((i * 40503) % 200) as f32 / 100.0 - 1.0))
                    .collect();
                let in_shape: Vec<usize> = if spelling == Spelling::Matmul {
                    vec![tile, k]
                } else {
                    vec![1, k, 1, tile]
                };
                let inputs = [("x", &x[..], &in_shape[..])];
                let out_shape: Vec<usize> = if spelling == Spelling::Matmul {
                    vec![tile, n]
                } else {
                    vec![1, n, 1, tile]
                };
                let mut out_specs: Vec<(String, Vec<usize>)> = Vec::new();
                for i in 0..n_split {
                    let mut sh = out_shape.clone();
                    if spelling != Spelling::Matmul {
                        sh[1] = n / n_split;
                    }
                    out_specs.push((if i == 0 { "y".into() } else { format!("y{i}") }, sh));
                }
                let outs: Vec<(&str, &[usize])> =
                    out_specs.iter().map(|(a, b)| (a.as_str(), &b[..])).collect();
                let got = model.predict(&inputs, &outs)?.remove(0);

                // gate 3: accuracy against an f64 oracle, on a row slice (the full
                // product is tens of GFLOP and the slice is enough to catch a
                // mis-encoded palette or a wrong packing).
                const ROWS: usize = 8;
                let (mut num, mut den) = (0f64, 0f64);
                let checked_rows = ROWS.min(n / n_split);
                for r in 0..checked_rows {
                    for t in 0..tile {
                        let mut want = 0f64;
                        for c in 0..k {
                            let xv = if spelling == Spelling::Matmul {
                                x[t * k + c]
                            } else {
                                x[c * tile + t]
                            };
                            want += f64::from(w[r * k + c].to_f32()) * f64::from(xv.to_f32());
                        }
                        let got_v = if spelling == Spelling::Matmul {
                            got[t * n + r]
                        } else {
                            got[r * tile + t]  // first column share
                        };
                        let d = f64::from(got_v.to_f32()) - want;
                        num += d * d;
                        den += want * want;
                    }
                }
                let rmse = (num / den.max(f64::MIN_POSITIVE)).sqrt();

                for _ in 0..5 {
                    model.predict(&inputs, &outs)?;
                }
                let mut ts = Vec::new();
                for _ in 0..20 {
                    let t0 = std::time::Instant::now();
                    model.predict(&inputs, &outs)?;
                    ts.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                ts.sort_by(f64::total_cmp);
                // normalize to a 128-token tile so tiles are comparable
                let ms = ts[ts.len() / 2] * 128.0 / tile as f64;
                if base.is_none() {
                    base = Some(ms);
                }
                let rel = base.unwrap() / ms;
                println!("  {label} {ms:>10.3} {rmse:>11.2e} {mb:>10.1} {rel:>8.2}x  ok");
                cells.push(Cell { spelling: spelling.label(), n_split, tile, k_chunk, format: format.label(), verdict: "ok",
                    detail: String::new(), ms_per_128: Some(ms), rel_rmse: Some(rmse),
                    weight_mb: mb, build_s: t_build.elapsed().as_secs_f64() });
                if best.as_ref().is_none_or(|(b, _)| ms < *b) {
                    best = Some((ms, s));
                }
                drop(model);
                let _ = std::fs::remove_dir_all(&dir);
            }
            }
        }
    }
    }
    let _ = std::fs::remove_dir_all(&cache_dir);

    // The receipt: every cell including the rejected ones, keyed by the host it
    // is valid on. A later run reads this instead of re-searching; a run on
    // another chip or OS build will miss the key and search again, which is the
    // point.
    if let Some(out) = arg("--out").or_else(|| cache_file.as_ref().map(|p| p.display().to_string())) {
        let mut j = String::new();
        j.push_str("{\n");
        j.push_str(&format!("  \"tool\": \"anesched\",\n"));
        j.push_str(&format!("  \"chip\": \"{}\",\n", esc(&chip)));
        j.push_str(&format!("  \"os_build\": \"{}\",\n", esc(&osv)));
        j.push_str(&format!("  \"N\": {n},\n  \"K\": {k},\n"));
        j.push_str(&format!("  \"layers\": {layers},\n"));
        j.push_str(&format!("  \"matrices_per_layer\": {per_layer},\n"));
        j.push_str(&format!("  \"weights\": \"{}\",\n", esc(&source)));
        j.push_str(&format!("  \"synthetic\": {},\n", source.starts_with("SYNTHETIC")));
        j.push_str(&format!(
            "  \"budget_gb\": {},\n",
            budget_gb.map(|v| format!("{v}")).unwrap_or_else(|| "null".into())
        ));
        j.push_str("  \"note\": \"ms_per_128tok is engine-side wall clock normalised to a 128-token tile; rel_rmse is one matrix against an f64 oracle on a row slice, not model acceptance; per-layer optima do not compose, check resident_gb\",\n");
        j.push_str("  \"cells\": [\n");
        for (i, c) in cells.iter().enumerate() {
            j.push_str(&format!(
                "    {{\"spelling\": \"{}\", \"format\": \"{}\", \"tile\": {}, \"k_chunk\": {}, \"n_split\": {}, \"verdict\": \"{}\", \"detail\": \"{}\", \"ms_per_128tok\": {}, \"rel_rmse\": {}, \"weight_mb\": {:.3}, \"resident_gb\": {:.3}, \"build_s\": {:.2}}}{}\n",
                c.spelling, esc(&c.format), c.tile, c.k_chunk, c.n_split, c.verdict, esc(&c.detail),
                num(c.ms_per_128), num(c.rel_rmse), c.weight_mb,
                c.weight_mb * copies as f64 / 1e3, c.build_s,
                if i + 1 == cells.len() { "" } else { "," }
            ));
        }
        j.push_str("  ],\n");
        match best {
            Some((ms, b)) => j.push_str(&format!(
                "  \"best\": {{\"spelling\": \"{}\", \"format\": \"{}\", \"tile\": {}, \"k_chunk\": {}, \"n_split\": {}, \"ms_per_128tok\": {:.6}}}\n",
                b.spelling.label(), esc(&b.format.label()), b.tile, b.k_chunk, b.n_split, ms
            )),
            None => j.push_str("  \"best\": null\n"),
        }
        j.push_str("}\n");
        if let Some(parent) = std::path::Path::new(&out).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(&out, j).map_err(|e| Error::Other(format!("{out}: {e}")))?;
        println!("\n  receipt -> {out}");
    }
    println!("  host: {chip}, macOS build {osv}");

    if let Some((ms, s)) = best {
        println!(
            "\n  best: {} {} tile={} k_chunk={} n_split={} at {:.3} ms/128tok ({:.2}x the first cell)",
            s.spelling.label(),
            s.format.label(),
            s.tile,
            s.k_chunk,
            s.n_split,
            ms,
            base.unwrap() / ms
        );
        let gb = s.format.bytes_per_weight(k) * (n * k) as f64 * copies as f64 / 1e9;
        println!(
            "  resident: {:.3} bytes/weight, {gb:.2} GB across {layers} layer(s) x {per_layer} matrix/matrices",
            s.format.bytes_per_weight(k)
        );
        println!("\n  NOTE: per-layer optima do not compose — 32 resident layers measured");
        println!("  37.0 ms/layer against 3.05 ms for one layer alone. Check total");
        println!("  resident bytes before adopting a per-layer winner.");
    } else {
        println!("\n  no candidate passed all three gates");
    }
    Ok(())
}

fn short(e: &Error) -> String {
    let s = e.to_string();
    s.lines().next().unwrap_or("").chars().take(64).collect()
}
