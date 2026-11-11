//! anesched — search the MIL-level schedule space for one layer of an op class.
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
//! Each op class carries only the axes that are physically meaningful for it;
//! the table and what each axis means per class live in `class.rs`.
//!
//!   cargo run -p apxinf-coreml --release --features native --example anesched -- \
//!       [--op linear] --n 4096 --k 5632 [--tiles 128,512] [--chunks 0,1408,2816] \
//!       [--formats fp16,int8,lut6:16] [--weights <f16 binary N*K>]
//!       --op conv2d --c <in> --n <out> --h <h> --w <w> --kernel <k>
//!       --op norm   --hidden <c> --tiles <t>
//!       --op moe    --experts <E> --m <tokens> --n <n> --k <k> [--n-splits <experts per predict>]
//!       --op attention --heads <h> --d <d> --seq <s> [--kv-heads <h>] [--kv-seq <n>] [--causal]
//!                   [--spellings folded,separate]
mod class;

use apxinf_core::{Error, Result};
use apxinf_coreml::{
    ComputeDeviceKind, ComputeUnits, Operand, Package, ProgramBuilder, ProgramCache, Var,
};
use class::{
    check_flags, check_format, check_spelling, Format, OpClass, Problem, Schedule, Spelling,
};
use half::f16;

/// What happened to one searched cell. A schedule the compiler refused, or
/// placed off the engine, is a result worth keeping — it is why the next
/// search does not retry it.
enum Outcome {
    OverBudget(String),
    BuildRefused(String),
    OffEngine {
        detail: String,
        weight_mb: f64,
    },
    Ok {
        ms: f64,
        rel_rmse: f64,
        weight_mb: f64,
        /// The operations of the first program, every one placed on the engine.
        engine_ops: Vec<String>,
    },
}

struct Cell {
    schedule: Schedule,
    outcome: Outcome,
    build_s: f64,
}

impl Outcome {
    fn verdict(&self) -> &'static str {
        match self {
            Self::OverBudget(_) => "over-budget",
            Self::BuildRefused(_) => "build-refused",
            Self::OffEngine { .. } => "off-engine",
            Self::Ok { .. } => "ok",
        }
    }
    fn detail(&self) -> &str {
        match self {
            Self::OverBudget(d) | Self::BuildRefused(d) | Self::OffEngine { detail: d, .. } => d,
            Self::Ok { .. } => "",
        }
    }
    fn weight_mb(&self) -> f64 {
        match self {
            Self::OffEngine { weight_mb, .. } | Self::Ok { weight_mb, .. } => *weight_mb,
            _ => 0.0,
        }
    }
}

/// Emits `conv(x, w)` with `w` encoded in `format`. `shape` is the weight's
/// `[rows, in channels, kh, kw]`, row-major in `w`.
fn weight_conv(
    b: &mut ProgramBuilder,
    x: &Var,
    w: &[f16],
    shape: [usize; 4],
    format: Format,
) -> Result<Var> {
    let rows = shape[0];
    let row_len = w.len() / rows;
    let wv = match format {
        Format::Fp16 => return b.conv_fp16(x, w, &shape),
        Format::Int8 => {
            let mut q = Vec::with_capacity(w.len());
            let mut scale = Vec::with_capacity(rows);
            for row in w.chunks_exact(row_len) {
                let peak = row
                    .iter()
                    .fold(0f32, |m, v| m.max(v.to_f32().abs()))
                    .max(f32::MIN_POSITIVE);
                let s = peak / 127.0;
                scale.push(f16::from_f32(s));
                q.extend(
                    row.iter()
                        .map(|v| (v.to_f32() / s).round().clamp(-127.0, 127.0) as i8),
                );
            }
            b.constexpr_blockwise_shift_scale(&q, &shape, &scale, &[rows, 1, 1, 1])?
        }
        Format::Lut { bits, group } => {
            if rows % group != 0 {
                return Err(Error::Other(format!(
                    "group {group} must divide {rows} rows"
                )));
            }
            let groups = rows / group;
            let levels = ((1u32 << (bits - 1)) - 1) as f32;
            let entries = 1usize << bits;
            let mut idx = vec![0u8; w.len()];
            let mut lut = Vec::with_capacity(groups * entries);
            for g in 0..groups {
                let block = &w[g * group * row_len..(g + 1) * group * row_len];
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
                    idx[g * group * row_len + i] = (q + levels) as u8;
                }
            }
            b.constexpr_lut_to_dense(&idx, &shape, &lut, groups, bits)?
        }
    };
    b.conv(x, &wv)
}

/// One conv program over input `x_shape`: the weight's in channels split into
/// `k_chunk`-wide convolutions that are summed, its rows into `out_split`
/// outputs. Linear, conv2d and every MoE predict are this program.
fn conv_program(
    w: &[f16],
    shape: [usize; 4],
    x_shape: &[usize],
    spelling: Spelling,
    k_chunk: usize,
    out_split: usize,
    format: Format,
) -> Result<Package> {
    let [rows, cin, kh, kw] = shape;
    let parts = if k_chunk == 0 { 1 } else { cin / k_chunk };
    let width = cin / parts;
    let n_part = rows / out_split;
    let area = kh * kw;
    let row_len = cin * area;

    if spelling == Spelling::Matmul {
        if parts > 1 || out_split > 1 {
            return Err(Error::Other(
                "matmul spelling is probed unsplit, as the placement question".into(),
            ));
        }
        if format != Format::Fp16 {
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

    let mut b = ProgramBuilder::new();
    let x = b.input("x", x_shape);
    let xs = if parts > 1 {
        b.split(&x, &vec![width; parts], 1)?
    } else {
        vec![x]
    };
    let mut columns: Vec<Var> = Vec::with_capacity(out_split);
    for c in 0..out_split {
        let mut acc: Option<Var> = None;
        for (p, xi) in xs.iter().enumerate() {
            // rows [c*n_part, (c+1)*n_part) x in channels [p*width, (p+1)*width)
            let slice: Vec<f16> = w[c * n_part * row_len..(c + 1) * n_part * row_len]
                .chunks_exact(row_len)
                .flat_map(|row| {
                    row[p * width * area..(p + 1) * width * area]
                        .iter()
                        .copied()
                })
                .collect();
            let y = weight_conv(&mut b, xi, &slice, [n_part, width, kh, kw], format)?;
            acc = Some(match acc {
                None => y,
                Some(a) => b.add(&a, Operand::Var(&y), None)?,
            });
        }
        columns.push(acc.expect("at least one part"));
    }
    // The builder has no concat, and adding one would measure the join rather
    // than the split. Each column share is its own output: the engine still runs
    // `out_split` narrower convolutions, which is the axis under test.
    let outs: Vec<Var> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let name = if i == 0 {
                "y".to_owned()
            } else {
                format!("y{i}")
            };
            b.identity(c, Some(&name))
        })
        .collect();
    b.finish(&outs.iter().collect::<Vec<_>>())
}

/// RMSNorm as `x * rsqrt(reduce_sum(x * x)) * gamma`, gamma a function input
/// with 1/C already folded in.
fn norm_program(hidden: usize, tile: usize) -> Result<Package> {
    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, hidden, 1, tile]);
    let gamma = b.input("gamma", &[1, hidden, 1, 1]);
    let sq = b.mul(&x, Operand::Var(&x), None)?;
    let sum = b.reduce_sum(&sq, &[1], true)?;
    let r = b.rsqrt(&sum);
    let xn = b.mul(&x, Operand::Var(&r), None)?;
    let y = b.mul(&xn, Operand::Var(&gamma), Some("y"))?;
    b.finish(&[&y])
}

/// One query tile of attention, `softmax(q k^T / sqrt(d) + mask) v`, with the
/// scale where `s.spelling` puts it: folded leaves it to the host, separate is
/// a `mul` over the scores. Grouped query heads are reshaped into the query
/// rows of their K/V head, so one matmul serves the group and K and V are
/// never repeated.
fn attention_program(
    heads: usize,
    kv_heads: usize,
    d: usize,
    kv_seq: usize,
    causal: bool,
    s: &Schedule,
) -> Result<Package> {
    let (t, group) = (s.tile, heads / kv_heads);
    let dims = |v: &[usize]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();
    let mut b = ProgramBuilder::new();
    let q = b.input("q", &[1, heads, t, d]);
    let k = b.input("k", &[1, kv_heads, kv_seq, d]);
    let v = b.input("v", &[1, kv_heads, kv_seq, d]);
    let mask = causal.then(|| b.input("mask", &[1, 1, group * t, kv_seq]));
    let q = match group {
        1 => q,
        _ => b.reshape(&q, &dims(&[1, kv_heads, group * t, d]))?,
    };
    let kt = b.transpose(&k, &[0, 1, 3, 2])?;
    let mut scores = b.matmul(&q, &kt)?;
    if s.spelling == Spelling::ScaleSeparate {
        let scale = f16::from_f32(1.0 / (d as f32).sqrt());
        scores = b.mul(&scores, Operand::Scalar(scale), None)?;
    }
    if let Some(m) = &mask {
        scores = b.add(&scores, Operand::Var(m), None)?;
    }
    let p = b.softmax(&scores, -1)?;
    let mut o = b.matmul(&p, &v)?;
    if group > 1 {
        o = b.reshape(&o, &dims(&[1, heads, t, d]))?;
    }
    let y = b.identity(&o, Some("y"));
    b.finish(&[&y])
}

/// The programs one round of schedule `s` runs: one, or one per MoE predict.
fn build(p: &Problem, w: &[f16], s: Schedule) -> Result<Vec<Package>> {
    p.validate(&s)?;
    let t = s.tile;
    match *p {
        Problem::Linear { k, .. } | Problem::Conv2d { c: k, .. } => {
            let x_shape = match *p {
                Problem::Conv2d { .. } => [1, k, t, t],
                _ => [1, k, 1, t],
            };
            let one = conv_program(
                w,
                p.weight_shape(),
                &x_shape,
                s.spelling,
                s.k_chunk,
                s.n_split,
                s.format,
            )?;
            Ok(vec![one])
        }
        Problem::Norm { hidden } => Ok(vec![norm_program(hidden, t)?]),
        Problem::Attention {
            heads,
            kv_heads,
            d,
            kv_seq,
            causal,
            ..
        } => Ok(vec![attention_program(
            heads, kv_heads, d, kv_seq, causal, &s,
        )?]),
        // `n_split` experts' [N, K] weights stacked into one [n_split*N, K] conv.
        Problem::Moe { n, k, .. } => w
            .chunks_exact(s.n_split * n * k)
            .map(|group| {
                conv_program(
                    group,
                    [s.n_split * n, k, 1, 1],
                    &[1, k, 1, t],
                    s.spelling,
                    s.k_chunk,
                    1,
                    s.format,
                )
            })
            .collect(),
    }
}

/// Runs the gates on one schedule and times it if it survives them.
fn evaluate(
    p: &Problem,
    w: &[f16],
    s: Schedule,
    budget_gb: Option<f64>,
    copies: usize,
    dir: &std::path::Path,
) -> Result<Cell> {
    let t_build = std::time::Instant::now();
    let cell = |outcome: Outcome| Cell {
        schedule: s,
        outcome,
        build_s: t_build.elapsed().as_secs_f64(),
    };

    // gate 0: resident bytes across every layer that will hold this schedule.
    // Cheapest gate of all — no compile needed.
    let (rows, cols) = p.gemm();
    let total_gb = s.format.bytes_per_weight(cols) * (rows * cols) as f64 * copies as f64 / 1e9;
    if let Some(cap) = budget_gb {
        if total_gb > cap {
            return Ok(Cell {
                build_s: 0.0,
                ..cell(Outcome::OverBudget(format!(
                    "{total_gb:.2} GB over the {cap:.2} GB budget"
                )))
            });
        }
    }

    // gate 1: the compiler may refuse, opaquely. That is a normal outcome.
    let packages = match build(p, w, s) {
        Ok(p) => p,
        Err(e) => return Ok(cell(Outcome::BuildRefused(short(&e)))),
    };
    let weight_mb = packages.iter().map(|p| p.weights.len()).sum::<usize>() as f64 / 1e6;

    let _ = std::fs::remove_dir_all(dir);
    let cache = ProgramCache::new(dir);

    // gate 2: placement is observed, not commanded. Reject anything the
    // compiler did not put entirely on the engine.
    let mut models = Vec::with_capacity(packages.len());
    let mut engine_ops: Vec<String> = Vec::new();
    for (n, package) in packages.iter().enumerate() {
        let mut off: Vec<String> = Vec::new();
        let model = cache.load_with_placement(
            package,
            ComputeUnits::CpuAndNeuralEngine,
            std::time::Duration::from_secs(180),
            |facts| {
                for f in &facts.functions {
                    for op in &f.operations {
                        if let Some(d) = op.usage.as_ref().and_then(|u| u.preferred.as_ref()) {
                            if d.kind != ComputeDeviceKind::NeuralEngine {
                                off.push(format!("{}:{:?}", op.operator, d.kind));
                            } else if n == 0 {
                                engine_ops.push(op.operator.clone());
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
        match model {
            Ok(m) => models.push(m),
            Err(e) => {
                let detail = short(&e);
                drop(models);
                let _ = std::fs::remove_dir_all(dir);
                return Ok(cell(Outcome::OffEngine { detail, weight_mb }));
            }
        }
    }

    let inputs = p.inputs(&s, w);
    let ins: Vec<(&str, &[f16], &[usize])> = inputs
        .iter()
        .map(|i| (i.name, &i.data[..], &i.shape[..]))
        .collect();
    let out_specs = p.outputs(&s);
    let outs: Vec<(&str, &[usize])> = out_specs
        .iter()
        .map(|(a, b)| (a.as_str(), &b[..]))
        .collect();

    // gate 3: accuracy against the class's f64 oracle on a sample.
    let mut got = Vec::with_capacity(models.len());
    for m in &models {
        got.push(m.predict(&ins, &outs)?.remove(0));
    }
    let rel_rmse = p.rel_rmse(&s, w, &inputs, &got);

    // One round is every predict the schedule needs, so a split that costs
    // more calls pays for them.
    let round = || -> Result<()> {
        for m in &models {
            m.predict(&ins, &outs)?;
        }
        Ok(())
    };
    for _ in 0..5 {
        round()?;
    }
    let mut ts = Vec::new();
    for _ in 0..20 {
        let t0 = std::time::Instant::now();
        round()?;
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(f64::total_cmp);
    let ms = p.ms_per_unit(&s, ts[ts.len() / 2]);
    drop(models);
    let _ = std::fs::remove_dir_all(dir);
    Ok(cell(Outcome::Ok {
        ms,
        rel_rmse,
        weight_mb,
        engine_ops,
    }))
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
    v.map(|x| format!("{x:.6}"))
        .unwrap_or_else(|| "null".into())
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|v| v == name)
        .and_then(|i| a.get(i + 1).cloned())
}

fn list(p: &Problem, name: &str) -> Vec<String> {
    arg(name)
        .unwrap_or_else(|| p.default_values(name))
        .split(',')
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect()
}

fn counts(p: &Problem, name: &str) -> Result<Vec<usize>> {
    list(p, name)
        .iter()
        .map(|v| {
            v.parse()
                .map_err(|_| Error::Other(format!("{name}: {v} is not a count")))
        })
        .collect()
}

/// The schedule's axes as table columns; an axis the class lacks prints `-`.
fn row_label(class: OpClass, s: &Schedule) -> String {
    let axes = class.axes();
    let dash = |on: bool, v: String| if on { v } else { "-".into() };
    format!(
        "{:<8} {:<10} {:>6} {:>8} {:>4}",
        dash(axes.has_spelling(), s.spelling.label().into()),
        s.format.label(),
        s.tile,
        dash(axes.k_chunk.is_some(), s.k_chunk.to_string()),
        dash(axes.n_split.is_some(), s.n_split.to_string()),
    )
}

/// The schedule as a sentence, naming only the axes the class has.
fn summary(class: OpClass, s: &Schedule) -> String {
    let axes = class.axes();
    let mut parts = Vec::new();
    if axes.has_spelling() {
        parts.push(s.spelling.label().to_owned());
    }
    parts.push(s.format.label());
    parts.push(format!("tile={}", s.tile));
    if axes.k_chunk.is_some() {
        parts.push(format!("k_chunk={}", s.k_chunk));
    }
    if axes.n_split.is_some() {
        parts.push(format!("n_split={}", s.n_split));
    }
    parts.join(" ")
}

/// The receipt fields of a schedule; an axis the class lacks is `null`.
fn schedule_json(class: OpClass, s: &Schedule) -> String {
    let axes = class.axes();
    let or_null = |on: bool, v: String| if on { v } else { "null".into() };
    format!(
        "\"spelling\": {}, \"format\": \"{}\", \"tile\": {}, \"k_chunk\": {}, \"n_split\": {}",
        or_null(axes.has_spelling(), format!("\"{}\"", s.spelling.label())),
        esc(&s.format.label()),
        s.tile,
        or_null(axes.k_chunk.is_some(), s.k_chunk.to_string()),
        or_null(axes.n_split.is_some(), s.n_split.to_string()),
    )
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let class = OpClass::parse(&arg("--op").unwrap_or_else(|| "linear".into()))?;
    check_flags(class, &argv)?;
    let problem = Problem::parse(class, arg, |f| argv.iter().any(|a| a == f))?;
    let axes = class.axes();

    let tiles = counts(&problem, "--tiles")?;
    let chunks = counts(&problem, "--chunks")?;
    let spellings: Vec<Spelling> = list(&problem, "--spellings")
        .iter()
        .map(|v| {
            let s = Spelling::parse(v)?;
            check_spelling(class, s).map(|()| s)
        })
        .collect::<Result<_>>()?;
    let n_splits = counts(&problem, "--n-splits")?;
    let formats: Vec<Format> = list(&problem, "--formats")
        .iter()
        .map(|s| {
            let f = Format::parse(s)?;
            check_format(class, f).map(|()| f)
        })
        .collect::<Result<_>>()?;
    // Per-layer optima do not compose: 32 resident layers measured 37.0 ms/layer
    // against 3.05 ms for one layer alone. `--layers` turns the winner's weight
    // bytes into a total, and `--budget-gb` rejects schedules whose total exceeds
    // what the host can hold resident.
    let layers: usize = arg("--layers")
        .unwrap_or_else(|| "1".into())
        .parse()
        .unwrap();
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
    let (rows, cols) = problem.gemm();
    let unit = problem.unit();

    // Real weights when given (linear only; `check_flags` refuses it elsewhere).
    // The synthetic fallback is labelled because it moved the split-K optimum and
    // overstated the gain by 1.4x.
    let (w, source) = match arg("--weights") {
        Some(p) => {
            let raw = std::fs::read(&p).map_err(|e| Error::Other(format!("{p}: {e}")))?;
            if raw.len() != rows * cols * 2 {
                return Err(Error::Other(format!(
                    "{p} holds {} bytes, need {} for {} f16",
                    raw.len(),
                    rows * cols * 2,
                    problem.describe()
                )));
            }
            let w: Vec<f16> = raw
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]))
                .collect();
            (w, format!("real weights {p}"))
        }
        None => problem.synthetic_weights(),
    };

    // A cached winner is only reused when the host matches: same chip and same OS
    // build. `--cache <dir>` reads `<dir>/<chip>-<os>-<shape>-L<layers>.json`, and
    // a hit prints the schedule without compiling anything. `--refresh` searches
    // anyway and overwrites.
    let (chip, osv) = host_key();
    let cache_key = format!(
        "{}-{}-{}-L{layers}x{per_layer}.json",
        chip.replace(' ', "_").replace('/', "_"),
        osv.replace(' ', "_").replace('/', "_"),
        problem.key()
    );
    let cache_file = arg("--cache").map(|d| std::path::PathBuf::from(d).join(&cache_key));
    let refresh = std::env::args().any(|a| a == "--refresh");
    if let (Some(path), false) = (cache_file.as_ref(), refresh) {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Some(best) = text.split("\"best\":").nth(1) {
                println!(
                    "anesched: cache hit for {} on {chip} / {osv}",
                    problem.describe()
                );
                println!("  {}", best.trim().trim_end_matches('}').trim());
                println!("  {}", path.display());
                println!("  (--refresh to search again)");
                return Ok(());
            }
        }
    }

    let mut candidates: Vec<Schedule> = Vec::new();
    for &spelling in &spellings {
        for &format in &formats {
            for &tile in &tiles {
                for &k_chunk in &chunks {
                    for &n_split in &n_splits {
                        candidates.push(Schedule {
                            spelling,
                            tile,
                            k_chunk,
                            n_split,
                            format,
                        });
                    }
                }
            }
        }
    }
    let mut sizes = Vec::new();
    if axes.has_spelling() {
        sizes.push(format!("{} spellings", spellings.len()));
    }
    sizes.push(format!("{} tiles", tiles.len()));
    if axes.k_chunk.is_some() {
        sizes.push(format!("{} chunks", chunks.len()));
    }
    if axes.n_split.is_some() {
        sizes.push(format!("{} n_splits", n_splits.len()));
    }
    sizes.push(format!("{} formats", formats.len()));

    println!("anesched: {}, {source}", problem.describe());
    if class != OpClass::Linear {
        let meaning =
            |name: &str, v: Option<&str>| v.map(|v| format!(", {name} = {v}")).unwrap_or_default();
        println!(
            "  axes: tile = {}{}{}",
            axes.tile,
            meaning("k_chunk over", axes.k_chunk),
            meaning("n_split", axes.n_split)
        );
    }
    println!(
        "  {} = {} candidates\n",
        sizes.join(" x "),
        candidates.len()
    );
    println!(
        "  {:<8} {:<10} {:>6} {:>8} {:>4} {:>10} {:>11} {:>10} {:>9}  {}",
        "spell",
        "format",
        "tile",
        "k_chunk",
        "nsp",
        format!("ms/{unit}"),
        "rel.rmse",
        "MB",
        "vs base",
        "gate"
    );

    let cache_dir = std::env::temp_dir().join(format!("anesched-{}", std::process::id()));
    let mut best: Option<(f64, Schedule)> = None;
    let mut base: Option<f64> = None;
    let mut cells: Vec<Cell> = Vec::new();

    for s in candidates {
        let dir = cache_dir.join(format!(
            "{}-{}-{}",
            s.format.label().replace(':', "_"),
            s.tile,
            s.k_chunk
        ));
        let c = evaluate(&problem, &w, s, budget_gb, copies, &dir)?;
        let label = row_label(class, &s);
        match &c.outcome {
            Outcome::OverBudget(d) => {
                println!(
                    "  {label} {:>10} {:>11} {:>10} {:>9}  budget: {d}",
                    "-", "-", "-", "-"
                )
            }
            Outcome::BuildRefused(d) => {
                println!(
                    "  {label} {:>10} {:>11} {:>10} {:>9}  build: {d}",
                    "-", "-", "-", "-"
                )
            }
            Outcome::OffEngine { detail, weight_mb } => {
                println!(
                    "  {label} {:>10} {:>11} {weight_mb:>10.1} {:>9}  {detail}",
                    "-", "-", "-"
                )
            }
            &Outcome::Ok {
                ms,
                rel_rmse,
                weight_mb,
                ..
            } => {
                let rel = *base.get_or_insert(ms) / ms;
                let calls = match problem.predicts(&s) {
                    1 => String::new(),
                    n => format!(" ({n} predicts)"),
                };
                println!("  {label} {ms:>10.3} {rel_rmse:>11.2e} {weight_mb:>10.1} {rel:>8.2}x  ok{calls}");
                if best.as_ref().is_none_or(|(b, _)| ms < *b) {
                    best = Some((ms, s));
                }
            }
        }
        cells.push(c);
    }
    let _ = std::fs::remove_dir_all(&cache_dir);

    // ms_per_128tok keeps its meaning: it is null for a class whose unit is not
    // a token tile, and ms_per_unit carries the class's own unit.
    let per_128 = |ms: Option<f64>| if unit == "128tok" { ms } else { None };

    // The receipt: every cell including the rejected ones, keyed by the host it
    // is valid on. A later run reads this instead of re-searching; a run on
    // another chip or OS build will miss the key and search again, which is the
    // point.
    if let Some(out) = arg("--out").or_else(|| cache_file.as_ref().map(|p| p.display().to_string()))
    {
        let mut j = String::new();
        j.push_str("{\n");
        j.push_str("  \"tool\": \"anesched\",\n");
        j.push_str(&format!("  \"class\": \"{}\",\n", class.label()));
        j.push_str(&format!("  \"chip\": \"{}\",\n", esc(&chip)));
        j.push_str(&format!("  \"os_build\": \"{}\",\n", esc(&osv)));
        j.push_str(&format!("  \"N\": {rows},\n  \"K\": {cols},\n"));
        j.push_str(&format!("  \"params\": {},\n", problem.params_json()));
        j.push_str(&format!("  \"layers\": {layers},\n"));
        j.push_str(&format!("  \"matrices_per_layer\": {per_layer},\n"));
        j.push_str(&format!("  \"weights\": \"{}\",\n", esc(&source)));
        j.push_str(&format!(
            "  \"synthetic\": {},\n",
            source.starts_with("SYNTHETIC")
        ));
        j.push_str(&format!(
            "  \"budget_gb\": {},\n",
            budget_gb
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| "null".into())
        ));
        j.push_str(&format!("  \"unit\": \"{unit}\",\n"));
        j.push_str(&format!("  \"note\": \"{}\",\n", problem.note()));
        j.push_str("  \"cells\": [\n");
        for (i, c) in cells.iter().enumerate() {
            let (ms, rmse, ops) = match &c.outcome {
                Outcome::Ok {
                    ms,
                    rel_rmse,
                    engine_ops,
                    ..
                } => (Some(*ms), Some(*rel_rmse), &engine_ops[..]),
                _ => (None, None, &[][..]),
            };
            let ops: Vec<String> = ops.iter().map(|o| format!("\"{}\"", esc(o))).collect();
            j.push_str(&format!(
                "    {{{}, \"verdict\": \"{}\", \"detail\": \"{}\", \"ms_per_128tok\": {}, \"rel_rmse\": {}, \"weight_mb\": {:.3}, \"resident_gb\": {:.3}, \"build_s\": {:.2}, \"ms_per_unit\": {}, \"engine_ops\": [{}]}}{}\n",
                schedule_json(class, &c.schedule), c.outcome.verdict(), esc(c.outcome.detail()),
                num(per_128(ms)), num(rmse), c.outcome.weight_mb(),
                c.outcome.weight_mb() * copies as f64 / 1e3, c.build_s, num(ms), ops.join(", "),
                if i + 1 == cells.len() { "" } else { "," }
            ));
        }
        j.push_str("  ],\n");
        match best {
            Some((ms, b)) => j.push_str(&format!(
                "  \"best\": {{{}, \"ms_per_128tok\": {}, \"ms_per_unit\": {ms:.6}}}\n",
                schedule_json(class, &b),
                num(per_128(Some(ms)))
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
            "\n  best: {} at {:.3} ms/{unit} ({:.2}x the first cell)",
            summary(class, &s),
            ms,
            base.unwrap() / ms
        );
        let bpw = s.format.bytes_per_weight(cols);
        let gb = bpw * (rows * cols) as f64 * copies as f64 / 1e9;
        println!(
            "  resident: {bpw:.3} bytes/weight, {gb:.2} GB across {layers} layer(s) x {per_layer} matrix/matrices"
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
