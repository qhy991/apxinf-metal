//! The op-class registry: the shape each class is searched at, the schedule
//! axes it physically has, the checks a schedule must pass before anything is
//! built, and the f64 oracle its output is judged against.
//!
//! Pure host code. The library's test build compiles this file too (see
//! `lib.rs`), so `cargo test --lib` covers the checks.
use apxinf_core::{Error, Result};
use half::f16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Fp16,
    Int8,
    /// `bits`-wide palette, one table per `group` rows. Group size is a
    /// first-class axis: at the same bit width, one table per row measured
    /// 2.9-9.7x slower than the grouped default.
    Lut {
        bits: u32,
        group: usize,
    },
}

impl Format {
    pub fn parse(s: &str) -> Result<Self> {
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
    pub fn label(&self) -> String {
        match self {
            Self::Fp16 => "fp16".into(),
            Self::Int8 => "int8".into(),
            Self::Lut { bits, group } => format!("lut{bits}:{group}"),
        }
    }
    /// Bytes per weight, for the resident-bytes constraint. `cols` is the
    /// number of weights one row's scale covers.
    pub fn bytes_per_weight(&self, cols: usize) -> f64 {
        match self {
            Self::Fp16 => 2.0,
            Self::Int8 => 1.0 + 2.0 / cols as f64,
            Self::Lut { bits, group } => {
                *bits as f64 / 8.0 + 2.0 * (1usize << bits) as f64 / (group * cols) as f64
            }
        }
    }
}

/// How the same arithmetic is written. At batch 1 `matmul` measured entirely on
/// the CPU while an equivalent 1x1 convolution reached the engine, so this is a
/// placement axis, not a style one — and it is the only axis whose outcome is
/// binary rather than a ratio.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Spelling {
    Conv1x1,
    Matmul,
}

impl Spelling {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "conv" | "conv1x1" => Ok(Self::Conv1x1),
            "matmul" => Ok(Self::Matmul),
            _ => Err(Error::Other(format!("unknown spelling {s}"))),
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            Self::Conv1x1 => "conv",
            Self::Matmul => "matmul",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpClass {
    Linear,
    Conv2d,
    Norm,
    Moe,
}

/// Which formats a class can search.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Formats {
    Any,
    /// The only tensor is gamma, an FP16 function input: a conv whose weight is
    /// an input measured on the CPU, so gamma cannot become a constexpr either.
    GammaFp16,
}

/// The schedule axes a class has. `None` is an axis the class does not have:
/// it takes its neutral value and its flag is refused. `Some` names what the
/// axis splits for this class.
pub struct Axes {
    pub spelling: bool,
    pub tile: &'static str,
    pub k_chunk: Option<&'static str>,
    pub n_split: Option<&'static str>,
    pub formats: Formats,
}

impl OpClass {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "linear" => Ok(Self::Linear),
            "conv2d" => Ok(Self::Conv2d),
            "norm" => Ok(Self::Norm),
            "moe" => Ok(Self::Moe),
            _ => Err(Error::Other(format!(
                "unknown --op {s}: expected linear, conv2d, norm or moe"
            ))),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::Conv2d => "conv2d",
            Self::Norm => "norm",
            Self::Moe => "moe",
        }
    }

    pub const fn axes(self) -> Axes {
        match self {
            Self::Linear => Axes {
                spelling: true,
                tile: "tokens",
                k_chunk: Some("K"),
                n_split: Some("output columns"),
                formats: Formats::Any,
            },
            Self::Conv2d => Axes {
                spelling: true,
                tile: "input H and W",
                k_chunk: Some("C"),
                n_split: Some("output channels"),
                formats: Formats::Any,
            },
            Self::Norm => Axes {
                spelling: false,
                tile: "tokens",
                k_chunk: None,
                n_split: None,
                formats: Formats::GammaFp16,
            },
            Self::Moe => Axes {
                spelling: false,
                tile: "M bucket",
                k_chunk: Some("K"),
                n_split: Some("experts per predict"),
                formats: Formats::Any,
            },
        }
    }

    fn shape_flags(self) -> &'static [&'static str] {
        match self {
            Self::Linear => &["--n", "--k", "--weights"],
            Self::Conv2d => &["--c", "--n", "--h", "--w", "--kernel"],
            Self::Norm => &["--hidden"],
            Self::Moe => &["--experts", "--m", "--n", "--k"],
        }
    }
}

const COMMON_FLAGS: &[&str] = &[
    "--op",
    "--tiles",
    "--formats",
    "--layers",
    "--matrices-per-layer",
    "--budget-gb",
    "--cache",
    "--refresh",
    "--out",
];

/// Refuses any flag the class has no use for: an axis it does not have, or a
/// parameter that belongs to another class.
pub fn check_flags(class: OpClass, args: &[String]) -> Result<()> {
    let axes = class.axes();
    for flag in args
        .iter()
        .map(String::as_str)
        .filter(|a| a.starts_with("--"))
    {
        let axis = match flag {
            "--spellings" => Some(("spelling", axes.spelling)),
            "--chunks" => Some(("k_chunk", axes.k_chunk.is_some())),
            "--n-splits" => Some(("n_split", axes.n_split.is_some())),
            _ => None,
        };
        match axis {
            Some((_, true)) => {}
            Some((name, false)) => {
                return Err(Error::Other(format!(
                    "{} has no {name} axis; drop {flag}",
                    class.label()
                )))
            }
            None if COMMON_FLAGS.contains(&flag) || class.shape_flags().contains(&flag) => {}
            None => {
                return Err(Error::Other(format!(
                    "{flag} is not a {} flag",
                    class.label()
                )))
            }
        }
    }
    Ok(())
}

pub fn check_format(class: OpClass, format: Format) -> Result<()> {
    match (class.axes().formats, format) {
        (Formats::Any, _) | (Formats::GammaFp16, Format::Fp16) => Ok(()),
        (Formats::GammaFp16, f) => Err(Error::Other(format!(
            "{} formats cover gamma only, an fp16 input; {} is not legal",
            class.label(),
            f.label()
        ))),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub spelling: Spelling,
    pub tile: usize,
    pub k_chunk: usize,
    /// Output-column split: N emitted as `n_split` separate convolutions that are
    /// concatenated, rather than one wide one. For `moe`, experts per predict.
    pub n_split: usize,
    pub format: Format,
}

/// The layer searched, one variant per class.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Problem {
    /// `[N, K]` weights over `[1, K, 1, tile]` tokens.
    Linear { n: usize, k: usize },
    /// `[N, C, kernel, kernel]` weights, valid padding, over a `[1, C, tile, tile]`
    /// spatial tile of an `[1, C, H, W]` image.
    Conv2d {
        c: usize,
        n: usize,
        h: usize,
        w: usize,
        kernel: usize,
    },
    /// RMSNorm over `[1, hidden, 1, tile]`, no bias and no eps.
    Norm { hidden: usize },
    /// `experts` resident `[N, K]` experts at decode: `m` tokens padded to a
    /// `tile`-wide M bucket, `n_split` experts packed into each predict's conv.
    Moe {
        experts: usize,
        m: usize,
        n: usize,
        k: usize,
    },
}

const WEIGHT_SEED: u64 = 0x5EED;
const INPUT_SEED: u64 = 0x1A7E;
/// Output rows the oracle checks per weight block.
const ORACLE_ROWS: usize = 8;

/// `len` deterministic values in `[-scale, scale)` from splitmix64.
pub fn splitmix(len: usize, seed: u64, scale: f32) -> Vec<f16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = (z >> 40) as f32 / (1u64 << 24) as f32;
            f16::from_f32((u * 2.0 - 1.0) * scale)
        })
        .collect()
}

/// A function input of a candidate program.
pub struct Input {
    pub name: &'static str,
    pub data: Vec<f16>,
    pub shape: Vec<usize>,
}

/// Relative RMSE accumulated over sampled outputs.
#[derive(Default)]
struct Rmse {
    num: f64,
    den: f64,
}

impl Rmse {
    fn add(&mut self, got: f16, want: f64) {
        let d = f64::from(got.to_f32()) - want;
        self.num += d * d;
        self.den += want * want;
    }
    fn value(&self) -> f64 {
        (self.num / self.den.max(f64::MIN_POSITIVE)).sqrt()
    }
}

fn dot(w: &[f16], x: &[f16], stride: usize, offset: usize) -> f64 {
    w.iter()
        .enumerate()
        .map(|(c, wv)| f64::from(wv.to_f32()) * f64::from(x[c * stride + offset].to_f32()))
        .sum()
}

fn divides(axis: &str, part: usize, of: &str, whole: usize) -> Result<()> {
    if part == 0 || whole % part != 0 {
        return Err(Error::Other(format!(
            "{axis} {part} does not divide {of}={whole}"
        )));
    }
    Ok(())
}

/// A requested chunk width is only valid if it divides the reduced axis
/// exactly: otherwise the parts would differ in width and the schedule is not
/// the one that was asked for. (div_ceil alone is not enough — K=1024 with
/// k_chunk=300 yields 4 parts of 256, silently a different schedule.)
fn chunk_divides(k_chunk: usize, of: &str, whole: usize) -> Result<()> {
    match k_chunk {
        0 => Ok(()),
        c => divides("k_chunk", c, of, whole),
    }
}

impl Problem {
    pub fn parse(class: OpClass, get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let count = |flag: &str, default: Option<usize>| -> Result<usize> {
            let v = match get(flag) {
                Some(v) => v
                    .parse()
                    .map_err(|_| Error::Other(format!("{flag} {v} is not a count")))?,
                None => default
                    .ok_or_else(|| Error::Other(format!("{} needs {flag}", class.label())))?,
            };
            match v {
                0 => Err(Error::Other(format!("{flag} must be positive"))),
                v => Ok(v),
            }
        };
        Ok(match class {
            OpClass::Linear => Self::Linear {
                n: count("--n", Some(4096))?,
                k: count("--k", Some(5632))?,
            },
            OpClass::Conv2d => Self::Conv2d {
                c: count("--c", None)?,
                n: count("--n", None)?,
                h: count("--h", None)?,
                w: count("--w", None)?,
                kernel: count("--kernel", None)?,
            },
            OpClass::Norm => Self::Norm {
                hidden: count("--hidden", None)?,
            },
            OpClass::Moe => Self::Moe {
                experts: count("--experts", None)?,
                m: count("--m", None)?,
                n: count("--n", None)?,
                k: count("--k", None)?,
            },
        })
    }

    /// What an axis flag searches when it is not given.
    pub fn default_values(&self, flag: &str) -> String {
        match (self, flag) {
            (Self::Linear { .. }, "--tiles") => "128,512".into(),
            (Self::Linear { .. }, "--chunks") => "0,1408,2816".into(),
            (Self::Linear { .. }, "--formats") => "fp16,int8,lut6:16".into(),
            (Self::Conv2d { h, w, .. }, "--tiles") => h.min(w).to_string(),
            (Self::Norm { .. }, "--tiles") => "128".into(),
            (Self::Moe { m, .. }, "--tiles") => m.to_string(),
            (_, "--chunks") => "0".into(),
            (_, "--n-splits") => "1".into(),
            (_, "--spellings") => "conv".into(),
            _ => "fp16".into(),
        }
    }

    /// The resident tensor the format encodes, as `[rows, in channels, kh, kw]`.
    /// Norm's is gamma.
    pub fn weight_shape(&self) -> [usize; 4] {
        match *self {
            Self::Linear { n, k } => [n, k, 1, 1],
            Self::Conv2d { c, n, kernel, .. } => [n, c, kernel, kernel],
            Self::Norm { hidden } => [hidden, 1, 1, 1],
            Self::Moe { experts, n, k, .. } => [experts * n, k, 1, 1],
        }
    }

    /// `(rows, weights per row)`: the GEMM a row scale is computed over.
    pub fn gemm(&self) -> (usize, usize) {
        let [rows, cin, kh, kw] = self.weight_shape();
        (rows, cin * kh * kw)
    }

    pub fn describe(&self) -> String {
        match *self {
            Self::Linear { n, k } => format!("[{n}, {k}]"),
            Self::Conv2d { c, n, h, w, kernel } => {
                format!("conv2d [C={c}, N={n}, H={h}, W={w}, kernel={kernel}]")
            }
            Self::Norm { hidden } => format!("norm [hidden={hidden}]"),
            Self::Moe { experts, m, n, k } => format!("moe [E={experts}, m={m}, N={n}, K={k}]"),
        }
    }

    /// The shape part of the cache key. Linear keeps the key it always had.
    pub fn key(&self) -> String {
        match *self {
            Self::Linear { n, k } => format!("{n}x{k}"),
            Self::Conv2d { c, n, h, w, kernel } => format!("conv2d-c{c}-n{n}-h{h}-w{w}-k{kernel}"),
            Self::Norm { hidden } => format!("norm-c{hidden}"),
            Self::Moe { experts, m, n, k } => format!("moe-e{experts}-m{m}-n{n}-k{k}"),
        }
    }

    pub fn params_json(&self) -> String {
        match *self {
            Self::Linear { n, k } => format!("{{\"N\": {n}, \"K\": {k}}}"),
            Self::Conv2d { c, n, h, w, kernel } => {
                format!("{{\"C\": {c}, \"N\": {n}, \"H\": {h}, \"W\": {w}, \"kernel\": {kernel}}}")
            }
            Self::Norm { hidden } => format!("{{\"hidden\": {hidden}}}"),
            Self::Moe { experts, m, n, k } => {
                format!("{{\"experts\": {experts}, \"m\": {m}, \"N\": {n}, \"K\": {k}}}")
            }
        }
    }

    /// What one `ms_per_unit` buys.
    pub fn unit(&self) -> &'static str {
        match self {
            Self::Linear { .. } | Self::Norm { .. } => "128tok",
            Self::Conv2d { .. } => "image",
            Self::Moe { .. } => "expert",
        }
    }

    pub fn note(&self) -> &'static str {
        match self {
            Self::Linear { .. } => "ms_per_128tok is engine-side wall clock normalised to a 128-token tile; rel_rmse is one matrix against an f64 oracle on a row slice, not model acceptance; per-layer optima do not compose, check resident_gb",
            Self::Conv2d { .. } => "tile is the side of a square input tile with its kernel halo; ms_per_unit is the predicts needed to cover the whole HxW output; rel_rmse samples 8 output channels at about 64 positions against an f64 oracle",
            Self::Norm { .. } => "y = x * rsqrt(reduce_sum(x*x)) * gamma with 1/C folded into gamma; no bias, no eps; ms_per_128tok is normalised to a 128-token tile; rel_rmse samples 16 tokens across every channel against an f64 RMSNorm",
            Self::Moe { .. } => "n_split is experts per predict, packed into one conv; tile is the M bucket the m tokens are padded to; ms_per_unit is one round over every resident expert divided by the experts; rel_rmse samples 8 rows of every expert on the m real tokens",
        }
    }

    pub fn synthetic_weights(&self) -> (Vec<f16>, String) {
        let (rows, cols) = self.gemm();
        let len = rows * cols;
        match self {
            // Linear keeps the fallback it always had, so its receipts stay
            // comparable across runs.
            Self::Linear { .. } => (
                (0..len)
                    .map(|i| {
                        f16::from_f32((((i * 2654435761) % 2000) as f32 / 1000.0 - 1.0) * 0.02)
                    })
                    .collect(),
                "SYNTHETIC weights (optima may not transfer)".to_owned(),
            ),
            Self::Norm { .. } => (
                splitmix(len, WEIGHT_SEED, 0.25)
                    .into_iter()
                    .map(|v| f16::from_f32(1.0 + v.to_f32()))
                    .collect(),
                "SYNTHETIC gamma, splitmix64 seed 0x5eed".to_owned(),
            ),
            _ => (
                splitmix(len, WEIGHT_SEED, 0.02),
                "SYNTHETIC weights, splitmix64 seed 0x5eed (optima may not transfer)".to_owned(),
            ),
        }
    }

    /// Program inputs for schedule `s`. `w` is the class's weight tensor; norm
    /// turns its gamma into the folded program input.
    pub fn inputs(&self, s: &Schedule, w: &[f16]) -> Vec<Input> {
        let t = s.tile;
        match *self {
            Self::Linear { k, .. } => vec![Input {
                name: "x",
                data: (0..k * t)
                    .map(|i| f16::from_f32(((i * 40503) % 200) as f32 / 100.0 - 1.0))
                    .collect(),
                shape: vec![1, k, 1, t],
            }],
            Self::Conv2d { c, .. } => vec![Input {
                name: "x",
                data: splitmix(c * t * t, INPUT_SEED, 1.0),
                shape: vec![1, c, t, t],
            }],
            Self::Norm { hidden } => {
                // mean(x^2) = sum(x^2) / C, so rsqrt(mean) = rsqrt(sum) * sqrt(C):
                // the 1/C folds into gamma and the program needs no divide.
                let fold = (hidden as f32).sqrt();
                vec![
                    Input {
                        name: "x",
                        data: splitmix(hidden * t, INPUT_SEED, 1.0),
                        shape: vec![1, hidden, 1, t],
                    },
                    Input {
                        name: "gamma",
                        data: w.iter().map(|g| f16::from_f32(g.to_f32() * fold)).collect(),
                        shape: vec![1, hidden, 1, 1],
                    },
                ]
            }
            Self::Moe { m, k, .. } => {
                // Bucket padding is zeros: only the first m columns are tokens.
                let mut x = splitmix(k * t, INPUT_SEED, 1.0);
                for (i, v) in x.iter_mut().enumerate() {
                    if i % t >= m {
                        *v = f16::ZERO;
                    }
                }
                vec![Input {
                    name: "x",
                    data: x,
                    shape: vec![1, k, 1, t],
                }]
            }
        }
    }

    /// The outputs of each predict, named `y`, `y1`, ... for a column split.
    pub fn outputs(&self, s: &Schedule) -> Vec<(String, Vec<usize>)> {
        let t = s.tile;
        let shares = |rows: usize, h: usize, w: usize| -> Vec<(String, Vec<usize>)> {
            (0..s.n_split)
                .map(|i| {
                    let name = if i == 0 {
                        "y".to_owned()
                    } else {
                        format!("y{i}")
                    };
                    (name, vec![1, rows / s.n_split, h, w])
                })
                .collect()
        };
        match *self {
            Self::Linear { n, .. } => shares(n, 1, t),
            Self::Conv2d { n, kernel, .. } => shares(n, t - kernel + 1, t - kernel + 1),
            Self::Norm { hidden } => vec![("y".into(), vec![1, hidden, 1, t])],
            Self::Moe { n, .. } => vec![("y".into(), vec![1, s.n_split * n, 1, t])],
        }
    }

    /// How many predicts one round of `s` takes.
    pub fn predicts(&self, s: &Schedule) -> usize {
        match *self {
            Self::Moe { experts, .. } => experts / s.n_split,
            _ => 1,
        }
    }

    /// Refuses a schedule that is not the one it claims to be, before anything
    /// is built.
    pub fn validate(&self, s: &Schedule) -> Result<()> {
        match *self {
            Self::Linear { n, k } => {
                chunk_divides(s.k_chunk, "K", k)?;
                divides("n_split", s.n_split, "N", n)
            }
            Self::Conv2d { c, n, h, w, kernel } => {
                if s.tile > h.min(w) {
                    return Err(Error::Other(format!(
                        "tile {} exceeds the {h}x{w} input",
                        s.tile
                    )));
                }
                if kernel > s.tile {
                    return Err(Error::Other(format!(
                        "kernel {kernel} is larger than the {t}x{t} input tile",
                        t = s.tile
                    )));
                }
                chunk_divides(s.k_chunk, "C", c)?;
                divides("n_split", s.n_split, "N", n)
            }
            Self::Norm { .. } => Ok(()),
            Self::Moe { experts, m, k, .. } => {
                if s.tile < m {
                    return Err(Error::Other(format!(
                        "M bucket {} cannot hold m={m} tokens",
                        s.tile
                    )));
                }
                chunk_divides(s.k_chunk, "K", k)?;
                divides("experts per predict", s.n_split, "E", experts)
            }
        }
    }

    /// One round's median wall clock in the class's unit.
    pub fn ms_per_unit(&self, s: &Schedule, round_ms: f64) -> f64 {
        match *self {
            Self::Linear { .. } | Self::Norm { .. } => round_ms * 128.0 / s.tile as f64,
            Self::Conv2d { h, w, kernel, .. } => {
                let step = s.tile - kernel + 1;
                let across = |extent: usize| (extent - kernel + 1).div_ceil(step);
                round_ms * (across(h) * across(w)) as f64
            }
            Self::Moe { experts, .. } => round_ms / experts as f64,
        }
    }

    /// Relative RMSE of `got` (the first output of each predict) against an f64
    /// oracle on a sample: a mis-encoded palette, a wrong packing or a wrong
    /// fold all show here, and the full product is too costly to check.
    pub fn rel_rmse(&self, s: &Schedule, w: &[f16], inputs: &[Input], got: &[Vec<f16>]) -> f64 {
        let x = &inputs[0].data;
        let t = s.tile;
        let mut e = Rmse::default();
        match *self {
            Self::Linear { n, k } => {
                for r in 0..ORACLE_ROWS.min(n / s.n_split) {
                    for tok in 0..t {
                        e.add(got[0][r * t + tok], dot(&w[r * k..(r + 1) * k], x, t, tok));
                    }
                }
            }
            Self::Conv2d { c, n, kernel, .. } => {
                let o = t - kernel + 1;
                let area = kernel * kernel;
                for r in 0..ORACLE_ROWS.min(n / s.n_split) {
                    for pos in (0..o * o).step_by((o * o / 64).max(1)) {
                        let (oy, ox) = (pos / o, pos % o);
                        let mut want = 0f64;
                        for ch in 0..c {
                            for i in 0..kernel {
                                for j in 0..kernel {
                                    let wv = w[(r * c + ch) * area + i * kernel + j];
                                    let xv = x[ch * t * t + (oy + i) * t + ox + j];
                                    want += f64::from(wv.to_f32()) * f64::from(xv.to_f32());
                                }
                            }
                        }
                        e.add(got[0][r * o * o + pos], want);
                    }
                }
            }
            Self::Norm { hidden } => {
                for tok in (0..t).step_by((t / 16).max(1)) {
                    let ss: f64 = (0..hidden)
                        .map(|ch| f64::from(x[ch * t + tok].to_f32()).powi(2))
                        .sum();
                    let rms = (ss / hidden as f64).sqrt();
                    for ch in 0..hidden {
                        let xv = f64::from(x[ch * t + tok].to_f32());
                        e.add(got[0][ch * t + tok], xv / rms * f64::from(w[ch].to_f32()));
                    }
                }
            }
            Self::Moe { m, n, k, .. } => {
                for (g, out) in got.iter().enumerate() {
                    for ex in 0..s.n_split {
                        for r in 0..ORACLE_ROWS.min(n) {
                            let row = (g * s.n_split + ex) * n + r;
                            for tok in 0..m {
                                let want = dot(&w[row * k..(row + 1) * k], x, t, tok);
                                e.add(out[(ex * n + r) * t + tok], want);
                            }
                        }
                    }
                }
            }
        }
        e.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn schedule(tile: usize, k_chunk: usize, n_split: usize) -> Schedule {
        Schedule {
            spelling: Spelling::Conv1x1,
            tile,
            k_chunk,
            n_split,
            format: Format::Fp16,
        }
    }

    fn refusal(r: Result<()>) -> String {
        r.expect_err("must be refused").to_string()
    }

    #[test]
    fn a_class_refuses_the_axes_it_does_not_have() {
        let norm = [
            "--op",
            "norm",
            "--hidden",
            "2048",
            "--tiles",
            "128",
            "--formats",
            "fp16",
        ];
        assert!(check_flags(OpClass::Norm, &args(&norm)).is_ok());
        for (flag, axis) in [
            ("--chunks", "k_chunk"),
            ("--n-splits", "n_split"),
            ("--spellings", "spelling"),
        ] {
            let e = refusal(check_flags(OpClass::Norm, &args(&[flag, "1"])));
            assert!(e.contains(&format!("norm has no {axis} axis")), "{e}");
        }
        let e = refusal(check_flags(OpClass::Moe, &args(&["--spellings", "conv"])));
        assert!(e.contains("moe has no spelling axis"), "{e}");
        assert!(check_flags(OpClass::Moe, &args(&["--chunks", "0", "--n-splits", "2"])).is_ok());
        let e = refusal(check_flags(OpClass::Linear, &args(&["--hidden", "2048"])));
        assert!(e.contains("--hidden is not a linear flag"), "{e}");
        let e = refusal(check_flags(OpClass::Norm, &args(&["--weights", "w.bin"])));
        assert!(e.contains("--weights is not a norm flag"), "{e}");

        let e = refusal(check_format(OpClass::Norm, Format::Int8));
        assert!(e.contains("int8 is not legal"), "{e}");
        assert!(check_format(OpClass::Norm, Format::Fp16).is_ok());
        assert!(check_format(OpClass::Moe, Format::Lut { bits: 6, group: 16 }).is_ok());
    }

    #[test]
    fn experts_per_predict_must_divide_the_experts() {
        let moe = Problem::Moe {
            experts: 8,
            m: 4,
            n: 64,
            k: 64,
        };
        for split in [1, 2, 4, 8] {
            assert!(moe.validate(&schedule(4, 0, split)).is_ok(), "{split}");
        }
        for split in [0, 3, 16] {
            let e = refusal(moe.validate(&schedule(4, 0, split)));
            assert_eq!(
                e,
                format!("experts per predict {split} does not divide E=8")
            );
        }
    }

    #[test]
    fn a_kernel_larger_than_the_input_is_refused() {
        let conv = Problem::Conv2d {
            c: 8,
            n: 8,
            h: 64,
            w: 48,
            kernel: 5,
        };
        assert!(conv.validate(&schedule(5, 0, 1)).is_ok());
        assert!(conv.validate(&schedule(48, 0, 1)).is_ok());
        let e = refusal(conv.validate(&schedule(4, 0, 1)));
        assert_eq!(e, "kernel 5 is larger than the 4x4 input tile");
        let e = refusal(conv.validate(&schedule(49, 0, 1)));
        assert_eq!(e, "tile 49 exceeds the 64x48 input");
    }
}
