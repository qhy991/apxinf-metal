//! The Neural Engine share of each image-block MLP (EngineTailor A007, #236).
//! MLP rows `[GPU_ROWS, MLP)` of gate/up and the matching columns of `out`
//! run as one Core ML program per block; the GPU keeps rows `[0, GPU_ROWS)`
//! on the fused kernels, and the two shares are summed on the GPU. Opt in
//! with `APXINF_QWEN_IMAGE_ANE=1` (default off).
//!
//! The program: gate and up as per-row symmetric INT8 1x1 convolutions on an
//! input pre-scaled by `2^a` (`r = 2^-a` undoes it), SiLU as
//! `0.5 v (1 + tanh(v / 2))`, the product scaled by `s = 2^b`, then `out` as an
//! INT8 convolution over its first `OUT_INT8` columns plus an FP16 one over the
//! rest. Token tiles of `TILE`. The GPU divides by `2^b` when it adds the share.
//!
//! Compiled programs are cached under `~/.cache/apxinf/ane/`, keyed by a
//! manifest of the weight identity, the block, the recipe and the macOS build,
//! so a warm process loads `.mlmodelc` directories without reading, quantizing
//! or compiling weights.
use super::weights::{Block, GateUp, Transformer, BLOCKS, DIM, MLP, Q4};
use apxinf_core::{Error, Result};
use apxinf_coreml::{ComputeUnits, Model, Operand, Package, ProgramBuilder, ProgramCache};
use apxinf_mlx::{Array, MlxDType, Stream};
use half::f16;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Rows the Neural Engine computes, from row `GPU_ROWS` on.
pub const ANE_ROWS: usize = 6656;
pub const GPU_ROWS: usize = MLP - ANE_ROWS;
/// `out` columns of the share stored as INT8; the remaining ones stay FP16.
const OUT_INT8: usize = 2560;
/// Token tile: 512² (1024 tokens) and 1024×576 (2304) pad to one or two tiles.
pub const TILE: usize = 1152;
/// Bumped whenever the program, its quantizer or the writer changes.
const RECIPE: &str = "qwen-image21/a007-mlp-columns/v1";

/// Whether the caller asked for the Neural Engine share.
pub fn requested() -> bool {
    std::env::var("APXINF_QWEN_IMAGE_ANE").is_ok_and(|v| v == "1")
}

fn env_mb(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn sysctl(name: &str) -> Option<String> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", name])
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Used swap in MiB, from `vm.swapusage` (`total = ... used = 123.45M ...`).
fn swap_used_mb() -> Option<f64> {
    parse_swap_used_mb(&sysctl("vm.swapusage")?)
}

fn parse_swap_used_mb(text: &str) -> Option<f64> {
    let used = text.split("used = ").nth(1)?;
    used.split('M').next()?.trim().parse().ok()
}

/// Reclaimable memory in MiB: free, inactive, speculative and purgeable pages.
fn available_mb() -> Option<f64> {
    let out = std::process::Command::new("/usr/bin/vm_stat")
        .output()
        .ok()?;
    parse_available_mb(&String::from_utf8_lossy(&out.stdout))
}

fn parse_available_mb(text: &str) -> Option<f64> {
    let page: f64 = text
        .split("page size of ")
        .nth(1)?
        .split(' ')
        .next()?
        .parse()
        .ok()?;
    let pages = |key: &str| -> f64 {
        text.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.rsplit(' ').next())
            .and_then(|v| v.trim_end_matches('.').parse::<f64>().ok())
            .unwrap_or(0.0)
    };
    let total = pages("Pages free:")
        + pages("Pages inactive:")
        + pages("Pages speculative:")
        + pages("Pages purgeable:");
    Some(total * page / (1024.0 * 1024.0))
}

/// The start-up memory gate: `Err(reason)` when swap use or reclaimable memory
/// says the ~3.1 GB of resident programs would push the host into paging.
pub fn memory_gate() -> std::result::Result<String, String> {
    let max_swap = env_mb("APXINF_QWEN_IMAGE_ANE_MAX_SWAP_MB", 4096) as f64;
    let min_avail = env_mb("APXINF_QWEN_IMAGE_ANE_MIN_AVAIL_MB", 4096) as f64;
    let (swap, avail) = (swap_used_mb(), available_mb());
    let facts = format!(
        "swap_used={:.0}MB (max {max_swap:.0}) available={:.0}MB (min {min_avail:.0})",
        swap.unwrap_or(f64::NAN),
        avail.unwrap_or(f64::NAN)
    );
    match (swap, avail) {
        (Some(s), Some(a)) if s <= max_swap && a >= min_avail => Ok(facts),
        _ => Err(facts),
    }
}

/// FP16 values of rows `rows` and columns `cols` of a Q4 group-64 weight,
/// dequantized on the device as `mx.dequantize` does (BF16), then FP16.
fn dense(q: &Q4, rows: (usize, usize), cols: (usize, usize)) -> Result<Vec<f16>> {
    let slice = |a: &Array, per: usize| -> Result<Array> {
        a.slice_axis(0, rows.0, rows.1)?
            .slice_axis(1, cols.0 / per, cols.1 / per)
    };
    let w = slice(&q.weight, 8)?
        .dequantize(&slice(&q.scales, 64)?, &slice(&q.biases, 64)?, 64, 4)?
        .cast(MlxDType::F16)?;
    Ok(w.to_bytes()?
        .chunks_exact(2)
        .map(|b| f16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// Symmetric per-row INT8 as A007's NumPy FP16 quantizer computes it: each
/// operation rounds to FP16, `scale = (max - -max) / 254`, then
/// `q = clip(rint(w / scale), -127, 127)`.
fn int8_rows(w: &[f16], cols: usize) -> (Vec<i8>, Vec<f16>) {
    let mut q = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(w.len() / cols);
    for row in w.chunks_exact(cols) {
        let max = row.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
        let span = f16::from_f32(max + max);
        let scale = f16::from_f32(span.to_f32() / 254.0);
        scales.push(scale);
        q.extend(row.iter().map(|v| {
            let x = f16::from_f32(v.to_f32() / scale.to_f32()).to_f32();
            x.round_ties_even().clamp(-127.0, 127.0) as i8
        }));
    }
    (q, scales)
}

/// Columns `[start, end)` of a row-major `[rows, cols]` matrix.
fn columns(w: &[f16], cols: usize, start: usize, end: usize) -> Vec<f16> {
    w.chunks_exact(cols)
        .flat_map(|row| row[start..end].iter().copied())
        .collect()
}

/// One block's Core ML program from its stored (separate) gate/up and `out`.
fn program(gate: &Q4, up: &Q4, out: &Q4) -> Result<Package> {
    let (qg, sg) = int8_rows(&dense(gate, (GPU_ROWS, MLP), (0, DIM))?, DIM);
    let (qu, su) = int8_rows(&dense(up, (GPU_ROWS, MLP), (0, DIM))?, DIM);
    let wo = dense(out, (0, DIM), (GPU_ROWS, MLP))?;
    let (qo, so) = int8_rows(&columns(&wo, ANE_ROWS, 0, OUT_INT8), OUT_INT8);
    let high = columns(&wo, ANE_ROWS, OUT_INT8, ANE_ROWS);
    drop(wo);

    let mut b = ProgramBuilder::new();
    let x = b.input("x", &[1, DIM, 1, TILE])?;
    let r = b.input("r", &[1, 1, 1, 1])?;
    let s = b.input("s", &[1, 1, 1, 1])?;
    let wg =
        b.constexpr_blockwise_shift_scale(&qg, &[ANE_ROWS, DIM, 1, 1], &sg, &[ANE_ROWS, 1, 1, 1])?;
    let wu =
        b.constexpr_blockwise_shift_scale(&qu, &[ANE_ROWS, DIM, 1, 1], &su, &[ANE_ROWS, 1, 1, 1])?;
    let wl =
        b.constexpr_blockwise_shift_scale(&qo, &[DIM, OUT_INT8, 1, 1], &so, &[DIM, 1, 1, 1])?;
    let g = b.conv(&x, &wg)?;
    let g = b.mul(&g, Operand::Var(&r), None)?;
    let u = b.conv(&x, &wu)?;
    let u = b.mul(&u, Operand::Var(&r), None)?;
    let half = b.mul(&g, Operand::Scalar(f16::from_f32(0.5)), None)?;
    let t = b.tanh(&half)?;
    let t1 = b.add(&t, Operand::Scalar(f16::ONE), None)?;
    let silu = b.mul(&half, Operand::Var(&t1), None)?;
    let m = b.mul(&silu, Operand::Var(&u), None)?;
    let m = b.mul(&m, Operand::Var(&s), None)?;
    let parts = b.split(&m, &[OUT_INT8, ANE_ROWS - OUT_INT8], 1)?;
    let low = b.conv(&parts[0], &wl)?;
    let high = b.conv_fp16(&parts[1], &high, &[DIM, ANE_ROWS - OUT_INT8, 1, 1])?;
    let y = b.add(&low, Operand::Var(&high), Some("y"))?;
    b.finish(&[&y])
}

/// Content identity of the transformer shards: the Hugging Face blob names
/// (SHA-256 of their contents) when the snapshot links to blobs, otherwise
/// each shard's size and modification time.
fn weight_identity(dir: &Path) -> Result<Vec<String>> {
    let io = |e: std::io::Error| Error::Other(format!("Qwen-Image ANE identity: {e}"));
    let mut names: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(io)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    names.sort();
    names
        .iter()
        .map(|p| {
            let shard = p.file_name().unwrap_or_default().to_string_lossy();
            let blob = std::fs::read_link(p).ok().and_then(|t| {
                let name = t.file_name()?.to_string_lossy().into_owned();
                (name.len() == 64 && name.bytes().all(|c| c.is_ascii_hexdigit())).then_some(name)
            });
            Ok(match blob {
                Some(sha) => format!("{shard}=sha256:{sha}"),
                None => {
                    let meta = std::fs::metadata(p).map_err(io)?;
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map_or(0, |d| d.as_nanos());
                    format!("{shard}=size:{}:mtime:{mtime}", meta.len())
                }
            })
        })
        .collect()
}

fn manifest(identity: &[String], block: usize, hardware: &str) -> String {
    let weights: Vec<String> = identity.iter().map(|s| format!("\"{s}\"")).collect();
    format!(
        "{{\"recipe\":\"{RECIPE}\",\"block\":{block},\"weights\":[{}],\
         \"ane_rows\":[{GPU_ROWS},{MLP}],\"out_int8\":{OUT_INT8},\"tile\":{TILE},\
         \"inputs\":\"x fp16[1,{DIM},1,{TILE}] r,s fp16[1,1,1,1]\",\
         \"program\":\"mlprogram spec9 CoreML8 fp16\",\"units\":\"cpu_and_ne\",\
         \"hardware\":\"{hardware}\"}}\n",
        weights.join(",")
    )
}

pub fn cache_dir() -> PathBuf {
    std::env::var_os("APXINF_QWEN_IMAGE_ANE_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/apxinf/ane")
        })
}

/// One loaded program per transformer block.
pub struct Programs {
    models: Vec<Model>,
}

impl Programs {
    /// Loads every block's program, building and compiling the missing ones
    /// from `transformer` (whose MLP weights must still be separate), then
    /// runs one warm prediction each. A stored program Core ML will not load
    /// is discarded and compiled again, so a damaged cache costs one cold
    /// install rather than failing the request.
    pub fn install(transformer: &Transformer, dir: &Path, timing: bool) -> Result<Self> {
        const UNITS: ComputeUnits = ComputeUnits::CpuAndNeuralEngine;
        let start = Instant::now();
        let identity = weight_identity(dir)?;
        let hardware = sysctl("hw.model").unwrap_or_default();
        let cache = ProgramCache::new(cache_dir());
        let package = |i: usize| -> Result<Package> {
            let block = &transformer.blocks[i];
            let GateUp::Separate { gate, up } = &block.gate_up else {
                return Err(Error::Contract(
                    "Qwen-Image ANE programs are installed before the MLP weights are packed",
                ));
            };
            program(gate, up, &block.out)
        };

        let (mut hits, mut built) = (0usize, 0.0f64);
        let mut entries = Vec::with_capacity(BLOCKS);
        for i in 0..transformer.blocks.len() {
            let manifest = manifest(&identity, i, &hardware);
            let miss = Instant::now();
            let (path, hit) = cache.compiled_for(&manifest, || package(i))?;
            if hit {
                hits += 1;
            } else {
                built += miss.elapsed().as_secs_f64();
            }
            entries.push((path, hit, manifest));
        }
        let resolved = start.elapsed().as_secs_f64();

        let mut rebuilt = 0usize;
        let mut models = Vec::with_capacity(entries.len());
        for (i, (path, hit, manifest)) in entries.into_iter().enumerate() {
            models.push(match Model::load(&path, UNITS) {
                Err(_) if hit => {
                    cache.evict(&path)?;
                    hits -= 1;
                    rebuilt += 1;
                    let miss = Instant::now();
                    let (path, _) = cache.compiled_for(&manifest, || package(i))?;
                    built += miss.elapsed().as_secs_f64();
                    Model::load(&path, UNITS)?
                }
                loaded => loaded?,
            });
        }
        let loaded = start.elapsed().as_secs_f64();

        let programs = Self { models };
        let zeros = vec![f16::ZERO; DIM * TILE];
        for i in 0..programs.models.len() {
            programs.predict(i, &zeros, f16::ONE, f16::ONE)?;
        }
        if timing {
            let total = start.elapsed().as_secs_f64();
            let source = if built > 0.0 { "compile" } else { "cache" };
            eprintln!(
                "qimage-timing: ane-install={total:.3} source={source} programs={} hits={hits} \
                 rebuilt={rebuilt} build+compile={built:.3}s resolve={resolved:.3}s \
                 load={:.3}s warm={:.3}s cache={}",
                programs.models.len(),
                loaded - resolved,
                total - loaded,
                cache_dir().display()
            );
        }
        Ok(programs)
    }

    /// One tile through block `block`'s program: `x` FP16 `[DIM, TILE]`
    /// (channels first), returning FP16 `[DIM, TILE]`.
    pub fn predict(&self, block: usize, x: &[f16], r: f16, s: f16) -> Result<Vec<f16>> {
        let one = [1usize, 1, 1, 1];
        let shape = [1, DIM, 1, TILE];
        Ok(self.models[block]
            .predict(
                &[("x", x, &shape), ("r", &[r], &one), ("s", &[s], &one)],
                &[("y", &shape)],
            )?
            .remove(0))
    }
}

/// Per-call input scales chosen on the GPU (A007): `2^a` is the largest of
/// `2^-8..2^8` with `top 2^a <= 8` (`2^-8` if none) and `2^b` the largest of
/// `2^-14..2^8` with `26.4 top^2 2^b <= 1024` (`2^-14` if none), for `top` the
/// largest magnitude of the MLP input.
pub struct Scales {
    up: Array,
    act: Array,
}

impl Scales {
    pub fn new(stream: &Stream) -> Result<Self> {
        let table = |low: i32| -> Result<Array> {
            let v: Vec<f32> = (low..=8).map(|e| 2f32.powi(e)).collect();
            Array::from_f32(stream, &[v.len()], &v)
        };
        Ok(Self {
            up: table(-8)?,
            act: table(-14)?,
        })
    }

    /// `(2^a, [2^-a, 2^b] as FP32 [2], 2^-b)` for `x` BF16.
    pub fn call(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let s = x.stream();
        let f = |v: f32| Array::scalar(s, v, MlxDType::F32);
        let top = x
            .abs()?
            .reshape(&[x.numel()])?
            .max(0)?
            .cast(MlxDType::F32)?;
        let pick = |table: &Array, product: &Array, limit: f32, floor: f32| -> Result<Array> {
            product
                .less_equal(&f(limit)?)?
                .where_select(table, &f(floor)?)?
                .max(0)
        };
        let a = pick(&self.up, &top.mul(&self.up)?, 8.0, 2f32.powi(-8))?;
        let act = top.mul(&top)?.mul(&f(26.4)?)?.mul(&self.act)?;
        let b = pick(&self.act, &act, 1024.0, 2f32.powi(-14))?;
        let one = f(1.0)?;
        let params = Array::concat(&[&one.div(&a)?.reshape(&[1])?, &b.reshape(&[1])?], 0)?;
        Ok((a, params, one.div(&b)?))
    }
}

/// The GPU share of a block after the text prefix: gate/up rows
/// `[0, GPU_ROWS)` interleaved for the fused SwiGLU, and `out` columns
/// `[0, GPU_ROWS)`. The full weights are dropped with the block's originals.
pub fn gpu_share(block: &mut Block) -> Result<()> {
    let GateUp::Separate { gate, up } = &block.gate_up else {
        return Ok(());
    };
    let rows = |q: &Q4| -> Result<Q4> {
        Ok(Q4 {
            weight: q.weight.slice_axis(0, 0, GPU_ROWS)?,
            scales: q.scales.slice_axis(0, 0, GPU_ROWS)?,
            biases: q.biases.slice_axis(0, 0, GPU_ROWS)?,
        })
    };
    let packed = super::dit::interleave_gate_up(&rows(gate)?, &rows(up)?)?;
    let o = &block.out;
    let out = Q4 {
        weight: o.weight.slice_axis(1, 0, GPU_ROWS / 8)?.contiguous()?,
        scales: o.scales.slice_axis(1, 0, GPU_ROWS / 64)?.contiguous()?,
        biases: o.biases.slice_axis(1, 0, GPU_ROWS / 64)?.contiguous()?,
    };
    out.weight.stream().eval(&[
        packed.weight.clone(),
        packed.scales.clone(),
        packed.biases.clone(),
        out.weight.clone(),
        out.scales.clone(),
        out.biases.clone(),
    ])?;
    block.gate_up = GateUp::AneSplit(packed);
    block.out = out;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_rows_match_the_fp16_numpy_quantizer() {
        // max 2.0: span 4.0, scale f16(4/254); 1.0 / scale rounds to f16 63.5
        // (exactly), which rint takes to 64 (ties to even).
        let w: Vec<f16> = [2.0f32, -1.0, 0.5, 0.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let (q, s) = int8_rows(&w, 4);
        assert_eq!(s[0], f16::from_f32(4.0 / 254.0));
        let expect: Vec<i8> = w
            .iter()
            .map(|v| {
                f16::from_f32(v.to_f32() / s[0].to_f32())
                    .to_f32()
                    .round_ties_even() as i8
            })
            .collect();
        assert_eq!(q, expect);
        assert_eq!(q[0], 127);
    }

    #[test]
    fn manifest_changes_with_block_and_weights() {
        let a = manifest(&["0.safetensors=sha256:aa".into()], 0, "Mac16,10");
        assert_ne!(
            a,
            manifest(&["0.safetensors=sha256:aa".into()], 1, "Mac16,10")
        );
        assert_ne!(
            a,
            manifest(&["0.safetensors=sha256:bb".into()], 0, "Mac16,10")
        );
        assert!(a.contains(RECIPE));
    }

    /// `vm_stat` on this M4 while another measurement lane held ~11 GB wired.
    const VM_STAT_LOADED: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     5833.
Pages active:                                  84276.
Pages inactive:                                77565.
Pages speculative:                              5455.
Pages throttled:                                   0.
Pages wired down:                             766651.
Pages purgeable:                                 472.
File-backed pages:                             73493.
Anonymous pages:                               93803.
";

    /// The same host minutes later, after that lane released its memory.
    const VM_STAT_FREED: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   408852.
Pages active:                                  60757.
Pages inactive:                                53936.
Pages speculative:                              5673.
Pages throttled:                                   0.
Pages wired down:                             150903.
Pages purgeable:                                 174.
";

    #[test]
    fn available_counts_only_the_reclaimable_pages() {
        // (5833 + 77565 + 5455 + 472) pages of 16 KiB.
        let loaded = parse_available_mb(VM_STAT_LOADED).unwrap();
        assert!((loaded - 1395.7).abs() < 0.1, "{loaded}");
        let freed = parse_available_mb(VM_STAT_FREED).unwrap();
        assert!((freed - 7322.4).abs() < 0.1, "{freed}");
        // Active and wired pages are not headroom; counting them would admit
        // the split on the very host that cannot hold it.
        assert!(loaded < 4096.0, "the loaded host must fail the default floor");
        assert!(freed > 4096.0, "the freed host must clear the default floor");
        assert_eq!(parse_available_mb("not vm_stat output"), None);
    }

    #[test]
    fn swap_used_reads_the_used_field_not_the_total_or_free() {
        assert_eq!(
            parse_swap_used_mb("total = 4096.00M  used = 3227.75M  free = 868.25M  (encrypted)"),
            Some(3227.75)
        );
        assert_eq!(
            parse_swap_used_mb("total = 10240.00M  used = 9627.94M  free = 612.06M  (encrypted)"),
            Some(9627.94)
        );
        assert_eq!(parse_swap_used_mb("total = 4096.00M"), None);
    }

    #[test]
    fn split_covers_the_mlp() {
        assert_eq!(GPU_ROWS + ANE_ROWS, MLP);
        assert_eq!(GPU_ROWS % 64, 0);
        assert!(OUT_INT8 < ANE_ROWS && ANE_ROWS - OUT_INT8 <= 4096);
    }
}
