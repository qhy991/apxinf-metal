//! Checkpoint loading for the mflux Qwen-Image-2.1 export: affine Q4 DiT, F32
//! VAE decoder and the BF16 text encoder (read one layer at a time).
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::io::Read;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

pub const DIM: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const BLOCKS: usize = 32;
pub const MLP: usize = 12288;
pub const LATENT: usize = 64;
pub const TEXT_LAYERS: usize = 36;
pub const TEXT_KV_HEADS: usize = 8;
pub const TEXT_VOCAB: usize = 151_936;

/// One affine Q4 group-64 linear layer without bias, kept as stored.
pub struct Q4 {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
}

impl Q4 {
    /// `nn.QuantizedLinear` without bias.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        x.quantized_matmul(&self.weight, &self.scales, &self.biases, true, 64, 4)
    }
}

pub struct Block {
    pub norm_q: Array,
    pub norm_k: Array,
    pub to_q: Q4,
    pub to_k: Q4,
    pub to_v: Q4,
    pub to_out: Q4,
    pub gate_up: GateUp,
    pub out: Q4,
}

/// The image MLP's gate and up projections: separate as stored, or
/// interleaved in 32-row pairs for the fused SwiGLU after the text prefix.
pub enum GateUp {
    Separate { gate: Q4, up: Q4 },
    Interleaved(Q4),
}

pub struct Transformer {
    pub img_in: Q4,
    pub txt_norm: Array,
    pub txt_in: Q4,
    pub txt_out: Q4,
    pub time_1: Q4,
    pub time_2: Q4,
    pub modulation: Q4,
    pub blocks: Vec<Block>,
    pub norm_out: Q4,
    pub proj_out: Q4,
}

/// `nn.Conv2d` weights `[O, KH, KW, I]` and bias `[O]`, F32.
pub struct Conv {
    pub weight: Array,
    pub bias: Array,
}

pub struct Residual {
    pub norm1: Array,
    pub conv1: Conv,
    pub norm2: Array,
    pub conv2: Conv,
    pub shortcut: Option<Conv>,
}

pub struct Up {
    pub resnets: Vec<Residual>,
    pub upsampler: Option<Conv>,
    pub in_dim: usize,
    pub out_dim: usize,
}

pub struct Decoder {
    pub post_quant: Conv,
    pub conv_in: Conv,
    pub mid: [Residual; 2],
    pub attn_norm: Array,
    pub attn_qkv: Conv,
    pub attn_proj: Conv,
    pub up: Vec<Up>,
    pub norm_out: Array,
    pub conv_out: Conv,
}

pub struct TextLayer {
    pub input_norm: Array,
    pub q: Array,
    pub k: Array,
    pub v: Array,
    pub o: Array,
    pub q_norm: Array,
    pub k_norm: Array,
    pub post_norm: Array,
    pub gate: Array,
    pub up: Array,
    pub down: Array,
}

struct Take {
    map: HashMap<String, Array>,
    what: &'static str,
}

impl Take {
    fn array(&mut self, name: &str, dtype: MlxDType, shape: &[usize]) -> Result<Array> {
        let array = self
            .map
            .remove(name)
            .ok_or_else(|| Error::Other(format!("{} is missing `{name}`", self.what)))?;
        if array.dtype() != dtype || array.shape() != shape {
            return Err(Error::Other(format!(
                "{} `{name}`: expected {dtype:?} {shape:?}, got {:?} {:?}",
                self.what,
                array.dtype(),
                array.shape()
            )));
        }
        Ok(array)
    }

    fn q4(&mut self, name: &str, out: usize, input: usize) -> Result<Q4> {
        Ok(Q4 {
            weight: self.array(&format!("{name}.weight"), MlxDType::U32, &[out, input / 8])?,
            scales: self.array(
                &format!("{name}.scales"),
                MlxDType::BF16,
                &[out, input / 64],
            )?,
            biases: self.array(
                &format!("{name}.biases"),
                MlxDType::BF16,
                &[out, input / 64],
            )?,
        })
    }

    fn conv(&mut self, name: &str, out: usize, kernel: usize, input: usize) -> Result<Conv> {
        Ok(Conv {
            weight: self.array(
                &format!("{name}.weight"),
                MlxDType::F32,
                &[out, kernel, kernel, input],
            )?,
            bias: self.array(&format!("{name}.bias"), MlxDType::F32, &[out])?,
        })
    }

    fn finish(self) -> Result<()> {
        let mut extra: Vec<_> = self.map.into_keys().collect();
        if extra.is_empty() {
            return Ok(());
        }
        extra.sort();
        extra.truncate(8);
        Err(Error::Other(format!(
            "{} has unrecognized tensors: {}",
            self.what,
            extra.join(", ")
        )))
    }
}

fn files(dir: &Path) -> Result<Vec<PathBuf>> {
    apxinf_loader::safetensors::safetensors_files(dir).map_err(Error::Other)
}

/// Deterministic buffers older exports store; mflux skips and recomputes them.
fn recomputed_buffer(name: &str) -> bool {
    name == "time_text_embed.time_proj.freqs" || name.starts_with("pos_embed.")
}

impl Transformer {
    pub fn load(stream: &Stream, dir: &Path) -> Result<Self> {
        let files = files(dir)?;
        headers(&files)?;
        let map = apxinf_mlx::load_safetensors(stream, &files, |n| !recomputed_buffer(n))?;
        let mut t = Take {
            map,
            what: "Qwen-Image transformer",
        };
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let p = format!("transformer_blocks.{i}");
            blocks.push(Block {
                norm_q: t.array(
                    &format!("{p}.attn.norm_q.weight"),
                    MlxDType::BF16,
                    &[HEAD_DIM],
                )?,
                norm_k: t.array(
                    &format!("{p}.attn.norm_k.weight"),
                    MlxDType::BF16,
                    &[HEAD_DIM],
                )?,
                to_q: t.q4(&format!("{p}.attn.to_q"), DIM, DIM)?,
                to_k: t.q4(&format!("{p}.attn.to_k"), DIM, DIM)?,
                to_v: t.q4(&format!("{p}.attn.to_v"), DIM, DIM)?,
                to_out: t.q4(&format!("{p}.attn.to_out.0"), DIM, DIM)?,
                gate_up: GateUp::Separate {
                    gate: t.q4(&format!("{p}.img_mlp.gate_layer"), MLP, DIM)?,
                    up: t.q4(&format!("{p}.img_mlp.proj"), MLP, DIM)?,
                },
                out: t.q4(&format!("{p}.img_mlp.out"), DIM, MLP)?,
            });
        }
        let transformer = Self {
            img_in: t.q4("img_in", DIM, LATENT)?,
            txt_norm: t.array("txt_in.text_norm.weight", MlxDType::BF16, &[DIM])?,
            txt_in: t.q4("txt_in.in_layer", DIM, DIM)?,
            txt_out: t.q4("txt_in.out_layer", DIM, DIM)?,
            time_1: t.q4("time_text_embed.timestep_embedder.linear_1", DIM, 256)?,
            time_2: t.q4("time_text_embed.timestep_embedder.linear_2", DIM, DIM)?,
            modulation: t.q4("modulation.layers.1", 4 * DIM, DIM)?,
            blocks,
            norm_out: t.q4("norm_out.linear", DIM, DIM)?,
            proj_out: t.q4("proj_out", LATENT, DIM)?,
        };
        t.finish()?;
        Ok(transformer)
    }
}

impl Decoder {
    /// Decoder channel widths: `144 * [8, 8, 8, 4, 2, 1]`.
    pub const DIMS: [usize; 6] = [1152, 1152, 1152, 576, 288, 144];

    pub fn load(stream: &Stream, dir: &Path) -> Result<Self> {
        let files = files(dir)?;
        headers(&files)?;
        let map = apxinf_mlx::load_safetensors(stream, &files, |n| {
            n.starts_with("decoder.") || n.starts_with("post_quant_conv.")
        })?;
        let mut t = Take {
            map,
            what: "Qwen-Image VAE decoder",
        };
        let residual = |t: &mut Take, p: &str, input: usize, out: usize| -> Result<Residual> {
            Ok(Residual {
                norm1: t.array(&format!("{p}.norm1.weight"), MlxDType::F32, &[input])?,
                conv1: t.conv(&format!("{p}.conv1.conv"), out, 3, input)?,
                norm2: t.array(&format!("{p}.norm2.weight"), MlxDType::F32, &[out])?,
                conv2: t.conv(&format!("{p}.conv2.conv"), out, 3, out)?,
                shortcut: if input == out {
                    None
                } else {
                    Some(t.conv(&format!("{p}.conv_shortcut.conv"), out, 1, input)?)
                },
            })
        };
        let top = Self::DIMS[0];
        let post_quant = t.conv("post_quant_conv.conv", LATENT, 1, LATENT)?;
        let conv_in = t.conv("decoder.conv_in.conv", top, 3, LATENT)?;
        let mid = [
            residual(&mut t, "decoder.mid_block.resnets.0", top, top)?,
            residual(&mut t, "decoder.mid_block.resnets.1", top, top)?,
        ];
        let a = "decoder.mid_block.attentions.0";
        let attn_norm = t.array(&format!("{a}.norm.weight"), MlxDType::F32, &[top])?;
        let attn_qkv = t.conv(&format!("{a}.to_qkv"), 3 * top, 1, top)?;
        let attn_proj = t.conv(&format!("{a}.proj"), top, 1, top)?;
        let mut up = Vec::new();
        for (i, pair) in Self::DIMS.windows(2).enumerate() {
            let (input, out) = (pair[0], pair[1]);
            let p = format!("decoder.up_blocks.{i}");
            let resnets = (0..3)
                .map(|j| {
                    let from = if j == 0 { input } else { out };
                    residual(&mut t, &format!("{p}.resnets.{j}"), from, out)
                })
                .collect::<Result<Vec<_>>>()?;
            let upsampler = if i < Self::DIMS.len() - 2 {
                Some(t.conv(&format!("{p}.upsampler.conv"), out, 3, out)?)
            } else {
                None
            };
            up.push(Up {
                resnets,
                upsampler,
                in_dim: input,
                out_dim: out,
            });
        }
        let last = Self::DIMS[Self::DIMS.len() - 1];
        let decoder = Self {
            post_quant,
            conv_in,
            mid,
            attn_norm,
            attn_qkv,
            attn_proj,
            up,
            norm_out: t.array("decoder.norm_out.weight", MlxDType::F32, &[last])?,
            conv_out: t.conv("decoder.conv_out.conv", 4, 3, last)?,
        };
        t.finish()?;
        Ok(decoder)
    }
}

/// The stored dtype and shape of each text-encoder tensor; `None` for names
/// the family does not use.
fn text_tensor(name: &str) -> Option<(MlxDType, Vec<usize>)> {
    let bf16 = |shape: &[usize]| Some((MlxDType::BF16, shape.to_vec()));
    let kv = TEXT_KV_HEADS * HEAD_DIM;
    match name {
        "embed_tokens.weight" => bf16(&[TEXT_VOCAB, DIM]),
        "norm.weight" => bf16(&[DIM]),
        // mflux loads the stored rotary frequencies instead of recomputing them.
        "rotary_emb.inv_freq" => Some((MlxDType::F32, vec![HEAD_DIM / 2])),
        _ => {
            let (layer, rest) = name.strip_prefix("layers.")?.split_once('.')?;
            let index: usize = layer.parse().ok()?;
            if index >= TEXT_LAYERS || index.to_string() != layer {
                return None;
            }
            match rest {
                "input_layernorm.weight" | "post_attention_layernorm.weight" => bf16(&[DIM]),
                "self_attn.q_norm.weight" | "self_attn.k_norm.weight" => bf16(&[HEAD_DIM]),
                "self_attn.q_proj.weight" | "self_attn.o_proj.weight" => bf16(&[DIM, DIM]),
                "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => bf16(&[kv, DIM]),
                "mlp.gate_proj.weight" | "mlp.up_proj.weight" => bf16(&[MLP, DIM]),
                "mlp.down_proj.weight" => bf16(&[DIM, MLP]),
                _ => None,
            }
        }
    }
}

/// SafeTensors dtype, shape, file and absolute payload byte range of one tensor.
type Header = (String, Vec<usize>, PathBuf, u64, u64);

fn header_entries(
    raw: &[u8],
    payload_bytes: u64,
) -> Result<HashMap<String, (String, Vec<usize>, u64, u64)>> {
    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(raw)
        .map_err(|error| Error::Other(format!("Qwen-Image SafeTensors header: {error}")))?;
    if let Some(value) = header.get("__metadata__") {
        let metadata = value
            .as_object()
            .ok_or_else(|| Error::Other("malformed SafeTensors metadata".into()))?;
        if let Some(level) = metadata.get("quantization_level") {
            if level.as_str() != Some("4") {
                return Err(Error::Other(
                    "Qwen-Image quantization_level must be the string 4 when present".into(),
                ));
            }
        }
    }
    let mut entries = HashMap::new();
    let mut ranges = Vec::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info.get("dtype").and_then(|value| value.as_str());
        let shape: Option<Vec<usize>> = info
            .get("shape")
            .and_then(|value| serde_json::from_value(value.clone()).ok());
        let (Some(dtype), Some(shape)) = (dtype, shape) else {
            return Err(Error::Other(format!("malformed header entry `{name}`")));
        };
        let size = match dtype {
            "BOOL" | "U8" | "I8" => 1u64,
            "U16" | "I16" | "F16" | "BF16" => 2,
            "U32" | "I32" | "F32" => 4,
            "U64" | "I64" | "F64" => 8,
            _ => {
                return Err(Error::Other(format!(
                    "unsupported dtype `{dtype}` for `{name}`"
                )))
            }
        };
        let bytes = shape
            .iter()
            .try_fold(size, |total, dimension| {
                total.checked_mul(*dimension as u64)
            })
            .ok_or_else(|| Error::Other(format!("tensor `{name}` byte extent overflows")))?;
        let offsets = info
            .get("data_offsets")
            .and_then(|value| serde_json::from_value::<[u64; 2]>(value.clone()).ok())
            .ok_or_else(|| Error::Other(format!("tensor `{name}` has malformed data_offsets")))?;
        let [start, end] = offsets;
        if end.checked_sub(start) != Some(bytes) || end > payload_bytes {
            return Err(Error::Other(format!("tensor `{name}` data_offsets [{start}, {end}] do not match {bytes} bytes within payload {payload_bytes}")));
        }
        ranges.push((start, end, name.clone()));
        entries.insert(name, (dtype.to_owned(), shape, start, end));
    }
    ranges.sort_unstable();
    let mut consumed = 0;
    for (start, end, name) in ranges {
        if start != consumed {
            return Err(Error::Other(format!(
                "tensor `{name}` data_offsets leave a hole or overlap"
            )));
        }
        consumed = end;
    }
    if consumed != payload_bytes {
        return Err(Error::Other(
            "SafeTensors data_offsets do not cover the complete payload".into(),
        ));
    }
    Ok(entries)
}

/// Tensor name -> header entry, read from file headers only.
fn headers(files: &[PathBuf]) -> Result<HashMap<String, Header>> {
    let mut out = HashMap::new();
    for file in files {
        let fail = |e: String| Error::Other(format!("read {}: {e}", file.display()));
        let mut reader = std::fs::File::open(file).map_err(|e| fail(e.to_string()))?;
        let mut length = [0; 8];
        reader
            .read_exact(&mut length)
            .map_err(|e| fail(e.to_string()))?;
        let length = u64::from_le_bytes(length);
        if length > 1 << 26 {
            return Err(fail("SafeTensors header exceeds 64 MiB".into()));
        }
        let mut raw = vec![0; length as usize];
        reader
            .read_exact(&mut raw)
            .map_err(|e| fail(e.to_string()))?;
        let payload_bytes = reader
            .metadata()
            .map_err(|error| fail(error.to_string()))?
            .len()
            .checked_sub(8 + length)
            .ok_or_else(|| fail("truncated SafeTensors header".into()))?;
        for (name, (dtype, shape, start, end)) in
            header_entries(&raw, payload_bytes).map_err(|error| fail(error.to_string()))?
        {
            let base = 8 + length;
            if out
                .insert(name.clone(), (dtype, shape, file.clone(), base + start, base + end))
                .is_some()
            {
                return Err(fail(format!("duplicate tensor `{name}`")));
            }
        }
    }
    Ok(out)
}

/// The BF16 text encoder. Opening validates every tensor header; requests
/// then read one decoder layer at a time.
pub struct TextEncoderFiles {
    /// Tensor name -> shard file.
    shards: HashMap<String, PathBuf>,
    /// Tensor name -> absolute payload byte range inside its shard file.
    spans: HashMap<String, (u64, u64)>,
    /// Fingerprint of the shard files (path, size, mtime and SafeTensors
    /// header bytes); the encode-output cache keys on it so a checkpoint
    /// replacement invalidates every entry.
    pin: [u8; 32],
}

/// Hashes each shard file's path, byte size, modification time and raw
/// SafeTensors header. Content is not hashed (that would re-read the full
/// 15.1 GB per model open); replacing or re-exporting the checkpoint always
/// changes size and mtime, and header bytes pin the tensor structure.
fn shard_pin(shards: &HashMap<String, PathBuf>) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    let files: BTreeSet<&PathBuf> = shards.values().collect();
    for file in files {
        let fail = |e: String| Error::Other(format!("read {}: {e}", file.display()));
        let mut reader = std::fs::File::open(file).map_err(|e| fail(e.to_string()))?;
        let mut length = [0u8; 8];
        reader
            .read_exact(&mut length)
            .map_err(|e| fail(e.to_string()))?;
        let length = u64::from_le_bytes(length);
        if length > 1 << 26 {
            return Err(fail("SafeTensors header exceeds 64 MiB".into()));
        }
        let mut raw = vec![0u8; length as usize];
        reader
            .read_exact(&mut raw)
            .map_err(|e| fail(e.to_string()))?;
        let meta = reader.metadata().map_err(|e| fail(e.to_string()))?;
        hasher.update(file.as_os_str().as_bytes());
        hasher.update(meta.len().to_le_bytes());
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        hasher.update(mtime.to_le_bytes());
        hasher.update(length.to_le_bytes());
        hasher.update(&raw);
    }
    Ok(hasher.finalize().into())
}

impl TextEncoderFiles {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut shards = HashMap::new();
        let mut spans = HashMap::new();
        for (name, (dtype, shape, file, start, end)) in headers(&files(dir)?)? {
            let Some((want_dtype, want_shape)) = text_tensor(&name) else {
                return Err(Error::Other(format!(
                    "Qwen-Image text encoder has an unrecognized tensor `{name}`"
                )));
            };
            let want = match want_dtype {
                MlxDType::F32 => "F32",
                _ => "BF16",
            };
            if dtype != want || shape != want_shape {
                return Err(Error::Other(format!(
                    "Qwen-Image text encoder `{name}`: expected {want} {want_shape:?}, \
                     got {dtype} {shape:?}"
                )));
            }
            shards.insert(name.clone(), file);
            spans.insert(name, (start, end));
        }
        // Every name is known and unique, so the count proves completeness.
        if shards.len() != 3 + 11 * TEXT_LAYERS {
            return Err(Error::Other(format!(
                "Qwen-Image text encoder has {} of {} tensors",
                shards.len(),
                3 + 11 * TEXT_LAYERS
            )));
        }
        let pin = shard_pin(&shards)?;
        Ok(Self { shards, spans, pin })
    }

    /// Weight fingerprint the encode-output cache keys on.
    pub fn pin(&self) -> &[u8; 32] {
        &self.pin
    }

    /// Absolute shard ranges of one decoder layer's tensors, merged per file.
    pub fn layer_ranges(&self, i: usize) -> Vec<(PathBuf, u64, u64)> {
        let p = format!("layers.{i}.");
        self.ranges(|name| name.starts_with(&p))
    }

    /// Absolute shard range of one tensor (e.g. `embed_tokens.weight`).
    pub fn tensor_ranges(&self, name: &str) -> Vec<(PathBuf, u64, u64)> {
        self.ranges(|n| n == name)
    }

    /// Group the selected tensors by shard and merge touching byte ranges.
    fn ranges(&self, keep: impl Fn(&str) -> bool) -> Vec<(PathBuf, u64, u64)> {
        let mut per_file: HashMap<&Path, Vec<(u64, u64)>> = HashMap::new();
        for (name, &(start, end)) in &self.spans {
            if keep(name) {
                per_file
                    .entry(self.shards[name].as_path())
                    .or_default()
                    .push((start, end));
            }
        }
        let mut out = Vec::new();
        for (file, mut spans) in per_file {
            spans.sort_unstable();
            let mut merged: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
            for (start, end) in spans {
                match merged.last_mut() {
                    Some((_, last_end)) if start <= *last_end => *last_end = end.max(*last_end),
                    _ => merged.push((start, end)),
                }
            }
            for (start, end) in merged {
                out.push((file.to_path_buf(), start, end));
            }
        }
        out
    }

    fn load(&self, stream: &Stream, keep: impl Fn(&str) -> bool) -> Result<Take> {
        let files: BTreeSet<_> = self
            .shards
            .iter()
            .filter(|(name, _)| keep(name))
            .map(|(_, file)| file)
            .collect();
        let files: Vec<_> = files.into_iter().collect();
        Ok(Take {
            map: apxinf_mlx::load_safetensors(stream, &files, keep)?,
            what: "Qwen-Image text encoder",
        })
    }

    fn take(t: &mut Take, name: &str) -> Result<Array> {
        let (dtype, shape) = text_tensor(name).expect("text tensor names are fixed");
        t.array(name, dtype, &shape)
    }

    pub fn tensor(&self, stream: &Stream, name: &str) -> Result<Array> {
        let mut t = self.load(stream, |n| n == name)?;
        let array = Self::take(&mut t, name)?;
        t.finish()?;
        Ok(array)
    }

    pub fn layer(&self, stream: &Stream, i: usize) -> Result<TextLayer> {
        let p = format!("layers.{i}.");
        let mut t = self.load(stream, |n| n.starts_with(&p))?;
        let mut b = |name: &str| Self::take(&mut t, &format!("{p}{name}"));
        let layer = TextLayer {
            input_norm: b("input_layernorm.weight")?,
            q: b("self_attn.q_proj.weight")?,
            k: b("self_attn.k_proj.weight")?,
            v: b("self_attn.v_proj.weight")?,
            o: b("self_attn.o_proj.weight")?,
            q_norm: b("self_attn.q_norm.weight")?,
            k_norm: b("self_attn.k_norm.weight")?,
            post_norm: b("post_attention_layernorm.weight")?,
            gate: b("mlp.gate_proj.weight")?,
            up: b("mlp.up_proj.weight")?,
            down: b("mlp.down_proj.weight")?,
        };
        t.finish()?;
        Ok(layer)
    }
}

/// Above this image-token count the text-encoder prefetch stays off.
/// Round 17 (EngineTailor #217) admitted only ≤2,304 tokens: warming all
/// 15.1 GB of shard pages into the cache triggered heavy paging at 1328² on
/// the 16 GB qualification host. Round 19 (#222) bounds the warmed window
/// (~0.8 GB) and evicts every consumed layer, so the encode-phase cache
/// footprint no longer depends on the image size; 1328² (6,889 tokens) is
/// admitted with paired swap-clean evidence. Tiers above 1328² remain
/// unmeasured and stay off.
pub const PREFETCH_MAX_IMAGE_TOKENS: usize = 6889;

/// Layers warmed ahead of the one being computed (0 = only the layer the
/// loader is about to read). At ~386 MB per layer the default keeps at most
/// ~0.8 GB of warmed pages resident (the next layer plus the one the worker
/// is reading). `APXINF_QWEN_IMAGE_PREFETCH_WINDOW` overrides it.
pub const DEFAULT_PREFETCH_WINDOW: usize = 1;
/// Sanity clamp: 8 layers ≈ 3.1 GB of warmed cache, beyond the design point.
pub const MAX_PREFETCH_WINDOW: usize = 8;
/// Memory-gate floors: prefetch admits only when the host has this much
/// free+inactive memory and at most this much swap in use (round-18 owner
/// criteria), so the cache warming can never push the host into paging.
pub const DEFAULT_PREFETCH_MIN_AVAIL_MB: u64 = 4096;
pub const DEFAULT_PREFETCH_MAX_SWAP_MB: u64 = 3072;

fn parse_window(value: Option<&str>) -> usize {
    value
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PREFETCH_WINDOW)
        .min(MAX_PREFETCH_WINDOW)
}

fn parse_mb(value: Option<&str>, default: u64) -> u64 {
    value
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
        .saturating_mul(1 << 20)
}

fn env_cached(name: &str, slot: &'static std::sync::OnceLock<Option<String>>) -> Option<String> {
    slot.get_or_init(|| std::env::var(name).ok()).clone()
}

fn forced_prefetch() -> Option<bool> {
    static FORCED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    env_cached("APXINF_QWEN_IMAGE_TEXT_PREFETCH", &FORCED).map(|v| {
        !matches!(v.to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no")
    })
}

fn prefetch_window() -> usize {
    static WINDOW: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    parse_window(env_cached("APXINF_QWEN_IMAGE_PREFETCH_WINDOW", &WINDOW).as_deref())
}

fn prefetch_min_avail_bytes() -> u64 {
    static FLOOR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    parse_mb(
        env_cached("APXINF_QWEN_IMAGE_PREFETCH_MIN_AVAIL_MB", &FLOOR).as_deref(),
        DEFAULT_PREFETCH_MIN_AVAIL_MB,
    )
}

fn prefetch_max_swap_bytes() -> u64 {
    static CEIL: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    parse_mb(
        env_cached("APXINF_QWEN_IMAGE_PREFETCH_MAX_SWAP_MB", &CEIL).as_deref(),
        DEFAULT_PREFETCH_MAX_SWAP_MB,
    )
}

/// Host memory at the encode start: pages the kernel can reuse immediately
/// without paging (free + inactive + purgeable + speculative) and swap
/// bytes in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemSnapshot {
    pub available_bytes: u64,
    pub swap_used_bytes: u64,
}

fn memory_snapshot() -> Option<MemSnapshot> {
    use std::mem::{size_of, zeroed};
    unsafe {
        let mut usage: libc::xsw_usage = zeroed();
        let mut len = size_of::<libc::xsw_usage>();
        if libc::sysctlbyname(
            b"vm.swapusage\0".as_ptr().cast(),
            &mut usage as *mut _ as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        let mut stats: libc::vm_statistics64 = zeroed();
        let mut count =
            (size_of::<libc::vm_statistics64>() / size_of::<libc::integer_t>()) as u32;
        if libc::host_statistics64(
            libc::mach_host_self(),
            libc::HOST_VM_INFO64,
            &mut stats as *mut _ as *mut libc::integer_t,
            &mut count,
        ) != libc::KERN_SUCCESS
        {
            return None;
        }
        let page = libc::sysconf(libc::_SC_PAGESIZE);
        if page <= 0 {
            return None;
        }
        Some(MemSnapshot {
            available_bytes: (u64::from(stats.free_count)
                + u64::from(stats.inactive_count)
                + u64::from(stats.purgeable_count)
                + u64::from(stats.speculative_count))
                * page as u64,
            swap_used_bytes: usage.xsu_used,
        })
    }
}

/// Why the prefetch is on or off for one encode call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchGate {
    /// `APXINF_QWEN_IMAGE_TEXT_PREFETCH` forced the direction (measurement,
    /// rollback); the tier cap and memory gate do not apply.
    ForcedOn,
    ForcedOff,
    /// Above `PREFETCH_MAX_IMAGE_TOKENS`: unmeasured tier, stays off.
    TierCap,
    /// The host is already low on memory or swapping: prefetch would risk
    /// paging, so the reads stay on demand.
    Memory(MemSnapshot),
    /// Admitted; the host stats when the decision was made (`None` when the
    /// query failed — the legacy behavior admits then).
    Admit(Option<MemSnapshot>),
}

fn decide_gate(
    forced: Option<bool>,
    image_tokens: usize,
    mem: Option<MemSnapshot>,
    min_avail: u64,
    max_swap: u64,
) -> PrefetchGate {
    if let Some(forced) = forced {
        return if forced {
            PrefetchGate::ForcedOn
        } else {
            PrefetchGate::ForcedOff
        };
    }
    if image_tokens > PREFETCH_MAX_IMAGE_TOKENS {
        return PrefetchGate::TierCap;
    }
    match mem {
        Some(mem) if mem.swap_used_bytes > max_swap || mem.available_bytes < min_avail => {
            PrefetchGate::Memory(mem)
        }
        mem => PrefetchGate::Admit(mem),
    }
}

/// What one encode does about the shard page cache.
pub enum PrefetchPlan {
    /// Legacy path (forced off or an unmeasured tier): no worker, no purge.
    Disabled(PrefetchGate),
    /// The worker runs: `warm` prefetches ahead, `purge` drops each consumed
    /// layer's pages. A memory-pressured host degrades to warm=false with
    /// purge still on — under pressure the encode must not grow the cache.
    Active(Box<TextPrefetchPlan>),
}

pub struct TextPrefetchPlan {
    pub worker: TextPrefetch,
    pub warm: bool,
    pub purge: bool,
    pub window: usize,
    pub gate: PrefetchGate,
}

/// Decide what this encode does with the shard cache, and start the worker
/// when it does anything at all.
///
/// `APXINF_QWEN_IMAGE_TEXT_PREFETCH` forces warming on/off for measurement
/// and rollback. Unset, warming admits tiers up to
/// `PREFETCH_MAX_IMAGE_TOKENS` subject to the memory gate
/// (`APXINF_QWEN_IMAGE_PREFETCH_MIN_AVAIL_MB`, default 4096;
/// `APXINF_QWEN_IMAGE_PREFETCH_MAX_SWAP_MB`, default 3072): a host already
/// under memory pressure skips the warming — with a log line — but still
/// purges consumed layers, so the encode never grows the cache it cannot
/// afford. `APXINF_QWEN_IMAGE_PREFETCH_PURGE` (default on) disables the
/// purge alone for isolation measurements.
pub fn prefetch_plan(image_tokens: usize) -> PrefetchPlan {
    let min_avail = prefetch_min_avail_bytes();
    let max_swap = prefetch_max_swap_bytes();
    let gate = decide_gate(
        forced_prefetch(),
        image_tokens,
        memory_snapshot(),
        min_avail,
        max_swap,
    );
    let purge = purge_enabled();
    match gate {
        PrefetchGate::ForcedOff | PrefetchGate::TierCap => PrefetchPlan::Disabled(gate),
        PrefetchGate::Memory(mem) => {
            // Always logged: degrading silently would hide the lever's
            // absence from the operator exactly when the host is pressured.
            eprintln!(
                "qwen-image: text prefetch disabled by the memory gate \
                 (available {:.2} GB < {} MB or swap used {:.2} GB > {} MB); \
                 reading weights on demand, still purging consumed layers",
                mem.available_bytes as f64 / (1u64 << 30) as f64,
                min_avail >> 20,
                mem.swap_used_bytes as f64 / (1u64 << 30) as f64,
                max_swap >> 20,
            );
            PrefetchPlan::Active(Box::new(TextPrefetchPlan {
                worker: TextPrefetch::new(),
                warm: false,
                purge,
                window: prefetch_window(),
                gate,
            }))
        }
        PrefetchGate::ForcedOn | PrefetchGate::Admit(_) => {
            PrefetchPlan::Active(Box::new(TextPrefetchPlan {
                worker: TextPrefetch::new(),
                warm: true,
                purge,
                window: prefetch_window(),
                gate,
            }))
        }
    }
}

fn purge_enabled() -> bool {
    static PURGE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    env_cached("APXINF_QWEN_IMAGE_PREFETCH_PURGE", &PURGE)
        .map(|v| !matches!(v.to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no"))
        .unwrap_or(true)
}

/// Whole `page`-sized pages fully inside `[start, end)`: layer boundaries
/// can share a page with a kept neighbor, so partial edge pages stay cached.
fn contained_pages(start: u64, end: u64, page: u64) -> Option<(u64, u64)> {
    let lo = start.checked_add(page - 1)? & !(page - 1);
    let hi = end & !(page - 1);
    (hi > lo).then_some((lo, hi - lo))
}

/// Drop already-consumed shard bytes from the OS page cache. On this host
/// (measured, EngineTailor #222 round 19) the effective userspace primitive
/// is `mmap(MAP_SHARED)` + `msync(MS_INVALIDATE)`: it returns the clean
/// file pages to the free pool deterministically (~2 ms per 386 MB layer),
/// while `madvise(MADV_DONTNEED)`/`MADV_FREE` leave them resident. The
/// shards are read-only model weights, so invalidation cannot discard
/// anybody's dirty data; a concurrent reader of a page simply re-faults it
/// from disk. Only whole pages fully inside a range are touched, so a page
/// shared with a kept neighbor layer is never dropped. Failures are
/// ignored: a missed purge only means the cache keeps the pages longer.
pub fn purge_cached_ranges(ranges: &[(PathBuf, u64, u64)]) {
    use std::os::unix::io::AsRawFd;
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return;
    }
    let page = page as u64;
    for (file, start, end) in ranges {
        let Some((offset, len)) = contained_pages(*start, *end, page) else {
            continue;
        };
        let Ok(input) = std::fs::File::open(file) else {
            continue;
        };
        let Ok(meta) = input.metadata() else {
            continue;
        };
        let len = len.min(meta.len().saturating_sub(offset));
        if len == 0 {
            continue;
        }
        unsafe {
            let addr = libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ,
                libc::MAP_SHARED,
                input.as_raw_fd(),
                offset as libc::off_t,
            );
            if addr == libc::MAP_FAILED {
                continue;
            }
            libc::msync(addr, len as usize, libc::MS_INVALIDATE);
            libc::munmap(addr, len as usize);
        }
    }
}

/// Work item for the prefetch thread: warm these bytes into the cache, or
/// purge these already-consumed bytes out of it.
enum Msg {
    Warm(Vec<(PathBuf, u64, u64)>),
    Purge(Vec<(PathBuf, u64, u64)>),
    Quit,
}

/// Background disk prefetcher for the text encoder's per-layer loads.
///
/// The encoder reads its weights one layer at a time; while a layer computes
/// on the device, this worker reads the *next* layers' exact shard bytes so
/// the OS page cache already holds them when the graph mmaps and evaluates
/// them (`warm`), and drops each consumed layer's pages again (`purge`), so
/// the cache holds a ~0.8 GB window instead of the full 15.1 GB. The bytes
/// the loader sees are unchanged — this moves the disk wait and the cache
/// footprint, not the computation — so outputs are bit-exact. Read failures
/// are ignored: every tensor is still read synchronously by the real load.
pub struct TextPrefetch {
    sender: Option<std::sync::mpsc::Sender<Msg>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl TextPrefetch {
    pub fn new() -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("qwen-image-text-prefetch".into())
            .spawn(move || Self::run(receiver));
        let worker = match worker {
            Ok(worker) => worker,
            // Prefetch is an optional hint: a thread that cannot spawn is no
            // reason to reject the request; the reads stay synchronous.
            Err(_) => return Self { sender: None, worker: None },
        };
        Self {
            sender: Some(sender),
            worker: Some(worker),
        }
    }

    /// Queue these absolute file ranges for a cache-warming read.
    pub fn warm(&self, ranges: Vec<(PathBuf, u64, u64)>) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(Msg::Warm(ranges));
        }
    }

    /// Queue these absolute file ranges for page-cache eviction.
    pub fn purge(&self, ranges: Vec<(PathBuf, u64, u64)>) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(Msg::Purge(ranges));
        }
    }

    fn run(receiver: std::sync::mpsc::Receiver<Msg>) {
        use std::io::{Read, Seek, SeekFrom};
        const CHUNK: usize = 8 << 20;
        let mut buf = vec![0_u8; CHUNK];
        while let Ok(msg) = receiver.recv() {
            match msg {
                Msg::Quit => break,
                Msg::Purge(ranges) => purge_cached_ranges(&ranges),
                Msg::Warm(ranges) => {
                    for (file, start, end) in ranges {
                        let Ok(mut input) = std::fs::File::open(&file) else {
                            continue;
                        };
                        if input.seek(SeekFrom::Start(start)).is_err() {
                            continue;
                        }
                        let mut left = end - start;
                        while left > 0 {
                            let want = left.min(CHUNK as u64) as usize;
                            match input.read(&mut buf[..want]) {
                                Ok(0) => break,
                                Ok(read) => left -= read as u64,
                                Err(_) => break,
                            }
                        }
                    }
                }
            }
        }
    }
}

impl Drop for TextPrefetch {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Msg::Quit);
        }
        if let Some(worker) = self.worker.take() {
            // Wait out the queued reads and purges; prefetch is on the
            // encode critical path's shadow, never on a later phase's.
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the pinned Qwen-Image checkpoint headers"]
    fn real_checkpoint_headers_validate_without_materializing_payloads() -> Result<()> {
        let root = PathBuf::from(
            std::env::var("APXINF_QWEN_IMAGE_CHECKPOINT")
                .expect("set APXINF_QWEN_IMAGE_CHECKPOINT"),
        );
        TextEncoderFiles::open(&root.join("text_encoder"))?;
        for component in ["transformer", "vae"] {
            assert!(!headers(&files(&root.join(component))?)?.is_empty());
        }
        Ok(())
    }

    fn parse_header(
        value: &serde_json::Value,
        payload_bytes: u64,
    ) -> Result<HashMap<String, (String, Vec<usize>, u64, u64)>> {
        header_entries(&serde_json::to_vec(value).unwrap(), payload_bytes)
    }

    #[test]
    fn headers_accept_valid_payload_and_optional_export_metadata() {
        let mut value =
            serde_json::json!({"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}});
        assert_eq!(
            parse_header(&value, 4).unwrap()["x"],
            ("BF16".into(), vec![2], 0, 4)
        );
        value["__metadata__"] =
            serde_json::json!({"quantization_level": "4", "mflux_version": "0.19.2"});
        assert!(parse_header(&value, 4).is_ok());
    }

    #[test]
    fn headers_reject_quantization_metadata_without_changing_tensors() {
        let mut value =
            serde_json::json!({"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}});
        for metadata in [
            serde_json::json!({"quantization_level": "8"}),
            serde_json::json!({"quantization_level": 4}),
        ] {
            value["__metadata__"] = metadata;
            assert!(parse_header(&value, 4).is_err());
        }
    }

    #[test]
    fn headers_reject_invalid_offsets_extents_and_overflow() {
        let original =
            serde_json::json!({"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}});
        for offsets in [
            serde_json::json!(null),
            serde_json::json!([3, 2]),
            serde_json::json!([0, 5]),
            serde_json::json!([0, 3]),
            serde_json::json!([-1, 4]),
            serde_json::json!([0, u64::MAX]),
        ] {
            let mut value = original.clone();
            value["x"]["data_offsets"] = offsets;
            assert!(parse_header(&value, 4).is_err());
        }
        let mut overflow = original;
        overflow["x"]["shape"] = serde_json::json!([usize::MAX, usize::MAX]);
        assert!(parse_header(&overflow, 4).is_err());
    }

    #[test]
    fn headers_reject_payload_overlap_holes_and_trailing_bytes() {
        let mut value = serde_json::json!({
            "x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]},
            "y": {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]}
        });
        assert!(parse_header(&value, 8).is_ok());
        assert!(parse_header(&value, 10).is_err());
        value["y"]["data_offsets"] = serde_json::json!([2, 6]);
        assert!(parse_header(&value, 6).is_err());
        value["y"]["data_offsets"] = serde_json::json!([6, 10]);
        assert!(parse_header(&value, 10).is_err());
    }

    #[test]
    fn prefetch_gate_admits_measured_tiers_and_forced_directions() {
        let roomy = Some(MemSnapshot {
            available_bytes: 8 << 30,
            swap_used_bytes: 0,
        });
        let (floor, ceil) = (4 << 30, 3 << 30);
        // Measured tiers up to 1328² (6,889 tokens) admit on a roomy host.
        for tokens in [1024, 2304, 4096, 6889] {
            assert_eq!(
                decide_gate(None, tokens, roomy, floor, ceil),
                PrefetchGate::Admit(roomy),
                "tokens={tokens}"
            );
        }
        // Above 1328² stays off; force-env overrides either direction.
        assert_eq!(
            decide_gate(None, 6890, roomy, floor, ceil),
            PrefetchGate::TierCap
        );
        assert_eq!(
            decide_gate(Some(true), 6890, roomy, floor, ceil),
            PrefetchGate::ForcedOn
        );
        assert_eq!(
            decide_gate(Some(false), 1024, roomy, floor, ceil),
            PrefetchGate::ForcedOff
        );
    }

    #[test]
    fn prefetch_gate_degrades_under_memory_pressure() {
        let (floor, ceil) = (4u64 << 30, 3u64 << 30);
        let low_free = MemSnapshot {
            available_bytes: (2u64 << 30),
            swap_used_bytes: 0,
        };
        let swapping = MemSnapshot {
            available_bytes: 12 << 30,
            swap_used_bytes: 4 << 30,
        };
        for pressured in [low_free, swapping] {
            assert_eq!(
                decide_gate(None, 1024, Some(pressured), floor, ceil),
                PrefetchGate::Memory(pressured)
            );
        }
        // Exactly at the floors still admits; a failed stat query keeps the
        // legacy admit rather than disabling the lever blindly.
        let at_floor = MemSnapshot {
            available_bytes: floor,
            swap_used_bytes: ceil,
        };
        assert_eq!(
            decide_gate(None, 1024, Some(at_floor), floor, ceil),
            PrefetchGate::Admit(Some(at_floor))
        );
        assert_eq!(
            decide_gate(None, 1024, None, floor, ceil),
            PrefetchGate::Admit(None)
        );
        // Forced-on bypasses the memory gate (explicit measurement override).
        assert_eq!(
            decide_gate(Some(true), 1024, Some(swapping), floor, ceil),
            PrefetchGate::ForcedOn
        );
    }

    #[test]
    fn prefetch_window_parses_default_and_clamps() {
        assert_eq!(parse_window(None), DEFAULT_PREFETCH_WINDOW);
        assert_eq!(parse_window(Some("0")), 0);
        assert_eq!(parse_window(Some("3")), 3);
        assert_eq!(parse_window(Some("junk")), DEFAULT_PREFETCH_WINDOW);
        assert_eq!(parse_window(Some("99")), MAX_PREFETCH_WINDOW);
    }

    #[test]
    fn contained_pages_evicts_only_fully_owned_pages() {
        let page = 16_384;
        // Exact whole-page range.
        assert_eq!(contained_pages(0, page, page), Some((0, page)));
        // Partial edge pages stay cached: a neighbor tensor may share them.
        assert_eq!(contained_pages(100, page, page), None);
        assert_eq!(contained_pages(100, 2 * page, page), Some((page, page)));
        assert_eq!(
            contained_pages(100, 3 * page + 7, page),
            Some((page, 2 * page))
        );
        // Ranges smaller than one page and overflowing arithmetic are inert.
        assert_eq!(contained_pages(1, 100, page), None);
        assert_eq!(contained_pages(u64::MAX - 10, u64::MAX, page), None);
    }

    #[test]
    fn text_tensor_names_are_exact() {
        assert!(text_tensor("layers.35.mlp.down_proj.weight").is_some());
        assert!(text_tensor("layers.36.mlp.down_proj.weight").is_none());
        assert!(text_tensor("layers.01.mlp.down_proj.weight").is_none());
        assert!(text_tensor("layers.0.mlp.down_proj.bias").is_none());
        assert_eq!(
            text_tensor("rotary_emb.inv_freq"),
            Some((MlxDType::F32, vec![64]))
        );
        let names = (0..TEXT_LAYERS)
            .flat_map(|i| {
                [
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                ]
                .map(|n| format!("layers.{i}.{n}"))
            })
            .chain(["embed_tokens.weight", "norm.weight", "rotary_emb.inv_freq"].map(String::from));
        assert_eq!(names.filter(|n| text_tensor(n).is_some()).count(), 399);
    }
}
