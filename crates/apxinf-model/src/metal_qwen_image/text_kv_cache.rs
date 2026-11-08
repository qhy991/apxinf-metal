//! Disk-persistent cross-request cache of the DiT text K/V
//! (EngineTailor round 23, qhy991/EngineTailor#239).
//!
//! `Dit::text_cache` projects the text-encoder output through the 32 DiT
//! blocks once per request and keeps each block's post-RoPE text K/V for the
//! whole denoise loop; at long prompts that phase costs ~9.8 s of the ~172 s
//! wall on an M4 (round 21 profile). The K/V are a pure function of the
//! truncated token ids, the text-encoder weights (through the prompt states)
//! and the DiT transformer weights, and the denoise loop only ever reads
//! them — every step concatenates them with the image K/V — so reloading the
//! exact bytes of a previous evaluation is bit-identical to recomputing
//! them. This cache persists those bytes (64 BF16 `[1, 32, T, 128]` tensors,
//! ~1.07 GB for T=2034) under a key of the truncated ids plus pins of both
//! checkpoint shard sets, and a hit skips both the text encode (round 22
//! cache) and the `text_cache` phase.
//!
//! Writes are atomic (tmp file + rename, fsync before the rename) and the
//! directory is bounded by a byte cap with least-recently-used eviction.
//! Cache problems — I/O errors, corrupt or stale entries, a missing index —
//! never fail a request: they log one stderr line and degrade to a miss.
//!
//! Environment:
//! - `APXINF_QWEN_IMAGE_TEXT_KV_CACHE=0` disables the cache (the exact
//!   legacy path; default enabled);
//! - `APXINF_QWEN_IMAGE_TEXT_KV_CACHE_DIR` overrides the default
//!   `~/.cache/apxinf/text-kv`;
//! - `APXINF_QWEN_IMAGE_TEXT_KV_CACHE_MAX_MB` overrides the 4096 MiB cap.

use super::dit::TextCache;
use super::disk_cache::{hex, now_unix_ms, short_key, Entry, Index};
use super::text::PromptTokens;
use super::weights::{dir_pin, TextEncoderFiles, BLOCKS, HEADS, HEAD_DIM};
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Bumped when the on-disk format or the tensor layout changes, so old
/// entries can never be served under new semantics.
const FORMAT_VERSION: u32 = 1;
/// Default capacity cap in MiB; `APXINF_QWEN_IMAGE_TEXT_KV_CACHE_MAX_MB`
/// overrides. One long entry is ~1.07 GB, so the default holds about three.
const DEFAULT_MAX_MB: u64 = 4096;
/// Value of the entries' `__metadata__.format` marker.
const FORMAT: &str = "apxinf-text-kv-cache";
/// Tensor names inside the value file: `k.{i}` / `v.{i}` for block `i`.
const K: &str = "k";
const V: &str = "v";

const ENV_SWITCH: &str = "APXINF_QWEN_IMAGE_TEXT_KV_CACHE";
const ENV_DIR: &str = "APXINF_QWEN_IMAGE_TEXT_KV_CACHE_DIR";
const ENV_MAX_MB: &str = "APXINF_QWEN_IMAGE_TEXT_KV_CACHE_MAX_MB";

struct Config {
    dir: PathBuf,
    max_bytes: u64,
}

fn cache_enabled() -> bool {
    std::env::var(ENV_SWITCH).map(|value| value != "0").unwrap_or(true)
}

fn log_disabled_once(reason: String) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("qimage-timing: text_kv_cache disabled ({reason})"));
}

fn config_from_env() -> Option<Config> {
    let dir = match std::env::var_os(ENV_DIR) {
        Some(dir) => PathBuf::from(dir),
        None => match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".cache/apxinf/text-kv"),
            None => {
                log_disabled_once(format!("no HOME and {ENV_DIR} unset"));
                return None;
            }
        },
    };
    let max_mb = std::env::var(ENV_MAX_MB)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MAX_MB);
    Some(Config {
        dir,
        max_bytes: max_mb.saturating_mul(1 << 20),
    })
}

fn text_len(tokens: &PromptTokens) -> usize {
    tokens.ids.len().saturating_sub(tokens.drop)
}

/// Cache identity of one text-K/V computation: the truncated template ids,
/// both checkpoint pins (text encoder through the prompt states, DiT
/// transformer through the projections) and the format version.
fn cache_key(
    encoder_pin: &[u8; 32],
    transformer_pin: &[u8; 32],
    tokens: &PromptTokens,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(encoder_pin);
    hasher.update(transformer_pin);
    hasher.update(FORMAT_VERSION.to_le_bytes());
    hasher.update((tokens.ids.len() as u64).to_le_bytes());
    hasher.update((tokens.drop as u64).to_le_bytes());
    for id in &tokens.ids {
        hasher.update(id.to_le_bytes());
    }
    hex(&hasher.finalize())
}

/// The DiT shard-set pin, computed best-effort: an unreadable shard set
/// degrades to no cache rather than failing the request.
fn transformer_pin(transformer_dir: &Path) -> Option<[u8; 32]> {
    match dir_pin(transformer_dir) {
        Ok(pin) => Some(pin),
        Err(error) => {
            eprintln!("qimage-timing: text_kv_cache pin failed: {error}");
            None
        }
    }
}

/// Returns the cached per-block text K/V for `tokens`, or `None` on any
/// miss. Degrades to `None` — never an error — on every cache problem.
pub fn load(
    files: &TextEncoderFiles,
    transformer_dir: &Path,
    tokens: &PromptTokens,
    stream: &Stream,
) -> Result<Option<TextCache>> {
    if !cache_enabled() {
        log_disabled_once(format!("{ENV_SWITCH}=0"));
        return Ok(None);
    }
    let Some(config) = config_from_env() else {
        return Ok(None);
    };
    let Some(transformer_pin) = transformer_pin(transformer_dir) else {
        return Ok(None);
    };
    let key = cache_key(files.pin(), &transformer_pin, tokens);
    let start = std::time::Instant::now();
    match load_at(&config, &key, files.pin(), &transformer_pin, tokens, stream) {
        Ok(Some((cache, bytes))) => {
            eprintln!(
                "qimage-timing: text_kv_cache hit=1 key={} blocks={} bytes={} load={:.3}s",
                short_key(&key),
                BLOCKS,
                bytes,
                start.elapsed().as_secs_f64()
            );
            Ok(Some(cache))
        }
        Ok(None) => {
            eprintln!("qimage-timing: text_kv_cache hit=0 key={}", short_key(&key));
            Ok(None)
        }
        Err(error) => {
            eprintln!(
                "qimage-timing: text_kv_cache hit=0 key={} error={error}",
                short_key(&key)
            );
            Ok(None)
        }
    }
}

/// Stores the freshly computed text K/V under `tokens`' cache identity.
/// Stores are best-effort: problems log one line and leave generation
/// unaffected.
pub fn store(
    files: &TextEncoderFiles,
    transformer_dir: &Path,
    tokens: &PromptTokens,
    cache: &TextCache,
) -> Result<()> {
    if !cache_enabled() {
        return Ok(());
    }
    let Some(config) = config_from_env() else {
        return Ok(());
    };
    let Some(transformer_pin) = transformer_pin(transformer_dir) else {
        return Ok(());
    };
    if cache.kv().len() != BLOCKS {
        eprintln!(
            "qimage-timing: text_kv_cache store skipped: {} blocks, want {BLOCKS}",
            cache.kv().len()
        );
        return Ok(());
    }
    let key = cache_key(files.pin(), &transformer_pin, tokens);
    let start = std::time::Instant::now();
    match store_at(
        &config,
        &key,
        files.pin(),
        &transformer_pin,
        tokens,
        cache.kv(),
    ) {
        Ok(bytes) => eprintln!(
            "qimage-timing: text_kv_cache store key={} bytes={bytes} took={:.3}s",
            short_key(&key),
            start.elapsed().as_secs_f64()
        ),
        Err(error) => eprintln!(
            "qimage-timing: text_kv_cache store key={} failed: {error}",
            short_key(&key)
        ),
    }
    Ok(())
}

fn load_at(
    config: &Config,
    key: &str,
    encoder_pin: &[u8; 32],
    transformer_pin: &[u8; 32],
    tokens: &PromptTokens,
    stream: &Stream,
) -> Result<Option<(TextCache, u64)>> {
    let mut index = Index::read(&config.dir)?;
    let Some(position) = index.entries.iter().position(|entry| entry.key == key) else {
        return Ok(None);
    };
    let path = config.dir.join(index.entries[position].file.clone());
    let bytes = index.entries[position].bytes;
    match read_entry(
        &path,
        key,
        encoder_pin,
        transformer_pin,
        tokens,
        stream,
    ) {
        Ok(kv) => {
            index.touch(key);
            let _ = index.write(&config.dir);
            Ok(Some((TextCache::from_kv(kv), bytes)))
        }
        Err(_) => {
            // Stale or corrupt entry: drop the file and the index row, then
            // let the caller recompute and repopulate.
            let _ = std::fs::remove_file(&path);
            index.remove(key);
            let _ = index.write(&config.dir);
            Ok(None)
        }
    }
}

fn store_at(
    config: &Config,
    key: &str,
    encoder_pin: &[u8; 32],
    transformer_pin: &[u8; 32],
    tokens: &PromptTokens,
    kv: &[(Array, Array)],
) -> Result<u64> {
    let bytes = payload_len(tokens)?;
    std::fs::create_dir_all(&config.dir)?;
    let mut index = Index::read(&config.dir)?;
    let file = format!("{key}.safetensors");
    let tmp = config
        .dir
        .join(format!(".{key}.tmp{}", std::process::id()));
    let written = write_entry(&tmp, key, encoder_pin, transformer_pin, tokens, kv)?;
    if written != bytes {
        let _ = std::fs::remove_file(&tmp);
        return Err(Error::Other(format!(
            "text-kv entry wrote {written} bytes, want {bytes}"
        )));
    }
    std::fs::rename(&tmp, config.dir.join(&file))?;
    index.upsert(Entry {
        key: key.to_owned(),
        file,
        bytes,
        created_unix_ms: now_unix_ms(),
        last_hit_unix_ms: now_unix_ms(),
    });
    index.evict(&config.dir, config.max_bytes);
    index.write(&config.dir)?;
    Ok(bytes)
}

fn tensor_name(kind: &str, block: usize) -> String {
    format!("{kind}.{block}")
}

/// Byte length of one K or V tensor at this prompt's `text_len`.
fn tensor_len(tokens: &PromptTokens) -> Result<u64> {
    text_len(tokens)
        .checked_mul(HEADS)
        .and_then(|n| n.checked_mul(HEAD_DIM))
        .and_then(|n| n.checked_mul(2))
        .map(|n| n as u64)
        .ok_or_else(|| Error::Other("text-kv tensor length overflow".into()))
}

fn payload_len(tokens: &PromptTokens) -> Result<u64> {
    tensor_len(tokens)?
        .checked_mul(BLOCKS as u64)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| Error::Other("text-kv payload length overflow".into()))
}

/// Serializes the per-block K/V as a multi-tensor SafeTensors file:
/// `[u64 LE header length][header JSON][BF16 payloads in name order]`.
/// Payloads stream one tensor at a time so the store path peaks at one
/// tensor's copy, not the whole entry.
fn write_entry(
    path: &Path,
    key: &str,
    encoder_pin: &[u8; 32],
    transformer_pin: &[u8; 32],
    tokens: &PromptTokens,
    kv: &[(Array, Array)],
) -> Result<u64> {
    let t = text_len(tokens);
    let want_shape = [1u64, HEADS as u64, t as u64, HEAD_DIM as u64];
    let tensor_len = tensor_len(tokens)? as usize;
    let mut tensors = serde_json::Map::new();
    let mut offset = 0u64;
    for block in 0..BLOCKS {
        for kind in [K, V] {
            tensors.insert(
                tensor_name(kind, block),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": want_shape,
                    "data_offsets": [offset, offset + tensor_len as u64],
                }),
            );
            offset += tensor_len as u64;
        }
    }
    let mut header = serde_json::Map::new();
    header.extend(tensors);
    header.insert(
        "__metadata__".into(),
        serde_json::json!({
            "format": FORMAT,
            "version": FORMAT_VERSION.to_string(),
            "key": key,
            "encoder_pin": hex(encoder_pin),
            "transformer_pin": hex(transformer_pin),
            "tokens_len": tokens.ids.len().to_string(),
            "drop": tokens.drop.to_string(),
            "text_len": t.to_string(),
            "blocks": BLOCKS.to_string(),
        }),
    );
    let header = serde_json::to_vec(&header)
        .map_err(|error| Error::Other(format!("serialize text-kv header: {error}")))?;
    let mut file = std::fs::File::create(path)?;
    use std::io::Write;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    for (k, v) in kv {
        for tensor in [k, v] {
            if tensor.dtype() != MlxDType::BF16 || tensor.shape() != want_shape.map(|n| n as usize)
            {
                return Err(Error::Other(format!(
                    "text-kv tensor is {} {:?}, want BF16 {want_shape:?}",
                    tensor.dtype() as u32,
                    tensor.shape()
                )));
            }
            file.write_all(&tensor.to_bytes()?)?;
        }
    }
    file.sync_all()?;
    Ok(offset)
}

/// Reads and fully validates one entry; any deviation is an error so the
/// caller drops the entry and degrades to a miss. Payloads stream one tensor
/// at a time for the same peak-memory reason as `write_entry`.
fn read_entry(
    path: &Path,
    key: &str,
    encoder_pin: &[u8; 32],
    transformer_pin: &[u8; 32],
    tokens: &PromptTokens,
    stream: &Stream,
) -> Result<Vec<(Array, Array)>> {
    use std::io::{Read, Seek};
    let fail = |why: String| Error::Other(format!("{}: {why}", path.display()));
    let mut file = std::fs::File::open(path).map_err(|error| fail(error.to_string()))?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .map_err(|error| fail(error.to_string()))?;
    let length = u64::from_le_bytes(length);
    if length > 1 << 26 {
        return Err(fail("SafeTensors header exceeds 64 MiB".into()));
    }
    let mut raw = vec![0u8; length as usize];
    file.read_exact(&mut raw)
        .map_err(|error| fail(error.to_string()))?;
    let file_size = file
        .metadata()
        .map_err(|error| fail(error.to_string()))?
        .len();
    let header: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|error| fail(error.to_string()))?;
    let metadata = header
        .get("__metadata__")
        .ok_or_else(|| fail("no __metadata__".into()))?;
    let meta = |name: &str| -> Result<String> {
        metadata
            .get(name)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| fail(format!("__metadata__.{name} missing")))
    };
    if meta("format")? != FORMAT || meta("version")? != FORMAT_VERSION.to_string() {
        return Err(fail("entry format/version mismatch".into()));
    }
    if meta("key")? != key
        || meta("encoder_pin")? != hex(encoder_pin)
        || meta("transformer_pin")? != hex(transformer_pin)
    {
        return Err(fail("entry identity mismatch".into()));
    }
    if meta("tokens_len")?.parse::<usize>() != Ok(tokens.ids.len())
        || meta("drop")?.parse::<usize>() != Ok(tokens.drop)
        || meta("text_len")?.parse::<usize>() != Ok(text_len(tokens))
        || meta("blocks")?.parse::<usize>() != Ok(BLOCKS)
    {
        return Err(fail("entry token geometry mismatch".into()));
    }
    let t = text_len(tokens);
    let want_shape = [1u64, HEADS as u64, t as u64, HEAD_DIM as u64];
    let tensor_len = tensor_len(tokens)? as usize;
    let payload_len = payload_len(tokens)?;
    if file_size != 8 + length + payload_len {
        return Err(fail("entry byte extent mismatch".into()));
    }

    let mut read_tensor = |name: &str, offset: u64| -> Result<Array> {
        let tensor = header
            .get(name)
            .ok_or_else(|| fail(format!("no {name} tensor")))?;
        if tensor.get("dtype").and_then(serde_json::Value::as_str) != Some("BF16") {
            return Err(fail(format!("{name} is not BF16")));
        }
        let shape = tensor
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| fail(format!("{name} has no shape")))?;
        if shape.len() != 4
            || shape
                .iter()
                .map(|value| value.as_u64().unwrap_or(u64::MAX))
                .ne(want_shape)
        {
            return Err(fail(format!("{name} shape {shape:?}")));
        }
        let offsets = tensor
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| fail(format!("{name} has no data_offsets")))?;
        let [start, end] = <[u64; 2]>::try_from(
            offsets
                .iter()
                .map(|value| value.as_u64().unwrap_or(u64::MAX))
                .collect::<Vec<_>>(),
        )
        .map_err(|_| fail(format!("{name} data_offsets malformed")))?
        else {
            return Err(fail(format!("{name} data_offsets must be a pair")));
        };
        if start != offset || end != offset + tensor_len as u64 {
            return Err(fail(format!("{name} offsets are not sequential")));
        }
        file.seek(std::io::SeekFrom::Start(8 + length + start))
            .map_err(|error| fail(error.to_string()))?;
        let mut payload = vec![0u8; tensor_len];
        file.read_exact(&mut payload)
            .map_err(|error| fail(error.to_string()))?;
        Array::from_bytes(stream, &want_shape.map(|n| n as usize), MlxDType::BF16, &payload)
    };

    let mut kv = Vec::with_capacity(BLOCKS);
    for block in 0..BLOCKS {
        let base = block as u64 * 2 * tensor_len as u64;
        let k = read_tensor(&tensor_name(K, block), base)?;
        let v = read_tensor(&tensor_name(V, block), base + tensor_len as u64)?;
        kv.push((k, v));
    }
    Ok(kv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_mlx::Stream;

    fn tokens(n: usize, drop: usize) -> PromptTokens {
        PromptTokens {
            ids: (0..n as u32).collect(),
            drop,
        }
    }

    fn tempdir(label: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "apxinf-text-kv-cache-test-{label}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn config(dir: &Path, max_mb: u64) -> Config {
        Config {
            dir: dir.to_path_buf(),
            max_bytes: max_mb * (1 << 20),
        }
    }

    fn kv_bytes(stream: &Stream, tokens: &PromptTokens, fill: u8) -> (Vec<(Array, Array)>, Vec<u8>) {
        let t = text_len(tokens);
        let n = t * HEADS * HEAD_DIM * 2;
        let bytes: Vec<u8> = (0..n).map(|i| fill ^ (i % 251) as u8).collect();
        let one = |block: usize, kind: &str| {
            let seed = if kind == K { block } else { block + 100 };
            let data: Vec<u8> = bytes.iter().map(|b| b ^ seed as u8).collect();
            Array::from_bytes(
                stream,
                &[1, HEADS, t, HEAD_DIM],
                MlxDType::BF16,
                &data,
            )
            .unwrap()
        };
        let kv: Vec<_> = (0..BLOCKS).map(|b| (one(b, K), one(b, V))).collect();
        (kv, bytes)
    }

    #[test]
    fn cache_key_is_stable_and_sensitive() {
        let pin_a = [7u8; 32];
        let pin_b = [8u8; 32];
        let t = tokens(24, 2);
        let mut edited = tokens(24, 2);
        edited.ids[10] = 999;
        assert_eq!(
            cache_key(&pin_a, &pin_b, &t),
            cache_key(&pin_a, &pin_b, &t)
        );
        assert_ne!(cache_key(&pin_a, &pin_b, &t), cache_key(&pin_b, &pin_b, &t));
        assert_ne!(cache_key(&pin_a, &pin_b, &t), cache_key(&pin_a, &pin_a, &t));
        assert_ne!(
            cache_key(&pin_a, &pin_b, &t),
            cache_key(&pin_a, &pin_b, &edited)
        );
        let mut retruncate = tokens(20, 2);
        assert_ne!(
            cache_key(&pin_a, &pin_b, &t),
            cache_key(&pin_a, &pin_b, &retruncate)
        );
        retruncate.drop = 1;
        assert_ne!(
            cache_key(&pin_a, &pin_b, &tokens(20, 2)),
            cache_key(&pin_a, &pin_b, &retruncate)
        );
    }

    #[test]
    fn store_then_load_roundtrips_bits() {
        let dir = tempdir("roundtrip");
        let stream = Stream::cpu().unwrap();
        let t = tokens(19, 2);
        let (kv, _) = kv_bytes(&stream, &t, 0x5a);
        let pin_a = [3u8; 32];
        let pin_b = [4u8; 32];
        let key = cache_key(&pin_a, &pin_b, &t);
        store_at(&config(&dir, 64), &key, &pin_a, &pin_b, &t, &kv).unwrap();
        let hit = load_at(&config(&dir, 64), &key, &pin_a, &pin_b, &t, &stream).unwrap();
        let Some((cache, bytes)) = hit else { panic!("expected a hit") };
        assert_eq!(bytes, payload_len(&t).unwrap());
        assert_eq!(cache.kv().len(), BLOCKS);
        for ((want_k, want_v), (got_k, got_v)) in kv.iter().zip(cache.kv().iter()) {
            assert_eq!(want_k.to_bytes().unwrap(), got_k.to_bytes().unwrap());
            assert_eq!(want_v.to_bytes().unwrap(), got_v.to_bytes().unwrap());
        }
    }

    #[test]
    fn lru_eviction_drops_the_stale_entry() {
        let dir = tempdir("lru");
        let stream = Stream::cpu().unwrap();
        let t = tokens(9, 1);
        let pin_a = [3u8; 32];
        let pin_b = [4u8; 32];
        let per_entry = payload_len(&t).unwrap() as f64 / (1 << 20) as f64;
        let max_mb = (per_entry * 1.5).ceil() as u64;
        let key_a = cache_key(&pin_a, &pin_b, &t);
        let (kv_a, _) = kv_bytes(&stream, &t, 0x11);
        store_at(&config(&dir, max_mb), &key_a, &pin_a, &pin_b, &t, &kv_a).unwrap();
        // A second, same-size entry must evict the first under a
        // 1.5-entry cap.
        let mut other = tokens(9, 1);
        other.ids[0] ^= 0x8000_0000;
        let key_b = cache_key(&pin_a, &pin_b, &other);
        let (kv_b, _) = kv_bytes(&stream, &other, 0x22);
        store_at(&config(&dir, max_mb), &key_b, &pin_a, &pin_b, &other, &kv_b).unwrap();
        assert!(
            load_at(&config(&dir, max_mb), &key_a, &pin_a, &pin_b, &t, &stream)
                .unwrap()
                .is_none(),
            "oldest entry should be evicted"
        );
        assert!(
            load_at(&config(&dir, max_mb), &key_b, &pin_a, &pin_b, &other, &stream)
                .unwrap()
                .is_some(),
            "newest entry should survive"
        );
    }

    #[test]
    fn tampered_shape_is_pruned_to_a_miss() {
        let dir = tempdir("tamper");
        let stream = Stream::cpu().unwrap();
        let t = tokens(9, 1);
        let pin_a = [3u8; 32];
        let pin_b = [4u8; 32];
        let key = cache_key(&pin_a, &pin_b, &t);
        let (kv, _) = kv_bytes(&stream, &t, 0x33);
        store_at(&config(&dir, 64), &key, &pin_a, &pin_b, &t, &kv).unwrap();
        // Corrupt one header byte (a stored shape); the entry must be
        // dropped, not served.
        let file = dir.join(format!("{key}.safetensors"));
        let mut raw = std::fs::read(&file).unwrap();
        raw[20] ^= 0xff;
        std::fs::write(&file, raw).unwrap();
        assert!(
            load_at(&config(&dir, 64), &key, &pin_a, &pin_b, &t, &stream)
                .unwrap()
                .is_none(),
            "corrupt entry should be pruned"
        );
        assert!(!file.exists(), "corrupt entry file should be removed");
    }

    #[test]
    fn mismatched_pin_or_prompt_misses() {
        let dir = tempdir("mismatch");
        let stream = Stream::cpu().unwrap();
        let t = tokens(9, 1);
        let pin_a = [3u8; 32];
        let pin_b = [4u8; 32];
        let key = cache_key(&pin_a, &pin_b, &t);
        let (kv, _) = kv_bytes(&stream, &t, 0x44);
        store_at(&config(&dir, 64), &key, &pin_a, &pin_b, &t, &kv).unwrap();
        assert!(
            load_at(&config(&dir, 64), &key, &[9u8; 32], &pin_b, &t, &stream)
                .unwrap()
                .is_none()
        );
        assert!(
            load_at(&config(&dir, 64), &key, &pin_a, &[9u8; 32], &t, &stream)
                .unwrap()
                .is_none()
        );
        let mut other = tokens(9, 1);
        other.ids[3] = 7;
        assert!(
            load_at(&config(&dir, 64), &key, &pin_a, &pin_b, &other, &stream)
                .unwrap()
                .is_none()
        );
    }
}
