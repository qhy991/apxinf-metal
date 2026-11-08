//! Disk-persistent output cache for the Qwen-Image text encoder
//! (EngineTailor round 22, qhy991/EngineTailor#231).
//!
//! The templated prompt is truncated to 2048 tokens before encoding, so two
//! prompts that differ only after the truncation point — or a re-rolled
//! identical prompt — produce the same token ids and therefore byte-identical
//! BF16 prompt states. This cache keys on a hash of the truncated ids plus a
//! pin of the text-encoder shard files, and stores the evaluated
//! `[1, T, 4096]` BF16 output of `text::encode` as a single-tensor
//! SafeTensors file. A hit loads those bytes back onto the device and skips
//! the whole encode; because the bytes are the exact result of a previous
//! evaluation on the same weights, a hit is bit-identical to a cold run by
//! construction (round 22 verifies this A/A and across fresh processes).
//!
//! Writes are atomic (tmp file + rename, fsync before the rename) and the
//! directory is bounded by a byte cap with least-recently-used eviction.
//! Cache problems — I/O errors, corrupt or stale entries, a missing index —
//! never fail a request: they log one stderr line and degrade to a miss.
//!
//! Environment:
//! - `APXINF_QWEN_IMAGE_TEXT_CACHE=0` disables the cache (the exact legacy
//!   path; default enabled);
//! - `APXINF_QWEN_IMAGE_TEXT_CACHE_DIR` overrides the default
//!   `~/.cache/apxinf/text-encode`;
//! - `APXINF_QWEN_IMAGE_TEXT_CACHE_MAX_MB` overrides the 2048 MiB cap.
//!
//! The pin covers each shard's path, byte size, modification time and
//! SafeTensors header bytes, so replacing or re-exporting the checkpoint
//! invalidates every entry. An in-place rewrite that keeps size, mtime and
//! headers byte-identical is outside this local cache's threat model; delete
//! the cache directory when swapping checkpoints underneath it.
use super::text::PromptTokens;
use super::weights::{TextEncoderFiles, DIM};
use apxinf_core::Result;
use apxinf_mlx::{Array, MlxDType, Stream};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Bumped when the on-disk format or the prompt template changes, so old
/// entries can never be served under new semantics.
const FORMAT_VERSION: u32 = 1;
/// Default capacity cap in MiB; `APXINF_QWEN_IMAGE_TEXT_CACHE_MAX_MB` overrides.
const DEFAULT_MAX_MB: u64 = 2048;
/// Tensor name used inside the single-tensor SafeTensors value file.
const TENSOR_NAME: &str = "prompt_states";

const ENV_SWITCH: &str = "APXINF_QWEN_IMAGE_TEXT_CACHE";
const ENV_DIR: &str = "APXINF_QWEN_IMAGE_TEXT_CACHE_DIR";
const ENV_MAX_MB: &str = "APXINF_QWEN_IMAGE_TEXT_CACHE_MAX_MB";

struct Config {
    dir: PathBuf,
    max_bytes: u64,
}

fn cache_enabled() -> bool {
    std::env::var(ENV_SWITCH).map(|value| value != "0").unwrap_or(true)
}

fn log_disabled_once(reason: String) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("qimage-timing: text_encode_cache disabled ({reason})"));
}

fn config_from_env() -> Option<Config> {
    let dir = match std::env::var_os(ENV_DIR) {
        Some(dir) => PathBuf::from(dir),
        None => match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".cache/apxinf/text-encode"),
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

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn short_key(key: &str) -> &str {
    key.get(..16).unwrap_or(key)
}

/// Cache identity of one prompt encoding: the truncated template ids, the
/// text length they imply, and the text-encoder weight pin.
fn cache_key(pin: &[u8; 32], tokens: &PromptTokens) -> String {
    let mut hasher = Sha256::new();
    hasher.update(pin);
    hasher.update(FORMAT_VERSION.to_le_bytes());
    hasher.update((tokens.ids.len() as u64).to_le_bytes());
    hasher.update((tokens.drop as u64).to_le_bytes());
    for id in &tokens.ids {
        hasher.update(id.to_le_bytes());
    }
    hex(&hasher.finalize())
}

fn text_len(tokens: &PromptTokens) -> usize {
    tokens.ids.len().saturating_sub(tokens.drop)
}

/// Returns the cached prompt states for `tokens`, or `None` on any miss.
/// Degrades to `None` — never an error — on every cache problem.
pub fn load(
    files: &TextEncoderFiles,
    tokens: &PromptTokens,
    stream: &Stream,
) -> Result<Option<Array>> {
    if !cache_enabled() {
        log_disabled_once(format!("{ENV_SWITCH}=0"));
        return Ok(None);
    }
    let Some(config) = config_from_env() else {
        return Ok(None);
    };
    let key = cache_key(files.pin(), tokens);
    let start = std::time::Instant::now();
    match load_at(&config, &key, files.pin(), tokens, stream) {
        Ok(Some((array, bytes))) => {
            eprintln!(
                "qimage-timing: text_encode_cache hit=1 key={} bytes={} load={:.3}s",
                short_key(&key),
                bytes,
                start.elapsed().as_secs_f64()
            );
            Ok(Some(array))
        }
        Ok(None) => {
            eprintln!("qimage-timing: text_encode_cache hit=0 key={}", short_key(&key));
            Ok(None)
        }
        Err(error) => {
            eprintln!(
                "qimage-timing: text_encode_cache hit=0 key={} error={error}",
                short_key(&key)
            );
            Ok(None)
        }
    }
}

/// Stores the evaluated prompt states under `tokens`' cache identity. Stores
/// are best-effort: problems log one line and leave generation unaffected.
pub fn store(files: &TextEncoderFiles, tokens: &PromptTokens, prompt: &Array) -> Result<()> {
    if !cache_enabled() {
        return Ok(());
    }
    let Some(config) = config_from_env() else {
        return Ok(());
    };
    if prompt.dtype() != MlxDType::BF16 || prompt.shape() != [1, text_len(tokens), DIM] {
        eprintln!(
            "qimage-timing: text_encode_cache store skipped: unexpected {} {:?}",
            prompt.dtype() as u32,
            prompt.shape()
        );
        return Ok(());
    }
    let key = cache_key(files.pin(), tokens);
    match store_at(&config, &key, files.pin(), tokens, prompt) {
        Ok(bytes) => eprintln!(
            "qimage-timing: text_encode_cache store key={} bytes={bytes}",
            short_key(&key)
        ),
        Err(error) => eprintln!(
            "qimage-timing: text_encode_cache store key={} failed: {error}",
            short_key(&key)
        ),
    }
    Ok(())
}

fn load_at(
    config: &Config,
    key: &str,
    pin: &[u8; 32],
    tokens: &PromptTokens,
    stream: &Stream,
) -> Result<Option<(Array, u64)>> {
    let mut index = Index::read(&config.dir)?;
    let Some(position) = index.entries.iter().position(|entry| entry.key == key) else {
        return Ok(None);
    };
    let path = config.dir.join(index.entries[position].file.clone());
    let bytes = index.entries[position].bytes;
    match read_entry(&path, key, pin, tokens, stream) {
        Ok(array) => {
            index.touch(key);
            let _ = index.write(&config.dir);
            Ok(Some((array, bytes)))
        }
        Err(_) => {
            // Stale or corrupt entry: drop the file and the index row, then
            // let the caller re-encode and repopulate.
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
    pin: &[u8; 32],
    tokens: &PromptTokens,
    prompt: &Array,
) -> Result<u64> {
    let bytes = prompt.to_bytes()?;
    std::fs::create_dir_all(&config.dir)?;
    let mut index = Index::read(&config.dir)?;
    let file = format!("{key}.safetensors");
    let tmp = config
        .dir
        .join(format!(".{key}.tmp{}", std::process::id()));
    write_entry(&tmp, key, pin, tokens, &bytes)?;
    std::fs::rename(&tmp, config.dir.join(&file))?;
    index.upsert(Entry {
        key: key.to_owned(),
        file,
        bytes: bytes.len() as u64,
        created_unix_ms: now_unix_ms(),
        last_hit_unix_ms: now_unix_ms(),
    });
    index.evict(&config.dir, config.max_bytes);
    index.write(&config.dir)?;
    Ok(bytes.len() as u64)
}

/// Serializes the prompt states as a one-tensor SafeTensors file:
/// `[u64 LE header length][header JSON][BF16 payload]`.
fn write_entry(
    path: &Path,
    key: &str,
    pin: &[u8; 32],
    tokens: &PromptTokens,
    bytes: &[u8],
) -> Result<()> {
    let header = serde_json::json!({
        TENSOR_NAME: {
            "dtype": "BF16",
            "shape": [1, text_len(tokens), DIM],
            "data_offsets": [0, bytes.len()],
        },
        "__metadata__": {
            "format": "apxinf-text-encode-cache",
            "version": FORMAT_VERSION.to_string(),
            "key": key,
            "pin": hex(pin),
            "tokens_len": tokens.ids.len().to_string(),
            "drop": tokens.drop.to_string(),
            "created_unix_ms": now_unix_ms().to_string(),
        },
    });
    let header = serde_json::to_vec(&header)
        .map_err(|error| apxinf_core::Error::Other(format!("serialize cache header: {error}")))?;
    let mut file = std::fs::File::create(path)?;
    use std::io::Write;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Reads and fully validates one entry; any deviation is an error so the
/// caller drops the entry and degrades to a miss.
fn read_entry(
    path: &Path,
    key: &str,
    pin: &[u8; 32],
    tokens: &PromptTokens,
    stream: &Stream,
) -> Result<Array> {
    use std::io::Read;
    let fail = |why: String| apxinf_core::Error::Other(format!("{}: {why}", path.display()));
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
    let tensor = header
        .get(TENSOR_NAME)
        .ok_or_else(|| fail("no prompt_states tensor".into()))?;
    if tensor.get("dtype").and_then(serde_json::Value::as_str) != Some("BF16") {
        return Err(fail("prompt_states is not BF16".into()));
    }
    let shape = tensor
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| fail("prompt_states has no shape".into()))?;
    let want_shape = [1u64, text_len(tokens) as u64, DIM as u64];
    if shape.len() != 3
        || shape
            .iter()
            .map(|value| value.as_u64().unwrap_or(u64::MAX))
            .ne(want_shape)
    {
        return Err(fail(format!("prompt_states shape {shape:?}")));
    }
    let offsets = tensor
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| fail("prompt_states has no data_offsets".into()))?;
    let [0, end] = <[u64; 2]>::try_from(
        offsets
            .iter()
            .map(|value| value.as_u64().unwrap_or(u64::MAX))
            .collect::<Vec<_>>(),
    )
    .map_err(|_| fail("prompt_states data_offsets malformed".into()))?
    else {
        return Err(fail("prompt_states data_offsets must start at 0".into()));
    };
    let payload_len = (text_len(tokens) * DIM * 2) as u64;
    if end != payload_len || file_size != 8 + length + payload_len {
        return Err(fail("prompt_states byte extent mismatch".into()));
    }
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
    if meta("key")? != key || meta("pin")? != hex(pin) {
        return Err(fail("entry identity mismatch".into()));
    }
    if meta("tokens_len")?.parse::<usize>() != Ok(tokens.ids.len()) {
        return Err(fail("entry token length mismatch".into()));
    }
    let mut payload = vec![0u8; payload_len as usize];
    file.read_exact(&mut payload)
        .map_err(|error| fail(error.to_string()))?;
    Ok(Array::from_bytes(
        stream,
        &[1, text_len(tokens), DIM],
        MlxDType::BF16,
        &payload,
    )?)
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Index {
    entries: Vec<Entry>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Entry {
    key: String,
    file: String,
    bytes: u64,
    created_unix_ms: u64,
    last_hit_unix_ms: u64,
}

impl Index {
    fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("index.json");
        let Ok(raw) = std::fs::read(&path) else {
            return Ok(Index { entries: vec![] });
        };
        match serde_json::from_slice(&raw) {
            Ok(index) => Ok(index),
            Err(_) => {
                // A corrupt index loses LRU order only; quarantine it and
                // start empty. Entries whose files still exist are
                // rediscoverable only through a fresh store, which is the
                // safe direction.
                let _ = std::fs::rename(&path, dir.join("index.json.bad"));
                Ok(Index { entries: vec![] })
            }
        }
    }

    fn write(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join(format!(".index.json.tmp{}", std::process::id()));
        let bytes = serde_json::to_vec(self)
            .map_err(|error| apxinf_core::Error::Other(format!("serialize cache index: {error}")))?;
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, dir.join("index.json"))?;
        Ok(())
    }

    fn upsert(&mut self, entry: Entry) {
        self.entries
            .retain(|existing| existing.key != entry.key && existing.file != entry.file);
        self.entries.push(entry);
    }

    fn touch(&mut self, key: &str) {
        let now = now_unix_ms();
        for entry in &mut self.entries {
            if entry.key == key {
                entry.last_hit_unix_ms = now;
                return;
            }
        }
    }

    fn remove(&mut self, key: &str) {
        self.entries.retain(|entry| entry.key != key);
    }

    /// Drops least-recently-hit entries until the total is within `max`.
    /// Ties on the hit timestamp fall back to insertion order (stable sort),
    /// so equal timestamps evict the earliest inserted entry first.
    fn evict(&mut self, dir: &Path, max: u64) {
        loop {
            let total: u64 = self.entries.iter().map(|entry| entry.bytes).sum();
            if total <= max || self.entries.is_empty() {
                return;
            }
            self.entries
                .sort_by_key(|entry| entry.last_hit_unix_ms);
            let victim = self.entries.remove(0);
            let _ = std::fs::remove_file(dir.join(&victim.file));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(n: usize, drop: usize) -> PromptTokens {
        PromptTokens {
            ids: (0..n as u32).collect(),
            drop,
        }
    }

    fn tempdir(label: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "apxinf-encode-cache-test-{label}-{}-{}",
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

    fn prompt_bytes(stream: &Stream, tokens: &PromptTokens, fill: u8) -> (Array, Vec<u8>) {
        let n = text_len(tokens) * DIM * 2;
        let bytes: Vec<u8> = (0..n).map(|i| fill ^ (i % 251) as u8).collect();
        (
            Array::from_bytes(stream, &[1, text_len(tokens), DIM], MlxDType::BF16, &bytes)
                .unwrap(),
            bytes,
        )
    }

    #[test]
    fn cache_key_is_stable_and_sensitive() {
        let pin_a = [7u8; 32];
        let pin_b = [8u8; 32];
        let t = tokens(24, 2);
        let mut edited = tokens(24, 2);
        edited.ids[10] = 999;
        assert_eq!(cache_key(&pin_a, &t), cache_key(&pin_a, &t));
        assert_ne!(cache_key(&pin_a, &t), cache_key(&pin_b, &t));
        assert_ne!(cache_key(&pin_a, &t), cache_key(&pin_a, &edited));
        let mut retruncate = tokens(20, 2);
        assert_ne!(cache_key(&pin_a, &t), cache_key(&pin_a, &retruncate));
        retruncate.drop = 1;
        assert_ne!(cache_key(&pin_a, &tokens(20, 2)), cache_key(&pin_a, &retruncate));
    }

    #[test]
    fn store_then_load_roundtrips_bits() {
        let dir = tempdir("roundtrip");
        let stream = Stream::cpu().unwrap();
        let t = tokens(19, 2);
        let pin = [3u8; 32];
        let key = cache_key(&pin, &t);
        let (array, bytes) = prompt_bytes(&stream, &t, 0x5a);
        store_at(&config(&dir, 64), &key, &pin, &t, &array).unwrap();
        let hit = load_at(&config(&dir, 64), &key, &pin, &t, &stream)
            .unwrap()
            .expect("stored entry must hit");
        assert_eq!(hit.0.to_bytes().unwrap(), bytes);
        assert_eq!(hit.1, bytes.len() as u64);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn different_prompts_have_distinct_entries() {
        let dir = tempdir("distinct");
        let stream = Stream::cpu().unwrap();
        let pin = [3u8; 32];
        let t1 = tokens(19, 2);
        let t2 = tokens(21, 2);
        let k1 = cache_key(&pin, &t1);
        let k2 = cache_key(&pin, &t2);
        assert_ne!(k1, k2);
        let (a1, _) = prompt_bytes(&stream, &t1, 0x11);
        let (a2, _) = prompt_bytes(&stream, &t2, 0x22);
        store_at(&config(&dir, 64), &k1, &pin, &t1, &a1).unwrap();
        store_at(&config(&dir, 64), &k2, &pin, &t2, &a2).unwrap();
        assert!(load_at(&config(&dir, 64), &k1, &pin, &t1, &stream)
            .unwrap()
            .is_some());
        assert!(load_at(&config(&dir, 64), &k2, &pin, &t2, &stream)
            .unwrap()
            .is_some());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn lru_evicts_least_recently_hit() {
        let dir = tempdir("lru");
        let stream = Stream::cpu().unwrap();
        let pin = [3u8; 32];
        let mut keys = Vec::new();
        for (i, n) in [19usize, 20, 21].into_iter().enumerate() {
            let t = tokens(n, 2);
            let key = cache_key(&pin, &t);
            let (array, _) = prompt_bytes(&stream, &t, i as u8);
            store_at(&config(&dir, 64), &key, &pin, &t, &array).unwrap();
            keys.push((key, t));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // All three fit (each ~150 KiB, cap 64 MiB): everyone hits.
        for (key, t) in &keys {
            assert!(load_at(&config(&dir, 64), key, &pin, t, &stream)
                .unwrap()
                .is_some());
        }
        // Shrink the cap below the total; the untouched... all were touched,
        // so re-touch the first, making the second the coldest.
        load_at(&config(&dir, 64), &keys[0].0, &pin, &keys[0].1, &stream).unwrap();
        // Cap = 2 entries worth (~300 KiB).
        let two = 2 * (19 * DIM * 2 + 4096) as u64;
        let mut index = Index::read(&dir).unwrap();
        index.evict(&dir, two);
        index.write(&dir).unwrap();
        assert!(load_at(&config(&dir, 64), &keys[0].0, &pin, &keys[0].1, &stream)
            .unwrap()
            .is_some(), "recently hit entry survives");
        assert!(load_at(&config(&dir, 64), &keys[1].0, &pin, &keys[1].1, &stream)
            .unwrap()
            .is_none(), "coldest entry evicted");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tampered_entry_is_pruned_to_a_miss() {
        let dir = tempdir("tamper");
        let stream = Stream::cpu().unwrap();
        let t = tokens(19, 2);
        let pin = [3u8; 32];
        let key = cache_key(&pin, &t);
        let (array, _) = prompt_bytes(&stream, &t, 0x77);
        store_at(&config(&dir, 64), &key, &pin, &t, &array).unwrap();
        // Rewrite the header with a wrong shape claim.
        let path = dir.join(format!("{key}.safetensors"));
        let raw = std::fs::read(&path).unwrap();
        let (header_len, rest) = raw.split_at(8);
        let header_len = u64::from_le_bytes(header_len.try_into().unwrap()) as usize;
        let (header, payload) = rest.split_at(header_len);
        let mut value: serde_json::Value = serde_json::from_slice(header).unwrap();
        value[TENSOR_NAME]["shape"] = serde_json::json!([1, 1, DIM]);
        let header = serde_json::to_vec(&value).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(payload);
        std::fs::write(&path, out).unwrap();
        assert!(load_at(&config(&dir, 64), &key, &pin, &t, &stream)
            .unwrap()
            .is_none(), "shape mismatch must miss");
        assert!(!path.exists(), "stale entry file is pruned");
        assert!(Index::read(&dir).unwrap().entries.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn foreign_pin_or_key_is_a_miss() {
        let dir = tempdir("pin");
        let stream = Stream::cpu().unwrap();
        let t = tokens(19, 2);
        let pin_a = [3u8; 32];
        let pin_b = [4u8; 32];
        let key = cache_key(&pin_a, &t);
        let (array, _) = prompt_bytes(&stream, &t, 0x42);
        store_at(&config(&dir, 64), &key, &pin_a, &t, &array).unwrap();
        assert!(load_at(&config(&dir, 64), &key, &pin_b, &t, &stream)
            .unwrap()
            .is_none(), "different weight pin must miss");
        let other = tokens(23, 2);
        assert!(load_at(&config(&dir, 64), &cache_key(&pin_a, &other), &pin_a, &other, &stream)
            .unwrap()
            .is_none(), "different prompt must miss");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn disabled_switch_short_circuits() {
        std::env::set_var(ENV_SWITCH, "0");
        assert!(!cache_enabled(), "{ENV_SWITCH}=0 must disable the cache");
        std::env::remove_var(ENV_SWITCH);
        assert!(cache_enabled(), "cache defaults to enabled");
    }
}
