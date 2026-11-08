//! Shared on-disk cache primitives for the Qwen-Image cross-request caches
//! (EngineTailor rounds 22 and 23): a JSON `Index` of entries with
//! atomic tmp+rename writes and least-recently-used eviction, plus the small
//! helpers both caches use. Cache-specific policy — the key, the pin set,
//! the value format, the environment names — stays in each cache module.

use apxinf_core::Result;
use std::path::Path;

/// Lowercase hex encoding of a byte string (cache keys and pins).
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// First 16 hex chars of a cache key for log lines.
pub fn short_key(key: &str) -> &str {
    key.get(..16).unwrap_or(key)
}

pub fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Index {
    pub entries: Vec<Entry>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct Entry {
    pub key: String,
    pub file: String,
    pub bytes: u64,
    pub created_unix_ms: u64,
    pub last_hit_unix_ms: u64,
}

impl Index {
    pub fn read(dir: &Path) -> Result<Self> {
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

    pub fn write(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join(format!(".index.json.tmp{}", std::process::id()));
        let bytes = serde_json::to_vec(self)
            .map_err(|error| apxinf_core::Error::Other(format!("serialize cache index: {error}")))?;
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, dir.join("index.json"))?;
        Ok(())
    }

    pub fn upsert(&mut self, entry: Entry) {
        self.entries
            .retain(|existing| existing.key != entry.key && existing.file != entry.file);
        self.entries.push(entry);
    }

    pub fn touch(&mut self, key: &str) {
        let now = now_unix_ms();
        for entry in &mut self.entries {
            if entry.key == key {
                entry.last_hit_unix_ms = now;
                return;
            }
        }
    }

    pub fn remove(&mut self, key: &str) {
        self.entries.retain(|entry| entry.key != key);
    }

    /// Drops least-recently-hit entries until the total is within `max`.
    /// Ties on the hit timestamp fall back to insertion order (stable sort),
    /// so equal timestamps evict the earliest inserted entry first.
    pub fn evict(&mut self, dir: &Path, max: u64) {
        loop {
            let total: u64 = self.entries.iter().map(|entry| entry.bytes).sum();
            if total <= max || self.entries.is_empty() {
                return;
            }
            self.entries.sort_by_key(|entry| entry.last_hit_unix_ms);
            let victim = self.entries.remove(0);
            let _ = std::fs::remove_file(dir.join(&victim.file));
        }
    }
}
