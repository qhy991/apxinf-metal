//! Fork-compatible helpers layered on top of upstream's concurrent ProgramCache.
//!
//! The upstream cache (LRU byte budget, cross-process entry locks, batch
//! prediction) owns [`super::ProgramCache`]. This module keeps the fork's
//! manifest-keyed compile API — `compiled_for`, `evict`, and the no-return
//! `seal` spelling — so existing Metal families and their tests keep working
//! without forking the cache implementation.

use std::path::{Path, PathBuf};

use apxinf_core::{Error, Result};
use sha2::{Digest, Sha256};

use crate::program::Package;
use crate::runtime::{ProgramCache, compile};

/// Cache a compiled program keyed by an explicit manifest string.
///
/// On a hit the manifest on disk must equal `manifest` byte-for-byte; a
/// directory without a matching record is treated as a missed publish and
/// rebuilt. The macOS build string is folded into the key so a cache directory
/// can hold entries from more than one OS build.
pub fn compiled_for(
    cache: &ProgramCache,
    manifest: &str,
    build: impl FnOnce() -> Result<Package>,
) -> Result<(PathBuf, bool)> {
    let mut hash = Sha256::new();
    hash.update(manifest.as_bytes());
    hash.update(system_version());
    let key = hex16(&hash.finalize());
    let dir = cache.cache_dir();
    let compiled = dir.join(format!("{key}.mlmodelc"));
    let record = dir.join(format!("{key}.json"));
    if compiled.is_dir() && std::fs::read(&record).is_ok_and(|r| r == manifest.as_bytes()) {
        return Ok((compiled, true));
    }
    // The upstream cache rejects misses after seal(); reusing its sealed
    // contract keeps both spellings consistent.
    if cache.is_sealed() {
        return Err(Error::Contract(
            "Core ML programs are compiled before the cache is sealed",
        ));
    }
    let package = build()?;
    let source_dir = dir.join(format!(".{key}-source"));
    let compiled_dir = dir.join(format!(".{key}-compiled"));
    std::fs::create_dir_all(&dir)
        .map_err(|e| Error::Other(format!("Core ML cache: {e}")))?;
    let _ = std::fs::remove_dir_all(&source_dir);
    let _ = std::fs::remove_dir_all(&compiled_dir);
    package.write(&source_dir)?;
    compile(&source_dir, &compiled_dir)?;
    // Publish atomically: a concurrent loader may have placed the same key
    // first, in which case its entry wins and ours is discarded.
    if std::fs::rename(&compiled_dir, &compiled).is_err() && !compiled.is_dir() {
        let _ = std::fs::remove_dir_all(&source_dir);
        let _ = std::fs::remove_dir_all(&compiled_dir);
        return Err(Error::Other(
            "cannot place the compiled Core ML program".into(),
        ));
    }
    std::fs::write(&record, manifest)
        .map_err(|e| Error::Other(format!("Core ML cache: {e}")))?;
    let _ = std::fs::remove_dir_all(&source_dir);
    Ok((compiled, false))
}

/// Discards the entry `compiled` and its manifest, so the next request for it
/// builds and compiles again. Core ML refusing to load a stored program is the
/// other way an entry turns out to be unusable.
pub fn evict(_cache: &ProgramCache, compiled: &Path) -> Result<()> {
    for path in [compiled.to_path_buf(), compiled.with_extension("json")] {
        let removed = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        match removed {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::Other(format!("Core ML cache: {e}"))),
        }
    }
    Ok(())
}

fn system_version() -> Vec<u8> {
    std::fs::read("/System/Library/CoreServices/SystemVersion.plist").unwrap_or_default()
}

fn hex16(digest: &[u8]) -> String {
    digest[..16].iter().map(|b| format!("{b:02x}")).collect()
}
