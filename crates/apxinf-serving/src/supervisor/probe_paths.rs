//! Scratch directories for CPU probe fixtures.
//!
//! Every probe run owns one directory under a single recognisable root. A
//! clean run removes its directory; an interrupted run leaves one recognisable
//! directory instead of files scattered through the system temp root.
use std::path::{Path, PathBuf};

/// Probe runs live under this directory inside the system temp directory.
pub(super) const ROOT: &str = "apxinf-serving-tests";

pub(super) struct ProbeDir {
    path: PathBuf,
}

impl ProbeDir {
    pub(super) fn new() -> Self {
        let path = std::env::temp_dir().join(ROOT).join(format!(
            "{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&path).expect("The probe directory must be created.");
        Self { path }
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn file(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    pub(super) fn cleanup(&self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

impl Drop for ProbeDir {
    fn drop(&mut self) {
        self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        std::env::temp_dir().join(ROOT)
    }

    #[test]
    fn probe_directories_are_unique_and_inside_one_root() {
        let first = ProbeDir::new();
        let second = ProbeDir::new();
        assert_eq!(first.path().parent(), Some(root().as_path()));
        assert_eq!(second.path().parent(), Some(root().as_path()));
        assert_ne!(first.path(), second.path());
        assert!(first.path().is_dir());
        assert!(first.file("trace.jsonl").starts_with(first.path()));
        first.cleanup();
        assert!(!first.path().exists());
        assert!(second.path().is_dir(), "Another run keeps its directory.");
    }

    #[test]
    fn an_interrupted_run_leaves_one_recognisable_directory() {
        // A killed test process never runs Drop, so the run leaves exactly one
        // directory under the known root instead of scattered files.
        let dir = ProbeDir::new();
        let path = dir.path().to_path_buf();
        std::mem::forget(dir);
        assert!(path.is_dir());
        assert_eq!(path.parent(), Some(root().as_path()));
        std::fs::remove_dir_all(&path).unwrap();
    }
}
