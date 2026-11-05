use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::Result;

pub const MODEL_REVISION: &str = "2fc06364715b967f1860aea9cf38778875588b17";

struct AssetSpec {
    name: &'static str,
    bytes: u64,
    sha256: &'static str,
}

const CONFIG: AssetSpec = AssetSpec {
    name: "config.json",
    bytes: 2907,
    sha256: "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
};
const TOKENIZER: AssetSpec = AssetSpec {
    name: "tokenizer.json",
    bytes: 12807982,
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
};
const TOKENIZER_CONFIG: AssetSpec = AssetSpec {
    name: "tokenizer_config.json",
    bytes: 16709,
    sha256: "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
};
const CHAT_TEMPLATE: AssetSpec = AssetSpec {
    name: "chat_template.jinja",
    bytes: 7755,
    sha256: "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
};
const WEIGHTS: AssetSpec = AssetSpec {
    name: "model.safetensors-00001-of-00001.safetensors",
    bytes: 1746942600,
    sha256: "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
};

pub struct Assets {
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub tokenizer_config: Vec<u8>,
    pub chat_template: Vec<u8>,
    pub weights: File,
}

impl Assets {
    pub fn load(directory: &Path) -> Result<Self> {
        let (_, config) = verify(directory, &CONFIG, true)?;
        let (_, tokenizer) = verify(directory, &TOKENIZER, true)?;
        let (_, tokenizer_config) = verify(directory, &TOKENIZER_CONFIG, true)?;
        let (_, chat_template) = verify(directory, &CHAT_TEMPLATE, true)?;
        let (weights, _) = verify(directory, &WEIGHTS, false)?;
        Ok(Self {
            config,
            tokenizer,
            tokenizer_config,
            chat_template,
            weights,
        })
    }
}

fn verify(directory: &Path, spec: &AssetSpec, retain_bytes: bool) -> Result<(File, Vec<u8>)> {
    let path = directory.join(spec.name);
    let file =
        File::open(&path).map_err(|error| format!("Open asset {}: {error}", path.display()))?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != spec.bytes {
        return Err(format!(
            "Asset size mismatch: {} (expected {} bytes)",
            spec.name, spec.bytes
        )
        .into());
    }
    let mut bytes = Vec::new();
    let mut hasher = Sha256::new();
    let mut reader = (&file).take(spec.bytes + 1);
    let mut buffer = [0_u8; 65536];
    let mut count = 0_u64;
    loop {
        let received = reader.read(&mut buffer)?;
        if received == 0 {
            break;
        }
        count += received as u64;
        hasher.update(&buffer[..received]);
        if retain_bytes {
            bytes.extend_from_slice(&buffer[..received]);
        }
    }
    if count != spec.bytes || format!("{:x}", hasher.finalize()) != spec.sha256 {
        return Err(format!("Asset SHA-256 mismatch: {}", spec.name).into());
    }
    Ok((file, bytes))
}

pub fn identity() -> Value {
    let mut assets = serde_json::Map::new();
    for spec in [
        &CONFIG,
        &TOKENIZER,
        &TOKENIZER_CONFIG,
        &CHAT_TEMPLATE,
        &WEIGHTS,
    ] {
        assets.insert(
            spec.name.to_owned(),
            json!({"bytes": spec.bytes, "sha256": spec.sha256}),
        );
    }
    json!({"model": "Qwen/Qwen3.5-0.8B", "revision": MODEL_REVISION, "assets": assets})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "apxinf-qwen35-assets-{}-{}",
                std::process::id(),
                NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const TEST_ASSET: AssetSpec = AssetSpec {
        name: "test.bin",
        bytes: 3,
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    };

    #[test]
    fn same_size_different_bytes_fail_the_asset_digest() {
        let directory = Directory::new();
        std::fs::write(directory.0.join(TEST_ASSET.name), b"abc").unwrap();
        assert_eq!(verify(&directory.0, &TEST_ASSET, true).unwrap().1, b"abc");
        std::fs::write(directory.0.join(TEST_ASSET.name), b"abd").unwrap();
        assert!(verify(&directory.0, &TEST_ASSET, false)
            .unwrap_err()
            .to_string()
            .contains("SHA-256 mismatch"));
        std::fs::write(directory.0.join(TEST_ASSET.name), b"abcd").unwrap();
        assert!(verify(&directory.0, &TEST_ASSET, false)
            .unwrap_err()
            .to_string()
            .contains("size mismatch"));
    }

    #[test]
    fn verified_file_remains_the_original_file_after_path_replacement() {
        let directory = Directory::new();
        let path = directory.0.join(TEST_ASSET.name);
        std::fs::write(&path, b"abc").unwrap();
        let (mut file, bytes) = verify(&directory.0, &TEST_ASSET, false).unwrap();
        assert!(bytes.is_empty());
        std::fs::rename(&path, directory.0.join("old.bin")).unwrap();
        std::fs::write(&path, b"abd").unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut actual = Vec::new();
        file.read_to_end(&mut actual).unwrap();
        assert_eq!(actual, b"abc");
    }
}
