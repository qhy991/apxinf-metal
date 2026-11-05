use std::process::Command;

fn executable() -> Command {
    Command::new(env!("CARGO_BIN_EXE_apxinf-qwen35-08b"))
}

#[test]
fn help_and_invalid_arguments_never_require_model_assets() {
    let help = executable().arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8(help.stdout).unwrap().contains("2048"));
    let invalid = executable().arg("--unknown").output().unwrap();
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
    assert!(String::from_utf8(invalid.stderr)
        .unwrap()
        .contains("Unknown argument"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn a_same_size_modified_config_is_rejected_before_model_construction() {
    let directory =
        std::env::temp_dir().join(format!("apxinf-qwen35-entry-assets-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("config.json"), vec![b' '; 2907]).unwrap();
    let result = executable()
        .args([
            "--model",
            directory.to_str().unwrap(),
            "--prompt",
            "hello",
            "--json",
        ])
        .output();
    std::fs::remove_dir_all(&directory).unwrap();
    let result = result.unwrap();
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8(result.stderr)
        .unwrap()
        .contains("Asset SHA-256 mismatch: config.json"));
}
