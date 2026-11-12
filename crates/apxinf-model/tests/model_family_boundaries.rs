use std::path::PathBuf;
use std::process::Command;

#[test]
fn model_families_do_not_import_each_other() {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/check_model_family_boundaries.sh");
    let status = Command::new("bash")
        .arg(&script)
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", script.display()));

    assert!(status.success(), "model-family boundary check failed");
}
