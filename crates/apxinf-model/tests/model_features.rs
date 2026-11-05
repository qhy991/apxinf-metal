#![cfg(feature = "model-registry")]

use std::path::Path;

use apxinf_core::Device;
use apxinf_model::{register_builtin_models, AutoModel, LoadOptions};

#[test]
fn builtin_registry_matches_compiled_model_families() {
    register_builtin_models();
    let expected = [
        ("llama", cfg!(feature = "model-llama")),
        ("qwen3_5", cfg!(feature = "model-qwen35")),
        ("qwen35", cfg!(feature = "model-qwen35")),
        ("qwen3_vl", cfg!(feature = "model-qwen3vl")),
        ("qwen3vl", cfg!(feature = "model-qwen3vl")),
        (
            "pi05-cuda",
            cfg!(all(feature = "model-pi05", feature = "cuda")),
        ),
    ];
    for (name, enabled) in expected {
        assert_eq!(apxinf_model::get(name).is_some(), enabled, "{name}");
    }
}

fn assert_unavailable(name: &str) {
    let options = LoadOptions {
        model_name: Some(name.to_owned()),
        ..LoadOptions::default()
    };
    let error = AutoModel::load_model(
        Device::Cpu,
        Path::new("/apxinf-model-feature-test/no-checkpoint"),
        &options,
    )
    .err()
    .expect("an unavailable model must fail before loading weights");
    assert!(
        error
            .to_string()
            .contains(&format!("no model implementation for `{name}`")),
        "{error}"
    );
}

#[test]
fn unknown_model_fails_without_weight_access() {
    assert_unavailable("unsupported-model-feature-test");
}

#[test]
fn excluded_model_families_fail_without_weight_access() {
    if !cfg!(feature = "model-llama") {
        assert_unavailable("llama");
    }
    if !cfg!(feature = "model-qwen35") {
        assert_unavailable("qwen3_5");
        assert_unavailable("qwen35");
    }
    if !cfg!(feature = "model-qwen3vl") {
        assert_unavailable("qwen3_vl");
        assert_unavailable("qwen3vl");
    }
    if !cfg!(feature = "model-pi05") {
        assert_unavailable("pi05");
        assert_unavailable("pi05-cuda");
    }
}

#[cfg(not(any(
    feature = "model-llama",
    feature = "model-qwen35",
    feature = "model-qwen3vl",
    feature = "model-pi05"
)))]
#[test]
fn no_model_build_has_no_builtin_loaders() {
    register_builtin_models();
    assert!(apxinf_model::list().is_empty());
    assert_unavailable("qwen3_5");
}
