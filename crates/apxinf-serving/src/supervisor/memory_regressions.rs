//! Memory policy checks that do not load a model or access a device.
use super::*;

pub(super) fn configuration(memory_budget: u64, sequence_reservation: u64) -> Config {
    Config {
        runtime: WorkerRuntime::Python,
        python: PathBuf::from("/missing/apxinf-memory-test-python"),
        worker: PathBuf::from("/missing/apxinf-memory-test-worker"),
        model: PathBuf::from("/missing/apxinf-memory-test-model"),
        model_id: "memory-test".into(),
        max_context: 64,
        max_output: 16,
        prefill_step_size: 8,
        kv_bits: None,
        // Keep the generic memory fixture free of the python-mlx-lm budget terms;
        // the dedicated budget tests below set them explicitly.
        quantized_kv_start: 0,
        safety_margin: 0,
        prefill_transient: 0,
        output_batch_tokens: 1,
        queue_capacity: 1,
        timeout: Duration::from_secs(1),
        memory_budget,
        sequence_reservation,
        host_pressure_policy: crate::host_pressure::HostPressurePolicy::Disabled,
    }
}

#[test]
fn native_manifest_checks_selected_runtime_and_execution_profile() {
    let mut config = configuration(4096, 1024);
    config.model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/serving/native-tokenizer");
    let executable = std::env::current_exe().unwrap();
    let directory =
        std::env::temp_dir().join(format!("apxinf-native-identity-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir(&directory).unwrap();
    let library = directory.join("libmlx.dylib");
    std::fs::write(&library, b"original runtime identity fixture").unwrap();
    config.runtime = WorkerRuntime::NativeMiniCpm5 {
        executable: executable.clone(),
        metal_library: library.clone(),
    };
    let runtime = crate::native_identity::runtime(&executable, &library).unwrap();
    let mut manifest = crate::native_identity::manifest(
        &config.model,
        runtime,
        crate::native_identity::execution(
            config.max_context,
            config.max_output,
            config.memory_budget,
        ),
    )
    .unwrap();
    verify_manifest(&manifest, &config).unwrap();
    manifest["runtime"]["executable_sha256"] = "0".repeat(64).into();
    assert!(verify_manifest(&manifest, &config).is_err());
    manifest["runtime"] = crate::native_identity::runtime(&executable, &library).unwrap();
    manifest["execution"]["precision"] = "auto".into();
    assert!(verify_manifest(&manifest, &config).is_err());
    manifest["execution"]["precision"] = "bf16".into();
    config.runtime = WorkerRuntime::NativeMiniCpm5Sampled {
        executable: executable.clone(),
        metal_library: library.clone(),
    };
    assert!(verify_manifest(&manifest, &config).is_err());
    manifest = crate::native_identity::manifest(
        &config.model,
        crate::native_identity::runtime(&executable, &library).unwrap(),
        crate::native_identity::sampled_execution(
            config.max_context,
            config.max_output,
            config.memory_budget,
        ),
    )
    .unwrap();
    verify_manifest(&manifest, &config).unwrap();
    manifest["execution"]["sampling"]["top_p"] = "1".into();
    assert!(verify_manifest(&manifest, &config).is_err());
    config.runtime = WorkerRuntime::NativeMiniCpm5Thinking {
        executable: executable.clone(),
        metal_library: library.clone(),
    };
    assert!(verify_manifest(&manifest, &config).is_err());
    manifest = crate::native_identity::manifest(
        &config.model,
        crate::native_identity::runtime(&executable, &library).unwrap(),
        crate::native_identity::thinking_execution(
            config.max_context,
            config.max_output,
            config.memory_budget,
        ),
    )
    .unwrap();
    verify_manifest(&manifest, &config).unwrap();
    manifest["execution"]["enable_thinking"] = false.into();
    assert!(verify_manifest(&manifest, &config).is_err());
    config.runtime = WorkerRuntime::NativeMiniCpm5BoundedThinking {
        executable: executable.clone(),
        metal_library: library.clone(),
    };
    assert!(verify_manifest(&manifest, &config).is_err());
    manifest = crate::native_identity::manifest(
        &config.model,
        crate::native_identity::runtime(&executable, &library).unwrap(),
        crate::native_identity::bounded_thinking_execution(
            config.max_context,
            config.max_output,
            config.memory_budget,
        ),
    )
    .unwrap();
    verify_manifest(&manifest, &config).unwrap();
    manifest["execution"]["initial_reasoning"]["budget_tokens"] = 1024.into();
    assert!(verify_manifest(&manifest, &config).is_err());
    manifest["execution"]["initial_reasoning"]["budget_tokens"] = 512.into();
    config.runtime = WorkerRuntime::NativeMiniCpm5Thinking {
        executable: executable.clone(),
        metal_library: library.clone(),
    };
    assert!(verify_manifest(&manifest, &config).is_err());
    config.runtime = WorkerRuntime::NativeMiniCpm5BoundedThinking {
        executable: executable.clone(),
        metal_library: library.clone(),
    };
    config.prefill_step_size = 256;
    config.max_context = 16384;
    assert!(config.validate_memory().is_ok());
    config.max_context = 16385;
    assert!(config.validate_memory().is_err());
    config.max_context = 4096;
    config.prefill_step_size = 128;
    assert!(config.validate_memory().is_err());
    assert!(crate::native_identity::runtime(
        &executable,
        &config.model.join("tokenizer_config.json")
    )
    .is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn memory_limits_fail_before_the_worker_can_start() {
    for (budget, reservation) in [
        (0, 1),
        (1024, 0),
        (contracts::MAX_SAFE_INTEGER + 1, 1),
        (1024, u64::MAX),
        (1024, 1025),
    ] {
        let result = start(configuration(budget, reservation)).await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("Invalid memory limits must fail before startup."),
        };
        assert!(
            !error.contains("Cannot start worker"),
            "Memory validation ran too late: {error}"
        );
    }
}

#[tokio::test]
async fn clef_compatibility_arguments_fail_before_the_worker_can_start() {
    let mut late_errors = Vec::new();
    for (prefill_step_size, output_batch_tokens, parameter) in [
        (128, 1, "--prefill-step-size"),
        (256, 2, "--output-batch-tokens"),
    ] {
        for with_shutdown in [false, true] {
            let mut config = configuration(4096, 1024);
            config.runtime = WorkerRuntime::PythonClef {
                loader: PathBuf::from("/missing/apxinf-clef-test-loader"),
            };
            config.max_context = 4096;
            config.max_output = 0;
            config.prefill_step_size = prefill_step_size;
            config.output_batch_tokens = output_batch_tokens;
            let result = if with_shutdown {
                start_with_shutdown(config, std::future::pending())
                    .await
                    .map(|_| ())
            } else {
                start(config).await.map(|_| ())
            };
            let error = result.expect_err("Unsupported Clef arguments must fail before startup.");
            if !error.contains("Clef") || !error.contains(parameter) {
                late_errors.push(format!("{parameter}, shutdown={with_shutdown}: {error}"));
            }
        }
    }
    assert!(
        late_errors.is_empty(),
        "Clef argument validation ran too late:\n{}",
        late_errors.join("\n")
    );
}

#[test]
fn output_limit_above_the_context_limit_fails_before_startup() {
    let mut config = configuration(4096, 1024);
    config.max_output = 2048;
    assert_eq!(
        config.validate_memory().unwrap_err(),
        "The output limit must not exceed the context limit."
    );
    config.max_context = 2048;
    assert!(config.validate_memory().is_ok());
    config.max_context = 128;
    assert!(config.validate_memory().is_err());
}

#[test]
fn context_limit_must_fit_the_default_output_budget() {
    let mut config = configuration(4096, 1024);
    assert!(config.validate_memory().is_ok());
    // An omitted max_tokens reserves the default budget, so the context must
    // also fit that budget plus one input token.
    config.max_context = 16;
    config.max_output = 16;
    assert_eq!(
        config.validate_memory().unwrap_err(),
        "The context limit must fit the default output budget and one input token; it must be at least 17 positions."
    );
    config.max_context = 17;
    assert!(config.validate_memory().is_ok());
    // A count-only deployment keeps its zero output limit.
    config.max_context = 1;
    config.max_output = 0;
    assert!(config.validate_memory().is_ok());
    // The minimum reads as a single position for a zero-sized deployment.
    config.max_context = 0;
    assert_eq!(
        config.validate_memory().unwrap_err(),
        "The context limit must fit the default output budget and one input token; it must be at least 1 position."
    );
}

#[tokio::test]
async fn unserviceable_context_limit_fails_before_the_worker_can_start() {
    for with_shutdown in [false, true] {
        let mut config = configuration(4096, 1024);
        config.max_context = 128;
        config.max_output = 128;
        let result = if with_shutdown {
            start_with_shutdown(config, std::future::pending())
                .await
                .map(|_| ())
        } else {
            start(config).await.map(|_| ())
        };
        let error = result.expect_err("An unserviceable context limit must fail before startup.");
        assert_eq!(
            error,
            "The context limit must fit the default output budget and one input token; it must be at least 129 positions.",
            "shutdown={with_shutdown}"
        );
    }
}

#[test]
fn clef_execution_limits_preserve_the_supported_boundaries() {
    let mut config = configuration(4096, 1024);
    config.runtime = WorkerRuntime::PythonClef {
        loader: PathBuf::from("/missing/apxinf-clef-test-loader"),
    };
    config.max_output = 0;
    config.prefill_step_size = 256;
    for max_context in [1, 4096] {
        config.max_context = max_context;
        assert!(config.validate_memory().is_ok());
    }
    for max_context in [0, 4097] {
        config.max_context = max_context;
        assert!(config.validate_memory().is_err());
    }
    config.max_context = 4096;
    config.max_output = 1;
    assert!(config.validate_memory().is_err());

    config.runtime = WorkerRuntime::Python;
    config.prefill_step_size = 128;
    config.output_batch_tokens = 2;
    assert!(config.validate_memory().is_ok());
}

#[test]
fn memory_limits_accept_only_the_documented_integer_boundaries() {
    for (budget, reservation) in [
        (1, 1),
        (contracts::MAX_SAFE_INTEGER, 1),
        (contracts::MAX_SAFE_INTEGER, contracts::MAX_SAFE_INTEGER),
    ] {
        assert!(configuration(budget, reservation).validate_memory().is_ok());
    }
    for (budget, reservation) in [(0, 0), (1, 0), (1, 2), (u64::MAX, 1), (u64::MAX, u64::MAX)] {
        assert!(configuration(budget, reservation)
            .validate_memory()
            .is_err());
    }
}

#[test]
fn memory_capacity_rejects_overflow_and_accepts_an_exact_fit() {
    let config = configuration(1024, 256);
    assert!(config.check_resident_capacity(768).is_ok());
    assert!(config.check_resident_capacity(769).is_err());
    assert!(configuration(u64::MAX, 1)
        .check_resident_capacity(u64::MAX)
        .unwrap_err()
        .contains("overflow"));
}

#[test]
fn memory_peak_uses_the_reported_peak_and_the_allocator_total() {
    assert_eq!(
        ready_memory_totals(&json!({"memory": {
            "active_bytes":100, "cache_bytes":20, "peak_bytes":500
        }}))
        .unwrap(),
        (120, 500)
    );
    assert_eq!(
        ready_memory_totals(&json!({"memory": {
            "active_bytes":100, "cache_bytes":400, "peak_bytes":200
        }}))
        .unwrap(),
        (500, 500)
    );
    assert!(ready_memory_totals(&json!({"memory": {
        "active_bytes":u64::MAX, "cache_bytes":1, "peak_bytes":0
    }}))
    .is_err());
    assert!(ready_memory_totals(&json!({"memory": {
        "active_bytes":100, "peak_bytes":500
    }}))
    .is_err());
}

#[test]
fn prepared_settlement_refreshes_residency_before_next_admission() {
    let ready = json!({"protocol":contracts::NATIVE_PREPARED_PROTOCOL,
        "memory":{"active_bytes":1024,"cache_bytes":0,"peak_bytes":2048}});
    let service = super::lifecycle_regressions::cpu_service(ready, Duration::from_secs(1), 1);
    service.require_request_capacity(1024).unwrap();
    let frame = json!({"protocol":contracts::NATIVE_PREPARED_PROTOCOL,
        "memory":{"active_bytes":3000,"cache_bytes":200,"peak_bytes":4000}});
    settle_memory(&service, &frame).unwrap();
    assert_eq!(service.ready_snapshot()["memory"], frame["memory"]);
    assert_eq!(
        service.require_request_capacity(1024).unwrap_err().code,
        "capacity_unavailable"
    );
    assert_eq!(service.metrics.peak_bytes.load(Ordering::Relaxed), 4000);
    settle_memory(
        &service,
        &json!({"protocol":contracts::NATIVE_PROTOCOL,"memory":{}}),
    )
    .unwrap();
    assert_eq!(service.ready_snapshot()["memory"], frame["memory"]);
}

#[test]
fn prepared_cache_policy_is_bound_to_the_selected_manifest() {
    let mut config = configuration(4096, 1024);
    config.prefill_step_size = 256;
    config.runtime = WorkerRuntime::NativeMiniCpm5Prepared {
        executable: "/worker".into(),
        metal_library: "/libmlx.dylib".into(),
        cache_limit_bytes: 2048,
    };
    config.validate_memory().unwrap();
    assert_eq!(
        config.runtime.protocol(),
        contracts::NATIVE_PREPARED_PROTOCOL
    );
    config.runtime = WorkerRuntime::NativeMiniCpm5Prepared {
        executable: "/worker".into(),
        metal_library: "/libmlx.dylib".into(),
        cache_limit_bytes: 4097,
    };
    assert!(config.validate_memory().is_err());
    config.sequence_reservation = 4096;
    assert!(config.validate_memory().is_err());
}

fn python_budget_config(memory_budget: u64) -> Config {
    let mut config = configuration(memory_budget, 1024);
    config.max_context = 40_960;
    config.safety_margin = DEFAULT_SAFETY_MARGIN_BYTES;
    config.prefill_transient = DEFAULT_PREFILL_TRANSIENT_BYTES;
    config
}

#[test]
fn context_budget_plan_allows_the_measured_context_inside_eight_gibibytes() {
    const RESIDENT: u64 = 1_192_000_000; // measured Qwen3-0.6B weights (~1137 MiB)
    let plan = context_budget_plan(
        RESIDENT,
        40_960,
        114_688,
        DEFAULT_PREFILL_TRANSIENT_BYTES,
        DEFAULT_SAFETY_MARGIN_BYTES,
        8 * 1024u64.pow(3),
    )
    .unwrap();
    assert_eq!(plan.aligned_context, 40_960);
    assert_eq!(plan.kv_bytes, 40_960 * 114_688);
    assert_eq!(
        plan.required,
        RESIDENT + 40_960 * 114_688 + DEFAULT_PREFILL_TRANSIENT_BYTES + DEFAULT_SAFETY_MARGIN_BYTES
    );
    assert!(plan.required <= plan.memory_budget);
}

#[test]
fn context_budget_plan_reports_the_largest_context_that_fits() {
    let error = context_budget_plan(
        1_192_000_000,
        40_960,
        114_688,
        DEFAULT_PREFILL_TRANSIENT_BYTES,
        DEFAULT_SAFETY_MARGIN_BYTES,
        4 * 1024u64.pow(3),
    )
    .unwrap_err();
    // fixed = 1_192_000_000 + 536_870_912 + 1_073_741_824 = 2_802_612_736
    // (4 GiB - fixed) / 114_688 = 13_012 tokens, floored to the 256 step.
    assert!(
        error.contains("maximum context that fits this budget is 12800 positions"),
        "{error}"
    );
    assert!(error.contains("bytes are required"), "{error}");
}

#[test]
fn context_budget_plan_aligns_the_context_to_the_kv_step() {
    let plan = context_budget_plan(1_000_000, 40_961, 114_688, 0, 0, 9 * 1024u64.pow(3)).unwrap();
    assert_eq!(plan.aligned_context, 41_216);
    assert_eq!(plan.kv_bytes, 41_216 * 114_688);
}

#[test]
fn context_budget_plan_rejects_an_empty_geometry() {
    assert!(context_budget_plan(1, 16, 0, 0, 0, 1024).is_err());
}

#[test]
fn python_context_budget_skips_workers_without_geometry() {
    let config = python_budget_config(8 * 1024u64.pow(3));
    let skipped = python_context_budget(
        &config,
        1_192_000_000,
        &serde_json::json!({"max_context": 40_960}),
    )
    .unwrap();
    assert!(skipped.is_none());
    let plan = python_context_budget(
        &config,
        1_192_000_000,
        &serde_json::json!({"max_context": 40_960, "kv_bytes_per_token": 114_688}),
    )
    .unwrap()
    .unwrap();
    assert_eq!(plan.aligned_context, 40_960);

    let mut over = python_budget_config(4 * 1024u64.pow(3));
    over.max_context = 40_960;
    let error = python_context_budget(
        &over,
        1_192_000_000,
        &serde_json::json!({"max_context": 40_960, "kv_bytes_per_token": 114_688}),
    )
    .unwrap_err();
    assert!(
        error.contains("maximum context that fits this budget is"),
        "{error}"
    );
}

#[test]
fn python_provider_accepts_kv_eight_and_rejects_other_widths() {
    let mut config = python_budget_config(8 * 1024u64.pow(3));
    config.kv_bits = Some(4);
    assert!(config
        .validate_memory()
        .unwrap_err()
        .contains("Only --kv-bits 8"));
    config.kv_bits = Some(8);
    config.validate_memory().unwrap();
    // A quantisation offset at or beyond the context keeps bf16 KV and must not
    // reject the launch; the forwarded offset is clamped instead.
    config.quantized_kv_start = 40_961;
    config.validate_memory().unwrap();
}

#[test]
fn non_python_providers_reject_the_python_budget_options() {
    let mut config = python_budget_config(8 * 1024u64.pow(3));
    config.runtime = WorkerRuntime::PythonMiniCpmVision;
    config.max_context = 16_384;
    config.max_output = 8_192;
    config.prefill_step_size = 256;
    config.kv_bits = Some(8);
    assert!(config
        .validate_memory()
        .unwrap_err()
        .contains("--kv-bits requires --provider python-mlx-lm"));
}

#[test]
fn python_provider_requires_budget_room_for_the_fixed_terms() {
    let mut config = python_budget_config(8 * 1024u64.pow(3));
    config.memory_budget = 1_073_741_824; // safety margin alone consumes the budget
    assert!(config
        .validate_memory()
        .unwrap_err()
        .contains("safety margin and prefill transient"));
}

#[test]
fn python_kv_options_clamp_the_offset_to_the_launch_context() {
    let mut config = python_budget_config(8 * 1024u64.pow(3));
    config.max_context = 4_096;
    assert_eq!(python_kv_options(&config), None);
    config.kv_bits = Some(8);
    config.quantized_kv_start = 5_000;
    // The default 5000 is clamped: legacy 4096-token launches stay usable.
    assert_eq!(python_kv_options(&config), Some((8, 4_096)));
    config.quantized_kv_start = 2_000;
    assert_eq!(python_kv_options(&config), Some((8, 2_000)));
}
