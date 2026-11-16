//! Real-checkpoint lifecycle checks, separate from numerical qualification.
use super::{config::Config, weights::Weights, MiniCpm5, Variant};
use crate::{LlmInput, LlmTrait, TextPreparationState};
use apxinf_core::{Error, Result, Tensor};
use apxinf_mlx::{clear_cache, memory_stats, MlxBackend};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{env, path::PathBuf};

#[derive(Deserialize)]
struct ReplayInputs {
    cases: Vec<ReplayCase>,
}

#[derive(Deserialize)]
struct ReplayCase {
    id: String,
    prompt_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LogitsDigest {
    bytes: usize,
    all_rows: [u8; 32],
    last_row: [u8; 32],
}

/// The download is intentional test observation, never a runtime model helper.
fn digest(backend: &MlxBackend, output: &Tensor) -> Result<LogitsDigest> {
    let bytes = backend.array(output)?.to_bytes()?;
    let row_bytes = output.shape().dims()[1] * output.dtype().size_in_bytes();
    assert!(bytes.len() >= row_bytes);
    Ok(LogitsDigest {
        bytes: bytes.len(),
        all_rows: Sha256::digest(&bytes).into(),
        last_row: Sha256::digest(&bytes[bytes.len() - row_bytes..]).into(),
    })
}

fn full_forward(model: &mut MiniCpm5, case: &ReplayCase) -> Result<Tensor> {
    model.reset();
    model.prepare(case.prompt_token_ids.len(), 1)?;
    assert_eq!(
        model.preparation_status().state,
        TextPreparationState::Ready
    );
    let output = model.forward(&case.prompt_token_ids, 0)?;
    assert_eq!(
        output.shape().dims(),
        [case.prompt_token_ids.len(), model.vocab_size()],
        "public forward must retain every logit row for {}",
        case.id
    );
    Ok(output)
}

#[test]
#[ignore = "requires official MiniCPM checkpoint, isolated process and exclusive shared Metal locks"]
fn real_checkpoint_long_chunk_and_bucket_boundary() -> Result<()> {
    let checkpoint = PathBuf::from(
        env::var_os("APXINF_MINICPM_CHECKPOINT")
            .ok_or(Error::Contract("set APXINF_MINICPM_CHECKPOINT"))?,
    );
    let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let backend = MlxBackend::new(0)?;
    let (map, _) =
        apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
    let weights = Weights::load(&config, backend.stream(), map)?;
    let mut model = MiniCpm5::new(config, weights, backend.clone(), Variant::Compiled)?;
    model.prepare(3, 1)?;
    let short = model.prefill(LlmInput::text(&[1, 7, 3]))?;
    let expected = digest(&backend, &short)?;
    model.reset();
    model.prepare(4351, 2)?;
    assert_eq!(model.preparation_status().kv_capacity, Some(4608));
    model.prefill(LlmInput::text(&vec![7; 4351]))?;
    let traces = backend.stream().counters().trace_callbacks;
    model.forward(&[2], 4351)?;
    model.forward(&[3], 4352)?;
    assert_eq!(backend.stream().counters().trace_callbacks, traces);
    assert!(model.forward(&[3], 4608).is_err());
    model.reset();
    model.prepare(3, 1)?;
    assert_eq!(model.preparation_status().kv_capacity, Some(256));
    let repeated = model.prefill(LlmInput::text(&[1, 7, 3]))?;
    assert_eq!(digest(&backend, &repeated)?, expected);
    assert_eq!(digest(&backend, &short)?, expected);
    Ok(())
}

/// Run only this test in an isolated process under the shared Metal lock:
/// `minicpm5::tests::real_checkpoint_full_forward_retention_and_release
/// --exact --ignored --test-threads=1 --nocapture`.
/// APXINF_MINICPM_CHECKPOINT names the checkpoint directory and
/// APXINF_DSPARK_REPLAY_INPUT names a source-bound JSON with two real prompts.
/// This checks lifecycle/determinism, not agreement with a numerical oracle.
#[test]
#[ignore = "requires official MiniCPM checkpoint, real replay inputs, isolated process and shared Metal lock"]
fn real_checkpoint_full_forward_retention_and_release() -> Result<()> {
    let path = |name| {
        env::var_os(name)
            .map(PathBuf::from)
            .ok_or_else(|| Error::Other(format!("set {name} for this ignored test")))
    };
    let checkpoint = path("APXINF_MINICPM_CHECKPOINT")?;
    let replay = path("APXINF_DSPARK_REPLAY_INPUT")?;
    let inputs: ReplayInputs = serde_json::from_slice(&std::fs::read(replay)?)
        .map_err(|e| Error::Other(format!("MiniCPM lifecycle replay inputs: {e}")))?;
    if inputs.cases.len() < 2
        || inputs.cases[..2]
            .iter()
            .any(|case| case.id.is_empty() || case.prompt_token_ids.is_empty())
        || inputs.cases[0].prompt_token_ids == inputs.cases[1].prompt_token_ids
    {
        return Err(Error::Contract(
            "MiniCPM lifecycle test needs two distinct nonempty real prompts",
        ));
    }
    let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let backend = MlxBackend::new(0)?;
    backend.stream().synchronize()?;
    clear_cache()?;
    let baseline = memory_stats()?;
    let mut expected: [Option<LogitsDigest>; 2] = [None, None];
    for cycle in 0..2 {
        let (map, _) =
            apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
        let weights = Weights::load(&config, backend.stream(), map)?;
        // The observer backend owns only the stream, never weights or a model.
        let mut model = MiniCpm5::new(config.clone(), weights, backend.clone(), Variant::Compiled)?;
        assert_eq!(
            model.preparation_status().state,
            TextPreparationState::Unprepared
        );
        // Reverse order on the second load: both prompts get a fresh-model run.
        let first = cycle;
        let changed = 1 - cycle;
        let retained = full_forward(&mut model, &inputs.cases[first])?;
        let original = digest(&backend, &retained)?;
        if let Some(previous) = &expected[first] {
            assert_eq!(&original, previous, "fresh-model logits changed");
        }
        expected[first] = Some(original.clone());

        let next = full_forward(&mut model, &inputs.cases[changed])?;
        let changed_digest = digest(&backend, &next)?;
        assert_ne!(
            original.last_row, changed_digest.last_row,
            "changed real input did not change final logits"
        );
        if let Some(previous) = &expected[changed] {
            assert_eq!(
                &changed_digest, previous,
                "reset leaked prior request state"
            );
        }
        expected[changed] = Some(changed_digest);
        drop(next);
        assert_eq!(
            digest(&backend, &retained)?,
            original,
            "subsequent forward mutated retained output"
        );

        let repeated = full_forward(&mut model, &inputs.cases[first])?;
        assert_eq!(
            digest(&backend, &repeated)?,
            original,
            "reset/reprepare changed the original logits"
        );
        drop(repeated);
        model.reset();
        assert_eq!(
            digest(&backend, &retained)?,
            original,
            "reset mutated retained output"
        );
        drop(model);
        backend.stream().synchronize()?;
        assert_eq!(
            digest(&backend, &retained)?,
            original,
            "model drop invalidated retained output"
        );
        drop(retained);
        backend.stream().synchronize()?;
        clear_cache()?;
        let released = memory_stats()?;
        assert_eq!(
            released.active_bytes, baseline.active_bytes,
            "model/output retained live MLX allocations in cycle {cycle}: {released:?}"
        );
        assert_eq!(released.cache_bytes, 0);
        eprintln!("cycle={cycle} first_case={} full_logits={original:?} baseline={baseline:?} released={released:?}", inputs.cases[first].id);
    }
    Ok(())
}

#[test]
#[ignore = "requires official MiniCPM checkpoint and exclusive shared Metal locks"]
fn real_checkpoint_prepared_code_reuses_with_fresh_request_state() -> Result<()> {
    let checkpoint = PathBuf::from(env::var_os("APXINF_MINICPM_CHECKPOINT")
        .ok_or(Error::Contract("set APXINF_MINICPM_CHECKPOINT"))?);
    let tokenizer = apxinf_tokenizer::Tokenizer::from_file(checkpoint.join("tokenizer.json"))?;
    let template_options = serde_json::Map::from_iter([("enable_thinking".into(), serde_json::Value::Bool(false))]);
    let mut cases = Vec::new();
    for (id, question) in [
        ("cache-en", "Explain why a cache can reduce repeated work. Answer in two sentences."),
        ("cache-zh", "请用两句话说明为什么缓存可以减少重复计算。"),
    ] {
        let prompt = tokenizer.apply_chat_template_with_options(&[apxinf_tokenizer::ChatMessage::user(question)], &template_options)?;
        cases.push(ReplayCase { id: id.into(), prompt_token_ids: tokenizer.encode(&prompt)? });
    }
    let inputs = ReplayInputs { cases };
    let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    apxinf_mlx::set_memory_limit(10 * 1024 * 1024 * 1024)?;
    apxinf_mlx::set_cache_limit(2 * 1024 * 1024 * 1024)?;
    let backend = MlxBackend::new(0)?;
    let (map, _) = apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
    let weights = Weights::load(&config, backend.stream(), map)?;
    let mut model = MiniCpm5::new(config, weights, backend.clone(), Variant::Compiled)?;
    let options = crate::GenerationOptions::greedy(8, None);
    let run = |model: &mut MiniCpm5, index: usize| {
        model.generate_streaming_with_options(crate::GenerationRequest {
            input: LlmInput::text(&inputs.cases[index].prompt_token_ids), options: &options,
        }, |_| {}).map(|output| output.token_ids())
    };
    let first = run(&mut model, 0)?;
    model.release_request_state_retaining_preparation()?;
    assert_eq!(model.preparation_status().state, TextPreparationState::Unprepared);
    let traces = backend.stream().counters().trace_callbacks;
    assert_eq!(run(&mut model, 0)?, first);
    assert_eq!(backend.stream().counters().trace_callbacks, traces,
        "same-profile request traced after prepared-code retention");
    model.release_request_state_retaining_preparation()?;
    let changed = run(&mut model, 1)?;
    model.release_request_state()?;
    assert_eq!(run(&mut model, 1)?, changed, "retained preparation leaked prior request state");
    model.release_request_state()?;
    assert_eq!(run(&mut model, 0)?, first, "changed profile altered the original output");
    let old_capacity = model.preparation_status().kv_capacity.unwrap();
    model.reset();
    model.prepare(old_capacity + 1, 8)?;
    assert!(model.preparation_status().kv_capacity.unwrap() > old_capacity);
    assert_eq!(run(&mut model, 1)?, changed, "ordinary capacity change leaked prior state");
    Ok(())
}

/// The M1 exit criterion for the ANE MLP: on real checkpoint weights, the
/// Neural Engine program must reproduce the engine's own GPU MLP to the
/// EngineTailor admission band (`corr >= 0.99`, relative RMSE `<= 0.02`)
/// before any schedule may use it. Prints both metrics so the receipt can be
/// recorded even when the assertion passes.
#[cfg(feature = "coreml")]
#[test]
#[ignore = "requires the MiniCPM5 checkpoint, ANE plan inspection, isolated process and shared Metal locks"]
fn real_checkpoint_ane_mlp_matches_the_gpu_mlp() -> Result<()> {
    use super::ane::{program, AneMlp, HIDDEN, INTERMEDIATE, TILE};
    use super::model::{ane_cache_dir, transposed_f16};
    use apxinf_mlx::{Array, MlxDType};

    let checkpoint = PathBuf::from(
        env::var_os("APXINF_MINICPM_CHECKPOINT")
            .ok_or(Error::Contract("set APXINF_MINICPM_CHECKPOINT"))?,
    );
    let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
    let backend = MlxBackend::new(0)?;
    let (map, _) =
        apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
    let weights = Weights::load(&config, backend.stream(), map)?;
    let w = &weights.layers[0];

    let mut values = vec![0f32; TILE * HIDDEN];
    for (index, value) in values.iter_mut().enumerate() {
        *value = (((index * 37) % 199) as f32 - 99.0) / 99.0;
    }
    let x = Array::from_f32(backend.stream(), &[1, TILE, HIDDEN], &values)?
        .cast(MlxDType::BF16)?;
    let gate = x.matmul(&w.gate)?;
    let up = x.matmul(&w.up)?;
    let reference = super::math::swiglu(&gate, &up)?.matmul(&w.down)?.to_f32_vec()?;

    let gate_h = transposed_f16(&w.gate, HIDDEN, INTERMEDIATE)?;
    let up_h = transposed_f16(&w.up, HIDDEN, INTERMEDIATE)?;
    let down_h = transposed_f16(&w.down, INTERMEDIATE, HIDDEN)?;
    let package = program(&gate_h, &up_h, &down_h)?;
    let cache = apxinf_coreml::ProgramCache::new(ane_cache_dir());
    let ane = AneMlp::load_package(&cache, &package)?;
    ane.warm()?;
    let packed = AneMlp::tiles(&x)?;
    let measured = ane
        .predict(backend.stream(), &packed, TILE)?
        .to_f32_vec()?;

    let count = reference.len() as f64;
    let mean_reference = reference.iter().map(|&v| v as f64).sum::<f64>() / count;
    let mean_measured = measured.iter().map(|&v| v as f64).sum::<f64>() / count;
    let mut covariance = 0f64;
    let mut variance_reference = 0f64;
    let mut variance_measured = 0f64;
    let mut squared_error = 0f64;
    let mut squared_reference = 0f64;
    let mut maximum = 0f64;
    for (&a, &b) in reference.iter().zip(&measured) {
        let (a, b) = (a as f64, b as f64);
        covariance += (a - mean_reference) * (b - mean_measured);
        variance_reference += (a - mean_reference).powi(2);
        variance_measured += (b - mean_measured).powi(2);
        squared_error += (a - b).powi(2);
        squared_reference += a * a;
        maximum = maximum.max((a - b).abs());
    }
    let correlation = covariance / (variance_reference.sqrt() * variance_measured.sqrt());
    let relative_rmse = (squared_error / squared_reference).sqrt();
    println!(
        "layer 0 ANE vs GPU MLP: corr={correlation:.6} rel_rmse={relative_rmse:.6} max_abs={maximum:.6}"
    );
    assert!(
        correlation >= 0.99 && relative_rmse <= 0.02,
        "ANE MLP outside the admission band: corr {correlation} rel_rmse {relative_rmse}"
    );
    Ok(())
}
