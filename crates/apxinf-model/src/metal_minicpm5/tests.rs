//! Attention boundary and real-checkpoint lifecycle checks.
use super::{config::Config, weights::Weights, MiniCpm5, Variant};
use crate::{LlmInput, LlmTrait, TextPreparationState};
use apxinf_core::{Error, Result, Tensor};
use apxinf_mlx::{clear_cache, memory_stats, Array, MlxBackend, MlxDType};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{env, path::PathBuf};

#[test]
#[ignore = "requires exclusive shared Metal locks"]
fn builtin_causal_matches_exact_prefix_mask() -> Result<()> {
    let backend = MlxBackend::new(0)?;
    let stream = backend.stream();
    let fixture = |heads: usize, length: usize, salt: usize| -> Result<Array> {
        let data = (0..heads * length * 128)
            .map(|index| {
                let head = index / (length * 128);
                let row = index / 128 % length;
                let channel = index % 128;
                ((head * 17 + row * 13 + channel * 7 + salt) % 61) as f32 / 32. - 30. / 32.
                    + row as f32 / 8.
            })
            .collect::<Vec<_>>();
        Array::from_f32(stream, &[1, heads, length, 128], &data)?.cast(MlxDType::BF16)
    };
    for (position, rows) in [(0, 4), (7, 4)] {
        let end = position + rows;
        let q = fixture(16, rows, 3)?;
        let k = fixture(2, end, 11)?;
        let v = fixture(2, end, 23)?;
        let k_positions =
            Array::arange(stream, 0., end as f32, 1., MlxDType::I32)?.reshape(&[1, end])?;
        let q_positions = Array::arange(stream, position as f32, end as f32, 1., MlxDType::I32)?
            .reshape(&[rows, 1])?;
        let mask = k_positions
            .less_equal(&q_positions)?
            .reshape(&[1, 1, rows, end])?;
        let expected = q.sdpa(&k, &v, 128f32.powf(-0.5), false, Some(&mask))?;
        let actual = q.sdpa(&k, &v, 128f32.powf(-0.5), true, None)?;
        assert_eq!(actual.shape(), &[1, 16, rows, 128]);
        assert_eq!(actual.dtype(), MlxDType::BF16);
        let expected = expected.to_f32_vec()?;
        let actual = actual.to_f32_vec()?;
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = 0.015625 + 0.0078125 * expected.abs();
            assert!(
                actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs() <= tolerance,
                "position={position}, rows={rows}, element={index}, actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
        if position > 0 {
            let upper_left_rows =
                Array::arange(stream, 0., rows as f32, 1., MlxDType::I32)?.reshape(&[rows, 1])?;
            let upper_left_mask = k_positions
                .less_equal(&upper_left_rows)?
                .reshape(&[1, 1, rows, end])?;
            let upper_left = q
                .sdpa(&k, &v, 128f32.powf(-0.5), false, Some(&upper_left_mask))?
                .to_f32_vec()?;
            let max_difference = expected
                .iter()
                .zip(upper_left)
                .map(|(expected, upper_left)| (expected - upper_left).abs())
                .fold(0f32, f32::max);
            assert!(
                max_difference > 0.0625,
                "fixture must distinguish lower-right and upper-left causal alignment"
            );
        }
    }
    Ok(())
}

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
#[ignore = "requires official MiniCPM checkpoint and exclusive shared Metal locks"]
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
