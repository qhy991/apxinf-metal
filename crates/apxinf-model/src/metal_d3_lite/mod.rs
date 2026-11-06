//! Native d3-lite on Apple Metal: the checkpoint's text decision path as an
//! ApxInf-owned MLX composition.
//!
//! This is the Metal port of the d3 family (registry name `metal_d3`, explicit
//! `--model-name metal_d3_lite` also accepted). It is deliberately independent
//! of the CUDA-only `d3` directory at `src/d3/`: ports are self-contained and copy
//! the checkpoint contract they need (see `doc/model-layer-architecture.md`,
//! "Where backend-native model code lives").
//!
//! Scope of this module: the **text** path — `language_model.*` (24 layers, 18
//! GatedDeltaNet + 6 non-causal full attention) and the FP32 255-code readout,
//! one compiled forward per request, no decode and no KV cache. The `visual.*`
//! tower (`d3::vision`'s counterpart) is parsed for package identity but is
//! never read or executed here; multimodal alignment is P1b.

mod config;
mod model;
mod weights;

pub use config::{Config, Geometry, VisionContract};
pub use model::Model;
pub use weights::Weights;

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::{MlxBackend, Stream};

use crate::decision::{
    CodeDecisionRuntime, DecisionInput, DecisionModel, DecisionOutput, DecisionScore,
};
use crate::llm_trait::LlmCapabilities;
use crate::{LoadOptions, LoadedModel, ModelPrecision};

/// Device-independent option admission, checked before the checkpoint is read.
/// One BF16 implementation: the checkpoint's own storage, no CUDA tuning,
/// calibration, synthetic weights, auxiliary assets or CUDA-graph profile
/// knobs (those belong to the CUDA `d3` loader).
fn admit(options: &LoadOptions) -> Result<()> {
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto" | "bf16" | "bf16-compiled")
    ) {
        return Err(Error::Contract(
            "d3-lite Metal admits the auto, bf16 and bf16-compiled variants",
        ));
    }
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || options.text_weight_dtype.is_some()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.autotune
        || options.uniform_fp8_scale.is_some()
        || !options.assets.is_empty()
        || options.d3_text_profile_limit.is_some()
        || options.d3_text_profile_token_budget.is_some()
    {
        return Err(Error::Contract(
            "d3-lite Metal loads the BF16 checkpoint as stored and compiles whole-request \
             profiles itself; CUDA tuning, calibration, synthetic weights, auxiliary assets \
             and d3 CUDA-graph profile limits are unsupported",
        ));
    }
    Ok(())
}

/// `registry::ModelFactory` entry point. Resolved by `AutoModel` for
/// `Device::Metal` under the names `metal_d3` (checkpoint detection) and
/// `metal_d3_lite` (explicit selection).
pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    admit(options)?;
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract(
            "d3-lite Metal requires the native MLX backend",
        ))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    if !path.is_dir() {
        return Err(Error::Contract("d3-lite loads a checkpoint directory"));
    }
    let config = Config::from_dir(path)?;
    let stream: Stream = native.stream().clone();
    let weights = Weights::load(config.geometry, &stream, path)?;
    let model = Model::new(&config, stream, weights)?;
    Ok(LoadedModel::Decision(DecisionModel::Code(Box::new(
        D3LiteRunner {
            codes: config.codes.clone(),
            token_ids: config.token_ids.clone(),
            config,
            model,
        },
    ))))
}

/// Encoded-prompt decision runtime: the caller (CLI / bench) owns tokenization
/// and the chat template; this runner scores the leading package codes of one
/// complete prompt.
pub struct D3LiteRunner {
    codes: Vec<String>,
    token_ids: Vec<u32>,
    config: Config,
    model: Model,
}

impl D3LiteRunner {
    /// Profiles prepared on the compiled whole-request callable, in order.
    pub fn prepared_profiles(&self) -> Vec<(usize, usize)> {
        self.model.prepared_profiles()
    }
}

impl CodeDecisionRuntime for D3LiteRunner {
    fn capabilities(&self) -> LlmCapabilities {
        // Text-only until P1b wires `visual.*` through the same request graph.
        LlmCapabilities::TEXT_ONLY
    }

    fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput> {
        if input.image.is_some() {
            return Err(Error::Other(
                "d3-lite Metal text path does not read images; the vision tower is P1b".into(),
            ));
        }
        let count = input.candidate_count;
        let probabilities = self.model.probabilities(input.token_ids, count)?;
        if probabilities.len() != count {
            return Err(Error::Other(format!(
                "d3-lite decision returned {} probabilities for {count} candidates",
                probabilities.len()
            )));
        }
        if probabilities.iter().any(|value| !value.is_finite()) {
            return Err(Error::Other(
                "d3-lite readout produced non-finite probabilities".into(),
            ));
        }
        // First maximum wins, matching the package reference and the CUDA path.
        let mut selected_index = 0usize;
        for (index, value) in probabilities.iter().enumerate() {
            if *value > probabilities[selected_index] {
                selected_index = index;
            }
        }
        let scores = self.codes[..count]
            .iter()
            .zip(&self.token_ids[..count])
            .zip(&probabilities)
            .map(|((code, &token_id), &probability)| DecisionScore {
                code: code.clone(),
                token_id,
                probability,
            })
            .collect();
        Ok(DecisionOutput {
            selected_index,
            scores,
        })
    }

    fn code_names(&self) -> &[String] {
        &self.codes
    }

    fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_needs_no_device_or_checkpoint() {
        assert!(admit(&LoadOptions::default()).is_ok());
        for variant in ["auto", "bf16", "bf16-compiled"] {
            assert!(
                admit(&LoadOptions {
                    model_variant: Some(variant.into()),
                    ..LoadOptions::default()
                })
                .is_ok(),
                "{variant}"
            );
        }
        for options in [
            LoadOptions {
                model_variant: Some("fast".into()),
                ..LoadOptions::default()
            },
            LoadOptions {
                autotune: true,
                ..LoadOptions::default()
            },
            LoadOptions {
                assets: [("draft".to_owned(), "draft".into())].into(),
                ..LoadOptions::default()
            },
            LoadOptions {
                d3_text_profile_limit: Some(1),
                ..LoadOptions::default()
            },
            LoadOptions {
                precision: ModelPrecision::Fp8,
                ..LoadOptions::default()
            },
        ] {
            assert!(admit(&options).is_err(), "{options:?}");
        }
    }
}
