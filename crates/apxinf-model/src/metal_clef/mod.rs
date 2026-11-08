//! Native Clef Flash on Apple Metal: Qwen3.5 hybrid backbone (affine Q4) in
//! the stock MLX-VLM composition or the optimized GPU composition, and the
//! BF16 joint schema head. No Python model execution.
// The macro must be imported before every module that expands it: `macro_use`
// scoping is textual, and `ane` is declared first below.
#[macro_use]
pub(crate) mod signpost;
#[cfg(feature = "coreml")]
mod ane;
#[cfg(feature = "coreml")]
mod ane_admission;
mod config;
mod encode;
#[cfg(test)]
pub mod host_probe;
mod model;
mod preprocess;
#[cfg_attr(not(feature = "coreml"), allow(dead_code))]
mod trace_json;
mod vision;
mod vision_ane;
mod weights;

pub use config::{Config, MAX_INPUT_TOKENS};

use crate::decision::{
    DecisionAnswer, DecisionImage, DecisionInput, DecisionOutput, DecisionRuntime, DecisionSchema,
    DecisionScore,
};
use crate::llm_trait::LlmCapabilities;
use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::{Array, MlxBackend, MlxDType};
use apxinf_tokenizer::Tokenizer;
use model::{Composition, Network};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

pub struct ClefRunner {
    tokenizer: Tokenizer,
    /// `Rc` so the prepared whole-backbone callable can hold a weak reference
    /// back to the model it replays (see `model::install_compiled`).
    network: Rc<Network>,
    /// Placeholder token the processor writes per merged image row.
    image_token_id: u32,
    /// Prompt IDs encoded for the most recent request, for parity evidence.
    last_input_ids: Vec<u32>,
}

/// MLX architectures where the optimized GPU composition passed its quality
/// and paired latency gates: the M1 Pro (2026-10-17). `auto` uses the
/// reference composition everywhere else.
const GPU_QUALIFIED_ARCHITECTURES: &[&str] = &["applegpu_g13s"];

/// MLX architectures where the GPU composition with MLP columns on the Neural
/// Engine passed its gates; `auto` prefers it there: the M1 Pro (2026-10-20;
/// programs equal to the reference build, at-risk decisions bit-identical to
/// it, quality against stock, memory, and paired latency against the
/// reference implementation).
#[cfg(feature = "coreml")]
const GPU_ANE_COLUMNS_QUALIFIED_ARCHITECTURES: &[&str] = &["applegpu_g13s"];

/// MLX architectures where the GPU composition with MLP neurons split onto the
/// Neural Engine passed its gates; `auto` prefers it there: the M4 (2026-10-18;
/// programs equal to the reference build, at-risk decisions bit-identical to
/// it, quality against stock answers frozen on the M4, paired latency median
/// 0.997x the reference implementation).
#[cfg(feature = "coreml")]
const GPU_ANE_NEURONS_QUALIFIED_ARCHITECTURES: &[&str] = &["applegpu_g16g"];

/// MLX allocator cache limit set when Clef loads (bytes).
const CACHE_LIMIT: usize = 1 << 30;

/// Device-independent option admission, checked before the checkpoint is read.
/// Returns the requested composition; `None` is `auto`, which picks the
/// composition qualified on the device.
fn admit(options: &LoadOptions) -> Result<Option<Composition>> {
    let requested = match options.model_variant.as_deref() {
        None | Some("auto") => None,
        Some("reference") => Some(Composition::Reference),
        Some(_) => {
            return Err(Error::Contract(
                "Clef MLX supports the auto and reference variants",
            ))
        }
    };
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
    {
        return Err(Error::Contract(
            "Clef MLX loads the affine Q4 checkpoint as stored; CUDA tuning, \
             calibration, synthetic weights and auxiliary assets are unsupported",
        ));
    }
    Ok(requested)
}

/// A composition name from the diagnostic gate. `reference` and `gpu` are
/// always available; the two Core ML compositions only with that feature.
fn parse_composition(value: &str) -> Result<Composition> {
    match value {
        "reference" => Ok(Composition::Reference),
        "gpu" => Ok(Composition::Gpu),
        #[cfg(feature = "coreml")]
        "gpu-ane-columns" => Ok(Composition::GpuAneColumns),
        #[cfg(feature = "coreml")]
        "gpu-ane-neurons" => Ok(Composition::GpuAneNeurons),
        other => Err(Error::Other(format!(
            "APXINF_CLEF_COMPOSITION must be {}, not {other:?}",
            if cfg!(feature = "coreml") {
                "reference, gpu, gpu-ane-columns or gpu-ane-neurons"
            } else {
                "reference or gpu"
            }
        ))),
    }
}

/// `APXINF_CLEF_COMPOSITION` forces a composition for a measured campaign or a
/// diagnostic run. Unset (or empty) leaves `auto`, which runs an optimized
/// composition only on an architecture where it qualified. A forced
/// composition bypasses that qualification gate, so it is a scan gate and
/// never a default; the column depth stays `APXINF_CLEF_ANE_COLUMNS`.
fn forced_composition() -> Result<Option<Composition>> {
    match std::env::var("APXINF_CLEF_COMPOSITION") {
        Ok(value) if !value.is_empty() => parse_composition(&value).map(Some),
        Ok(_) => Ok(None),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::Contract(
            "APXINF_CLEF_COMPOSITION is not valid UTF-8",
        )),
    }
}

/// `auto` runs an optimized composition only on an architecture where it
/// qualified. A forced composition applies in place of `auto`; an explicit
/// [`LoadOptions::model_variant`] request wins over the environment.
fn select(
    requested: Option<Composition>,
    architecture: &str,
    forced: Option<Composition>,
) -> Composition {
    if requested.is_none() {
        if let Some(composition) = forced {
            return composition;
        }
    }
    #[cfg(feature = "coreml")]
    if requested.is_none() {
        if GPU_ANE_NEURONS_QUALIFIED_ARCHITECTURES.contains(&architecture) {
            return Composition::GpuAneNeurons;
        }
        if GPU_ANE_COLUMNS_QUALIFIED_ARCHITECTURES.contains(&architecture) {
            return Composition::GpuAneColumns;
        }
    }
    requested.unwrap_or(if GPU_QUALIFIED_ARCHITECTURES.contains(&architecture) {
        Composition::Gpu
    } else {
        Composition::Reference
    })
}

/// `APXINF_CLEF_LOAD_TIMING` prints the load stages to stderr.
pub(crate) fn load_timing() -> bool {
    std::env::var_os("APXINF_CLEF_LOAD_TIMING").is_some()
}

/// One named load stage in milliseconds (no-op unless the gate is set).
pub(crate) fn load_stage(name: &str, started: std::time::Instant) {
    if load_timing() {
        eprintln!(
            "clef load: {name} {:.3}",
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}

pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    let requested = admit(options)?;
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract("Clef requires the native MLX backend"))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    if !path.is_dir() {
        return Err(Error::Contract("Clef loads a checkpoint directory"));
    }
    let read = |name: &str| {
        std::fs::read_to_string(path.join(name))
            .map_err(|e| Error::Other(format!("cannot read Clef {name}: {e}")))
    };
    let config = Config::from_json(&read("config.json")?, &read("joint_head_config.json")?)?;
    let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))?;
    let stream = native.stream().clone();
    let architecture = stream
        .metal_caps()
        .map_or("", |caps| caps.architecture.as_str());
    let forced = forced_composition()?;
    let composition = select(requested, architecture, forced);
    if requested.is_none() && forced.is_none() && composition == Composition::Reference {
        eprintln!(
            "Clef: no optimized composition is qualified on {architecture:?}; using the reference composition"
        );
    }
    // Process-wide, as the reference engine sets it: an unbounded allocator cache grew to 4 GiB and slowed
    // requests under memory pressure on a 16 GB M1 Pro.
    apxinf_mlx::set_cache_limit(CACHE_LIMIT)?;
    let started = std::time::Instant::now();
    let backbone = weights::Backbone::load(&config, &stream, path)?;
    load_stage("backbone", started);
    let started = std::time::Instant::now();
    let head = weights::Head::load(&config, &stream, &path.join("joint_head.safetensors"))?;
    load_stage("head", started);
    let started = std::time::Instant::now();
    let vision = vision::Vision::load(&config, &stream, path)?;
    load_stage("vision", started);
    let image_token_id = config.image_token_id;
    let started = std::time::Instant::now();
    let network = Rc::new(Network::new(
        config, stream, backbone, head, vision, composition,
    )?);
    load_stage("network", started);
    let started = std::time::Instant::now();
    model::install_compiled(&network)?;
    load_stage("install", started);
    Ok(LoadedModel::Decision(Box::new(ClefRunner {
        tokenizer,
        network,
        image_token_id,
        last_input_ids: Vec::new(),
    })))
}

impl ClefRunner {
    fn decide_schema(
        &mut self,
        schema: &DecisionSchema,
        images: &[DecisionImage<'_>],
    ) -> Result<DecisionOutput> {
        // Decode and preprocess inside the request, as the MLX reference does:
        // the caller's timer covers this cost on both sides.
        let chrome = trace_json::enabled();
        let request_start = std::time::Instant::now();
        // Host signpost for the whole request; the guard closes it on every
        // exit path. Args are the request shape known at entry.
        let _sp = signpost_interval!(signpost::REQUEST, images.len());
        let ts_request = if chrome {
            trace_json::to_us(request_start)
        } else {
            None
        };
        let prepared = if images.is_empty() {
            None
        } else {
            let t = std::time::Instant::now();
            let _sp = signpost_interval!(signpost::PREPROCESS, images.len());
            let prepared = Some(preprocess::prepare(images)?);
            if let Some(ts0) = ts_request {
                trace_json::record(
                    "preprocess",
                    "host",
                    "cpu",
                    ts0,
                    t.elapsed().as_secs_f64() * 1_000_000.0,
                    serde_json::json!({"images": images.len()}),
                );
            }
            prepared
        };
        let enc_start = std::time::Instant::now();
        // The token count is only known after the prompt is built, so the
        // encode interval carries the request shape instead. The guard is
        // scoped to the block: a function-scoped `_sp` would keep the
        // interval open until `decide_schema` returns (first recording hit
        // exactly that, spanning the whole request).
        let encoded = {
            let _sp = signpost_interval!(signpost::ENCODE, images.len(), schema.questions.len());
            match &prepared {
                None => encode::encode(&self.tokenizer, schema)?,
                Some(prepared) => encode::encode_with_images(
                    &self.tokenizer,
                    schema,
                    &prepared.grids,
                    self.image_token_id,
                )?,
            }
        };
        if let Some(ts0) = trace_json::to_us(enc_start) {
            trace_json::record(
                "encode",
                "host",
                "cpu",
                ts0,
                enc_start.elapsed().as_secs_f64() * 1_000_000.0,
                serde_json::json!({"tokens": encoded.input_ids.len()}),
            );
        }
        let logits = match &prepared {
            None => self.network.logits(&encoded)?,
            Some(prepared) => {
                let pixels = Array::from_f32(
                    self.network.stream(),
                    &[prepared.rows(), preprocess::PATCH_WIDTH],
                    &prepared.pixels,
                )?;
                self.network.logits_with_vision(
                    &encoded,
                    Some((&pixels, &prepared.grids, &prepared.image_key)),
                )?
            }
        };
        self.last_input_ids.clone_from(&encoded.input_ids);
        // No eager eval here: the softmax chain below ends in
        // `to_f32_vec` -> `to_bytes`, which synchronizes the full stream and
        // materializes the same logits. Evaluating logits first would drain
        // and refill the queue for the same data (redundant ~0.1-1ms).
        let probabilities = logits
            .iter()
            .map(|l| l.cast(MlxDType::F32)?.softmax_with(-1, false))
            .collect::<Result<Vec<_>>>()?;
        let sync_start = std::time::Instant::now();
        // The GPU drain: host wall time of the `to_f32_vec` stream sync that
        // materialises the lazy graph. Signpost args = the queued logits.
        // Scoped to the drain itself (see the encode guard above).
        let values = {
            let _sp = signpost_interval!(signpost::SYNC, logits.len());
            Array::concat(&probabilities.iter().collect::<Vec<_>>(), 0)?.to_f32_vec()?
        };
        if let Some(ts0) = trace_json::to_us(sync_start) {
            // This is the `to_f32_vec` -> `to_bytes` stream-sync (mod.rs:209);
            // it sits inside W1's gap0 (head+sync+preprocess ~110ms) and is
            // the Q21 attack surface.
            trace_json::record(
                "logits_sync",
                "gpu",
                "stream",
                ts0,
                sync_start.elapsed().as_secs_f64() * 1_000_000.0,
                serde_json::json!({"logits": logits.len()}),
            );
        }
        let mut offset = 0;
        let mut answers = Vec::with_capacity(encoded.questions.len());
        for question in encoded.questions {
            let count = question.option_ids.len();
            let row = &values[offset..offset + count];
            offset += count;
            if row.iter().any(|p| !p.is_finite()) {
                return Err(Error::Other(format!(
                    "Clef produced non-finite probabilities for `{}`",
                    question.id
                )));
            }
            let mut selected_index = 0;
            for (i, &p) in row.iter().enumerate() {
                if p > row[selected_index] {
                    selected_index = i;
                }
            }
            answers.push(DecisionAnswer {
                question_id: Some(question.id),
                selected_index,
                scores: question
                    .option_ids
                    .into_iter()
                    .zip(row)
                    .map(|(code, &probability)| DecisionScore {
                        code,
                        token_id: None,
                        probability,
                    })
                    .collect(),
            });
        }
        if let Some(ts0) = ts_request {
            trace_json::record(
                "request",
                "timeline",
                "req",
                ts0,
                request_start.elapsed().as_secs_f64() * 1_000_000.0,
                serde_json::json!({"images": images.len(), "tokens": encoded.input_ids.len()}),
            );
        }
        Ok(DecisionOutput { answers })
    }
}

impl DecisionRuntime for ClefRunner {
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::VISION
    }

    fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput> {
        // Flush on every exit path (including `?` errors) so the Chrome
        // Tracing file is always valid JSON for Perfetto / chrome://tracing.
        let result = match input {
            DecisionInput::Schema(schema) => self.decide_schema(schema, &[]),
            DecisionInput::SchemaWithImages { schema, images } => {
                self.decide_schema(schema, images)
            }
            DecisionInput::Encoded { .. } => Err(Error::Other(
                "Clef encodes its own prompt; pass a schema request".into(),
            )),
        };
        trace_json::flush_request();
        result
    }

    fn encoded_input_ids(&self) -> Option<&[u32]> {
        (!self.last_input_ids.is_empty()).then_some(self.last_input_ids.as_slice())
    }

    fn code_names(&self) -> &[String] {
        &[]
    }

    fn token_ids(&self) -> &[u32] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_needs_no_device_or_checkpoint() {
        assert_eq!(admit(&LoadOptions::default()).unwrap(), None);
        for (variant, composition) in [("auto", None), ("reference", Some(Composition::Reference))]
        {
            let options = LoadOptions {
                model_variant: Some(variant.into()),
                ..LoadOptions::default()
            };
            assert_eq!(admit(&options).unwrap(), composition, "{variant}");
        }
        for options in [
            LoadOptions {
                model_variant: Some("bf16".into()),
                ..LoadOptions::default()
            },
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
        ] {
            assert!(admit(&options).is_err());
        }
    }

    #[test]
    fn auto_runs_each_composition_only_on_its_qualified_architectures() {
        #[cfg(feature = "coreml")]
        assert_eq!(
            select(None, "applegpu_g13s", None),
            Composition::GpuAneColumns
        );
        #[cfg(not(feature = "coreml"))]
        assert_eq!(select(None, "applegpu_g13s", None), Composition::Gpu);
        #[cfg(feature = "coreml")]
        assert_eq!(
            select(None, "applegpu_g16g", None),
            Composition::GpuAneNeurons
        );
        #[cfg(not(feature = "coreml"))]
        assert_eq!(select(None, "applegpu_g16g", None), Composition::Reference);
        for other in ["applegpu_g14g", "applegpu_g13g", ""] {
            assert_eq!(
                select(None, other, None),
                Composition::Reference,
                "{other}"
            );
        }
        for architecture in ["applegpu_g13s", "applegpu_g16g"] {
            assert_eq!(
                select(Some(Composition::Reference), architecture, None),
                Composition::Reference
            );
        }
        // The gate replaces auto, while an explicit variant request wins.
        assert_eq!(
            select(None, "applegpu_g16g", Some(Composition::GpuAneColumns)),
            Composition::GpuAneColumns
        );
        assert_eq!(
            select(None, "applegpu_g13s", Some(Composition::Gpu)),
            Composition::Gpu
        );
        assert_eq!(
            select(
                Some(Composition::Reference),
                "applegpu_g16g",
                Some(Composition::Gpu)
            ),
            Composition::Reference
        );
    }

    #[test]
    fn composition_gate_parses_only_named_compositions() {
        let mut cases = vec![
            ("reference", Composition::Reference),
            ("gpu", Composition::Gpu),
        ];
        #[cfg(feature = "coreml")]
        cases.extend([
            ("gpu-ane-columns", Composition::GpuAneColumns),
            ("gpu-ane-neurons", Composition::GpuAneNeurons),
        ]);
        for (value, composition) in cases {
            assert_eq!(parse_composition(value).unwrap(), composition, "{value}");
        }
        for value in ["", "auto", "columns", "GPU", "gpu-ane", "1", "gpu-ane-column"] {
            assert!(parse_composition(value).is_err(), "accepted {value:?}");
        }
    }

    #[test]
    fn forced_composition_reads_the_gate_and_absent_means_auto() {
        // `APXINF_CLEF_COMPOSITION` is process-global: this is the only test
        // that sets it, and it restores the prior value.
        let prior = std::env::var_os("APXINF_CLEF_COMPOSITION");
        std::env::remove_var("APXINF_CLEF_COMPOSITION");
        assert_eq!(forced_composition().unwrap(), None);
        std::env::set_var("APXINF_CLEF_COMPOSITION", "");
        assert_eq!(forced_composition().unwrap(), None);
        std::env::set_var("APXINF_CLEF_COMPOSITION", "gpu");
        assert_eq!(forced_composition().unwrap(), Some(Composition::Gpu));
        std::env::set_var("APXINF_CLEF_COMPOSITION", "auto");
        assert!(forced_composition().is_err());
        match prior {
            Some(value) => std::env::set_var("APXINF_CLEF_COMPOSITION", value),
            None => std::env::remove_var("APXINF_CLEF_COMPOSITION"),
        }
    }
}
