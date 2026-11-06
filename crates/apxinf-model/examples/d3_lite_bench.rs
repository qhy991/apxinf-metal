//! Reusable d3-lite parity and latency bench over the Metal port.
//!
//! Reads a workload of pre-tokenized cases (the reference harness's
//! `workload.json` shape: `cases[].{id,bucket,timing,token_ids,option_count}`),
//! runs the native `AutoModel` decision path, and writes two JSON files:
//!
//! - `--records` — a gate-compatible receipt (`cases[].{id,bucket,count,
//!   probabilities,hex,decision,margin}`, plus `repeat` and `byte_identical`),
//!   so `EngineTailor`'s `gate.py` can consume it directly as `--candidate`;
//! - `--summary` — parity against an optional `--reference` receipt (decision
//!   agreement, max |Δp|, mean KL), per-round latency with `vm_stat` swap
//!   deltas, and per-bucket aggregates.
//!
//! Usage:
//! `cargo run -p apxinf-model --features mlx --example d3_lite_bench -- \
//!   --model-dir MODEL --workload WORKLOAD.json --set gate|timing \
//!   --bucket all|150|400|1000|2400 [--reference probs.json] \
//!   [--warmup N] [--repeat N] [--rounds N] \
//!   --records out/probs.json --summary out/summary.json`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use apxinf_core::{Device, Tensor};
use apxinf_model::metal_d3_lite;
use apxinf_model::{AutoModel, DecisionInput, ImageInput, LoadOptions};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
struct Workload {
    cases: Vec<WorkloadCase>,
}

#[derive(Debug, Deserialize)]
struct WorkloadCase {
    id: String,
    #[serde(default)]
    bucket: usize,
    #[serde(default)]
    timing: bool,
    /// Image path; required for the `image` set, absent for text cases.
    #[serde(default)]
    image: Option<String>,
    token_ids: Vec<u32>,
    option_count: usize,
    #[serde(default)]
    tokens: usize,
}

/// The reference's image receipt (`cases.json`): probabilities plus the
/// preprocessed tensors the element-wise comparison reads.
#[derive(Debug, Deserialize)]
struct ImageReference {
    cases: Vec<ImageReferenceCase>,
}

#[derive(Debug, Deserialize)]
struct ImageReferenceCase {
    id: String,
    probabilities: Vec<f32>,
    #[serde(default)]
    grid_thw: Option<Vec<u32>>,
    #[serde(default)]
    arrays: Option<ImageReferenceArrays>,
}

#[derive(Debug, Deserialize)]
struct ImageReferenceArrays {
    #[serde(default)]
    pixel_values: Option<ImageReferenceArray>,
}

#[derive(Debug, Deserialize)]
struct ImageReferenceArray {
    /// Path relative to the reference JSON's directory.
    file: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct ReferenceReceipt {
    cases: Vec<ReferenceCase>,
}

#[derive(Debug, Deserialize)]
struct ReferenceCase {
    id: String,
    probabilities: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct CaseRecord {
    id: String,
    bucket: usize,
    tokens: usize,
    count: usize,
    probabilities: Vec<f32>,
    hex: String,
    decision: usize,
    margin: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    latency_ms: Option<Vec<f64>>,
    /// Host-side PNG decode + checkpoint-fixed preprocess, before `decide`.
    #[serde(skip_serializing_if = "Option::is_none")]
    input_prep_ms: Option<f64>,
    /// The same prompt's text stack without the vision tower (image-arm
    /// attribution; not an optimization comparison).
    #[serde(skip_serializing_if = "Option::is_none")]
    text_latency_ms: Option<Vec<f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    swapouts: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    swapins: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_swapouts: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_swapins: Option<Vec<u64>>,
}

#[derive(Debug)]
struct Options {
    model_dir: PathBuf,
    model_name: Option<String>,
    workload: PathBuf,
    set: String,
    bucket: String,
    reference: Option<PathBuf>,
    warmup: usize,
    repeat: usize,
    rounds: usize,
    records: PathBuf,
    summary: PathBuf,
}

fn usage() -> &'static str {
    "usage: d3_lite_bench --model-dir DIR --set gate|timing|image --workload FILE \
     [--images FILE (sets --set image and --workload)] [--model-name NAME] \
     [--bucket all|150|400|1000|2400] [--reference FILE] \
     [--warmup N] [--repeat N] [--rounds N] --records FILE --summary FILE"
}

fn parse_options() -> Result<Options, String> {
    let mut model_dir = None;
    let mut model_name = None;
    let mut workload = None;
    let mut set = None;
    let mut bucket = "all".to_owned();
    let mut reference = None;
    let mut warmup = 1usize;
    let mut repeat = 1usize;
    let mut rounds = 1usize;
    let mut records = None;
    let mut summary = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(value()?)),
            "--model-name" => model_name = Some(value()?),
            "--workload" => workload = Some(PathBuf::from(value()?)),
            // The image-suite entry point: implies `--set image`.
            "--images" => {
                set = Some("image".to_owned());
                workload = Some(PathBuf::from(value()?));
            }
            "--set" => set = Some(value()?),
            "--bucket" => bucket = value()?,
            "--reference" => reference = Some(PathBuf::from(value()?)),
            "--warmup" => {
                warmup = value()?
                    .parse()
                    .map_err(|_| "--warmup must be an integer")?
            }
            "--repeat" => {
                repeat = value()?
                    .parse()
                    .map_err(|_| "--repeat must be an integer")?
            }
            "--rounds" => {
                rounds = value()?
                    .parse()
                    .map_err(|_| "--rounds must be an integer")?
            }
            "--records" => records = Some(PathBuf::from(value()?)),
            "--summary" => summary = Some(PathBuf::from(value()?)),
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unknown flag {other:?}\n{}", usage())),
        }
    }
    let set = set.ok_or_else(|| format!("--set is required\n{}", usage()))?;
    if set != "gate" && set != "timing" && set != "image" {
        return Err(format!("--set must be gate, timing or image, not {set}"));
    }
    if warmup == 0 || repeat == 0 || rounds == 0 {
        return Err("--warmup, --repeat and --rounds must be positive".into());
    }
    Ok(Options {
        model_dir: model_dir.ok_or_else(|| format!("--model-dir is required\n{}", usage()))?,
        model_name,
        workload: workload.ok_or_else(|| format!("--workload is required\n{}", usage()))?,
        set,
        bucket,
        reference,
        warmup,
        repeat,
        rounds,
        records: records.ok_or_else(|| format!("--records is required\n{}", usage()))?,
        summary: summary.ok_or_else(|| format!("--summary is required\n{}", usage()))?,
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(
    path: &std::path::Path,
) -> Result<T, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

fn write_json(
    path: &std::path::Path,
    value: &impl Serialize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

/// `vm_stat` Swapins/Swapouts, in pages. A run whose Swapouts delta is
/// non-zero is swap-contaminated and must not enter a latency gate.
fn vm_stat_metric(name: &str) -> Option<u64> {
    let output = Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.trim() != name {
            return None;
        }
        value.trim().trim_end_matches('.').parse().ok()
    })
}

fn swapouts() -> u64 {
    vm_stat_metric("Swapouts").unwrap_or(0)
}

fn swapins() -> u64 {
    vm_stat_metric("Swapins").unwrap_or(0)
}

fn hex_f32(values: &[f32]) -> String {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn decision_of(probabilities: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, value) in probabilities.iter().enumerate() {
        if *value > probabilities[best] {
            best = index;
        }
    }
    best
}

fn margin_of(probabilities: &[f32]) -> f32 {
    if probabilities.len() < 2 {
        return 0.0;
    }
    let mut order = (0..probabilities.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        probabilities[right]
            .partial_cmp(&probabilities[left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    probabilities[order[0]] - probabilities[order[1]]
}

fn max_delta(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0.0f64, f64::max)
}

/// `KL(p || q)` with `0 log 0 = 0`; `q = 0` where `p > 0` is infinite.
fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    let mut total = 0.0;
    for (&a, &b) in p.iter().zip(q) {
        if a <= 0.0 {
            continue;
        }
        if b <= 0.0 {
            return f64::INFINITY;
        }
        total += a as f64 * ((a as f64) / (b as f64)).ln();
    }
    total
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let index = ((sorted.len() as f64 - 1.0) * fraction).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn timing_stats(values: &[f64]) -> serde_json::Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = if sorted.is_empty() {
        f64::NAN
    } else {
        sorted.iter().sum::<f64>() / sorted.len() as f64
    };
    json!({
        "samples": sorted.len(),
        "min_ms": sorted.first().copied().unwrap_or(f64::NAN),
        "median_ms": percentile(&sorted, 0.5),
        "mean_ms": mean,
        "p95_ms": percentile(&sorted, 0.95),
        "max_ms": sorted.last().copied().unwrap_or(f64::NAN),
    })
}

/// Image-container decoding is application-owned; the model runtime takes the
/// decoded RGB bytes and performs the checkpoint-fixed resize/normalize/patchify.
fn decode_rgb(path: &Path) -> Result<(Vec<u8>, usize, usize), String> {
    let decoded =
        image::open(path).map_err(|error| format!("decode {}: {error}", path.display()))?;
    let rgb = decoded.to_rgb8();
    let (width, height) = rgb.dimensions();
    Ok((rgb.into_raw(), width as usize, height as usize))
}

fn read_f32_file(base: &Path, file: &str) -> Result<Vec<f32>, String> {
    let path = base.join(file);
    let bytes =
        std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{} is not a whole number of f32 values",
            path.display()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn parity_summary(rows: &[serde_json::Value]) -> serde_json::Value {
    let compared = rows
        .iter()
        .filter(|row| row.get("status") == Some(&json!("compared")))
        .collect::<Vec<_>>();
    let flips = compared
        .iter()
        .filter(|row| row.get("decisions_agree") == Some(&json!(false)))
        .count();
    let deltas = compared
        .iter()
        .filter_map(|row| row.get("max_abs_delta").and_then(serde_json::Value::as_f64))
        .collect::<Vec<_>>();
    json!({
        "compared": compared.len(),
        "flips": flips,
        "decision_agreements": compared.len() - flips,
        "max_abs_delta": deltas.iter().copied().fold(0.0f64, f64::max),
        "mean_case_max_abs_delta": if deltas.is_empty() {
            f64::NAN
        } else {
            deltas.iter().sum::<f64>() / deltas.len() as f64
        },
        "rows": rows,
    })
}

fn preprocess_summary(rows: &[serde_json::Value]) -> serde_json::Value {
    let compared = rows
        .iter()
        .filter(|row| row.get("status") == Some(&json!("compared")))
        .collect::<Vec<_>>();
    let exact = compared
        .iter()
        .filter(|row| row.get("max_abs_delta") == Some(&json!(0.0)))
        .count();
    json!({
        "compared": compared.len(),
        "exact": exact,
        "max_abs_delta": compared
            .iter()
            .filter_map(|row| row.get("max_abs_delta").and_then(serde_json::Value::as_f64))
            .fold(0.0f64, f64::max),
        "rows": rows,
    })
}

/// `--set image` / `--images SUITE`: decode each case's image, run the
/// checkpoint-fixed preprocessing and the full vision + text forward, and
/// report latency with the same swap gate as the text baseline. When
/// `--reference` points at the checkpoint's PyTorch receipt (`cases.json`),
/// the summary also carries the end-to-end probability parity, the
/// preprocessor element-wise comparison and the model-only parity (reference
/// pixels fed to the native vision tower).
fn image_main(options: Options) -> Result<(), Box<dyn std::error::Error>> {
    let workload: Workload = read_json(&options.workload)?;
    let cases = workload
        .cases
        .iter()
        .filter(|case| options.bucket == "all" || case.bucket.to_string() == options.bucket)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        return Err(format!(
            "image suite {} has no cases for --bucket {}",
            options.workload.display(),
            options.bucket
        )
        .into());
    }

    let mut load_options = LoadOptions::default();
    load_options.model_name = options.model_name.clone();
    let mut model = AutoModel::load_model(Device::Metal(0), &options.model_dir, &load_options)?;
    let code_count = model.decision_code_names()?.len();
    let config = metal_d3_lite::Config::from_dir(&options.model_dir)?;
    let patch_dim = 3 * config.vision.temporal_patch * config.vision.patch * config.vision.patch;

    let reference: Option<(PathBuf, HashMap<String, ImageReferenceCase>)> = match &options.reference
    {
        Some(path) => {
            let receipt: ImageReference = read_json(path)?;
            let base = path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            Some((
                base,
                receipt
                    .cases
                    .into_iter()
                    .map(|case| (case.id.clone(), case))
                    .collect(),
            ))
        }
        None => None,
    };

    let mut records = Vec::with_capacity(cases.len());
    let mut all_latency = Vec::new();
    let mut all_text_latency = Vec::new();
    let mut all_input_prep = Vec::new();
    let mut worst_swapouts = 0u64;
    let mut worst_swapins = 0u64;
    let mut worst_text_swapouts = 0u64;
    let mut worst_text_swapins = 0u64;
    let mut parity_rows = Vec::new();
    let mut model_only_rows = Vec::new();
    let mut preprocess_rows = Vec::new();
    let mut byte_identical = true;

    for case in &cases {
        let count = case.option_count;
        if count == 0 || count > code_count {
            return Err(format!(
                "case {}: option_count {count} is outside 1..={code_count}",
                case.id
            )
            .into());
        }
        let image = case
            .image
            .as_deref()
            .ok_or_else(|| format!("image case {} has no `image` path", case.id))?;
        let input_prep_started = Instant::now();
        let (rgb, width, height) = decode_rgb(Path::new(image))?;
        let prepared =
            metal_d3_lite::preprocess(&config.preprocessor, &config.vision, &rgb, width, height)?;
        let input_prep_ms = input_prep_started.elapsed().as_secs_f64() * 1e3;
        let rows = prepared.grid_thw[1] as usize * prepared.grid_thw[2] as usize;
        if prepared.pixel_values.len() != rows * patch_dim {
            return Err(format!(
                "case {}: preprocessing produced {} values, expected {rows} x {patch_dim}",
                case.id,
                prepared.pixel_values.len()
            )
            .into());
        }
        let pixels = Tensor::from_f32(vec![rows, patch_dim], &prepared.pixel_values)?;
        let grids = vec![prepared.grid_thw];
        let input =
            DecisionInput::with_vision(&case.token_ids, ImageInput::new(&pixels, &grids), count);
        // Same prompt without the vision tower: the image placeholders keep
        // their embedding vectors, so this measures the text stack + readout
        // the image request also pays. Attribution, not an optimization arm.
        let text_input = DecisionInput::text(&case.token_ids, count);

        for _ in 0..options.warmup {
            model.decide(input)?;
            model.decide(text_input)?;
        }
        let mut first: Option<Vec<f32>> = None;
        for _ in 0..options.repeat {
            let output = model.decide(input)?;
            let probabilities = output
                .scores
                .iter()
                .map(|score| score.probability)
                .collect::<Vec<_>>();
            match &first {
                None => first = Some(probabilities),
                Some(known) => {
                    if known.len() != probabilities.len()
                        || known
                            .iter()
                            .zip(&probabilities)
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        byte_identical = false;
                    }
                }
            }
        }
        let probabilities = first.ok_or("repeat produced no output")?;

        if let Some((base, reference)) = &reference {
            match reference.get(&case.id) {
                None => parity_rows.push(json!({
                    "id": case.id, "status": "missing-from-reference"
                })),
                Some(expected) => {
                    let expected_probabilities =
                        &expected.probabilities[..count.min(expected.probabilities.len())];
                    let decision = decision_of(&probabilities);
                    let reference_decision = decision_of(expected_probabilities);
                    parity_rows.push(json!({
                        "id": case.id,
                        "status": "compared",
                        "decisions_agree": decision == reference_decision,
                        "decision": decision,
                        "reference_decision": reference_decision,
                        "max_abs_delta": max_delta(&probabilities, expected_probabilities),
                        "kl": kl_divergence(expected_probabilities, &probabilities),
                    }));
                    if let Some(array) = expected
                        .arrays
                        .as_ref()
                        .and_then(|arrays| arrays.pixel_values.as_ref())
                    {
                        let reference_pixels = read_f32_file(base, &array.file)?;
                        let reference_grid =
                            expected.grid_thw.as_ref().map(Vec::as_slice).unwrap_or(&[]);
                        let same_grid = reference_grid == prepared.grid_thw;
                        let same_shape = same_grid && array.shape == vec![rows, patch_dim];
                        preprocess_rows.push(json!({
                            "id": case.id,
                            "status": if same_shape { "compared" } else { "shape-mismatch" },
                            "shape": [rows, patch_dim],
                            "reference_shape": array.shape,
                            "grid": prepared.grid_thw,
                            "reference_grid": reference_grid,
                            "elements": prepared.pixel_values.len(),
                            "max_abs_delta": if same_shape {
                                max_delta(&prepared.pixel_values, &reference_pixels)
                            } else {
                                f64::NAN
                            },
                        }));
                        if same_shape {
                            // Model-only parity: the native vision tower
                            // consumes the reference's preprocessed pixels,
                            // isolating vision + text numerics.
                            let reference_pixels_tensor =
                                Tensor::from_f32(vec![rows, patch_dim], &reference_pixels)?;
                            let model_only_input = DecisionInput::with_vision(
                                &case.token_ids,
                                ImageInput::new(&reference_pixels_tensor, &grids),
                                count,
                            );
                            let output = model.decide(model_only_input)?;
                            let model_probabilities = output
                                .scores
                                .iter()
                                .map(|score| score.probability)
                                .collect::<Vec<_>>();
                            let model_decision = decision_of(&model_probabilities);
                            model_only_rows.push(json!({
                                "id": case.id,
                                "status": "compared",
                                "decisions_agree": model_decision == reference_decision,
                                "decision": model_decision,
                                "reference_decision": reference_decision,
                                "max_abs_delta": max_delta(&model_probabilities, expected_probabilities),
                                "kl": kl_divergence(expected_probabilities, &model_probabilities),
                            }));
                        }
                    }
                }
            }
        }

        // Two series, one window: `text_plus_image` is the headline arm
        // (the full public image decision); `text` is the same prompt's text
        // stack without the vision tower, reported separately so the
        // vision-tower increment is visible. Fixed order (image first),
        // per-series swap deltas; no ABBA because there is no optimization
        // candidate in this phase.
        let mut latencies = Vec::with_capacity(options.rounds);
        let mut text_latencies = Vec::with_capacity(options.rounds);
        let mut swapout_deltas = Vec::with_capacity(options.rounds);
        let mut swapin_deltas = Vec::with_capacity(options.rounds);
        let mut text_swapout_deltas = Vec::with_capacity(options.rounds);
        let mut text_swapin_deltas = Vec::with_capacity(options.rounds);
        for _ in 0..options.rounds {
            let outs_before = swapouts();
            let ins_before = swapins();
            let start = Instant::now();
            model.decide(input)?;
            latencies.push(start.elapsed().as_secs_f64() * 1e3);
            let outs_mid = swapouts();
            let ins_mid = swapins();
            swapout_deltas.push(outs_mid.saturating_sub(outs_before));
            swapin_deltas.push(ins_mid.saturating_sub(ins_before));

            let start = Instant::now();
            model.decide(text_input)?;
            text_latencies.push(start.elapsed().as_secs_f64() * 1e3);
            let outs_after = swapouts();
            let ins_after = swapins();
            text_swapout_deltas.push(outs_after.saturating_sub(outs_mid));
            text_swapin_deltas.push(ins_after.saturating_sub(ins_mid));
        }
        worst_swapouts = worst_swapouts.max(swapout_deltas.iter().copied().max().unwrap_or(0));
        worst_swapins = worst_swapins.max(swapin_deltas.iter().copied().max().unwrap_or(0));
        worst_text_swapouts =
            worst_text_swapouts.max(text_swapout_deltas.iter().copied().max().unwrap_or(0));
        worst_text_swapins =
            worst_text_swapins.max(text_swapin_deltas.iter().copied().max().unwrap_or(0));
        all_latency.extend(latencies.iter().copied());
        all_text_latency.extend(text_latencies.iter().copied());
        all_input_prep.push(input_prep_ms);

        records.push(CaseRecord {
            id: case.id.clone(),
            bucket: case.bucket,
            tokens: case.token_ids.len(),
            count,
            probabilities: probabilities.clone(),
            hex: hex_f32(&probabilities),
            decision: decision_of(&probabilities),
            margin: margin_of(&probabilities),
            latency_ms: Some(latencies),
            input_prep_ms: Some(input_prep_ms),
            text_latency_ms: Some(text_latencies),
            swapouts: Some(swapout_deltas),
            swapins: Some(swapin_deltas),
            text_swapouts: Some(text_swapout_deltas),
            text_swapins: Some(text_swapin_deltas),
        });
    }

    let receipt = json!({
        "arm": "apxinf-metal",
        "set": "image",
        "model_dir": options.model_dir.display().to_string(),
        "model_name": options.model_name,
        "workload": {
            "path": options.workload.display().to_string(),
            "cases": cases.len(),
        },
        "repeat": options.repeat,
        "byte_identical": byte_identical,
        "cases": records,
    });
    write_json(&options.records, &receipt)?;

    let summary = json!({
        "engine": "apxinf-metal metal_d3_lite",
        "model_dir": options.model_dir.display().to_string(),
        "device": "metal:0",
        "set": "image",
        "cases": cases.len(),
        "warmup": options.warmup,
        "repeat": options.repeat,
        "rounds": options.rounds,
        "byte_identical": byte_identical,
        "reference": options.reference.as_ref().map(|path| path.display().to_string()),
        "parity": parity_summary(&parity_rows),
        "model_only_parity": parity_summary(&model_only_rows),
        "preprocessor": preprocess_summary(&preprocess_rows),
        // `text` and `text_plus_image` are reported separately (the
        // requester's baseline rule); `input_prep` is the host stage before
        // `decide`. The headline arm is `text_plus_image`.
        "timing": {
            "text": timing_stats(&all_text_latency),
            "text_plus_image": timing_stats(&all_latency),
            "input_prep": timing_stats(&all_input_prep),
        },
        "swap": {
            "text": {
                "worst_swapouts_pages": worst_text_swapouts,
                "worst_swapins_pages": worst_text_swapins,
                "clean": worst_text_swapouts == 0,
            },
            "text_plus_image": {
                "worst_swapouts_pages": worst_swapouts,
                "worst_swapins_pages": worst_swapins,
                "clean": worst_swapouts == 0,
            },
            "clean": worst_swapouts == 0 && worst_text_swapouts == 0,
        },
    });
    write_json(&options.summary, &summary)?;
    println!(
        "{}",
        json!({
            "records": options.records.display().to_string(),
            "summary": options.summary.display().to_string(),
            "cases": cases.len(),
            "byte_identical": byte_identical,
            "parity": summary["parity"],
            "preprocessor": {
                "compared": summary["preprocessor"]["compared"],
                "exact": summary["preprocessor"]["exact"],
                "max_abs_delta": summary["preprocessor"]["max_abs_delta"],
            },
            "median_ms": summary["timing"]["text_plus_image"]["median_ms"],
            "text_median_ms": summary["timing"]["text"]["median_ms"],
            "input_prep_median_ms": summary["timing"]["input_prep"]["median_ms"],
            "worst_swapouts": worst_swapouts,
        })
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options =
        parse_options().map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    if options.set == "image" {
        return image_main(options);
    }
    let workload: Workload = read_json(&options.workload)?;
    let cases = workload
        .cases
        .iter()
        // Mirrors the reference collector: the gate set is every case; the
        // timing set is the flagged subset (the 8 timing cases).
        .filter(|case| options.set == "gate" || case.timing)
        .filter(|case| options.bucket == "all" || case.bucket.to_string() == options.bucket)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        return Err(format!(
            "workload {} has no cases for --set {} --bucket {}",
            options.workload.display(),
            options.set,
            options.bucket
        )
        .into());
    }

    let mut load_options = LoadOptions::default();
    load_options.model_name = options.model_name.clone();
    let device = Device::Metal(0);
    let mut model = AutoModel::load_model(device, &options.model_dir, &load_options)?;
    let code_count = model.decision_code_names()?.len();

    let reference: Option<HashMap<String, Vec<f32>>> = match &options.reference {
        Some(path) => {
            let receipt: ReferenceReceipt = read_json(path)?;
            Some(
                receipt
                    .cases
                    .into_iter()
                    .map(|case| (case.id, case.probabilities))
                    .collect(),
            )
        }
        None => None,
    };

    let mut records = Vec::with_capacity(cases.len());
    let mut all_latency = Vec::new();
    let mut bucket_latency: HashMap<usize, Vec<f64>> = HashMap::new();
    let mut worst_swapouts = 0u64;
    let mut worst_swapins = 0u64;
    let mut parity_rows = Vec::new();
    let mut byte_identical = true;

    for case in &cases {
        let count = case.option_count;
        if count == 0 || count > code_count {
            return Err(format!(
                "case {}: option_count {count} is outside 1..={code_count}",
                case.id
            )
            .into());
        }
        for _ in 0..options.warmup {
            model.decide(DecisionInput::text(&case.token_ids, count))?;
        }
        let mut first: Option<Vec<f32>> = None;
        for _ in 0..options.repeat {
            let output = model.decide(DecisionInput::text(&case.token_ids, count))?;
            let probabilities = output
                .scores
                .iter()
                .map(|score| score.probability)
                .collect::<Vec<_>>();
            match &first {
                None => first = Some(probabilities),
                Some(known) => {
                    if known.len() != probabilities.len()
                        || known
                            .iter()
                            .zip(&probabilities)
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        byte_identical = false;
                    }
                }
            }
        }
        let probabilities = first.ok_or("repeat produced no output")?;

        let mut latencies = Vec::with_capacity(options.rounds);
        let mut swapout_deltas = Vec::with_capacity(options.rounds);
        let mut swapin_deltas = Vec::with_capacity(options.rounds);
        for _ in 0..options.rounds {
            let outs_before = swapouts();
            let ins_before = swapins();
            let start = Instant::now();
            model.decide(DecisionInput::text(&case.token_ids, count))?;
            latencies.push(start.elapsed().as_secs_f64() * 1e3);
            let outs_after = swapouts();
            let ins_after = swapins();
            swapout_deltas.push(outs_after.saturating_sub(outs_before));
            swapin_deltas.push(ins_after.saturating_sub(ins_before));
        }
        worst_swapouts = worst_swapouts.max(swapout_deltas.iter().copied().max().unwrap_or(0));
        worst_swapins = worst_swapins.max(swapin_deltas.iter().copied().max().unwrap_or(0));
        all_latency.extend(latencies.iter().copied());
        bucket_latency
            .entry(case.bucket)
            .or_default()
            .extend(latencies.iter().copied());

        if let Some(reference) = &reference {
            match reference.get(&case.id) {
                None => {
                    parity_rows.push(json!({
                        "id": case.id, "status": "missing-from-reference"
                    }));
                }
                Some(expected) => {
                    let expected = &expected[..count.min(expected.len())];
                    let decisions_agree = decision_of(&probabilities) == decision_of(expected);
                    parity_rows.push(json!({
                        "id": case.id,
                        "status": "compared",
                        "decisions_agree": decisions_agree,
                        "decision": decision_of(&probabilities),
                        "reference_decision": decision_of(expected),
                        "max_abs_delta": max_delta(&probabilities, expected),
                        "kl": kl_divergence(expected, &probabilities),
                    }));
                }
            }
        }

        records.push(CaseRecord {
            id: case.id.clone(),
            bucket: case.bucket,
            tokens: case.tokens,
            count,
            probabilities: probabilities.clone(),
            hex: hex_f32(&probabilities),
            decision: decision_of(&probabilities),
            margin: margin_of(&probabilities),
            latency_ms: if options.rounds > 0 {
                Some(latencies)
            } else {
                None
            },
            input_prep_ms: None,
            text_latency_ms: None,
            swapouts: Some(swapout_deltas),
            swapins: Some(swapin_deltas),
            text_swapouts: None,
            text_swapins: None,
        });
    }

    let receipt = json!({
        "arm": "apxinf-metal",
        "set": options.set,
        "model_dir": options.model_dir.display().to_string(),
        "model_name": options.model_name,
        "workload": {
            "path": options.workload.display().to_string(),
            "cases": cases.len(),
        },
        "repeat": options.repeat,
        "byte_identical": byte_identical,
        "cases": records,
    });
    write_json(&options.records, &receipt)?;

    let compared = parity_rows
        .iter()
        .filter(|row| row.get("status") == Some(&json!("compared")))
        .collect::<Vec<_>>();
    let flips = compared
        .iter()
        .filter(|row| row.get("decisions_agree") == Some(&json!(false)))
        .collect::<Vec<_>>();
    let max_abs_delta = compared
        .iter()
        .filter_map(|row| row.get("max_abs_delta").and_then(serde_json::Value::as_f64))
        .fold(0.0f64, f64::max);
    let mean_abs_delta = if compared.is_empty() {
        f64::NAN
    } else {
        compared
            .iter()
            .filter_map(|row| row.get("max_abs_delta").and_then(serde_json::Value::as_f64))
            .sum::<f64>()
            / compared.len() as f64
    };
    let mean_kl = if compared.is_empty() {
        f64::NAN
    } else {
        compared
            .iter()
            .filter_map(|row| row.get("kl").and_then(serde_json::Value::as_f64))
            .sum::<f64>()
            / compared.len() as f64
    };
    let mut by_bucket = serde_json::Map::new();
    for (bucket, values) in &bucket_latency {
        by_bucket.insert(bucket.to_string(), timing_stats(values));
    }
    let summary = json!({
        "engine": "apxinf-metal metal_d3_lite",
        "model_dir": options.model_dir.display().to_string(),
        "device": "metal:0",
        "set": options.set,
        "bucket": options.bucket,
        "cases": cases.len(),
        "warmup": options.warmup,
        "repeat": options.repeat,
        "rounds": options.rounds,
        "byte_identical": byte_identical,
        "parity": {
            "reference": options.reference.as_ref().map(|p| p.display().to_string()),
            "compared": compared.len(),
            "flips": flips.len(),
            "decision_agreements": compared.len() - flips.len(),
            "max_abs_delta": max_abs_delta,
            "mean_case_max_abs_delta": mean_abs_delta,
            "mean_kl": mean_kl,
            "rows": parity_rows,
        },
        "timing": {
            "overall": timing_stats(&all_latency),
            "by_bucket": by_bucket,
        },
        "swap": {
            "worst_swapouts_pages": worst_swapouts,
            "worst_swapins_pages": worst_swapins,
            "clean": worst_swapouts == 0,
        },
    });
    write_json(&options.summary, &summary)?;

    println!(
        "{}",
        json!({
            "records": options.records.display().to_string(),
            "summary": options.summary.display().to_string(),
            "cases": cases.len(),
            "byte_identical": byte_identical,
            "compared": compared.len(),
            "flips": flips.len(),
            "max_abs_delta": max_abs_delta,
            "median_ms": summary["timing"]["overall"]["median_ms"],
            "worst_swapouts": worst_swapouts,
        })
    );
    Ok(())
}
