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
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::{AutoModel, DecisionInput, LoadOptions};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize)]
struct Workload {
    cases: Vec<WorkloadCase>,
}

#[derive(Debug, Deserialize)]
struct WorkloadCase {
    id: String,
    bucket: usize,
    #[serde(default)]
    timing: bool,
    token_ids: Vec<u32>,
    option_count: usize,
    #[serde(default)]
    tokens: usize,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    swapouts: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    swapins: Option<Vec<u64>>,
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
    "usage: d3_lite_bench --model-dir DIR --workload FILE --set gate|timing \
     [--model-name NAME] [--bucket all|150|400|1000|2400] [--reference FILE] \
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
            "--set" => set = Some(value()?),
            "--bucket" => bucket = value()?,
            "--reference" => reference = Some(PathBuf::from(value()?)),
            "--warmup" => warmup = value()?.parse().map_err(|_| "--warmup must be an integer")?,
            "--repeat" => repeat = value()?.parse().map_err(|_| "--repeat must be an integer")?,
            "--rounds" => rounds = value()?.parse().map_err(|_| "--rounds must be an integer")?,
            "--records" => records = Some(PathBuf::from(value()?)),
            "--summary" => summary = Some(PathBuf::from(value()?)),
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unknown flag {other:?}\n{}", usage())),
        }
    }
    let set = set.ok_or_else(|| format!("--set is required\n{}", usage()))?;
    if set != "gate" && set != "timing" {
        return Err(format!("--set must be gate or timing, not {set}"));
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

fn read_json<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> Result<T, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&raw)?)
}

fn write_json(path: &std::path::Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options().map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
    let workload: Workload = read_json(&options.workload)?;
    let cases = workload
        .cases
        .iter()
        // Mirrors the reference collector: the gate set is every case; the
        // timing set is the flagged subset (the 8 timing cases).
        .filter(|case| options.set == "gate" || case.timing)
        .filter(|case| {
            options.bucket == "all" || case.bucket.to_string() == options.bucket
        })
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
            swapouts: Some(swapout_deltas),
            swapins: Some(swapin_deltas),
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
