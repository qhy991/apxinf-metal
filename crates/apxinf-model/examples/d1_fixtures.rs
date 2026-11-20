//! d1 P1a fixture harness: run the frozen P0 corpus through `metal_d1` and
//! write native answers in the reference's JSON shape, for
//! `devlocal/d1-2b-port/scripts/compare_answers.py`.
//!
//! Text-only (P1a): image fixtures are rejected explicitly, `system_one_batch`
//! fixtures run one forward per request (the reference's padded batching is a
//! P2 comparison point, noted in the port doc).
//!
//! Usage: d1_fixtures --model DIR --corpus corpus.json --out DIR [--only pfx,...]

use apxinf_core::{Backend, Device};
use apxinf_model::metal_d1::questions::{self, Question};
use apxinf_model::metal_d1::runner::D1Runner;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<(), String> {
    let mut model: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut only: Vec<String> = Vec::new();
    let mut prompts_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs a value")?)),
            "--corpus" => corpus = Some(PathBuf::from(args.next().ok_or("--corpus needs a value")?)),
            "--out" => out = Some(PathBuf::from(args.next().ok_or("--out needs a value")?)),
            "--emit-prompts" => prompts_only = true,
            "--only" => {
                only = args
                    .next()
                    .ok_or("--only needs a value")?
                    .split(',')
                    .map(str::to_string)
                    .collect()
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let model = model.ok_or("--model is required")?;
    let corpus = corpus.ok_or("--corpus is required")?;
    let out = out.ok_or("--out is required")?;
    std::fs::create_dir_all(&out).map_err(|e| format!("create {out:?}: {e}"))?;

    let corpus_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&corpus).map_err(|e| format!("read {corpus:?}: {e}"))?,
    )
    .map_err(|e| format!("parse {corpus:?}: {e}"))?;
    let fixtures = corpus_json["fixtures"]
        .as_array()
        .ok_or("corpus has no fixtures array")?
        .clone();

    let _ = Device::Metal(0);
    if prompts_only {
        return emit_prompts(&model, &fixtures, &out, &only);
    }
    let backend: Arc<dyn Backend> = Arc::new(
        apxinf_mlx::MlxBackend::new(0).map_err(|e| format!("MLX backend: {e}"))?,
    );
    let started = Instant::now();
    let model_obj = apxinf_model::metal_d1::load(&model, backend).map_err(|e| format!("load: {e}"))?;
    let mut runner = D1Runner::new(model_obj, &model).map_err(|e| format!("runner: {e}"))?;
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f64());

    let mut index = Vec::new();
    for fixture in &fixtures {
        let id = fixture["id"].as_str().ok_or("fixture without id")?;
        if !only.is_empty() && !only.iter().any(|p| id.starts_with(p.as_str())) {
            continue;
        }
        let payload = run_fixture(&mut runner, fixture);
        let text = match &payload {
            Ok(value) => serde_json::to_string_pretty(value).unwrap(),
            Err(error) => serde_json::to_string_pretty(&serde_json::json!({
                "id": id,
                "error": error,
            }))
            .unwrap(),
        };
        std::fs::write(out.join(format!("{id}.json")), format!("{text}\n"))
            .map_err(|e| format!("write {id}: {e}"))?;
        match &payload {
            Ok(_) => eprintln!("[{id}] ok"),
            Err(error) => eprintln!("[{id}] ERROR {error}"),
        }
        index.push(id.to_string());
    }
    let meta = serde_json::json!({
        "device": "metal",
        "dtype": "bf16",
        "implementation": "metal_d1 P1a (text-only, per-request forwards)",
        "fixtures": index,
    });
    std::fs::write(
        out.join("_run.json"),
        serde_json::to_string_pretty(&meta).unwrap() + "\n",
    )
    .map_err(|e| format!("write _run.json: {e}"))?;
    Ok(())
}

fn run_fixture(
    runner: &mut D1Runner,
    fixture: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let id = fixture["id"].as_str().unwrap_or("?");
    if fixture.get("media").is_some_and(|m| !m.is_null()) {
        return Err("P1a is text-only: media fixtures are rejected explicitly".into());
    }
    let started = Instant::now();
    let call = fixture["call"].as_str().ok_or("fixture without call")?;
    let set_size = fixture
        .get("set_size")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let parse_questions = |value: &serde_json::Value| -> Result<Vec<(String, Question)>, String> {
        let map = value
            .as_object()
            .ok_or_else(|| format!("{id}: questions must be an object"))?;
        let mut out = Vec::with_capacity(map.len());
        for (name, q) in map {
            let question = questions::from_json(q).map_err(|e| format!("{id}/{name}: {e}"))?;
            out.push((name.clone(), question));
        }
        Ok(out)
    };
    let result = match call {
        "system_one" => {
            let questions = parse_questions(&fixture["questions"])?;
            let state = fixture.get("state").filter(|s| !s.is_null());
            let (answers, tokens) = runner
                .system_one(state, &questions, set_size, "median")
                .map_err(|e| format!("{id}: {e}"))?;
            serde_json::json!({
                "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0},
            })
        }
        "system_one_batch" => {
            let requests = fixture["requests"]
                .as_array()
                .ok_or_else(|| format!("{id}: batch fixture without requests"))?;
            let mut results = Vec::with_capacity(requests.len());
            for request in requests {
                let questions = parse_questions(&request["questions"])?;
                let state = request.get("state").filter(|s| !s.is_null());
                if request.get("media").is_some_and(|m| !m.is_null()) {
                    return Err("P1a is text-only".into());
                }
                let (answers, tokens) = runner
                    .system_one(state, &questions, set_size, "median")
                    .map_err(|e| format!("{id}: {e}"))?;
                results.push(serde_json::json!({
                    "answers": answers,
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                }));
            }
            serde_json::Value::Array(results)
        }
        other => return Err(format!("{id}: unknown call {other}")),
    };
    let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(serde_json::json!({
        "id": id,
        "device": "metal",
        "dtype": "bf16",
        "result": result,
        "wall_ms": wall_ms,
    }))
}

/// Write each fixture's packed sequences (ids, state_len, question, ends)
/// without loading the model — the prompt-construction gate against
/// `fixtures/prompts/`.
fn emit_prompts(
    model: &Path,
    fixtures: &[serde_json::Value],
    out: &Path,
    only: &[String],
) -> Result<(), String> {
    use apxinf_model::metal_d1::prompt;
    use apxinf_model::metal_d1::questions;
    let tokenizer = apxinf_tokenizer::Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|e| format!("tokenizer: {e}"))?;
    let specials = prompt::specials_from_tokenizer_json(&model.join("tokenizer.json"))
        .map_err(|e| format!("specials: {e}"))?;
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(model.join("config.json")).map_err(|e| format!("config: {e}"))?,
    )
    .map_err(|e| format!("config: {e}"))?;
    let d1_numbers = config["d1_numbers"].as_bool().unwrap_or(false);
    let prompt_version = config["d1_prompt"].as_u64().unwrap_or(1) as u32;
    let set_size_cfg = config["d1_set_size"].as_u64().unwrap_or(16) as usize;
    let bos = "<|startoftext|>";
    for fixture in fixtures {
        let id = fixture["id"].as_str().ok_or("fixture without id")?;
        if !only.is_empty() && !only.iter().any(|p| id.starts_with(p.as_str())) {
            continue;
        }
        if fixture.get("media").is_some_and(|m| !m.is_null()) {
            continue;
        }
        let call = fixture["call"].as_str().ok_or("fixture without call")?;
        let set_size = fixture
            .get("set_size")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(set_size_cfg);
        let cases: Vec<(Option<&serde_json::Value>, &serde_json::Value)> = match call {
            "system_one" => vec![(
                fixture.get("state").filter(|s| !s.is_null()),
                &fixture["questions"],
            )],
            "system_one_batch" => fixture["requests"]
                .as_array()
                .ok_or("batch without requests")?
                .iter()
                .map(|r| (r.get("state").filter(|s| !s.is_null()), &r["questions"]))
                .collect(),
            other => return Err(format!("unknown call {other}")),
        };
        let mut sequences = Vec::new();
        for (state, questions_value) in cases {
            let map = questions_value.as_object().ok_or("questions not an object")?;
            let mut parsed = Vec::with_capacity(map.len());
            for (name, q) in map {
                parsed.push((name.clone(), questions::from_json(q).map_err(|e| format!("{id}/{name}: {e}"))?));
            }
            let prefix = prompt::prefix_text(&tokenizer, &specials, bos, state)
                .map_err(|e| format!("{id} prefix: {e}"))?;
            let layouts: Vec<(String, Vec<i64>)> = parsed
                .iter()
                .map(|(_, q)| {
                    prompt::suffix_layout(&tokenizer, &specials, q, d1_numbers, prompt_version, 2)
                        .map_err(|e| format!("{id} suffix: {e}"))
                })
                .collect::<Result<_, String>>()?;
            let header = prompt::question_header(state.is_none());
            let questions: Vec<&str> = parsed.iter().map(|(n, _)| n.as_str()).collect();
            let mut k = 0;
            while k < parsed.len() {
                let end = (k + set_size).min(parsed.len());
                let suffixes: Vec<String> = (k..end).map(|i| layouts[i].0.clone()).collect();
                let offsets: Vec<Vec<i64>> = (k..end).map(|i| layouts[i].1.clone()).collect();
                let packed = prompt::pack_set(&tokenizer, &prefix, &suffixes, header, &offsets)
                    .map_err(|e| format!("{id} pack: {e}"))?;
                let text = tokenizer
                    .decode_with_options(&packed.ids, false)
                    .map_err(|e| format!("{id} decode: {e}"))?;
                sequences.push(serde_json::json!({
                    "questions": questions[k..end],
                    "text": text,
                    "ids": packed.ids,
                    "state_len": packed.state_len,
                    "question_per_token": packed.question,
                    "answer_slot_positions": packed.ends,
                    "n_tokens": packed.ids.len(),
                }));
                k = end;
            }
        }
        let total: usize = sequences
            .iter()
            .map(|s| s["n_tokens"].as_u64().unwrap() as usize)
            .sum();
        let payload = serde_json::json!({
            "id": id,
            "set_size": set_size,
            "sequences": sequences,
            "total_input_tokens": total,
        });
        std::fs::write(
            out.join(format!("{id}.json")),
            serde_json::to_string_pretty(&payload).unwrap() + "\n",
        )
        .map_err(|e| format!("write {id}: {e}"))?;
        eprintln!("[{id}] {} seq, {total} tokens", sequences.len());
    }
    Ok(())
}
