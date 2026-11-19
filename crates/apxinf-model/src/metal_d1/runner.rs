//! The D1 request runner: named typed questions over a state, one packed
//! sequence per set group, answers assembled through the readout. Mirrors
//! the reference `runner.SlotReader`/`SystemOne` for text-only requests
//! (P1a): no media, no request batching — one sequence per forward.

use super::calibration::Calibration;
use super::config::D1Config;
use super::model::{D1Model, SetAttention};
use super::prompt;
use super::questions::Question;
use super::readout::{self, AnswerProbs, Slot};
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType};
use apxinf_tokenizer::Tokenizer;
use std::path::Path;

fn err(message: impl Into<String>) -> Error {
    Error::Other(format!("d1 runner: {}", message.into()))
}

pub struct D1Runner {
    model: D1Model,
    tokenizer: Tokenizer,
    specials: Vec<String>,
    calibration: Calibration,
    bos: String,
    set_size: usize,
    d1_numbers: bool,
    prompt_version: u32,
    box_edges: String,
}

impl D1Runner {
    pub fn new(model: D1Model, checkpoint: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(checkpoint.join("tokenizer.json"))
            .map_err(|e| err(format!("tokenizer: {e}")))?;
        let specials = prompt::specials_from_tokenizer_json(&checkpoint.join("tokenizer.json"))?;
        let calibration = Calibration::from_json(
            &std::fs::read_to_string(checkpoint.join("calibration.json"))
                .map_err(|e| err(format!("cannot read calibration.json: {e}")))?,
        )?;
        let config = model.config();
        let bos = bos_text(checkpoint)?;
        Ok(Self {
            set_size: config.set_size(),
            d1_numbers: config.d1_numbers,
            prompt_version: config.prompt_version(),
            box_edges: config.box_edges().to_string(),
            model,
            tokenizer,
            specials,
            calibration,
            bos,
        })
    }

    pub fn config(&self) -> &D1Config {
        self.model.config()
    }

    /// One `(state, questions)` request, text-only. Returns the answers JSON
    /// (`{name: answer}`) and the tokens the request read.
    pub fn system_one(
        &mut self,
        state: Option<&serde_json::Value>,
        questions: &[(String, Question)],
        set_size: Option<usize>,
        score_readout: &str,
    ) -> Result<(serde_json::Value, usize)> {
        if questions.is_empty() {
            return Err(err("at least one question is required"));
        }
        let step = set_size.unwrap_or(self.set_size);
        if step == 0 {
            return Err(err("set_size must be at least 1"));
        }
        let prefix = prompt::prefix_text(&self.tokenizer, &self.specials, &self.bos, state)?;
        let layouts: Vec<(String, Vec<i64>)> = questions
            .iter()
            .map(|(_, q)| {
                prompt::suffix_layout(
                    &self.tokenizer,
                    &self.specials,
                    q,
                    self.d1_numbers,
                    self.prompt_version,
                    self.model.config().d1_codes.unwrap_or(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let slot_plans: Vec<Vec<Slot>> = questions
            .iter()
            .map(|(_, q)| {
                readout::question_slots(
                    &self.tokenizer,
                    q,
                    self.d1_numbers,
                    self.model.config().d1_codes.unwrap_or(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let header = prompt::question_header(state.is_none());
        let context = match state {
            Some(state) => prompt::state_text(state)?,
            None => String::new(),
        };
        let mut answers = serde_json::Map::with_capacity(questions.len());
        let mut read_tokens = 0usize;
        let mut k = 0usize;
        while k < questions.len() {
            let group: Vec<usize> = (k..(k + step).min(questions.len())).collect();
            let suffixes: Vec<String> = group
                .iter()
                .map(|&i| layouts[i].0.clone())
                .collect();
            let offsets: Vec<Vec<i64>> = group
                .iter()
                .map(|&i| layouts[i].1.clone())
                .collect();
            let packed = prompt::pack_set(&self.tokenizer, &prefix, &suffixes, header, &offsets)?;
            read_tokens += packed.ids.len();
            // The pointer's state positions, only when a read is in this group.
            let positions = if group
                .iter()
                .any(|&i| matches!(questions[i].1, Question::Read { .. }))
            {
                let full_text = format!("{prefix}{}", suffixes[0]);
                Some(readout::state_positions(
                    &self.tokenizer,
                    &self.specials,
                    state,
                    &prefix,
                    &full_text,
                )?)
            } else {
                None
            };
            let set = SetAttention {
                state_len: packed.state_len,
                question: packed.question.clone(),
            };
            let hidden = self.model.forward_hidden(&packed.ids, Some(&set))?;
            let probs = self.read_slots(
                &hidden,
                &packed,
                &group,
                questions,
                &slot_plans,
                positions.as_ref(),
            )?;
            for (gi, p) in group.iter().zip(probs) {
                let (name, q) = &questions[*gi];
                let answer = readout::answer(
                    q,
                    &p,
                    &self.calibration,
                    &context,
                    &self.box_edges,
                    score_readout,
                )?;
                answers.insert(name.clone(), answer);
            }
            k += step;
        }
        Ok((serde_json::Value::Object(answers), read_tokens))
    }

    /// Gather the group's slot logits and read each question's probabilities.
    fn read_slots(
        &self,
        hidden: &Array,
        packed: &prompt::PackedSequence,
        group: &[usize],
        questions: &[(String, Question)],
        slot_plans: &[Vec<Slot>],
        positions: Option<&readout::StatePositions>,
    ) -> Result<Vec<AnswerProbs>> {
        let stream = self.model.stream().clone();
        let len = packed.ids.len();
        let hidden_dim = hidden.shape()[2];
        let hidden = hidden.reshape(&[1, len, hidden_dim])?;
        // The flat slot plan: (group position, slot) pairs with their token
        // positions, in marker order.
        let mut code_positions: Vec<i32> = Vec::new();
        let mut code_groups: Vec<Vec<Vec<u32>>> = Vec::new();
        let mut code_owner: Vec<usize> = Vec::new();
        let mut read_positions: Vec<(usize, i32)> = Vec::new(); // (group pos, token)
        for (gi, &qi) in group.iter().enumerate() {
            let plan = &slot_plans[qi];
            let ends = &packed.ends[gi];
            if plan.len() != ends.len() {
                return Err(err(format!(
                    "question `{}`: {} slot plans for {} slot positions",
                    questions[qi].0,
                    plan.len(),
                    ends.len()
                )));
            }
            for (slot, &position) in plan.iter().zip(ends) {
                match slot {
                    Slot::Codes(groups) => {
                        code_positions.push(position as i32);
                        code_groups.push(groups.clone());
                        code_owner.push(gi);
                    }
                    Slot::Position => read_positions.push((gi, position as i32)),
                }
            }
        }
        // Code slots: gather rows, project, FP32 log-softmax row-wise, then
        // one deduped gather of every readout token id.
        let mut slot_logz: Vec<Vec<f32>> = Vec::with_capacity(code_positions.len());
        if !code_positions.is_empty() {
            let index = Array::from_i32(&stream, &[code_positions.len()], &code_positions)?;
            let rows = hidden.take(&index, 1)?; // [1, S, H]
            let logits = self.model.project(&rows)?.cast(MlxDType::F32)?; // [1, S, V]
            let count = code_positions.len();
            let vocab = logits.shape()[2];
            let logits = logits.reshape(&[count, vocab])?;
            let max = logits.max(1)?.reshape(&[count, 1])?; // [S, 1]
            let shifted = logits.sub(&max.broadcast_to(&[count, vocab])?)?;
            let sumexp = shifted.exp()?.sum(&[1], false)?; // [S]
            let mut ids: Vec<u32> = Vec::new();
            for groups in &code_groups {
                for group in groups {
                    for &id in group {
                        if !ids.contains(&id) {
                            ids.push(id);
                        }
                    }
                }
            }
            let id_i32: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            let id_array = Array::from_i32(&stream, &[ids.len()], &id_i32)?;
            let gathered = logits.take(&id_array, 1)?; // [S, G]
            let gathered = gathered.to_f32_vec()?;
            let max_v = max.reshape(&[count])?.to_f32_vec()?;
            let sumexp_v = sumexp.to_f32_vec()?;
            for row in 0..count {
                let lse = max_v[row] + sumexp_v[row].ln();
                let base = row * ids.len();
                slot_logz.push(
                    (0..ids.len())
                        .map(|col| gathered[base + col] - lse)
                        .collect(),
                );
            }
            // Replace the dedup-indexed log-probs with a per-slot token map.
            let id_index = |token: u32| ids.iter().position(|&t| t == token).unwrap();
            slot_logz = slot_logz
                .iter()
                .zip(&code_groups)
                .map(|(logz, groups)| {
                    let mut flat = Vec::new();
                    for group in groups {
                        for &id in group {
                            flat.push(logz[id_index(id)]);
                        }
                    }
                    flat
                })
                .collect();
        }
        // Pointer slots: start/end pairs per read.
        let mut spans_out: std::collections::HashMap<usize, (usize, usize, f32)> =
            std::collections::HashMap::new();
        if !read_positions.is_empty() {
            let positions = positions
                .ok_or_else(|| err("a read question needs its state positions"))?;
            let mask = positions.pointer_mask();
            let width = mask.len();
            let keys = hidden
                .slice_axis(1, 0, width)?
                .cast(MlxDType::F32)?
                .reshape(&[width, hidden_dim])?
                .transpose(&[1, 0])?;
            let scale = Array::scalar(&stream, (hidden_dim as f64).sqrt() as f32, MlxDType::F32)?;
            for pair in read_positions.chunks_exact(2) {
                let (gi, start_pos) = pair[0];
                let (_, end_pos) = pair[1];
                let index = Array::from_i32(&stream, &[2], &[start_pos, end_pos])?;
                let queries = hidden
                    .take(&index, 1)?
                    .cast(MlxDType::F32)?
                    .reshape(&[2, hidden_dim])?;
                let logits = queries.matmul(&keys)?.div(&scale)?; // [2, width]
                let mut host = logits.reshape(&[2, width])?.to_f32_vec()?;
                for (j, valid) in mask.iter().enumerate() {
                    if !valid {
                        host[j] = f32::NEG_INFINITY;
                        host[width + j] = f32::NEG_INFINITY;
                    }
                }
                let kind = match &questions[group[gi]].1 {
                    Question::Read { kind, .. } => *kind,
                    _ => return Err(err("a pointer slot belongs to a read question")),
                };
                let (first, last, found) =
                    readout::spans(&host[..width], &host[width..], kind.span_cap());
                spans_out.insert(gi, (first, last, found));
            }
        }
        // Assemble per-question probabilities (calibrated), then AnswerProbs.
        let mut code_row = 0usize;
        let mut out: Vec<AnswerProbs> = Vec::with_capacity(group.len());
        for (gi, &qi) in group.iter().enumerate() {
            let q = &questions[qi].1;
            let plan = &slot_plans[qi];
            let mut slots_probs: Vec<Vec<f64>> = Vec::with_capacity(plan.len());
            for slot in plan {
                if let Slot::Codes(groups) = slot {
                    let logz = &slot_logz[code_row];
                    let mut at = 0usize;
                    let scores: Vec<f64> = groups
                        .iter()
                        .map(|group| {
                            let mut best = f32::NEG_INFINITY;
                            for _ in group {
                                best = best.max(logz[at]);
                                at += 1;
                            }
                            best as f64
                        })
                        .collect();
                    // Typed slots read raw (their tables calibrate); option
                    // slots go through the temperature.
                    let temperature = if q.is_typed() {
                        1.0
                    } else {
                        self.calibration.temperature(q, cardinality(q))
                    };
                    slots_probs.push(readout::probabilities(&scores, temperature));
                    code_row += 1;
                }
            }
            let answer = match q {
                Question::Noul { .. } | Question::Score { .. } => {
                    AnswerProbs::Codes(slots_probs[0].clone())
                }
                Question::Choice { multiple, .. } => {
                    if *multiple {
                        AnswerProbs::Multiple(slots_probs.iter().map(|p| p[0]).collect())
                    } else {
                        AnswerProbs::Codes(slots_probs[0].clone())
                    }
                }
                Question::Estimate { .. } => AnswerProbs::Codes(slots_probs[0].clone()),
                Question::Read { kind, .. } => {
                    let (first, last, found) = spans_out[&gi];
                    let span = positions
                        .unwrap()
                        .read(first, last, kind.span_cap())?;
                    AnswerProbs::Read {
                        found,
                        span: span.span,
                        text: span.text,
                        path: span.path,
                    }
                }
                Question::Box { .. } => AnswerProbs::Box {
                    found: slots_probs[0][0],
                    edges: slots_probs[1..].to_vec(),
                },
                Question::Boxes { .. } => AnswerProbs::Boxes(
                    slots_probs
                        .chunks_exact(5)
                        .map(|block| readout::BoxBlock {
                            found: block[0][0],
                            edges: block[1..].to_vec(),
                        })
                        .collect(),
                ),
            };
            out.push(answer);
        }
        let _ = code_owner;
        Ok(out)
    }
}

fn cardinality(q: &Question) -> usize {
    match q {
        Question::Noul { .. } => 2,
        Question::Choice { criteria, .. } => criteria.len(),
        Question::Score { levels, .. } => levels.len(),
        _ => 0,
    }
}

/// The prompt's BOS text from `tokenizer_config.json` (`bos_token`).
fn bos_text(checkpoint: &Path) -> Result<String> {
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(checkpoint.join("tokenizer_config.json"))
            .map_err(|e| err(format!("cannot read tokenizer_config.json: {e}")))?,
    )
    .map_err(|e| err(format!("cannot parse tokenizer_config.json: {e}")))?;
    config
        .get("bos_token")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| err("tokenizer_config.json has no bos_token"))
}
