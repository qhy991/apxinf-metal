//! D1 readout: slot logits → calibrated probabilities → typed answers.
//! Mirrors the reference's `prompt.readout*`, `pointer.py`, `box.py`,
//! `api.py answer()` and `json_spans.py`. Host math is FP64 exactly as the
//! reference's numpy path; device work is limited to log-softmax over the
//! vocabulary and the pointer's query·state product.

use super::calibration::Calibration;
use super::parsing;
use super::prompt;
use super::questions::{Grid, Question, ReadKind};
use apxinf_core::{Error, Result};
use apxinf_tokenizer::Tokenizer;

fn err(message: impl Into<String>) -> Error {
    Error::Other(format!("d1 readout: {}", message.into()))
}

/// One answer slot's readout plan (`prompt.question_slots`): code slots are
/// scored through the tied head; position slots are pointer reads into the
/// state (a read's start then its end).
#[derive(Clone, Debug)]
pub enum Slot {
    /// Score this slot's log-probs over these token-id groups (max per group).
    Codes(Vec<Vec<u32>>),
    /// A pointer position slot.
    Position,
}

/// `prompt.question_slots` resolved to readout groups.
pub fn question_slots(
    tokenizer: &Tokenizer,
    q: &Question,
    d1_numbers: bool,
    codes_scheme: u32,
) -> Result<Vec<Slot>> {
    match q {
        Question::Read { .. } => Ok(vec![Slot::Position, Slot::Position]),
        Question::Estimate { grid, .. } => Ok(vec![Slot::Codes(grid_groups(
            tokenizer,
            grid,
            1,
            d1_numbers,
            codes_scheme,
        )?)]),
        Question::Box { .. } => {
            let (grid, scale) = prompt::edge_grid(tokenizer, d1_numbers);
            let groups = grid_groups(tokenizer, &grid, scale, d1_numbers, codes_scheme)?;
            let mut slots = vec![Slot::Codes(noul_groups(tokenizer)?)];
            for _ in EDGES {
                slots.push(Slot::Codes(groups.clone()));
            }
            Ok(slots)
        }
        Question::Boxes { max, .. } => {
            let blocks = max.unwrap_or(16);
            let (grid, scale) = prompt::edge_grid(tokenizer, d1_numbers);
            let groups = grid_groups(tokenizer, &grid, scale, d1_numbers, codes_scheme)?;
            let mut slots = Vec::with_capacity(blocks * 5);
            for _ in 0..blocks {
                slots.push(Slot::Codes(noul_groups(tokenizer)?));
                for _ in EDGES {
                    slots.push(Slot::Codes(groups.clone()));
                }
            }
            Ok(slots)
        }
        Question::Choice {
            multiple: true,
            criteria,
            ..
        } => {
            if criteria.is_empty() {
                return Err(err("choice with no options"));
            }
            Ok((0..criteria.len())
                .map(|_| noul_groups(tokenizer).map(Slot::Codes))
                .collect::<Result<Vec<_>>>()?)
        }
        Question::Noul { .. } => Ok(vec![Slot::Codes(noul_groups(tokenizer)?)]),
        Question::Score { levels, .. } => Ok(vec![Slot::Codes(score_groups(tokenizer, levels.len())?)]),
        Question::Choice { criteria, .. } => Ok(vec![Slot::Codes(choice_groups(
            tokenizer,
            criteria,
            codes_scheme,
        )?)]),
    }
}

/// `_ids`: distinct single-token ids of these texts, in order.
fn distinct_ids(tokenizer: &Tokenizer, texts: &[String]) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for text in texts {
        let ids = tokenizer
            .encode(text)
            .map_err(|e| err(format!("encode {text:?}: {e}")))?;
        if ids.len() == 1 && !out.contains(&ids[0]) {
            out.push(ids[0]);
        }
    }
    Ok(out)
}

fn noul_groups(tokenizer: &Tokenizer) -> Result<Vec<Vec<u32>>> {
    let yes = distinct_ids(tokenizer, &["yes".into(), "Yes".into(), "YES".into()])?;
    let no = distinct_ids(tokenizer, &["no".into(), "No".into(), "NO".into()])?;
    if yes.is_empty() || no.is_empty() {
        return Err(err("tokenizer has no single-token yes/no"));
    }
    Ok(vec![yes, no])
}

fn score_groups(tokenizer: &Tokenizer, levels: usize) -> Result<Vec<Vec<u32>>> {
    let mut groups = Vec::with_capacity(levels);
    for i in 0..levels {
        let group = distinct_ids(tokenizer, &[i.to_string()])?;
        if group.is_empty() {
            return Err(err(format!(
                "score with {levels} levels needs single-token digits"
            )));
        }
        groups.push(group);
    }
    Ok(groups)
}

fn choice_groups(
    tokenizer: &Tokenizer,
    criteria: &[(String, Option<String>)],
    codes_scheme: u32,
) -> Result<Vec<Vec<u32>>> {
    let labels: Vec<String> = criteria.iter().map(|(name, _)| name.clone()).collect();
    let mut groups = Vec::with_capacity(labels.len());
    for (_, tid) in prompt::aliases(tokenizer, &labels, codes_scheme)? {
        // Each option scores its best form: the bare code and its spaced form.
        let code = prompt::option_codes(&labels, codes_scheme)
            .into_iter()
            .nth(groups.len())
            .unwrap();
        let mut group = vec![tid];
        for id in distinct_ids(tokenizer, &[format!(" {code}")])? {
            if id != tid {
                group.push(id);
            }
        }
        groups.push(group);
    }
    if groups.is_empty() {
        return Err(err("choice with no options"));
    }
    Ok(groups)
}

/// A grid slot's groups: the tokenizer's own numbers where the checkpoint
/// reads them so, else the grid's codes (`prompt.grid_slot`/`grid_question`).
fn grid_groups(
    tokenizer: &Tokenizer,
    grid: &Grid,
    scale: u64,
    d1_numbers: bool,
    codes_scheme: u32,
) -> Result<Vec<Vec<u32>>> {
    if d1_numbers {
        if let Some(groups) = prompt::number_ids(tokenizer, grid, scale) {
            return Ok(groups);
        }
    }
    if grid.size <= 10 {
        return score_groups(tokenizer, grid.size);
    }
    // `grid_question`: Choice over criteria keys "0".."N-1"; each option
    // scores its best form (bare code and spaced code), as `readout_ids`.
    let labels: Vec<String> = (0..grid.size).map(|i| i.to_string()).collect();
    let codes = prompt::option_codes(&labels, codes_scheme);
    let mut groups = Vec::with_capacity(grid.size);
    for ((_, tid), code) in prompt::aliases(tokenizer, &labels, codes_scheme)?
        .into_iter()
        .zip(&codes)
    {
        let mut group = vec![tid];
        for id in distinct_ids(tokenizer, &[format!(" {code}")])? {
            if id != tid {
                group.push(id);
            }
        }
        groups.push(group);
    }
    Ok(groups)
}

/// `prompt.probabilities`: each option's best score through the calibration
/// temperature, as an FP64 softmax. `logz` is the slot's FP32
/// log-probabilities over the vocabulary, gathered on the device.
pub fn probabilities(scores: &[f64], temperature: f64) -> Vec<f64> {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s / temperature - m / temperature).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}

// ---------------------------------------------------------------- pointer

/// `pointer.StatePositions`: token positions mapped to the caller's rendered
/// state text (byte spans); position 0 is a template token ("BOS0").
#[derive(Clone, Debug)]
pub struct StatePositions {
    pub text: String,
    pub offsets: Vec<Option<(usize, usize)>>,
    pub paths: Vec<(usize, usize, String)>,
}

impl StatePositions {
    pub fn pointer_mask(&self) -> Vec<bool> {
        let valid: Vec<bool> = self
            .offsets
            .iter()
            .enumerate()
            .map(|(i, span)| i == 0 || span.is_some())
            .collect();
        let last = valid
            .iter()
            .rposition(|v| *v)
            .map(|i| i + 1)
            .unwrap_or(0);
        valid[..last].to_vec()
    }

    /// `StatePositions.read`: the span `[a.0, b.1]` of the rendered text,
    /// trimmed of boundary space and clipped to one JSON string when the span
    /// starts or ends in exactly one (`json_spans`).
    pub fn read(&self, first: usize, last: usize, max_tokens: usize) -> Result<ReadSpan> {
        if first == 0 && last == 0 {
            return Ok(ReadSpan {
                text: None,
                span: None,
                path: None,
            });
        }
        if first == 0 || first > last || last >= self.offsets.len() || last - first >= max_tokens {
            return Err(err("invalid pointer endpoints"));
        }
        let a = self.offsets[first].ok_or_else(|| err("pointer leaves the state text"))?;
        let b = self.offsets[last].ok_or_else(|| err("pointer leaves the state text"))?;
        let text_bytes = self.text.as_bytes();
        let mut span = (a.0, b.1);
        // Trim boundary space; never alter a character inside.
        while span.0 < span.1 && (text_bytes[span.0] as char).is_whitespace() {
            span.0 += 1;
        }
        while span.1 > span.0 && (text_bytes[span.1 - 1] as char).is_whitespace() {
            span.1 -= 1;
        }
        let held: Vec<&(usize, usize, String)> = self
            .paths
            .iter()
            .filter(|(start, end, _)| *start < span.1 && span.0 < *end)
            .collect();
        if held.len() == 1 {
            let (start, end, path) = held[0];
            span = (span.0.max(*start), span.1.min(*end));
            return Ok(ReadSpan {
                text: Some(json_text(&self.text[span.0..span.1])),
                span: Some((self.text[..span.0].chars().count(), self.text[..span.1].chars().count())),
                path: Some(path.clone()),
            });
        }
        Ok(ReadSpan {
            text: Some(self.text[span.0..span.1].to_string()),
            span: Some((self.text[..span.0].chars().count(), self.text[..span.1].chars().count())),
            path: None,
        })
    }
}

pub struct ReadSpan {
    pub text: Option<String>,
    /// Character offsets into the rendered state text (the reference's units).
    pub span: Option<(usize, usize)>,
    pub path: Option<String>,
}

/// `json_spans.string_paths`: string-value spans in rendered JSON (excluding
/// quotes), paths RFC6901-escaped.
pub fn string_paths(text: &str) -> Result<Vec<(usize, usize, String)>> {
    let bytes = text.as_bytes();
    let mut ranges = Vec::new();

    fn skip(bytes: &[u8], mut at: usize) -> usize {
        while at < bytes.len() && (bytes[at] as char).is_whitespace() {
            at += 1;
        }
        at
    }

    fn parse_string(bytes: &[u8], at: usize) -> Result<(usize, usize)> {
        // `at` is the opening quote; returns (end_of_string_excl_quote_plus1, closing_quote).
        let mut i = at + 1;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 2,
                b'"' => return Ok((i + 1, i)),
                _ => i += 1,
            }
        }
        Err(err("unterminated JSON string"))
    }

    fn value(
        bytes: &[u8],
        mut at: usize,
        path: &str,
        ranges: &mut Vec<(usize, usize, String)>,
    ) -> Result<usize> {
        at = skip(bytes, at);
        if at >= bytes.len() {
            return Err(err("unexpected end of JSON"));
        }
        match bytes[at] {
            b'{' => {
                at = skip(bytes, at + 1);
                while bytes[at] != b'}' {
                    let (after, closing) = parse_string(bytes, at)?;
                    let key = json_text(&String::from_utf8_lossy(&bytes[at + 1..closing]));
                    at = skip(bytes, after);
                    if bytes[at] != b':' {
                        return Err(err("invalid JSON object"));
                    }
                    let name = key.replace('~', "~0").replace('/', "~1");
                    at = skip(bytes, value(bytes, skip(bytes, at + 1), &format!("{path}/{name}"), ranges)?);
                    if bytes[at] == b',' {
                        at = skip(bytes, at + 1);
                    } else if bytes[at] != b'}' {
                        return Err(err("invalid JSON object separator"));
                    }
                }
                Ok(at + 1)
            }
            b'[' => {
                at = skip(bytes, at + 1);
                let mut index = 0;
                while bytes[at] != b']' {
                    at = skip(bytes, value(bytes, at, &format!("{path}/{index}"), ranges)?);
                    index += 1;
                    if bytes[at] == b',' {
                        at = skip(bytes, at + 1);
                    } else if bytes[at] != b']' {
                        return Err(err("invalid JSON array separator"));
                    }
                }
                Ok(at + 1)
            }
            b'"' => {
                let (end, closing) = parse_string(bytes, at)?;
                ranges.push((at + 1, closing, path.to_string()));
                Ok(end)
            }
            _ => {
                // Numbers, booleans, null: scan to the next structural char.
                let mut i = at;
                while i < bytes.len() && !matches!(bytes[i], b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r') {
                    i += 1;
                }
                Ok(i)
            }
        }
    }

    let end = value(bytes, 0, "", &mut ranges)?;
    if skip(bytes, end) != bytes.len() {
        return Err(err("trailing JSON text"));
    }
    Ok(ranges)
}

/// `json_spans.json_text`: a slice of a rendered JSON string as the
/// characters it encodes (unchanged when it cuts an escape).
pub fn json_text(escaped: &str) -> String {
    serde_json::from_str::<String>(&format!("\"{escaped}\"")).unwrap_or_else(|_| escaped.to_string())
}

/// `pointer.state_positions`: align the rendered prompt to state-text tokens;
/// template and question tokens stay `None`.
pub fn state_positions(
    tokenizer: &Tokenizer,
    specials: &[String],
    state: Option<&serde_json::Value>,
    prefix: &str,
    full_text: &str,
) -> Result<StatePositions> {
    let (text, start) = match state {
        None => (String::new(), prefix.len()),
        Some(state) => {
            let body = super::prompt::state_block(state)?;
            let (escaped, _) = prompt::plain_offsets(&body, specials);
            let tail = format!("{escaped}\n{}", prompt::question_header(false));
            if !prefix.ends_with(&tail) {
                return Err(err("state rendering differs from the prompt prefix"));
            }
            (prompt::state_text(state)?, prefix.len() - tail.len())
        }
    };
    let (_, boundaries) = match state {
        None => (String::new(), vec![0usize]),
        Some(state) => prompt::plain_offsets(&prompt::state_block(state)?, specials),
    };
    let text_bytes = text.as_bytes().len();
    let stop = start
        + boundaries
            .partition_point(|&boundary| boundary <= text_bytes)
            .saturating_sub(1);
    let (_, mapping) = tokenizer
        .encode_with_offsets(full_text)
        .map_err(|e| err(format!("encode prompt for positions: {e}")))?;
    let whitespace = |range: &str| range.trim().is_empty();
    let mut offsets = Vec::with_capacity(mapping.len());
    for (a, b) in mapping {
        let mut span = None;
        let lo = a.max(start);
        let hi = b.min(stop);
        if lo < hi
            && whitespace(&full_text[a..lo])
            && whitespace(&full_text[hi..b])
        {
            let left = boundaries[lo - start];
            let right = boundaries[hi - start];
            if left < right && right <= text_bytes && !text[left..right].trim().is_empty() {
                span = Some((left, right));
            }
        }
        offsets.push(span);
    }
    if offsets.is_empty() || offsets[0].is_some() {
        return Err(err("position 0 must be a template token for the nothing readout"));
    }
    let paths = match state {
        Some(value) if !value.is_string() => string_paths(&text)?,
        _ => Vec::new(),
    };
    Ok(StatePositions { text, offsets, paths })
}

/// `pointer.spans`: each read's span from its start and end log-probs
/// `[2, T]`, within `max_tokens`; `(0, 0)` for nothing. Returns
/// `(first, last, found)`.
pub fn spans(start_logits: &[f32], end_logits: &[f32], max_tokens: usize) -> (usize, usize, f32) {
    // log_softmax over each row (host FP32, matching the reference's dtype).
    let log_softmax = |row: &[f32]| -> Vec<f32> {
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let total: f32 = row.iter().map(|x| (x - m).exp()).sum();
        row.iter().map(|x| x - m - total.ln()).collect()
    };
    let start = log_softmax(start_logits);
    let end = log_softmax(end_logits);
    let found = -start[0].exp_m1(); // -expm1(start[0]) = 1 - P(start = BOS0)
    if start.len() == 1 {
        return (0, 0, found);
    }
    let cap = max_tokens.min(start.len() - 1);
    // max_pool1d over the padded end logits, window `cap`.
    let mut best_index = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    let mut best_last = 0usize;
    for i in 0..start.len() - 1 {
        // window end positions j in [i, i+cap)
        let mut window_best = f32::NEG_INFINITY;
        let mut window_last = i;
        for j in i..(i + cap).min(start.len() - 1) {
            if end[j + 1] > window_best {
                window_best = end[j + 1];
                window_last = j;
            }
        }
        let score = start[i + 1] + window_best;
        if score > best_score {
            best_score = score;
            best_index = i;
            best_last = window_last;
        }
    }
    let first = best_index + 1;
    let last = best_last + 1;
    if found < 0.5 {
        (0, 0, found)
    } else {
        (first, last, found)
    }
}

// ---------------------------------------------------------------- answers

/// The assembled answer JSON for one question (`api.answer`).
pub fn answer(
    q: &Question,
    probs: &AnswerProbs,
    calibration: &Calibration,
    context: &str,
    edges_read: &str,
    score_readout: &str,
) -> Result<serde_json::Value> {
    match (q, probs) {
        (Question::Noul { .. }, AnswerProbs::Codes(p)) => Ok(serde_json::json!({
            "type": "noul",
            "noul": p[0],
        })),
        (Question::Choice { multiple: false, criteria, .. }, AnswerProbs::Codes(p)) => {
            let best = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let mut probabilities = serde_json::Map::new();
            for ((name, _), prob) in criteria.iter().zip(p) {
                probabilities.insert(name.clone(), (*prob).into());
            }
            Ok(serde_json::json!({
                "type": "choice",
                "choice": criteria[best].0,
                "confidence": p[best],
                "probabilities": probabilities,
            }))
        }
        (Question::Choice { multiple: true, criteria, .. }, AnswerProbs::Multiple(ps)) => {
            let calibrated: Vec<f64> = ps
                .iter()
                .map(|p| calibration.through("multiple", *p))
                .collect();
            let choices: Vec<&str> = criteria
                .iter()
                .zip(&calibrated)
                .filter(|(_, p)| **p >= 0.5)
                .map(|((name, _), _)| name.as_str())
                .collect();
            let mut probabilities = serde_json::Map::new();
            for ((name, _), prob) in criteria.iter().zip(&calibrated) {
                probabilities.insert(name.clone(), (*prob).into());
            }
            Ok(serde_json::json!({
                "type": "choice",
                "choices": choices,
                "probabilities": probabilities,
            }))
        }
        (Question::Score { levels, .. }, AnswerProbs::Codes(p)) => {
            let best = p
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            // `api.answer`: the mean readout answers the unrounded expected
            // level; median/argmax answer a level index as float.
            let level = if score_readout == "mean" {
                p.iter().enumerate().map(|(i, v)| i as f64 * v).sum()
            } else {
                score_level(p, score_readout)?
            };
            let mut probabilities = serde_json::Map::new();
            let mut legend = serde_json::Map::new();
            for (i, (name, prob)) in levels.iter().zip(p).enumerate() {
                probabilities.insert(i.to_string(), (*prob).into());
                legend.insert(i.to_string(), name.clone().into());
            }
            Ok(serde_json::json!({
                "type": "score",
                "score": level,
                "confidence": p[best],
                "probabilities": probabilities,
                "legend": legend,
            }))
        }
        (Question::Estimate { grid, unit, .. }, AnswerProbs::Codes(p)) => {
            let value = grid.read(p, Some(&|share| calibration.level("estimate", share)));
            Ok(serde_json::json!({
                "type": "number",
                "number": value.median,
                "interval": [value.p10, value.p90],
                "unit": unit,
            }))
        }
        (
            Question::Read { kind, unit, .. },
            AnswerProbs::Read {
                found,
                span,
                text,
                path,
            },
        ) => {
            let found = calibration.through(&format!("found:{}", kind.name()), *found as f64);
            let present = found >= 0.5;
            let text = if present { text.clone() } else { None };
            let span = if present { *span } else { None };
            let mut result = serde_json::Map::new();
            result.insert("type".into(), kind.name().into());
            let mut value = match &text {
                Some(text) => match kind {
                    ReadKind::Text => serde_json::Value::String(text.clone()),
                    ReadKind::Number => {
                        let parsed = parsing::parse_number(text, context, unit.as_deref());
                        match parsed {
                            parsing::ParsedNumber::Int(i) => i.into(),
                            parsing::ParsedNumber::Float(f) => f.into(),
                            parsing::ParsedNumber::None => serde_json::Value::Null,
                        }
                    }
                    ReadKind::Date => {
                        let (date, _) = parsing::parse_date(text, context);
                        date.map(serde_json::Value::String).unwrap_or(serde_json::Value::Null)
                    }
                },
                None => serde_json::Value::Null,
            };
            let out_unit = match kind {
                ReadKind::Number => {
                    let written = text.as_deref().and_then(parsing::written_unit);
                    let u = unit.clone().or(written);
                    u.map(serde_json::Value::String).unwrap_or(serde_json::Value::Null)
                }
                _ => serde_json::Value::Null,
            };
            result.insert(kind.name().into(), std::mem::take(&mut value));
            result.insert(
                "span".into(),
                span.map(|(a, b)| serde_json::json!([a, b])).unwrap_or(serde_json::Value::Null),
            );
            result.insert("found".into(), found.into());
            if *kind != ReadKind::Text {
                result.insert(
                    "text".into(),
                    text.map(serde_json::Value::String).unwrap_or(serde_json::Value::Null),
                );
            }
            if present {
                if let Some(path) = path {
                    result.insert("path".into(), path.clone().into());
                }
            }
            if *kind == ReadKind::Number {
                result.insert("unit".into(), out_unit);
            }
            Ok(serde_json::Value::Object(result))
        }
        (Question::Box { .. }, AnswerProbs::Box { found, edges }) => {
            let found = calibration.through("found:box", *found);
            if found < 0.5 {
                return Ok(serde_json::json!({
                    "type": "box",
                    "box": null,
                    "found": found,
                    "confidence": 0.0,
                }));
            }
            let rect = medians(edges, edges_read)?;
            let confidence = calibration.through("box", confidence(edges, &rect));
            Ok(serde_json::json!({
                "type": "box",
                "box": rect_json(&rect),
                "found": found,
                "confidence": confidence,
            }))
        }
        (Question::Boxes { .. }, AnswerProbs::Boxes(blocks)) => {
            let mut presence = Vec::with_capacity(blocks.len());
            let mut below = 1.0f64;
            for block in blocks {
                below = below.min(calibration.through("found:boxes", block.found));
                presence.push(below);
            }
            let mut out = Vec::new();
            for (block, found) in blocks.iter().zip(&presence) {
                if *found < 0.5 {
                    break;
                }
                let rect = medians(&block.edges, edges_read)?;
                let confidence = calibration.through("boxes", confidence(&block.edges, &rect));
                out.push(serde_json::json!({
                    "box": rect_json(&rect),
                    "found": found,
                    "confidence": confidence,
                }));
            }
            out.sort_by(|a, b| {
                let la = a["box"]["left"].as_f64().unwrap_or(0.0);
                let lb = b["box"]["left"].as_f64().unwrap_or(0.0);
                la.total_cmp(&lb).then(
                    a["box"]["top"]
                        .as_f64()
                        .unwrap_or(0.0)
                        .total_cmp(&b["box"]["top"].as_f64().unwrap_or(0.0)),
                )
            });
            Ok(serde_json::json!({
                "type": "boxes",
                "boxes": out,
                "count": out.len(),
                "at_least": !out.is_empty() && out.len() == presence.len(),
                "presence": presence,
            }))
        }
        _ => Err(err("answer/question probability shape mismatch")),
    }
}

pub enum AnswerProbs {
    Codes(Vec<f64>),
    Multiple(Vec<f64>),
    Read {
        found: f32,
        span: Option<(usize, usize)>,
        text: Option<String>,
        path: Option<String>,
    },
    Box {
        found: f64,
        edges: Vec<Vec<f64>>,
    },
    Boxes(Vec<BoxBlock>),
}

pub struct BoxBlock {
    pub found: f64,
    pub edges: Vec<Vec<f64>>,
}

/// `api.score_level`.
fn score_level(probs: &[f64], readout: &str) -> Result<f64> {
    let total: f64 = probs.iter().sum();
    let total = if total == 0.0 { 1.0 } else { total };
    match readout {
        "mean" => {
            let mean = probs
                .iter()
                .enumerate()
                .map(|(i, p)| i as f64 * p / total)
                .sum::<f64>();
            Ok(mean.round().clamp(0.0, (probs.len() - 1) as f64))
        }
        "median" => {
            let mut mass = 0.0;
            for (i, p) in probs.iter().enumerate() {
                mass += p / total;
                if mass >= 0.5 {
                    return Ok(i as f64);
                }
            }
            Ok((probs.len() - 1) as f64)
        }
        "argmax" => Ok(probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as f64)
            .unwrap_or(0.0)),
        other => Err(err(format!("score_readout {other:?}: one of mean, median, argmax"))),
    }
}

// ---------------------------------------------------------------- boxes

/// The four edge names, in question/slot order (`box.EDGES`).
pub const EDGES: [&str; 4] = ["top", "bottom", "left", "right"];

pub struct Rect {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
}

fn rect_json(rect: &Rect) -> serde_json::Value {
    serde_json::json!({
        "top": rect.top,
        "bottom": rect.bottom,
        "left": rect.left,
        "right": rect.right,
    })
}

/// The grid an edge distribution is over, by its length (`box.read_as`).
fn read_as(distribution: &[f64]) -> Grid {
    if distribution.len() == 1000 {
        Grid::new(0.0, 0.999, 1000).unwrap()
    } else {
        Grid::new(0.0, 1.0, 128).unwrap()
    }
}

/// `box.ordered`: crossed edges become one bin about their middle.
fn ordered(a: f64, b: f64, grid: &Grid) -> (f64, f64) {
    if a < b {
        return (a, b);
    }
    let half = grid.step() / 2.0;
    let middle = ((a + b) / 2.0).max(half).min(1.0 - half);
    (middle - half, middle + half)
}

/// `box.crossing` (the `interp` edge read): where the distribution's CDF
/// crosses one half, mass spread evenly over each bin.
fn crossing(probabilities: &[f64], grid: &Grid) -> f64 {
    let total: f64 = probabilities.iter().sum();
    let values = grid.values();
    let mut mass = 0.0;
    for (value, p) in values.iter().zip(probabilities) {
        let share = p / total;
        if share > 0.0 && mass + share >= 0.5 {
            return (value - grid.step() / 2.0 + (0.5 - mass) / share * grid.step())
                .max(grid.low)
                .min(grid.high);
        }
        mass += share;
    }
    grid.high
}

/// `box.medians`: the box at the four edge distributions' medians.
pub fn medians(edges: &[Vec<f64>], read: &str) -> Result<Rect> {
    if read != "median" && read != "interp" {
        return Err(err(format!("box edges read as one of median, interp, not {read:?}")));
    }
    let grid = read_as(&edges[0]);
    let values: Vec<f64> = edges
        .iter()
        .map(|row| {
            if read == "interp" {
                crossing(row, &grid)
            } else {
                grid.read(row, None).median
            }
        })
        .collect();
    let (top, bottom) = ordered(values[0], values[1], &grid);
    let (left, right) = ordered(values[2], values[3], &grid);
    Ok(Rect {
        top,
        bottom,
        left,
        right,
    })
}

fn area(rect: &Rect) -> f64 {
    (rect.bottom - rect.top) * (rect.right - rect.left)
}

fn iou(a: &Rect, b: &Rect) -> f64 {
    let overlap = (a.bottom.min(b.bottom) - a.top.max(b.top)).max(0.0)
        * (a.right.min(b.right) - a.left.max(b.left)).max(0.0);
    overlap / (area(a) + area(b) - overlap)
}

/// `box.samples`: 512 Halton points in 4 dimensions (deterministic).
fn halton_samples() -> Vec<[f64; 4]> {
    fn radical_inverse(mut i: usize, base: usize) -> f64 {
        let mut value = 0.0;
        let mut scale = 1.0;
        while i > 0 {
            let digit = i % base;
            i /= base;
            scale /= base as f64;
            value += digit as f64 * scale;
        }
        value
    }
    (1..=512)
        .map(|i| {
            [
                radical_inverse(i, 2),
                radical_inverse(i, 3),
                radical_inverse(i, 5),
                radical_inverse(i, 7),
            ]
        })
        .collect()
}

/// `box.confidence`: the share of boxes drawn from the edge distributions
/// (through each one's CDF) with IoU >= 0.5 against `rect`.
pub fn confidence(edges: &[Vec<f64>], rect: &Rect) -> f64 {
    let cdfs: Vec<Vec<f64>> = edges
        .iter()
        .map(|row| {
            let total: f64 = row.iter().sum();
            let mut acc = 0.0;
            row.iter()
                .map(|p| {
                    acc += p / total;
                    acc
                })
                .collect()
        })
        .collect();
    let grid = read_as(&edges[0]);
    let values = grid.values();
    let hits = halton_samples()
        .iter()
        .filter(|point| {
            let edges_at: Vec<f64> = cdfs
                .iter()
                .zip(point.iter())
                .map(|(cdf, u)| {
                    let idx = cdf.partition_point(|c| *c < *u).min(grid.size - 1);
                    values[idx]
                })
                .collect();
            let (top, bottom) = ordered(edges_at[0], edges_at[1], &grid);
            let (left, right) = ordered(edges_at[2], edges_at[3], &grid);
            iou(
                &Rect {
                    top,
                    bottom,
                    left,
                    right,
                },
                rect,
            ) >= 0.5
        })
        .count();
    hits as f64 / 512.0
}
