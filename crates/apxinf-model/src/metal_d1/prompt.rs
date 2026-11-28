//! D1 prompt construction, mirroring the reference `prompt.py` +
//! `attention.pack_set` byte-for-byte: state rendering, question blocks,
//! option codes, slot markers, special-token breaking and set packing.
//!
//! Everything here is host-side string/token work against the frozen P0
//! prompts (`devlocal/d1-2b-port/fixtures/prompts/`): any divergence shows up
//! as a token-id mismatch before a forward ever runs.

use super::questions::{Grid, Question, ReadKind};
use apxinf_core::{Error, Result};
use apxinf_tokenizer::Tokenizer;

fn err(message: impl Into<String>) -> Error {
    Error::Other(format!("d1 prompt: {}", message.into()))
}

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const BREAK: char = '\u{200b}';
pub const STATE_STYLE: &str = "json_only";

const LETTERS: [char; 26] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
];

/// The family lead between the assistant header and the answer slot
/// (`LEADS["lfm3_vl"] = "<think></think>"`).
pub const LEAD: &str = "<think></think>";

// ---------------------------------------------------------------- floats

/// Python's `round()`: half to even.
pub fn py_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        x.trunc()
    } else {
        r
    }
}

/// Python's `repr`/`str` of a float, for JSON rendering and `:g` legends.
pub fn py_float(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let ax = x.abs();
    if x == x.trunc() && ax < 1e16 {
        return format!("{x:.1}");
    }
    if (1e-4..1e16).contains(&ax) {
        let s = format!("{x}");
        return if s.contains('.') { s } else { format!("{s}.0") };
    }
    // Exponent form, Python style: `1e+20`, `1.5e-07`.
    let s = format!("{x:e}");
    let (mantissa, exponent) = s.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let mantissa = if mantissa.contains('.') {
        mantissa.to_string()
    } else {
        format!("{mantissa}.0")
    };
    format!("{mantissa}e{}{:02}", if exponent < 0 { '-' } else { '+' }, exponent.abs())
}

/// Python's `f"{x:g}"` (6 significant digits, trailing zeros stripped).
pub fn g_format(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let ax = x.abs();
    let exp = ax.log10().floor() as i32;
    if !(-4..6).contains(&exp) {
        let s = format!("{x:.5e}");
        let (mantissa, exponent) = s.split_once('e').unwrap();
        let mantissa = mantissa
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let exponent: i32 = exponent.parse().unwrap();
        return format!("{mantissa}e{}{:02}", if exponent < 0 { '-' } else { '+' }, exponent.abs());
    }
    let precision = (5 - exp).max(0) as usize;
    let s = format!("{x:.precision$}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    s.to_string()
}

// ---------------------------------------------------------------- specials

/// The checkpoint's special-token strings that `plain()` must break in caller
/// text: `added_tokens` with `special: true`, longest first, mirroring
/// `prompt._specials_cached`.
pub fn specials_from_tokenizer_json(path: &std::path::Path) -> Result<Vec<String>> {
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path)
            .map_err(|e| err(format!("cannot read tokenizer.json: {e}")))?,
    )
    .map_err(|e| err(format!("cannot parse tokenizer.json: {e}")))?;
    let mut names: Vec<String> = json
        .get("added_tokens")
        .and_then(|a| a.as_array())
        .map(|tokens| {
            tokens
                .iter()
                .filter(|t| t.get("special").and_then(|s| s.as_bool()) == Some(true))
                .filter_map(|t| t.get("content").and_then(|c| c.as_str()).map(str::to_string))
                .filter(|c| c.chars().count() > 1 && !c.contains(BREAK))
                .collect()
        })
        .unwrap_or_default();
    names.sort_by_key(|b| std::cmp::Reverse(b.len()));
    Ok(names)
}

/// `plain` + `plain_offsets`: insert a zero-width space after the first
/// character of every special-token occurrence, leftmost-longest, until none
/// remain. With offsets, also return `boundaries`: escaped byte index →
/// original byte index (len = escaped.len() + 1).
pub fn plain_offsets(text: &str, specials: &[String]) -> (String, Vec<usize>) {
    let mut escaped = text.to_string();
    let mut boundaries: Vec<usize> = (0..=text.len()).collect();
    loop {
        // Leftmost-longest non-overlapping matches, like `re.finditer` over a
        // length-descending alternation.
        let mut matches: Vec<(usize, usize)> = Vec::new();
        let mut pos = 0;
        let bytes = escaped.as_bytes();
        while pos < bytes.len() {
            let mut found = None;
            for special in specials {
                if bytes[pos..].starts_with(special.as_bytes()) {
                    found = Some((pos, pos + special.len()));
                    break;
                }
            }
            match found {
                Some((start, end)) => {
                    matches.push((start, end));
                    pos = end;
                }
                None => pos += 1,
            }
        }
        if matches.is_empty() {
            break;
        }
        // Rebuild with the break after each match's first character, mapping
        // boundaries at every inserted position.
        let mut out = String::with_capacity(escaped.len() + matches.len() * 3);
        let mut new_boundaries = Vec::with_capacity(boundaries.len() + matches.len());
        let mut last = 0;
        for (start, end) in matches {
            out.push_str(&escaped[last..start]);
            new_boundaries.extend_from_slice(&boundaries[last..start]);
            let first_len = escaped[start..end].chars().next().unwrap().len_utf8();
            out.push_str(&escaped[start..start + first_len]);
            new_boundaries.extend_from_slice(&boundaries[start..start + first_len]);
            out.push(BREAK);
            // The inserted break maps to the original position after the first
            // character (it sits between that char and the rest).
            new_boundaries.push(boundaries[start + first_len]);
            out.push_str(&escaped[start + first_len..end]);
            new_boundaries.extend_from_slice(&boundaries[start + first_len..end]);
            last = end;
        }
        out.push_str(&escaped[last..]);
        new_boundaries.extend_from_slice(&boundaries[last..]);
        escaped = out;
        boundaries = new_boundaries;
    }
    (escaped, boundaries)
}

pub fn plain(text: &str, specials: &[String]) -> String {
    plain_offsets(text, specials).0
}

// ---------------------------------------------------------------- state

/// `prompt.state_block` for the default `json_only` style: the caller's state
/// as `json.dumps(state, ensure_ascii=False, indent=2)` + `\n\n`; a plain
/// string state passes through with the same trailer.
pub fn state_block(state: &serde_json::Value) -> Result<String> {
    match state {
        serde_json::Value::String(text) => Ok(format!("{text}\n\n")),
        serde_json::Value::Null => Err(err("a null state has no state block")),
        other => Ok(format!("{}\n\n", json_dumps_indent2(other)?)),
    }
}

/// The state as a read's span offsets index it: `state_block` minus the
/// trailing blank line.
pub fn state_text(state: &serde_json::Value) -> Result<String> {
    let block = state_block(state)?;
    Ok(block[..block.len() - 2].to_string())
}

/// `json.dumps(value, ensure_ascii=False, indent=2)`.
pub fn json_dumps_indent2(value: &serde_json::Value) -> Result<String> {
    let mut out = String::new();
    write_json(value, 0, &mut out)?;
    Ok(out)
}

fn write_json(value: &serde_json::Value, depth: usize, out: &mut String) -> Result<()> {
    let indent = |out: &mut String, n: usize| out.push_str(&"  ".repeat(n));
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(f) = n.as_f64() {
                if !f.is_finite() {
                    return Err(err("JSON states cannot carry NaN or Infinity"));
                }
                out.push_str(&py_float(f));
            }
        }
        serde_json::Value::String(s) => write_json_string(s, out),
        serde_json::Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                indent(out, depth + 1);
                write_json(item, depth + 1, out)?;
                if i + 1 < items.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            indent(out, depth);
            out.push(']');
        }
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            out.push_str("{\n");
            for (i, (key, item)) in map.iter().enumerate() {
                indent(out, depth + 1);
                write_json_string(key, out);
                out.push_str(": ");
                write_json(item, depth + 1, out)?;
                if i + 1 < map.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            indent(out, depth);
            out.push('}');
        }
    }
    Ok(())
}

/// `json.dumps` of one string (Python escaping: the control set plus `\"`/`\\`).
fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

// ---------------------------------------------------------------- options

/// `prompt.option_codes`: the labels themselves when they already are single
/// letters, else A..Z, else the scheme's codes (2: AA.. after Z; 1: `00`..).
pub fn option_codes(labels: &[String], scheme: u32) -> Vec<String> {
    let labs: Vec<String> = labels.iter().map(|x| x.trim().to_string()).collect();
    if !labs.is_empty()
        && labs
            .iter()
            .all(|k| k.chars().count() == 1 && k.chars().next().is_some_and(char::is_alphabetic))
    {
        return labs;
    }
    if labs.len() <= 26 {
        return LETTERS[..labs.len()].iter().map(|c| c.to_string()).collect();
    }
    if scheme == 2 {
        let mut codes: Vec<String> = LETTERS.iter().map(|c| c.to_string()).collect();
        for a in LETTERS {
            for b in LETTERS {
                codes.push(format!("{a}{b}"));
            }
        }
        return codes[..labs.len()].to_vec();
    }
    (0..labs.len()).map(|i| format!("{i:02}")).collect()
}

/// One option's distinct single-token code: `[(code, token_id)]`, with the
/// fallback pool when a code does not tokenize to one unused id
/// (`prompt.aliases`).
pub fn aliases(tokenizer: &Tokenizer, labels: &[String], scheme: u32) -> Result<Vec<(String, u32)>> {
    let codes = option_codes(labels, scheme);
    let mut used = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(codes.len());
    let mut take = |raw: &str| -> Result<Option<u32>> {
        let ids = tokenizer
            .encode(raw)
            .map_err(|e| err(format!("encode {raw:?}: {e}")))?;
        if ids.len() == 1 && !used.contains(&ids[0]) {
            used.insert(ids[0]);
            Ok(Some(ids[0]))
        } else {
            Ok(None)
        }
    };
    let mut pool: Vec<String> = Vec::new();
    for code in &codes {
        let id = match take(code)? {
            Some(id) => id,
            None => {
                if pool.is_empty() {
                    pool = fallback_pool();
                }
                let mut found = None;
                for raw in &pool {
                    if let Some(id) = take(raw)? {
                        found = Some(id);
                        break;
                    }
                }
                found.ok_or_else(|| {
                    err(format!("no single-token alias left for {} options", codes.len()))
                })?
            }
        };
        out.push((code.clone(), id));
    }
    Ok(out)
}

fn fallback_pool() -> Vec<String> {
    let mut pool: Vec<String> = LETTERS.iter().map(|c| c.to_string()).collect();
    pool.extend((0..100).map(|i| format!("{i:02}")));
    pool.extend(('a'..='z').map(|c| c.to_string()));
    pool.extend((0..200).map(|i| format!("#{i}")));
    for a in LETTERS {
        for b in LETTERS {
            pool.push(format!("{a}{b}"));
        }
    }
    pool
}

// ---------------------------------------------------------------- blocks

/// The box-edge grid and scale (`box.edge_grid`): thousandths where the
/// checkpoint reads numbers and the tokenizer writes each in one token.
pub fn edge_grid(tokenizer: &Tokenizer, d1_numbers: bool) -> (Grid, u64) {
    let thousandths = Grid::new(0.0, 0.999, 1000).unwrap();
    if d1_numbers && number_ids(tokenizer, &thousandths, 1000).is_some() {
        (thousandths, 1000)
    } else {
        (Grid::new(0.0, 1.0, 128).unwrap(), 1)
    }
}

pub fn edge_legend(tokenizer: &Tokenizer, d1_numbers: bool, scheme: u32) -> String {
    let (grid, scale) = edge_grid(tokenizer, d1_numbers);
    if scale > 1 {
        return "Each edge is a number from 0 to 999, in thousandths of the picture: top and \
                bottom from its top edge, left and right from its left edge."
            .into();
    }
    // `grid_question`: Choice over criteria keys "0".."N-1" (size 128 > 10).
    let labels: Vec<String> = (0..grid.size).map(|i| i.to_string()).collect();
    let codes = option_codes(&labels, scheme);
    format!(
        "Each edge is a code on a {}-value scale: {} is 0 (top or left), {} is 1 (bottom or \
         right). Values are equally spaced.",
        grid.size,
        codes[0],
        codes[codes.len() - 1]
    )
}

/// `prompt.number_ids`: each grid value times `scale` as the tokenizer writes
/// it, plain and space-prefixed, one token a form; `None` when a value is no
/// integer or takes more than one token.
pub fn number_ids(tokenizer: &Tokenizer, grid: &Grid, scale: u64) -> Option<Vec<Vec<u32>>> {
    let mut groups = Vec::with_capacity(grid.size);
    for value in grid.values() {
        let n = py_round(value * scale as f64);
        if (n - value * scale as f64).abs() > 1e-6 {
            return None;
        }
        let plain_form = tokenizer.encode(&format!("{}", n as i64)).ok()?;
        let spaced_form = tokenizer.encode(&format!(" {}", n as i64)).ok()?;
        if plain_form.len() != 1 {
            return None;
        }
        let mut group = Vec::new();
        for ids in [plain_form, spaced_form] {
            if ids.len() == 1 && !group.contains(&ids[0]) {
                group.push(ids[0]);
            }
        }
        if group.is_empty() {
            return None;
        }
        groups.push(group);
    }
    Some(groups)
}

/// `prompt.question_block`: the question text as the prompt writes it.
pub fn question_block(
    tokenizer: &Tokenizer,
    q: &Question,
    d1_numbers: bool,
    prompt_version: u32,
    codes_scheme: u32,
) -> Result<String> {
    match q {
        Question::Box { instructions } => {
            let (_, scale) = edge_grid(tokenizer, d1_numbers);
            let one = if scale > 1 { "one number" } else { "one code" };
            Ok(format!(
                "{instructions}\n\nPresence is yes or no. {} Reply with yes or no at present, then {one} at top, bottom, left and right.",
                edge_legend(tokenizer, d1_numbers, codes_scheme)
            ))
        }
        Question::Boxes { instructions, .. } => {
            let most = if prompt_version == 1 { "16 at most, " } else { "" };
            Ok(format!(
                "{instructions}\n\nEvery one of them, {most}in reading order (left edge, then top). Present k is yes when there are at least k. {} Reply with yes or no at each present, then its top, bottom, left and right.",
                edge_legend(tokenizer, d1_numbers, codes_scheme)
            ))
        }
        Question::Estimate {
            instructions,
            grid,
            unit,
            ..
        } => {
            let unit = unit.as_deref().map(|u| format!(" {u}")).unwrap_or_default();
            if grid_slot_is_number(tokenizer, d1_numbers, grid, 1) {
                Ok(format!(
                    "{instructions}\n\nEstimate from {} to {}{unit}, in steps of {}. Reply with the number only.",
                    g_format(grid.low),
                    g_format(grid.high),
                    g_format(grid.step())
                ))
            } else {
                let size = grid.size;
                let codes: Vec<String> = if size <= 10 {
                    (0..size).map(|i| i.to_string()).collect()
                } else {
                    // `grid_question`: Choice with criteria keys "0".."N-1".
                    let labels: Vec<String> = (0..size).map(|i| i.to_string()).collect();
                    aliases(tokenizer, &labels, codes_scheme)?.into_iter().map(|(c, _)| c).collect()
                };
                Ok(format!(
                    "{instructions}\n\nEstimate from {} to {}{unit} on {} equally spaced values, step {}: {} is {}, {} is {}. Reply with the code only.",
                    g_format(grid.low),
                    g_format(grid.high),
                    size,
                    g_format(grid.step()),
                    codes[0],
                    g_format(grid.low),
                    codes[size - 1],
                    g_format(grid.high)
                ))
            }
        }
        Question::Read {
            instructions,
            kind,
            unit,
        } => {
            let unit = match (kind, unit) {
                (ReadKind::Number, Some(u)) => format!(" (unit: {u})"),
                _ => String::new(),
            };
            Ok(format!(
                "{instructions}\n\nRead the {}{unit} from the state, or nothing if it is absent.",
                kind.name()
            ))
        }
        Question::Choice {
            instructions,
            criteria,
            multiple,
        } => {
            if *multiple {
                let mut lines = Vec::with_capacity(criteria.len());
                for (name, desc) in criteria {
                    let quoted = json_dumps_indent2(&serde_json::Value::String(name.clone()))?;
                    lines.push(format!(
                        "{quoted}: {}",
                        desc.clone().unwrap_or_else(|| name.clone())
                    ));
                }
                Ok(format!(
                    "{instructions}\n\nOptions:\n{}\n\nReply with yes or no for each option.",
                    lines.join("\n")
                ))
            } else {
                let labels: Vec<String> = criteria.iter().map(|(name, _)| name.clone()).collect();
                let codes = aliases(tokenizer, &labels, codes_scheme)?;
                let lines: Vec<String> = criteria
                    .iter()
                    .zip(&codes)
                    .map(|((label, desc), (code, _))| {
                        let text = desc.clone().unwrap_or_else(|| label.replace('_', " "));
                        format!("{code} {text}")
                    })
                    .collect();
                Ok(format!(
                    "{instructions}\n\nOptions:\n{}\n\nReply with the option code only.",
                    lines.join("\n")
                ))
            }
        }
        Question::Noul {
            instructions,
            criteria,
        } => {
            let extra = match criteria {
                Some((yes, no)) => format!(
                    "\nYes: {}\nNo: {}",
                    yes.as_deref().unwrap_or("None"),
                    no.as_deref().unwrap_or("None")
                ),
                None => String::new(),
            };
            Ok(format!("{instructions}{extra}\n\nReply with yes or no only."))
        }
        Question::Score {
            instructions,
            levels,
        } => {
            let legend: Vec<String> = levels
                .iter()
                .enumerate()
                .map(|(i, name)| format!("{i} {name}"))
                .collect();
            Ok(format!(
                "{instructions}\n\n{}\n\nReply with a single digit 0-{} only.",
                legend.join("\n"),
                levels.len() - 1
            ))
        }
    }
}

/// Whether an estimate's grid resolves to number slots (`prompt.grid_slot`).
fn grid_slot_is_number(tokenizer: &Tokenizer, d1_numbers: bool, grid: &Grid, scale: u64) -> bool {
    d1_numbers && number_ids(tokenizer, grid, scale).is_some()
}

// ---------------------------------------------------------------- layout

pub fn question_header(state_is_null: bool) -> &'static str {
    if state_is_null {
        ""
    } else {
        "QUESTION:\n"
    }
}

/// `prompt.prefix_text` for a text-only request (no media markup): bos, the
/// user turn, the (special-broken) state block and the question header.
pub fn prefix_text(
    tokenizer: &Tokenizer,
    specials: &[String],
    bos: &str,
    state: Option<&serde_json::Value>,
) -> Result<String> {
    let _ = tokenizer;
    match state {
        None => Ok(format!("{bos}{IM_START}user\n")),
        Some(state) => {
            let block = state_block(state)?;
            let escaped = plain(&block, specials);
            Ok(format!(
                "{bos}{IM_START}user\n{escaped}\n{}",
                question_header(false)
            ))
        }
    }
}

/// One question's suffix text and its slot offsets from the suffix's last
/// token (`prompt.suffix_layout` + `marked_suffix`).
pub fn suffix_layout(
    tokenizer: &Tokenizer,
    specials: &[String],
    q: &Question,
    d1_numbers: bool,
    prompt_version: u32,
    codes_scheme: u32,
) -> Result<(String, Vec<i64>)> {
    let body = question_block(tokenizer, q, d1_numbers, prompt_version, codes_scheme)?;
    match q {
        Question::Read { .. } => marked_suffix(tokenizer, specials, &body, &["start", "end"], ""),
        Question::Box { .. } => marked_suffix(
            tokenizer,
            specials,
            &body,
            &["present", "top", "bottom", "left", "right"],
            "",
        ),
        Question::Boxes { max, .. } => {
            let blocks = max.unwrap_or(16);
            let mut names = Vec::with_capacity(blocks * 5);
            for k in 1..=blocks {
                for name in ["present", "top", "bottom", "left", "right"] {
                    names.push(format!("{name} {k}"));
                }
            }
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let after = if blocks < 16 || prompt_version > 1 { "\n" } else { "" };
            marked_suffix(tokenizer, specials, &body, &refs, after)
        }
        Question::Choice {
            multiple: true,
            criteria,
            ..
        } => {
            let names: Vec<String> = criteria
                .iter()
                .map(|(name, _)| json_dumps_indent2(&serde_json::Value::String(name.clone())))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            marked_suffix(tokenizer, specials, &body, &refs, "")
        }
        _ => Ok((
            format!("{}{IM_END}\n{IM_START}assistant\n{LEAD}", plain(&body, specials)),
            vec![0],
        )),
    }
}

/// `prompt.marked_suffix`: the body followed by named `name:` markers; slots
/// are the colon tokens, given as offsets from the sequence's last token.
fn marked_suffix(
    tokenizer: &Tokenizer,
    specials: &[String],
    body: &str,
    names: &[&str],
    after: &str,
) -> Result<(String, Vec<i64>)> {
    if names.is_empty()
        || names.iter().collect::<std::collections::HashSet<_>>().len() != names.len()
        || names.iter().any(|n| n.is_empty() || n.contains('\n'))
    {
        return Err(err("slot names must be distinct nonempty single lines"));
    }
    let mut text = format!("{}{IM_END}\n{IM_START}assistant\n{LEAD}", plain(body, specials));
    let mut ends = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            text.push('\n');
        }
        text.push_str(&plain(name, specials));
        text.push(':');
        ends.push(text.len());
    }
    if !after.is_empty() {
        let ids = tokenizer
            .encode(&format!("{text}{after}"))
            .map_err(|e| err(format!("encode suffix tail: {e}")))?;
        let last = &ids[ids.len() - 1..];
        let decoded = tokenizer
            .decode_with_options(last, false)
            .map_err(|e| err(format!("decode suffix tail: {e}")))?;
        if decoded.trim_end().ends_with(':') {
            text.push_str(after);
        }
    }
    let (ids, offsets) = tokenizer
        .encode_with_offsets(&text)
        .map_err(|e| err(format!("encode suffix: {e}")))?;
    let mut slots = Vec::with_capacity(ends.len());
    for chars in ends {
        let matching: Vec<usize> = offsets
            .iter()
            .enumerate()
            .filter(|(_, (start, stop))| *start <= chars - 1 && chars - 1 < *stop)
            .map(|(i, _)| i)
            .collect();
        if matching.len() != 1 {
            return Err(err("a slot marker must have an exact colon token position"));
        }
        let end = matching[0];
        let stop = offsets[end].1;
        // The colon may share its token with following whitespace, never with
        // the next marker's text.
        if !text[chars..stop].trim().is_empty() {
            return Err(err(
                "a slot marker token must end with its colon and optional whitespace",
            ));
        }
        let decoded = tokenizer
            .decode_with_options(&ids[end..end + 1], false)
            .map_err(|e| err(format!("decode slot token: {e}")))?;
        if !decoded.trim_end().ends_with(':') {
            return Err(err(
                "a slot marker token must end with its colon and optional whitespace",
            ));
        }
        slots.push(end as i64 - ids.len() as i64 + 1);
    }
    Ok((text, slots))
}

// ---------------------------------------------------------------- packing

/// `attention.pack_set`: `[prefix + suffix0][header + suffix1]...`, the state
/// length, each token's question (1..=N, -1 on the state) and each question's
/// absolute answer-slot positions.
pub struct PackedSequence {
    pub ids: Vec<u32>,
    pub state_len: usize,
    pub question: Vec<i32>,
    pub ends: Vec<Vec<usize>>,
}

pub fn pack_set(
    tokenizer: &Tokenizer,
    prefix: &str,
    suffixes: &[String],
    header: &str,
    slot_offsets: &[Vec<i64>],
) -> Result<PackedSequence> {
    if suffixes.is_empty() {
        return Err(err("a question set cannot be empty"));
    }
    if slot_offsets.len() != suffixes.len() {
        return Err(err("one slot layout is required per question"));
    }
    let encode = |text: &str| -> Result<Vec<u32>> {
        tokenizer
            .encode(text)
            .map_err(|e| err(format!("encode: {e}")))
    };
    let positions = |own: &[i64], length: usize, end: usize| -> Result<Vec<usize>> {
        if own.is_empty()
            || own
                .iter()
                .any(|&o| o <= -(length as i64) || o > 0)
            || own.last() != Some(&0)
            || {
                let mut sorted = own.to_vec();
                sorted.sort_unstable();
                sorted.dedup();
                sorted != own
            }
        {
            return Err(err(
                "slots must be increasing positions inside their question, ending at its last token",
            ));
        }
        Ok(own.iter().map(|&o| (end as i64 + o) as usize).collect())
    };
    let mut ids = encode(&format!("{prefix}{}", suffixes[0]))?;
    let state_len = ids.len() - encode(&suffixes[0])?.len();
    let mut question = vec![-1; state_len];
    question.extend(std::iter::repeat(1).take(ids.len() - state_len));
    let mut ends = vec![positions(&slot_offsets[0], ids.len() - state_len, ids.len() - 1)?];
    for (k, suffix) in suffixes.iter().enumerate().skip(1) {
        let own = encode(&format!("{header}{suffix}"))?;
        let start = ids.len();
        ids.extend(own);
        question.extend(std::iter::repeat(k as i32 + 1).take(ids.len() - start));
        ends.push(positions(
            &slot_offsets[k],
            ids.len() - start,
            ids.len() - 1,
        )?);
    }
    Ok(PackedSequence {
        ids,
        state_len,
        question,
        ends,
    })
}
