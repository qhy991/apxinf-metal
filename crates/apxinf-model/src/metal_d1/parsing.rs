//! Strict readings of written values — a faithful std-only port of the
//! reference `parsing.py` (no arithmetic, unit conversion or invented date
//! components).
//!
//! Scope note: `_fold` approximates Python's `unicodedata.casefold` + NFKD
//! combining-mark strip with a lowercase map plus an explicit accent table
//! covering the characters the month/weekday/currency tables use. Everything
//! else (matchers, grouping rules, scales, date orders) mirrors the Python
//! semantics one for one.

/// A parsed written magnitude: Python returns `int` when the value is
/// integral, else a finite `float`, else `None`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParsedNumber {
    Int(i64),
    Float(f64),
    None,
}

// ---------------------------------------------------------------- month tables

/// `_MONTHS`: month names (accents folded; genitive forms a date uses).
const MONTH_NAMES: [&[&str]; 12] = [
    &["january", "janvier", "januar", "enero", "gennaio", "janeiro", "januari", "stycznia", "styczen", "января"],
    &["february", "fevrier", "februar", "febrero", "febbraio", "fevereiro", "februari", "lutego", "luty", "февраля"],
    &["march", "mars", "marz", "marzo", "marco", "maart", "marca", "марта"],
    &["april", "avril", "abril", "aprile", "kwietnia", "kwiecien", "апреля"],
    &["may", "mai", "mayo", "maggio", "maio", "mei", "maj", "maja", "мая"],
    &["june", "juin", "juni", "junio", "giugno", "junho", "czerwca", "czerwiec", "июня"],
    &["july", "juillet", "juli", "julio", "luglio", "julho", "lipca", "lipiec", "июля"],
    &["august", "aout", "agosto", "augustus", "augusti", "sierpnia", "sierpien", "августа"],
    &["september", "septembre", "septiembre", "settembre", "setembro", "wrzesnia", "wrzesien", "сентября"],
    &["october", "octobre", "oktober", "octubre", "ottobre", "outubro", "pazdziernika", "pazdziernik", "октября"],
    &["november", "novembre", "noviembre", "novembro", "listopada", "listopad", "ноября"],
    &["december", "decembre", "dezember", "diciembre", "dicembre", "dezembro", "grudnia", "grudzien", "декабря"],
];

const WEEKDAYS: &[&str] = &[
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "mon", "tue",
    "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun", "lundi", "mardi", "mercredi",
    "jeudi", "vendredi", "samedi", "dimanche", "montag", "dienstag", "mittwoch", "donnerstag",
    "freitag", "samstag", "sonntag", "lunes", "martes", "miercoles", "jueves", "viernes", "sabado",
    "domingo", "lunedi", "martedi", "mercoledi", "giovedi", "venerdi", "sabato", "domenica",
    "segunda", "terca", "quarta", "quinta", "sexta",
];

fn months() -> &'static std::collections::HashMap<String, u32> {
    use std::sync::OnceLock;
    static MONTHS: OnceLock<std::collections::HashMap<String, u32>> = OnceLock::new();
    MONTHS.get_or_init(|| {
        let mut map = std::collections::HashMap::new();
        for (i, names) in MONTH_NAMES.iter().enumerate() {
            for name in *names {
                map.insert(name.to_string(), i as u32 + 1);
            }
        }
        // `_ABBREVIATIONS`: 3-letter prefixes that map to exactly one month.
        let mut abbrevs: std::collections::HashMap<String, Vec<u32>> =
            std::collections::HashMap::new();
        for (name, month) in map.iter() {
            if name.len() >= 3 {
                let prefix = name.chars().take(3).collect::<String>();
                abbrevs.entry(prefix).or_default().push(*month);
            }
        }
        for (prefix, found) in abbrevs {
            if found.len() == 1 {
                map.insert(prefix, found[0]);
            }
        }
        for (name, month) in [("janv", 1), ("fevr", 2), ("juil", 7), ("sept", 9)] {
            map.insert(name.to_string(), month);
        }
        map
    })
}

// ---------------------------------------------------------------- fold

/// `_fold`: casefold + NFKD combining-mark strip, approximated for the
/// characters the tables use.
fn fold(text: &str) -> String {
    text.chars()
        .flat_map(|c| c.to_lowercase())
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ä' | 'ã' | 'å' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'ö' | 'õ' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ý' | 'ÿ' => 'y',
            'ñ' => 'n',
            'ç' => 'c',
            'ß' => 's',
            'ă' => 'a',
            'ș' | 'ś' => 's',
            'ț' => 't',
            'ź' | 'ż' => 'z',
            'ć' => 'c',
            'ł' => 'l',
            'ń' => 'n',
            'ę' => 'e',
            'ą' => 'a',
            c => c,
        })
        .collect()
}

// ---------------------------------------------------------------- currencies

/// `CURRENCIES` by decimal mark: `""` whole units, `None` both in use.
const CURRENCIES_DOT: &[&str] = &[
    "$", "£", "₹", "USD", "GBP", "CAD", "AUD", "NZD", "INR", "Rs.", "Rs", "MXN", "CNY", "RMB", "元", "HKD", "SGD",
];
const CURRENCIES_COMMA: &[&str] = &["R$", "BRL", "₽", "RUB", "zł", "PLN", "kr", "SEK", "NOK", "DKK", "₺", "TRY"];
const CURRENCIES_NONE: &[&str] = &["¥", "￥", "円", "₩", "₫", "Rp.", "Rp", "IDR", "JPY", "KRW", "VND", "CLP"];
const CURRENCIES_BOTH: &[&str] = &["€", "EUR", "CHF"];

fn currency_names() -> &'static Vec<(String, String, Option<&'static str>)> {
    use std::sync::OnceLock;
    static NAMES: OnceLock<Vec<(String, String, Option<&'static str>)>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let mut out: Vec<(String, String, Option<&'static str>)> = Vec::new();
        for (mark, list) in [
            (Some("."), CURRENCIES_DOT),
            (Some(","), CURRENCIES_COMMA),
            (Some(""), CURRENCIES_NONE),
            (None, CURRENCIES_BOTH),
        ] {
            for name in list {
                out.push((name.to_lowercase(), name.to_string(), mark));
            }
        }
        // Longest first, as the reference's alternation.
        out.sort_by_key(|entry| std::cmp::Reverse(entry.0.len()));
        out
    })
}

/// `_CURRENCY` matches: markers sorted longest-first; an ASCII-letter marker
/// needs a non-letter boundary on the right (`\bCUR(?![a-z])`) and left.
fn currency_matches(text: &str) -> Vec<(usize, usize, String, Option<&'static str>)> {
    let mut out = Vec::new();
    let folded = text.to_lowercase();
    for (lower, original, mark) in currency_names() {
        let needle = lower.as_bytes();
        let bytes = folded.as_bytes();
        let mut at = 0;
        while at + needle.len() <= bytes.len() {
            if bytes[at..at + needle.len()] == *needle {
                let ascii_alpha_start = lower.chars().next().is_some_and(|c| c.is_ascii_alphabetic());
                let left_ok =
                    !ascii_alpha_start || at == 0 || !bytes[at - 1].is_ascii_lowercase();
                let right_ok = !ascii_alpha_start
                    || at + needle.len() >= bytes.len()
                    || !bytes[at + needle.len()].is_ascii_lowercase();
                if left_ok && right_ok {
                    out.push((at, at + needle.len(), original.clone(), *mark));
                    at += needle.len();
                    continue;
                }
            }
            at += 1;
        }
    }
    out.sort_by_key(|(start, _, _, _)| *start);
    out
}

fn is_currency(text: &str) -> bool {
    let trimmed = text.trim().to_lowercase();
    currency_names().iter().any(|(lower, _, _)| *lower == trimmed)
}

/// `written_unit`: only an explicitly written percent or currency marker.
pub fn written_unit(text: &str) -> Option<String> {
    if text.trim_end().ends_with('%') {
        return Some("%".into());
    }
    let matches = currency_matches(text);
    if matches.len() == 1 {
        return Some(matches[0].2.clone());
    }
    None
}

fn currency_marks(text: &str) -> std::collections::HashSet<&'static str> {
    currency_matches(text)
        .into_iter()
        .filter_map(|(_, _, _, mark)| mark)
        .collect()
}

// ---------------------------------------------------------------- scales

const SCALES: &[(&str, i64)] = &[
    ("thousand", 1_000), ("mil", 1_000), ("tausend", 1_000), ("тыс", 1_000), ("千", 1_000),
    ("万", 10_000), ("萬", 10_000),
    ("lakh", 100_000), ("lakhs", 100_000),
    ("crore", 10_000_000), ("crores", 10_000_000),
    ("million", 1_000_000), ("millions", 1_000_000), ("mio", 1_000_000), ("mln", 1_000_000),
    ("millón", 1_000_000), ("millones", 1_000_000), ("milhão", 1_000_000), ("milhões", 1_000_000),
    ("milione", 1_000_000), ("milioni", 1_000_000), ("млн", 1_000_000),
    ("亿", 100_000_000), ("億", 100_000_000),
    ("billion", 1_000_000_000), ("billions", 1_000_000_000), ("milliard", 1_000_000_000),
    ("milliards", 1_000_000_000), ("milliarde", 1_000_000_000), ("milliarden", 1_000_000_000),
    ("mrd", 1_000_000_000), ("miliardo", 1_000_000_000), ("miliardi", 1_000_000_000),
    ("млрд", 1_000_000_000),
    ("trillion", 1_000_000_000_000),
];

/// `_SCALE.search`: `\s*(word)\.?$` at the end, case-folded, longest first.
fn scale_word(text: &str) -> Option<(i64, usize)> {
    let folded = fold(text);
    let mut words: Vec<(&str, i64)> = SCALES.to_vec();
    words.sort_by_key(|(w, _)| std::cmp::Reverse(w.len()));
    for (word, value) in words {
        if let Some(prefix) = folded.strip_suffix(&format!("{word}.")) {
            return Some((value, prefix.len()));
        }
        if let Some(prefix) = folded.strip_suffix(word) {
            return Some((value, prefix.len()));
        }
    }
    None
}

/// `_SUFFIX.search`: `(?<=\d)(k|m|bn|b)$` case-insensitive.
fn suffix_scale(text: &str) -> Option<(i64, usize)> {
    let folded = fold(text);
    let candidates: Vec<(&str, i64)> =
        vec![("bn", 1_000_000_000), ("k", 1_000), ("m", 1_000_000), ("b", 1_000_000_000)];
    for (suffix, value) in candidates {
        if let Some(start) = folded.len().checked_sub(suffix.len()) {
            if &folded[start..] == suffix && start > 0 && folded.as_bytes()[start - 1].is_ascii_digit() {
                return Some((value, start));
            }
        }
    }
    None
}

const COUNTED_WORDS: &[&str] = &[
    "money", "price", "invoice", "revenue", "cost", "count", "quantity", "population", "people",
    "items", "units", "users", "downloads",
];

fn counted_context(context: &str) -> bool {
    let folded = fold(context);
    let tokens: Vec<&str> = folded
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    COUNTED_WORDS.iter().any(|word| tokens.contains(word))
}

// ---------------------------------------------------------------- digits

fn digits_of(text: &str, written: &str, context: &str) -> Option<String> {
    // `\d[\d., '’]*` (NBSP/narrow NBSP already normalized away before calls).
    let bytes = text.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_digit() {
        return None;
    }
    if !bytes
        .iter()
        .all(|&b| b.is_ascii_digit() || matches!(b, b'.' | b',' | b' ' | b'\'') || b == 0xE2)
    {
        return None;
    }
    // A space or an apostrophe only groups: `\d{1,3}([ '’]\d{3})+([.,]\d+)?`.
    if text.bytes().any(|b| matches!(b, b' ' | b'\'' | 0xE2)) {
        return group_digits(text, written, context);
    }
    // Indian lakh grouping `12,34,567.89` beside a rupee word.
    let parts: Vec<&str> = text.split(',').collect();
    let lakh_shape = parts.len() >= 3
        && !parts[0].is_empty()
        && parts[0].len() <= 2
        && parts[0].bytes().all(|b| b.is_ascii_digit())
        && parts[1..parts.len() - 1]
            .iter()
            .all(|p| p.len() == 2 && p.bytes().all(|b| b.is_ascii_digit()))
        && {
            let last = parts[parts.len() - 1];
            let (head, tail) = match last.split_once('.') {
                Some((h, t)) => (h, Some(t)),
                None => (last, None),
            };
            head.len() == 3
                && head.bytes().all(|b| b.is_ascii_digit())
                && tail.is_none_or(|t| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit()))
        };
    if lakh_shape {
        let near = fold(&format!("{written}\n{context}"));
        if near.contains('₹')
            || near
                .split(|c: char| !c.is_alphanumeric())
                .any(|tok| matches!(tok, "inr" | "rs" | "lakh" | "crore"))
        {
            return Some(text.replace(',', ""));
        }
    }
    if text.contains('.') && text.contains(',') {
        let (point, group) = if text.rfind('.') > text.rfind(',') {
            ('.', ',')
        } else {
            (',', '.')
        };
        // The group side must be complete threes: `\d{1,3}(g\d{3})+p\d+`.
        let groups: Vec<&str> = text.split(group).collect();
        let last = groups[groups.len() - 1];
        let (last_head, frac) = match last.split_once(point) {
            Some((h, f)) => (h, Some(f)),
            None => (last, None),
        };
        if groups.len() < 2
            || groups[0].is_empty()
            || groups[0].len() > 3
            || !groups[0].bytes().all(|b| b.is_ascii_digit())
            || !groups[1..groups.len() - 1]
                .iter()
                .all(|p| p.len() == 3 && p.bytes().all(|b| b.is_ascii_digit()))
            || last_head.len() != 3
            || !last_head.bytes().all(|b| b.is_ascii_digit())
            || frac.is_some_and(|f| f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()))
        {
            return None;
        }
        // The point is the decimal mark: `int` = every complete group plus
        // the last group's head, `frac` = the fraction digits.
        let mut body = groups[..groups.len() - 1].concat();
        body.push_str(last_head);
        if let Some(frac) = frac {
            body.push('.');
            body.push_str(frac);
        }
        return Some(body);
    }
    if !text.contains('.') && !text.contains(',') {
        return Some(text.to_string());
    }
    let mark = if text.contains('.') { '.' } else { ',' };
    let parts: Vec<&str> = text.split(mark).collect();
    if parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    let point = decimal_mark(written, context);
    if parts.len() > 2 || (point.is_some() && point != Some(mark)) {
        // The mark groups: complete threes.
        if !parts[0].is_empty()
            && parts[0].len() <= 3
            && parts[1..]
                .iter()
                .all(|p| p.len() == 3 && p.bytes().all(|b| b.is_ascii_digit()))
        {
            return Some(parts.concat());
        }
        return None;
    }
    if point.is_none()
        && parts.len() == 2
        && parts[1].len() == 3
        && !parts[0].is_empty()
        && parts[0].len() <= 3
    {
        // 1,250: a group or a decimal, unsettled.
        return None;
    }
    Some(parts.join("."))
}

fn group_digits(text: &str, written: &str, context: &str) -> Option<String> {
    // `\d{1,3}(g\d{3})+([.,]\d+)?` with group char g in {space, ', ’}.
    let head_end = text
        .char_indices()
        .find(|(_, c)| *c == '.' || *c == ',')
        .map(|(i, _)| i)
        .unwrap_or(text.len());
    let (grouped, frac) = if head_end < text.len() {
        (&text[..head_end], Some(&text[head_end..]))
    } else {
        (text, None)
    };
    let sep = grouped.chars().find(|c| !c.is_ascii_digit())?;
    let parts: Vec<&str> = grouped.split(sep).collect();
    let threes =
        |p: &str| !p.is_empty() && p.len() == 3 && p.bytes().all(|b| b.is_ascii_digit());
    if parts.len() < 2
        || parts[0].is_empty()
        || parts[0].len() > 3
        || !parts[0].bytes().all(|b| b.is_ascii_digit())
        || !parts[1..].iter().all(|p| threes(p))
    {
        return None;
    }
    if let Some(frac) = frac {
        let f = &frac[frac.len().min(1)..];
        if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
    }
    let mut plain = parts.concat();
    if let Some(frac) = frac {
        plain.push('.');
        plain.push_str(&frac[frac.len().min(1)..]);
    }
    digits_of(&plain, written, context)
}

/// `_decimal_mark`: the first evidence that speaks — the span's own currency,
/// the document's other numbers, the document's currencies, a declared locale.
fn decimal_mark(written: &str, context: &str) -> Option<char> {
    let context = strip_xml_decl(context);
    for marks in [currency_marks(written), currency_marks(&context)] {
        if marks.len() == 1 {
            return marks.into_iter().next().and_then(|c| c.chars().next());
        }
    }
    for marks in [number_marks(&context), locale_marks(&context)] {
        if marks.len() == 1 {
            return marks.into_iter().next();
        }
    }
    None
}

fn strip_xml_decl(text: &str) -> String {
    let mut out = text.to_string();
    while let Some(start) = out.find("<?") {
        match out[start..].find("?>") {
            Some(end) => out.replace_range(start..start + end + 2, ""),
            None => break,
        }
    }
    out
}

/// The decimal marks evidenced by the context's numbers.
fn number_marks(context: &str) -> std::collections::HashSet<char> {
    let mut marks = std::collections::HashSet::new();
    let bytes = context.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b',' | b' ' | 0xE2 | b'\''))
            {
                i += 1;
            }
            let token = &context[start..i];
            let trimmed = token.trim_end_matches([' ', '.', ',']);
            for mark in [
                grouped_decimal_mark(trimmed),
                short_decimal_mark(trimmed),
                repeated_group_mark(trimmed),
            ]
            .into_iter()
            .flatten()
            {
                marks.insert(mark);
            }
        } else {
            i += 1;
        }
    }
    marks
}

/// `\d{1,3}(g\d{3})+m\d+` — complete groups then a decimal fraction.
fn grouped_decimal_mark(token: &str) -> Option<char> {
    let bytes = token.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'.' || b == b',' {
            let head = &token[..i];
            let tail = &token[i + 1..];
            if tail.is_empty() || !tail.bytes().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let group_char = head.chars().find(|c| !c.is_ascii_digit())?;
            let parts: Vec<&str> = head.split(group_char).collect();
            if parts.len() >= 2
                && !parts[0].is_empty()
                && parts[0].len() <= 3
                && parts[1..]
                    .iter()
                    .all(|p| p.len() == 3 && p.bytes().all(|c| c.is_ascii_digit()))
            {
                return Some(b as char);
            }
        }
    }
    None
}

/// `\d+m\d{1,2}` — a decimal, never a group.
fn short_decimal_mark(token: &str) -> Option<char> {
    let bytes = token.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_digit() {
        return None;
    }
    let mut mark_at = None;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'.' || b == b',' {
            if mark_at.is_some() {
                return None;
            }
            mark_at = Some(i);
        }
    }
    let i = mark_at?;
    let tail = &token[i + 1..];
    if tail.len() <= 2
        && !tail.is_empty()
        && tail.bytes().all(|b| b.is_ascii_digit())
        && token[..i].bytes().all(|b| b.is_ascii_digit())
    {
        Some(token.as_bytes()[i] as char)
    } else {
        None
    }
}

/// `\d{1,3}(g\d{3}){2,}` — a group, never a decimal; the other mark is implied.
fn repeated_group_mark(token: &str) -> Option<char> {
    let group_char = token.chars().find(|c| !c.is_ascii_digit())?;
    if group_char != '.' && group_char != ',' {
        return None;
    }
    let parts: Vec<&str> = token.split(group_char).collect();
    if parts.len() >= 3
        && !parts[0].is_empty()
        && parts[0].len() <= 3
        && parts[1..]
            .iter()
            .all(|p| p.len() == 3 && p.bytes().all(|b| b.is_ascii_digit()))
    {
        return Some(if group_char == '.' { ',' } else { '.' });
    }
    None
}

fn locale_marks(context: &str) -> std::collections::HashSet<char> {
    let mut marks = std::collections::HashSet::new();
    let folded = fold(context);
    for lang in ["en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru", "sv", "ja", "zh", "ko"] {
        for key in ["locale", "language", "lang"] {
            let mut at = 0;
            while let Some(pos) = folded[at..].find(key) {
                let pos = at + pos + key.len();
                let after = folded[pos..].trim_start_matches(['"', '\'', ' ', '=', ':']);
                if let Some(rest) = after.strip_prefix(lang) {
                    let rest = rest.trim_start_matches(['-', '_', '"', '\'']);
                    let rest = rest.trim_start_matches(|c: char| c.is_ascii_alphabetic());
                    let _ = rest;
                    marks.insert(if matches!(lang, "en" | "ja" | "zh" | "ko") { '.' } else { ',' });
                }
                at = (pos).max(at + 1);
            }
        }
    }
    marks
}

// ---------------------------------------------------------------- number

/// `parse_number`: the written magnitude, percent points included; its
/// currency and percent sign set apart, its scale word applied; `None` when
/// the writing does not settle one value.
pub fn parse_number(text: &str, context: &str, unit: Option<&str>) -> ParsedNumber {
    let context = match unit {
        Some(u) if is_currency(u) => format!("{context}\n{u}"),
        _ => context.to_string(),
    };
    let context = context.as_str();
    let written = text;
    let mut text = text
        .trim()
        .replace('\u{2212}', "-")
        .replace('\u{00a0}', " ")
        .replace('\u{202f}', " ");
    if let Some(stripped) = text.strip_suffix('%') {
        text = stripped.trim_end().to_string();
    }
    let negative = text.starts_with('(') && text.ends_with(')');
    if negative {
        text = text[1..text.len() - 1].trim().to_string();
    }
    let mut sign: Option<char> = None;
    let mut currency: Option<String> = None;
    for _ in 0..2 {
        // one currency and one sign, in either order
        if currency.is_none() {
            let trimmed = text.trim().to_lowercase();
            let mut found = None;
            for (lower, original, _) in currency_names() {
                if trimmed.starts_with(lower.as_str()) {
                    let rest = trimmed[lower.len()..].trim_start();
                    if lower.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                        && !rest.is_empty()
                        && rest.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                    {
                        continue; // `\bCUR(?![a-z])`
                    }
                    found = Some((original.clone(), rest.to_string()));
                    break;
                }
                if trimmed.ends_with(lower.as_str()) {
                    let rest = trimmed[..trimmed.len() - lower.len()].trim_end();
                    if lower.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                        && !rest.is_empty()
                        && rest.chars().last().is_some_and(|c| c.is_ascii_lowercase())
                    {
                        continue;
                    }
                    found = Some((original.clone(), rest.to_string()));
                    break;
                }
            }
            if let Some((name, rest)) = found {
                currency = Some(name);
                text = rest.trim().to_string();
            }
        }
        if sign.is_none() && (text.starts_with('-') || text.starts_with('+')) {
            sign = Some(text.chars().next().unwrap());
            text = text[1..].trim().to_string();
        }
    }
    if negative && sign.is_some() {
        return ParsedNumber::None;
    }
    let mut scale: i64 = 1;
    if let Some((value, start)) = scale_word(&text) {
        scale = value;
        text = text[..start].trim_end().to_string();
    } else if let Some((value, start)) = suffix_scale(&text) {
        let combined = format!("{written}\n{context}");
        if currency.is_none() && !counted_context(context) && currency_matches(&combined).is_empty()
        {
            return ParsedNumber::None;
        }
        scale = value;
        text = text[..start].trim_end().to_string();
    }
    let digits = match digits_of(&text, written, context) {
        Some(digits) => digits,
        None => return ParsedNumber::None,
    };
    let negative_value = negative || sign == Some('-');
    // Decimal reconstruction: `int + frac` digits over `10^exponent`, like
    // `Decimal(digits) * scale` (the sign lives in the leading '-', if any).
    let exponent = digits.split_once('.').map(|(_, f)| f.len()).unwrap_or(0);
    let cleaned: String = digits.chars().filter(|&c| c != '.').collect();
    let mut integer = parse_big_int(&cleaned);
    if exponent > 0 {
        let denom = pow10(exponent as u32);
        integer = (if negative_value { -integer } else { integer }) * scale as i128;
        if integer % denom == 0 {
            return ParsedNumber::Int((integer / denom) as i64);
        }
        let value = integer as f64 / denom as f64;
        return if value.is_finite() {
            ParsedNumber::Float(value)
        } else {
            ParsedNumber::None
        };
    }
    let integer = (if negative_value { -integer } else { integer }) * scale as i128;
    ParsedNumber::Int(integer as i64)
}

fn parse_big_int(digits: &str) -> i128 {
    let mut value: i128 = 0;
    let mut negative = false;
    for ch in digits.chars() {
        match ch {
            '-' => negative = true,
            '+' => {}
            c if c.is_ascii_digit() => value = value * 10 + (c as i128 - '0' as i128),
            _ => {}
        }
    }
    if negative {
        -value
    } else {
        value
    }
}

fn pow10(n: u32) -> i128 {
    let mut v: i128 = 1;
    for _ in 0..n {
        v *= 10;
    }
    v
}

// ---------------------------------------------------------------- dates

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn valid_date(year: i64, month: u32, day: u32) -> bool {
    year > 0 && (1..=12).contains(&month) && day >= 1 && day <= days_in_month(year, month)
}

fn iso_date(year: i64, month: u32, day: u32) -> String {
    format!("{year:04}-{month:02}-{day:02}")
}

/// `date_order`: declared format/locale first, then the context's other
/// unambiguous numeric dates; conflicts stay unresolved.
pub fn date_order(context: &str) -> Option<&'static str> {
    let folded = fold(context);
    let mut explicit = std::collections::HashSet::new();
    if format_pattern(&folded, true) {
        explicit.insert("dmy");
    }
    if format_pattern(&folded, false) {
        explicit.insert("mdy");
    }
    if explicit.len() == 1 {
        return explicit.into_iter().next();
    }
    if !explicit.is_empty() {
        return None;
    }
    let locales = locale_decl(&folded);
    if locales.len() == 1 {
        return locales.into_iter().next();
    }
    let mut orders = std::collections::HashSet::new();
    for (a, b, year, year_digits) in numeric_dates(context) {
        if year_digits != 4 || a > 31 || b > 31 || a < 1 || b < 1 {
            continue;
        }
        for (day, month, order) in [(a, b, "dmy"), (b, a, "mdy")] {
            if valid_date(year, month, day) && (a > 12) != (b > 12) {
                orders.insert(order);
            }
        }
    }
    if orders.len() == 1 {
        orders.into_iter().next()
    } else {
        None
    }
}

/// `\bDAY [/.-] MONTH [/.-] YEAR\b` (or the reverse) in folded text.
fn format_pattern(folded: &str, day_first: bool) -> bool {
    let days: &[&str] = &[
        "dd", "d", "jj", "j", "tt", "t", "gg", "g", "day", "jour", "tag", "dia", "giorno",
    ];
    let months: &[&str] = &["mm", "m", "month", "mois", "monat", "mes", "mese"];
    let years = [
        "yy", "yyyy", "aa", "aaaa", "jj", "jjjj", "year", "annee", "jahr", "ano", "anno",
    ];
    let (first, second) = if day_first { (days, months) } else { (months, days) };
    for a in first {
        for b in second {
            for y in years {
                for sep in ["/", ".", "-"] {
                    for space in ["", " "] {
                        let pat = format!("{a}{space}{sep}{space}{b}{space}{sep}{space}{y}");
                        if folded.contains(&pat) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

fn locale_decl(folded: &str) -> std::collections::HashSet<&'static str> {
    let mut orders = std::collections::HashSet::new();
    for key in [
        "locale", "language", "lang", "langue", "sprache", "idioma", "lingua",
    ] {
        let mut at = 0;
        while let Some(pos) = folded[at..].find(key) {
            let pos = at + pos + key.len();
            let after = folded[pos..].trim_start_matches(['"', '\'', ' ', '=', ':']);
            for (lang, order) in [
                ("en", "mdy"),
                ("en-us", "mdy"),
                ("fr", "dmy"),
                ("fr-fr", "dmy"),
                ("de", "dmy"),
                ("de-de", "dmy"),
                ("es", "dmy"),
                ("es-es", "dmy"),
                ("it", "dmy"),
                ("it-it", "dmy"),
            ] {
                if let Some(rest) = after.strip_prefix(lang) {
                    let rest = rest.trim_start_matches(['-', '_']);
                    let rest = rest.trim_start_matches(|c: char| c.is_ascii_alphabetic());
                    let _ = rest;
                    orders.insert(order);
                }
            }
            at = pos.max(at + 1);
        }
    }
    orders
}

/// `NUMERIC_DATE.finditer` candidates `(a, b, year, year_digits)`.
fn numeric_dates(text: &str) -> Vec<(u32, u32, i64, usize)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit() || matches!(bytes[i], b'/' | b'.' | b'-'))
            {
                i += 1;
            }
            let token = &text[start..i];
            let parts: Vec<&str> = token.split(|c| c == '/' || c == '.' || c == '-').collect();
            if parts.len() == 3
                && parts
                    .iter()
                    .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
                && parts[0].len() <= 4
                && parts[2].len() >= 2
            {
                let seps: Vec<char> = token
                    .chars()
                    .filter(|c| matches!(c, '/' | '.' | '-'))
                    .collect();
                if seps.len() == 2 && seps[0] == seps[1] {
                    out.push((
                        parts[0].parse().unwrap_or(0),
                        parts[1].parse().unwrap_or(0),
                        parts[2].parse().unwrap_or(0),
                        parts[2].len(),
                    ));
                }
            }
        } else {
            i += 1;
        }
    }
    out
}

/// `parse_date`: ISO value and unresolved-ambiguity flag.
pub fn parse_date(text: &str, context: &str) -> (Option<String>, bool) {
    let order = date_order(context);
    parse_date_inner(text, context, order)
}

fn parse_date_inner(
    text: &str,
    context: &str,
    order: Option<&'static str>,
) -> (Option<String>, bool) {
    let text = text.trim();
    if text.is_empty() {
        return (None, false);
    }
    // Timed: keep an explicitly written time and timezone.
    if let Some((date_part, clock)) = split_timed(text) {
        let (date, ambiguous) = parse_date_inner(&date_part, context, order);
        return match date {
            None => (None, ambiguous),
            Some(date) => {
                let clock = if clock.len() == 4 && clock.chars().nth(1) == Some(':') {
                    format!("0{clock}")
                } else {
                    clock
                };
                match parse_iso_datetime(&format!("{date}T{clock}")) {
                    Some(value) => (Some(value), false),
                    None => (None, false),
                }
            }
        };
    }
    // Numeric.
    if let Some((a, b, c)) = numeric_fullmatch(text) {
        let first_len = text.split(['/', '.', '-']).next().unwrap().len();
        let last_len = text.rsplit(['/', '.', '-']).next().unwrap().len();
        let candidates: Vec<(i64, u32, u32)> = if first_len == 4 {
            vec![(a as i64, b as u32, c as u32)]
        } else if last_len == 4 {
            candidates_for(c as i64, a, b, order)
        } else {
            // A century is never invented: another unambiguous complete date
            // must establish it.
            if last_len != 2 {
                return (None, false);
            }
            let mut centuries = std::collections::HashSet::new();
            let mut complete: Vec<String> = numeric_dates(context)
                .into_iter()
                .filter(|(_, _, _, ya)| *ya == 4)
                .map(|(a, b, c, _)| format!("{a}/{b}/{c}"))
                .collect();
            complete.extend(word_date_strings(&fold(context)));
            for candidate in complete {
                let (value, ambiguous) = parse_date_inner(&candidate, context, order);
                if let Some(value) = value {
                    if !ambiguous {
                        let year: i64 = value[..4].parse().unwrap_or(0);
                        centuries.insert(year / 100 * 100);
                    }
                }
            }
            if centuries.len() != 1 {
                return (None, false);
            }
            candidates_for(
                centuries.into_iter().next().unwrap() + c as i64,
                a,
                b,
                order,
            )
        };
        let mut values = std::collections::HashSet::new();
        for (year, month, day) in candidates {
            if valid_date(year, month, day) {
                values.insert(iso_date(year, month, day));
            }
        }
        if values.len() > 1 {
            return (None, true);
        }
        return (values.into_iter().next(), false);
    }
    // CJK.
    if let Some((year, month, day)) = cjk_fullmatch(text) {
        return if valid_date(year, month, day) {
            (Some(iso_date(year, month, day)), false)
        } else {
            (None, false)
        };
    }
    // Word dates.
    let mut words = fold(text);
    words = strip_ordinal_suffixes(&words);
    words = words
        .split_whitespace()
        .filter(|w| !matches!(*w, "de" | "of" | "the" | "le" | "el" | "il" | "den"))
        .collect::<Vec<_>>()
        .join(" ");
    let tokens: Vec<&str> = words
        .split(|c: char| c == ',' || c == '.' || c == ' ')
        .filter(|w| !w.is_empty())
        .collect();
    let tokens = if !tokens.is_empty()
        && WEEKDAYS.contains(&tokens[0].trim_end_matches("-feira"))
    {
        &tokens[1..]
    } else {
        &tokens[..]
    };
    if tokens.len() != 3 {
        return (None, false);
    }
    let months = months();
    let (day_s, month, year_s) = if let Some(&m) = months.get(tokens[0]) {
        (tokens[1], m, tokens[2])
    } else if let Some(&m) = months.get(tokens[1]) {
        (tokens[0], m, tokens[2])
    } else {
        return (None, false);
    };
    if !day_s.bytes().all(|b| b.is_ascii_digit())
        || year_s.len() != 4
        || !year_s.bytes().all(|b| b.is_ascii_digit())
    {
        return (None, false);
    }
    let (day, year) = (day_s.parse().unwrap_or(0), year_s.parse().unwrap_or(0));
    if valid_date(year, month, day) {
        (Some(iso_date(year, month, day)), false)
    } else {
        (None, false)
    }
}

fn candidates_for(year: i64, a: u32, b: u32, order: Option<&str>) -> Vec<(i64, u32, u32)> {
    let dmy = (year, b, a);
    let mdy = (year, a, b);
    match order {
        None => vec![dmy, mdy],
        Some("dmy") => vec![dmy],
        _ => vec![mdy],
    }
}

/// `NUMERIC_DATE.fullmatch`: `a{1,4} sep b{1,2} sep c{2,4}`, same separator.
fn numeric_fullmatch(text: &str) -> Option<(u32, u32, u32)> {
    let sep = text.chars().find(|c| matches!(c, '/' | '.' | '-'))?;
    if text.matches(sep).count() != 2 {
        return None;
    }
    let parts: Vec<&str> = text.split(sep).collect();
    if parts.len() != 3
        || parts[0].is_empty()
        || parts[0].len() > 4
        || parts[1].is_empty()
        || parts[1].len() > 2
        || parts[2].len() < 2
        || parts[2].len() > 4
        || !parts.iter().all(|p| p.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    Some((
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
    ))
}

/// `CJK_DATE.fullmatch`: `(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]`.
fn cjk_fullmatch(text: &str) -> Option<(i64, u32, u32)> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let take_digits = |i: &mut usize, min: usize, max: usize| -> Option<u32> {
        let start = *i;
        while *i < chars.len() && chars[*i].is_ascii_digit() {
            *i += 1;
        }
        let n = *i - start;
        if n < min || n > max {
            return None;
        }
        chars[start..*i].iter().collect::<String>().parse().ok()
    };
    let skip_ws = |i: &mut usize| {
        while *i < chars.len() && chars[*i].is_whitespace() {
            *i += 1;
        }
    };
    let year = take_digits(&mut i, 4, 4)? as i64;
    skip_ws(&mut i);
    if i >= chars.len() || !matches!(chars[i], '年' | '년') {
        return None;
    }
    i += 1;
    skip_ws(&mut i);
    let month = take_digits(&mut i, 1, 2)?;
    skip_ws(&mut i);
    if i >= chars.len() || !matches!(chars[i], '月' | '월') {
        return None;
    }
    i += 1;
    skip_ws(&mut i);
    let day = take_digits(&mut i, 1, 2)?;
    skip_ws(&mut i);
    if i >= chars.len() || !matches!(chars[i], '日' | '일') {
        return None;
    }
    i += 1;
    if i != chars.len() {
        return None;
    }
    Some((year, month, day))
}

/// `(date) [T or space] (clock)$` — the trailing time keeps its timezone.
fn split_timed(text: &str) -> Option<(String, String)> {
    let chars: Vec<char> = text.chars().collect();
    let is_clock = |s: &[char]| -> bool {
        let read = |i: &mut usize, min: usize, max: usize| -> bool {
            let start = *i;
            while *i < s.len() && s[*i].is_ascii_digit() {
                *i += 1;
            }
            let n = *i - start;
            n >= min && n <= max
        };
        let mut i = 0;
        if !read(&mut i, 1, 2) || i >= s.len() || s[i] != ':' {
            return false;
        }
        i += 1;
        if !read(&mut i, 2, 2) {
            return false;
        }
        if i < s.len() && s[i] == ':' {
            i += 1;
            if !read(&mut i, 2, 2) {
                return false;
            }
            if i < s.len() && s[i] == '.' {
                i += 1;
                if !read(&mut i, 1, 10) {
                    return false;
                }
            }
        }
        if i < s.len() && s[i] == 'Z' {
            i += 1;
        } else if i < s.len() && (s[i] == '+' || s[i] == '-') {
            i += 1;
            if !read(&mut i, 2, 2) || i >= s.len() || s[i] != ':' {
                return false;
            }
            i += 1;
            if !read(&mut i, 2, 2) {
                return false;
            }
        }
        i == s.len()
    };
    for i in (1..chars.len()).rev() {
        if chars[i] == 'T' || chars[i].is_whitespace() {
            if i + 1 < chars.len() && is_clock(&chars[i + 1..]) {
                let date: String = chars[..i].iter().collect();
                let clock: String = chars[i + 1..].iter().collect();
                if !date.trim().is_empty() {
                    return Some((date.trim().to_string(), clock));
                }
            }
        }
    }
    None
}

/// `dt.datetime.fromisoformat(...).isoformat()` for the kept shapes.
fn parse_iso_datetime(text: &str) -> Option<String> {
    let (date, clock) = text.split_once('T')?;
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let (year, month, day): (i64, u32, u32) = (
        parts[0].parse().ok()?,
        parts[1].parse().ok()?,
        parts[2].parse().ok()?,
    );
    if !valid_date(year, month, day) {
        return None;
    }
    let (clock_main, zone) = match clock.find(['Z', '+']) {
        Some(at) => (&clock[..at], &clock[at..]),
        None => match clock.rfind('-') {
            Some(at) if at > 0 => (&clock[..at], &clock[at..]),
            _ => (clock, ""),
        },
    };
    let hms: Vec<&str> = clock_main.split(':').collect();
    if hms.len() < 2 || hms.len() > 3 {
        return None;
    }
    let hour: u32 = hms[0].parse().ok()?;
    let minute: u32 = hms[1].parse().ok()?;
    let (second, micros) = if hms.len() == 3 {
        match hms[2].split_once('.') {
            Some((s, f)) => (s.parse().ok()?, f.len() as u32),
            None => (hms[2].parse().ok()?, 0),
        }
    } else {
        (0, 0)
    };
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let mut out = format!(
        "{}T{hour:02}:{minute:02}:{second:02}",
        iso_date(year, month, day)
    );
    if micros > 0 {
        out.push_str(&format!(".{:0width$}", 0, width = micros as usize));
    }
    out.push_str(zone);
    Some(out)
}

fn strip_ordinal_suffixes(words: &str) -> String {
    // `(?<=\d)(st|nd|rd|th|er)\b` and `(?<=\d)\.?(o|a|°)\b`
    let mut out = String::with_capacity(words.len());
    let chars: Vec<char> = words.chars().collect();
    let mut i = 0;
    let is_ord = |j: usize, suffix: &str| -> bool {
        j + suffix.len() <= chars.len()
            && chars[j..j + suffix.len()].iter().collect::<String>() == suffix
            && j > 0
            && chars[j - 1].is_ascii_digit()
            && (j + suffix.len() == chars.len() || !chars[j + suffix.len()].is_alphabetic())
    };
    while i < chars.len() {
        if (is_ord(i, "st")
            || is_ord(i, "nd")
            || is_ord(i, "rd")
            || is_ord(i, "th")
            || is_ord(i, "er"))
            && (i == 0 || chars[i - 1].is_ascii_digit())
        {
            i += 2;
            continue;
        }
        if (chars[i] == 'o' || chars[i] == 'a' || chars[i] == '°')
            && i > 0
            && (chars[i - 1].is_ascii_digit()
                || (chars[i - 1] == '.' && i > 1 && chars[i - 2].is_ascii_digit()))
            && (i + 1 == chars.len() || !chars[i + 1].is_alphabetic())
        {
            i += 1;
            continue;
        }
        if chars[i] == '.'
            && i + 1 < chars.len()
            && (chars[i + 1] == 'o' || chars[i + 1] == 'a' || chars[i + 1] == '°')
            && i > 0
            && chars[i - 1].is_ascii_digit()
        {
            i += 2;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `WORD_DATE.finditer` candidates over the folded context (the century rule).
fn word_date_strings(folded: &str) -> Vec<String> {
    let months = months();
    let mut out = Vec::new();
    let words: Vec<&str> = folded
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    for window in words.windows(3) {
        let (a, b, c) = (window[0], window[1], window[2]);
        let monthish = |w: &str| months.contains_key(w);
        let digitish = |w: &str| !w.is_empty() && w.bytes().all(|x| x.is_ascii_digit());
        if ((monthish(a) && digitish(b) && digitish(c))
            || (digitish(a) && monthish(b) && digitish(c)))
            && c.len() == 4
        {
            out.push(window.join(" "));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn num(text: &str) -> ParsedNumber {
        parse_number(text, "", None)
    }

    #[test]
    fn fixture_spans() {
        assert_eq!(num("42000 EUR"), ParsedNumber::Int(42000));
        assert_eq!(num("3560 万元"), ParsedNumber::Int(35600000));
        assert_eq!(num("200"), ParsedNumber::Int(200));
        assert_eq!(num("5"), ParsedNumber::Int(5));
        assert_eq!(parse_date("March 4, 2026", "").0.as_deref(), Some("2026-03-04"));
        assert_eq!(parse_date("2026 年 9 月 21 日", "").0.as_deref(), Some("2026-09-21"));
        assert_eq!(parse_date("2026-11-18", "").0.as_deref(), Some("2026-11-18"));
        assert_eq!(parse_date("2026-12-03", "").0.as_deref(), Some("2026-12-03"));
        assert_eq!(num("1 234,50"), ParsedNumber::Float(1234.5));
        assert_eq!(
            parse_date("03/04/2026", "Pay before (day/month/year)").0.as_deref(),
            Some("2026-04-03")
        );
        assert_eq!(num("12.5%"), ParsedNumber::Float(12.5));
        assert_eq!(num("-1,250.00"), ParsedNumber::Int(-1250));
        assert_eq!(
            parse_number("12k", "about 12k downloads", None),
            ParsedNumber::Int(12000)
        );
        assert_eq!(num("€1.2M"), ParsedNumber::Int(1200000));
        assert_eq!(written_unit("12.5%").as_deref(), Some("%"));
        assert_eq!(written_unit("42000 EUR").as_deref(), Some("EUR"));
        assert_eq!(written_unit("3560 万元").as_deref(), Some("元"));
    }

    #[test]
    fn invalid_or_unsettled() {
        assert_eq!(num("1,250"), ParsedNumber::None); // group or decimal, unsettled
        assert_eq!(parse_date("2026-13-01", ""), (None, false));
        assert_eq!(parse_date("", ""), (None, false));
    }
}
