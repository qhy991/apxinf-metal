use std::collections::HashSet;
use std::io::{BufRead, Read};

use serde_json::Value;

use crate::Result;

pub const MAX_CONTEXT: usize = 2048;
pub const MAX_LINE_BYTES: usize = 65536;

pub struct Request {
    pub prompt: String,
    pub max_tokens: usize,
}

pub fn validate_prompt(prompt: &str) -> Result<()> {
    if prompt.trim().is_empty() {
        return Err("Prompt must not be empty".into());
    }
    Ok(())
}

pub fn validate_token_budget(max_tokens: usize) -> Result<()> {
    if !(1..MAX_CONTEXT).contains(&max_tokens) {
        return Err("Token budget must be between 1 and 2047".into());
    }
    Ok(())
}

pub fn validate_context(prompt_tokens: usize, max_tokens: usize) -> Result<()> {
    validate_token_budget(max_tokens)?;
    if prompt_tokens == 0 {
        return Err("Tokenized prompt must not be empty".into());
    }
    if prompt_tokens
        .checked_add(max_tokens)
        .is_none_or(|total| total > MAX_CONTEXT)
    {
        return Err("Prompt and output budget exceed the 2048-token context".into());
    }
    Ok(())
}

pub fn read_line(input: &mut impl BufRead) -> Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    let count = input
        .take((MAX_LINE_BYTES + 2) as u64)
        .read_until(b'\n', &mut line)?;
    if count == 0 {
        return Ok(None);
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.len() > MAX_LINE_BYTES {
        return Err("JSONL request exceeds 65536 bytes before its newline".into());
    }
    Ok(Some(line))
}

pub fn parse_request(line: &[u8], default_max_tokens: usize) -> Result<Request> {
    if line.len() > MAX_LINE_BYTES {
        return Err("JSONL request exceeds 65536 bytes before its newline".into());
    }
    let value: Value = serde_json::from_slice(line)?;
    let fields = value
        .as_object()
        .ok_or("Each JSONL request must be an object")?;
    if fields
        .keys()
        .any(|key| key != "prompt" && key != "max_tokens")
    {
        return Err("JSONL requests accept only prompt and max_tokens".into());
    }
    let prompt = fields
        .get("prompt")
        .and_then(Value::as_str)
        .ok_or("Request prompt must be a string")?;
    validate_prompt(prompt)?;
    let max_tokens = match fields.get("max_tokens") {
        None => default_max_tokens,
        Some(value) => usize::try_from(
            value
                .as_u64()
                .ok_or("Request max_tokens must be a positive integer")?,
        )?,
    };
    validate_token_budget(max_tokens)?;
    reject_duplicate_fields(line)?;
    Ok(Request {
        prompt: prompt.to_owned(),
        max_tokens,
    })
}

// Type validation above leaves only a flat object with string and integer values.
// Scan its JSON string boundaries so escaped member names cannot hide duplicates.
fn reject_duplicate_fields(line: &[u8]) -> Result<()> {
    let mut keys = HashSet::new();
    let mut index = 0;
    while index < line.len() {
        if line[index] != b'"' {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < line.len() {
            match line[index] {
                b'\\' => index += 2,
                b'"' => {
                    index += 1;
                    break;
                }
                _ => index += 1,
            }
        }
        let end = index;
        while index < line.len() && line[index].is_ascii_whitespace() {
            index += 1;
        }
        if line.get(index) == Some(&b':') {
            let key: String = serde_json::from_slice(&line[start..end])?;
            if !keys.insert(key) {
                return Err("Duplicate JSONL request field".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn rejects_unknown_duplicate_and_invalid_request_fields() {
        for input in [
            "[]",
            "{}",
            "null",
            "{\"prompt\":null}",
            "{\"prompt\":\" \"}",
            "{\"prompt\":\"x\",\"model\":\"other\"}",
            "{\"prompt\":\"x\",\"max_tokens\":0}",
            "{\"prompt\":\"x\",\"max_tokens\":2048}",
            "{\"prompt\":\"x\",\"max_tokens\":1.5}",
            "{\"prompt\":\"x\",\"max_tokens\":\"2\"}",
            "{\"prompt\":\"x\",\"max_tokens\":true}",
            "{\"prompt\":\"x\",\"prompt\":\"y\"}",
            "{\"prompt\":\"x\",\"pr\\u006fmpt\":\"y\"}",
            "{\"prompt\":\"x\",\"max_tokens\":2,\"max_tokens\":3}",
        ] {
            assert!(
                parse_request(input.as_bytes(), 64).is_err(),
                "Accepted: {input}"
            );
        }
    }

    #[test]
    fn accepts_escaped_text_without_treating_its_contents_as_fields() {
        let request =
            parse_request(br#"{"prompt":"say \"prompt\": 3","max_tokens":2}"#, 64).unwrap();
        assert_eq!(request.prompt, "say \"prompt\": 3");
        assert_eq!(request.max_tokens, 2);
        assert_eq!(
            parse_request(br#"{"prompt":"hello"}"#, 7)
                .unwrap()
                .max_tokens,
            7
        );
    }

    #[test]
    fn line_reading_is_bounded_and_does_not_consume_the_next_request() {
        let mut input = Cursor::new(b"first\nsecond\n");
        assert_eq!(read_line(&mut input).unwrap().unwrap(), b"first");
        assert_eq!(read_line(&mut input).unwrap().unwrap(), b"second");
        assert!(read_line(&mut input).unwrap().is_none());
        let mut allowed = vec![b' '; MAX_LINE_BYTES];
        allowed.push(b'\n');
        assert_eq!(
            read_line(&mut Cursor::new(allowed)).unwrap().unwrap().len(),
            MAX_LINE_BYTES
        );
        assert!(read_line(&mut Cursor::new(vec![b'x'; MAX_LINE_BYTES + 1])).is_err());
    }

    #[test]
    fn context_budget_includes_prompt_and_all_requested_output_tokens() {
        assert!(validate_context(2047, 1).is_ok());
        assert!(validate_context(2047, 2).is_err());
        assert!(validate_context(0, 1).is_err());
        assert!(validate_context(usize::MAX, 1).is_err());
    }
}
