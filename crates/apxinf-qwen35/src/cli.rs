use std::collections::HashSet;
use std::ffi::OsString;
use std::path::PathBuf;

use crate::protocol::{validate_prompt, validate_token_budget};
use crate::Result;

pub const HELP: &str = "apxinf-qwen35-08b --model PATH --prompt TEXT [--max-tokens N] [--json]
apxinf-qwen35-08b --model PATH --jsonl [--max-tokens N]

Fixed Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17.
Requires macOS Apple Silicon, Metal, and the five matching local model assets.
Uses CPU/Accelerate F32 with the existing Metal W8 head and 24 decode MLP blocks.
The context limit is 2048 tokens, including the output budget.

  --model PATH    Directory containing the fixed model assets.
  --prompt TEXT   One nonempty user prompt.
  --max-tokens N  Output budget: 1 through 2047; default 64.
  --json          Emit one complete JSON result.
  --jsonl         Read independent JSON requests from standard input.
  --help          Show this help without loading assets.

JSONL requests contain prompt and optional max_tokens. Each line is limited
to 65536 bytes before its newline. The process exits on the first error.
Plain output streams decoded text. JSON modes emit complete result records.";

#[derive(Debug)]
pub enum Command {
    Help,
    Run(Settings),
}

#[derive(Debug)]
pub struct Settings {
    pub model: PathBuf,
    pub mode: Mode,
    pub max_tokens: usize,
}

#[derive(Debug)]
pub enum Mode {
    Single { prompt: String, json: bool },
    JsonLines,
}

pub fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Command> {
    let mut arguments = arguments.into_iter();
    let mut seen = HashSet::new();
    let mut model = None;
    let mut prompt = None;
    let mut max_tokens = 64;
    let mut json = false;
    let mut jsonl = false;
    let mut help = false;
    while let Some(argument) = arguments.next() {
        let flag = argument.to_str().ok_or("Argument names must be UTF-8")?;
        if !matches!(
            flag,
            "--model" | "--prompt" | "--max-tokens" | "--json" | "--jsonl" | "--help"
        ) {
            return Err(format!("Unknown argument: {}", argument.to_string_lossy()).into());
        }
        if !seen.insert(flag.to_owned()) {
            return Err(format!("Duplicate argument: {flag}").into());
        }
        match flag {
            "--model" => {
                let value = arguments.next().ok_or("Missing value for --model")?;
                if value.is_empty() || value.to_str().is_some_and(|value| value.starts_with("--")) {
                    return Err("Missing value for --model".into());
                }
                model = Some(PathBuf::from(value));
            }
            "--prompt" => {
                let value = arguments.next().ok_or("Missing value for --prompt")?;
                let text = value.into_string().map_err(|_| "Prompt must be UTF-8")?;
                if text.starts_with("--") {
                    return Err("Missing value for --prompt".into());
                }
                validate_prompt(&text)?;
                prompt = Some(text);
            }
            "--max-tokens" => {
                let value = arguments.next().ok_or("Missing value for --max-tokens")?;
                let text = value.to_str().ok_or("Token budget must be an integer")?;
                if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err("Token budget must be a positive integer".into());
                }
                max_tokens = text.parse().map_err(|_| "Token budget is out of range")?;
                validate_token_budget(max_tokens)?;
            }
            "--json" => json = true,
            "--jsonl" => jsonl = true,
            "--help" => help = true,
            _ => unreachable!(),
        }
    }
    if help {
        return if seen.len() == 1 {
            Ok(Command::Help)
        } else {
            Err("--help cannot be combined with other arguments".into())
        };
    }
    if json && jsonl {
        return Err("--json and --jsonl are mutually exclusive".into());
    }
    let mode = match (prompt, jsonl) {
        (Some(prompt), false) => Mode::Single { prompt, json },
        (None, true) => Mode::JsonLines,
        (Some(_), true) => return Err("--prompt and --jsonl are mutually exclusive".into()),
        (None, false) => return Err("Specify --prompt or --jsonl".into()),
    };
    Ok(Command::Run(Settings {
        model: model.ok_or("Missing --model")?,
        mode,
        max_tokens,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(arguments: &[&str]) -> Result<Command> {
        parse(arguments.iter().map(OsString::from))
    }

    #[test]
    fn help_needs_no_model_and_single_prompt_has_fixed_defaults() {
        assert!(matches!(args(&["--help"]).unwrap(), Command::Help));
        let Command::Run(settings) = args(&["--model", "model", "--prompt", "hello"]).unwrap()
        else {
            panic!("Expected settings");
        };
        assert_eq!(settings.max_tokens, 64);
        assert!(matches!(settings.mode, Mode::Single { json: false, .. }));
    }

    #[test]
    fn rejects_missing_duplicate_unknown_and_conflicting_arguments() {
        let cases: &[&[&str]] = &[
            &[],
            &["--model"],
            &["--model", "model", "--prompt"],
            &["--model", "model", "--prompt", "--json"],
            &["--model", "model", "--prompt", ""],
            &["--model", "model", "--prompt", "hello", "--json", "--json"],
            &["--model", "model", "--prompt", "hello", "--jsonl"],
            &["--model", "model", "--jsonl", "--json"],
            &["--model", "model", "--jsonl", "--max-tokens", "0"],
            &["--model", "model", "--jsonl", "--max-tokens", "2048"],
            &["--model", "model", "--jsonl", "--max-tokens", "+2"],
            &["--model", "model", "--jsonl", "--max-tokens", "2.0"],
            &["--model", "model", "--jsonl", "--provider", "mlx"],
            &["--help", "--model", "model"],
        ];
        for case in cases {
            assert!(args(case).is_err(), "Accepted invalid arguments: {case:?}");
        }
    }
}
