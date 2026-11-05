mod assets;
mod cli;
mod protocol;
mod runtime;

use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use serde_json::Value;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn write_json(output: &mut impl Write, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    output.write_all(&bytes)?;
    output.flush()?;
    Ok(())
}

fn process_jsonl(
    input: &mut impl BufRead,
    output: &mut impl Write,
    default_max_tokens: usize,
    mut on_request: impl FnMut(&protocol::Request) -> Result<Value>,
) -> Result<()> {
    while let Some(line) = protocol::read_line(input)? {
        let request = protocol::parse_request(&line, default_max_tokens)?;
        write_json(output, &on_request(&request)?)?;
    }
    Ok(())
}

fn run() -> Result<()> {
    let settings = match cli::parse(std::env::args_os().skip(1))? {
        cli::Command::Help => {
            println!("{}", cli::HELP);
            return Ok(());
        }
        cli::Command::Run(settings) => settings,
    };
    let mut runtime = runtime::Runtime::load(&settings.model)?;
    let mut output = io::stdout().lock();
    match settings.mode {
        cli::Mode::Single { prompt, json } => {
            let request = protocol::Request {
                prompt,
                max_tokens: settings.max_tokens,
            };
            if json {
                let mut result = runtime.request(&request)?;
                result["startup"] = runtime.startup.clone();
                result["asset_identity"] = assets::identity();
                write_json(&mut output, &result)?;
            } else {
                runtime.request_plain(&request, &mut output)?;
            }
        }
        cli::Mode::JsonLines => {
            write_json(&mut output, &runtime.ready())?;
            let mut input = io::stdin().lock();
            process_jsonl(&mut input, &mut output, settings.max_tokens, |request| {
                runtime.request(request)
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Cursor;

    #[test]
    fn jsonl_stops_on_invalid_input_without_a_second_success_record() {
        let mut input = Cursor::new(b"{\"prompt\":\"first\"}\n{\"prompt\":\"bad\",\"unknown\":true}\n{\"prompt\":\"third\"}\n");
        let mut output = Vec::new();
        let mut calls = 0;
        let result = process_jsonl(&mut input, &mut output, 64, |request| {
            calls += 1;
            Ok(json!({"kind": "result", "prompt": request.prompt}))
        });
        assert!(result.is_err());
        assert_eq!(calls, 1);
        let records = String::from_utf8(output).unwrap();
        assert_eq!(records.lines().count(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(records.trim()).unwrap()["prompt"],
            "first"
        );
        assert_eq!(
            protocol::read_line(&mut input).unwrap().unwrap(),
            b"{\"prompt\":\"third\"}"
        );
    }

    #[test]
    fn jsonl_inference_error_writes_no_result_and_leaves_the_next_line_unread() {
        let mut input = Cursor::new(b"{\"prompt\":\"first\"}\n{\"prompt\":\"second\"}\n");
        let mut output = Vec::new();
        assert!(process_jsonl(&mut input, &mut output, 64, |_| Err(
            "Inference failed".into()
        ))
        .is_err());
        assert!(output.is_empty());
        assert_eq!(
            protocol::read_line(&mut input).unwrap().unwrap(),
            b"{\"prompt\":\"second\"}"
        );
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("apxinf-qwen35-08b: {error}");
            ExitCode::FAILURE
        }
    }
}
