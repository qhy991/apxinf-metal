//! Incremental output parsing for the declared Qwen text profile.
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Text(String),
    Tool {
        id: String,
        name: String,
        input: Value,
    },
}

/// Hold only a suffix that can still become a stop sequence.
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
    passed_bytes: usize,
    pub matched: Option<String>,
    pub cutoff: usize,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            pending: String::new(),
            passed_bytes: 0,
            matched: None,
            cutoff: 0,
        }
    }
    pub fn push(&mut self, text: &str, last: bool) -> String {
        if self.matched.is_some() {
            return String::new();
        }
        self.pending.push_str(text);
        let found = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s).map(|p| (p, s.clone())))
            .min_by_key(|v| v.0);
        if let Some((offset, stop)) = found {
            let result = self.pending[..offset].to_owned();
            self.cutoff = self.passed_bytes + offset;
            self.matched = Some(stop);
            self.pending.clear();
            return result;
        }
        let keep = if last {
            0
        } else {
            suffix_prefix_len(
                &self.pending,
                &self.stops.iter().map(String::as_str).collect::<Vec<_>>(),
            )
        };
        let split = self.pending.len() - keep;
        let result = self.pending[..split].to_owned();
        self.pending.drain(..split);
        self.passed_bytes += result.len();
        result
    }
}

fn suffix_prefix_len(value: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|marker| {
            (1..marker.len())
                .filter(|&n| marker.is_char_boundary(n))
                .filter(|&n| value.ends_with(&marker[..n]))
        })
        .max()
        .unwrap_or(0)
}

enum Mode {
    Text,
    Thinking,
    Tool(ToolBoundary),
}

#[derive(Default)]
struct ToolBoundary {
    scanned: usize,
    is_json: Option<bool>,
    in_string: bool,
    escaped: bool,
    closer: &'static str,
}

impl ToolBoundary {
    fn new(closer: &'static str) -> Self {
        Self { closer, ..Self::default() }
    }

    fn find_end(&mut self, pending: &str) -> Option<usize> {
        let end = self.closer;
        while self.scanned < pending.len() {
            let remaining = &pending[self.scanned..];
            let character = remaining.chars().next().unwrap();
            if self.is_json.is_none() && !character.is_whitespace() {
                self.is_json = Some(character == '{');
            }
            if !self.in_string && (character == '<' || character == '\n' || character == '`') {
                if remaining.starts_with(end) {
                    return Some(self.scanned);
                }
                if end.starts_with(remaining) {
                    // Only an unfinished delimiter needs inspection on the next push.
                    return None;
                }
            }
            if self.is_json == Some(true) {
                if self.escaped {
                    self.escaped = false;
                } else if self.in_string && character == '\\' {
                    self.escaped = true;
                } else if character == '"' {
                    self.in_string = !self.in_string;
                }
            }
            self.scanned += character.len_utf8();
        }
        None
    }
}

/// How tool calls are recognised in the model's text.
///
/// `Qwen` is the historical grammar (`<tool_call>` with JSON or `<function=…>`).
/// `BashFence` accepts the shape small instruction-tuned models actually emit -
/// a fenced ```bash / ```sh block or a `<bash>…</bash>` span - and maps it onto
/// whichever declared tool takes a single `command` string.  It exists so a
/// tool-use client can drive a model that has not learned the Qwen markers;
/// it is opt-in through `APXINF_SERVING_TOOL_GRAMMAR` and the default keeps
/// every byte of the Qwen behaviour.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Grammar {
    Qwen,
    BashFence,
    /// Both shapes at once: the model may answer with the Qwen markers or with a
    /// fenced/tagged command, and the parser accepts whichever arrives.  Tool
    /// names are matched case-insensitively here because small models copy the
    /// schema's casing inconsistently (`read` for `Read`).
    Auto,
}

impl Grammar {
    pub fn from_env() -> Self {
        match std::env::var("APXINF_SERVING_TOOL_GRAMMAR").ok().as_deref() {
            Some("bash_fence") => Grammar::BashFence,
            Some("auto") => Grammar::Auto,
            _ => Grammar::Qwen,
        }
    }
}

pub struct OutputParser {
    pending: String,
    mode: Mode,
    tools: Vec<Value>,
    grammar: Grammar,
    /// Treat a terminal `length` truncation inside a structured block as plain
    /// text instead of a protocol fault (default: true; set false to restore
    /// the pre-D4 behaviour where max_tokens mid-block returns 500).
    truncate_is_error: bool,
    pub tool_count: usize,
}

impl OutputParser {
    pub fn new(tools: Vec<Value>) -> Self {
        Self::with_grammar(tools, Grammar::Qwen)
    }

    pub fn with_grammar(tools: Vec<Value>, grammar: Grammar) -> Self {
        Self {
            pending: String::new(),
            mode: Mode::Text,
            tools,
            grammar,
            truncate_is_error: true,
            tool_count: 0,
        }
    }

    /// Downgrade an open structured block at `length` to plain text (D4 fix).
    /// After this call the terminal push returns Ok and the supervisor maps
    /// the frame to a normal `length` finish.
    pub fn allow_length_truncation(&mut self) {
        self.truncate_is_error = false;
    }

    pub fn push(&mut self, text: &str, last: bool) -> Result<Vec<Part>, String> {
        self.pending.push_str(text);
        if self.pending.len() > 262_144 {
            return Err("The output block exceeds the byte limit.".into());
        }
        let mut parts = Vec::new();
        loop {
            match &mut self.mode {
                Mode::Text => {
                    // (opener, closer) pairs; the closer tells the scanner where the body stops.
                    let openers: &[(&str, &str)] = match self.grammar {
                        Grammar::Qwen => &[("<tool_call>", "</tool_call>")],
                        Grammar::Auto => &[
                            ("<tool_call>", "</tool_call>"),
                            ("```bash\n", "\n```"),
                            ("```sh\n", "\n```"),
                            ("```mswea_bash_command\n", "\n```"),
                            ("<bash>", "</bash>"),
                        ],
                        Grammar::BashFence => &[
                            ("```bash\n", "\n```"),
                            ("```sh\n", "\n```"),
                            ("```mswea_bash_command\n", "\n```"),
                            ("<bash>", "</bash>"),
                        ],
                    };
                    let mut markers: Vec<&str> = openers.iter().map(|(open, _)| *open).collect();
                    markers.push("<think>");
                    let found = markers
                        .iter()
                        .filter_map(|m| self.pending.find(m).map(|i| (i, *m)))
                        .min_by_key(|v| v.0);
                    if let Some((i, marker)) = found {
                        if i > 0 {
                            parts.push(Part::Text(self.pending[..i].to_owned()));
                        }
                        self.pending.drain(..i + marker.len());
                        self.mode = if marker == "<think>" {
                            Mode::Thinking
                        } else {
                            let closer = openers
                                .iter()
                                .find(|(open, _)| *open == marker)
                                .map(|(_, close)| *close)
                                .unwrap_or("</tool_call>");
                            Mode::Tool(ToolBoundary::new(closer))
                        };
                    } else {
                        let keep = if last {
                            0
                        } else {
                            suffix_prefix_len(&self.pending, &markers)
                        };
                        let split = self.pending.len() - keep;
                        if split > 0 {
                            parts.push(Part::Text(self.pending[..split].to_owned()));
                            self.pending.drain(..split);
                        }
                        break;
                    }
                }
                Mode::Thinking => {
                    if let Some(i) = self.pending.find("</think>") {
                        self.pending.drain(..i + "</think>".len());
                        self.mode = Mode::Text;
                    } else {
                        let keep = suffix_prefix_len(&self.pending, &["</think>"]);
                        self.pending.drain(..self.pending.len() - keep);
                        break;
                    }
                }
                Mode::Tool(boundary) => {
                    if let Some(i) = boundary.find_end(&self.pending) {
                        let closer = boundary.closer;
                        let body = self.pending[..i].to_owned();
                        let (name, input) = if closer == "</tool_call>" {
                            parse_tool(&body, &self.tools, self.grammar == Grammar::Auto)?
                        } else {
                            parse_command_tool(&body, &self.tools, self.grammar == Grammar::Auto)?
                        };
                        self.pending.drain(..i + closer.len());
                        parts.push(Part::Tool {
                            id: format!("toolu_{}", uuid::Uuid::new_v4().simple()),
                            name,
                            input,
                        });
                        self.tool_count += 1;
                        self.mode = Mode::Text;
                    } else {
                        break;
                    }
                }
            }
        }
        if last && !matches!(self.mode, Mode::Text) {
            if self.truncate_is_error {
                return Err("The model output contains an incomplete structured block.".into());
            }
            // D4 fix: a max_tokens truncation is a normal finish, not a
            // protocol fault. Drop the unfinished tail (the worker ledger
            // keeps the raw bytes) and let the supervisor emit Done.
            self.pending.clear();
            self.mode = Mode::Text;
        }
        Ok(parts)
    }
}

fn parse_tool(body: &str, tools: &[Value], fold_case: bool) -> Result<(String, Value), String> {
    let body = body.trim();
    let (name, input) = if body.starts_with('{') {
        let value = crate::contracts::parse_document(body.as_bytes())
            .map_err(|_| "The model emitted invalid tool JSON.")?;
        let object = value
            .as_object()
            .ok_or("The tool call must be an object.")?;
        if object.keys().any(|k| k != "name" && k != "arguments") {
            return Err("The tool call contains an unknown field.".into());
        }
        let name = value["name"]
            .as_str()
            .ok_or("The tool call has no name.")?
            .to_owned();
        (name, value["arguments"].clone())
    } else {
        let rest = body
            .strip_prefix("<function=")
            .ok_or("The tool call has no function marker.")?;
        let end = rest.find('>').ok_or("The function marker has no end.")?;
        let name = rest[..end].to_owned();
        let schema = find_tool(tools, &name, fold_case)?;
        let mut rest = rest[end + 1..].trim();
        let mut args = Map::new();
        while !rest.starts_with("</function>") {
            rest = rest
                .strip_prefix("<parameter=")
                .ok_or("The tool parameter marker is invalid.")?;
            let end = rest.find('>').ok_or("The parameter marker has no end.")?;
            let key = rest[..end].to_owned();
            rest = &rest[end + 1..];
            let end = rest
                .find("</parameter>")
                .ok_or("The parameter value has no end.")?;
            let raw = rest[..end].strip_prefix('\n').unwrap_or(&rest[..end]);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            let kind = schema["parameters"]["properties"][&key]["type"].as_str();
            let value = if kind == Some("string") {
                Value::String(raw.to_owned())
            } else {
                crate::contracts::parse_document(raw.as_bytes()).map_err(|error| {
                    format!("The tool parameter {key} contains invalid JSON: {error}")
                })?
            };
            if args.insert(key, value).is_some() {
                return Err("The tool call repeats a parameter.".into());
            }
            rest = rest[end + "</parameter>".len()..].trim();
        }
        if rest != "</function>" {
            return Err("The tool call contains trailing data.".into());
        }
        (name, Value::Object(args))
    };
    let schema = find_tool(tools, &name, fold_case)?;
    if !input.is_object() {
        return Err("Tool arguments must be an object.".into());
    }
    validate_arguments(&input, &schema["parameters"])?;
    Ok((canonical_tool_name(tools, &name, fold_case), input))
}

/// Map a fenced command block onto the declared tool that takes one command string.
///
/// A tool-use client such as Claude Code declares `Bash` with a required
/// `command` string; small models emit the command but not the Qwen markers, so
/// the grammar adapter rewrites the fenced body into that tool's arguments and
/// then runs the same declaration and type validation as the JSON path.
fn parse_command_tool(body: &str, tools: &[Value], fold_case: bool) -> Result<(String, Value), String> {
    let command = body.trim();
    if command.is_empty() {
        return Err("The tool call has an empty command.".into());
    }
    let name = tool_for_command(tools)?;
    let mut arguments = Map::new();
    arguments.insert("command".to_owned(), Value::String(command.to_owned()));
    let input = Value::Object(arguments);
    let schema = find_tool(tools, &name, fold_case)?;
    validate_arguments(&input, &schema["parameters"])?;
    Ok((name, input))
}

fn tool_for_command(tools: &[Value]) -> Result<String, String> {
    let names: Vec<&str> = tools
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap_or(""))
        .collect();
    if names.iter().any(|name| *name == "Bash") {
        return Ok("Bash".to_owned());
    }
    tools
        .iter()
        .find(|tool| {
            tool["function"]["parameters"]["properties"]["command"]["type"].as_str() == Some("string")
        })
        .and_then(|tool| tool["function"]["name"].as_str())
        .map(str::to_owned)
        .ok_or_else(|| "No declared tool takes a command string.".into())
}

fn find_tool<'a>(tools: &'a [Value], name: &str, fold_case: bool) -> Result<&'a Value, String> {
    tools
        .iter()
        .map(|t| &t["function"])
        .find(|t| match t["name"].as_str() {
            Some(declared) if fold_case => declared.eq_ignore_ascii_case(name),
            Some(declared) => declared == name,
            None => false,
        })
        .ok_or_else(|| "The model selected an undeclared tool.".into())
}

/// The declared spelling of a tool the model named, so the wire response keeps
/// the schema's casing even when the model wrote `read` for `Read`.
fn canonical_tool_name(tools: &[Value], name: &str, fold_case: bool) -> String {
    find_tool(tools, name, fold_case)
        .ok()
        .and_then(|tool| tool["name"].as_str())
        .unwrap_or(name)
        .to_owned()
}

fn validate_arguments(value: &Value, schema: &Value) -> Result<(), String> {
    if let Some(k) = schema["type"].as_str() {
        let valid = match k {
            "string" => value.is_string(),
            "integer" => value.is_i64() || value.is_u64(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "object" => value.is_object(),
            "array" => value.is_array(),
            "null" => value.is_null(),
            _ => true,
        };
        if !valid {
            return Err(format!("A tool argument does not have type {k}."));
        }
    }
    if let Some(variants) = schema["enum"].as_array() {
        if !variants.contains(value) {
            return Err("A tool argument is outside its enum.".into());
        }
    }
    if let Some(required) = schema["required"].as_array() {
        for key in required {
            if let Some(key) = key.as_str() {
                if value.get(key).is_none() {
                    return Err(format!("The tool call omits required argument {key}."));
                }
            }
        }
    }
    if let Some(object) = value.as_object() {
        for (key, member) in object {
            if let Some(child) = schema["properties"].get(key) {
                validate_arguments(member, child)?;
            } else if schema["additionalProperties"] == false {
                return Err("The tool call contains an undeclared argument.".into());
            }
        }
    }
    if let Some(array) = value.as_array() {
        for item in array {
            validate_arguments(item, &schema["items"])?;
        }
    }
    Ok(())
}

pub fn anthropic_blocks(parts: &[Part]) -> Vec<Value> {
    let mut blocks = Vec::new();
    for part in parts {
        match part {
            Part::Text(text) => {
                if let Some(Value::Object(last)) = blocks.last_mut() {
                    if last.get("type") == Some(&json!("text")) {
                        if let Some(Value::String(previous)) = last.get_mut("text") {
                            previous.push_str(text);
                            continue;
                        }
                    }
                }
                blocks.push(json!({"type":"text","text":text}));
            }
            Part::Tool { id, name, input } => {
                blocks.push(json!({"type":"tool_use","id":id,"name":name,"input":input}))
            }
        }
    }
    blocks
}


fn bash_tools() -> Vec<Value> {
    vec![json!({"type": "function", "function": {"name": "Bash", "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}, "description": {"type": "string"}},
        "required": ["command"]
    }}})]
}

fn collect(grammar: Grammar, pieces: &[&str]) -> (String, Vec<Part>) {
    let mut parser = OutputParser::with_grammar(bash_tools(), grammar);
    let mut parts = Vec::new();
    for (i, piece) in pieces.iter().enumerate() {
        parts.extend(parser.push(piece, i + 1 == pieces.len()).unwrap());
    }
    (parser.pending.clone(), parts)
}

#[test]
fn bash_fence_becomes_a_tool_call() {
    let (_, parts) = collect(Grammar::BashFence, &["I will look.\n```bash\nls -la\n```\ndone"]);
    assert_eq!(parts[0], Part::Text("I will look.\n".into()));
    match &parts[1] {
        Part::Tool { name, input, .. } => {
            assert_eq!(name, "Bash");
            assert_eq!(input["command"], json!("ls -la"));
        }
        other => panic!("expected a tool part, got {other:?}"),
    }
    assert!(matches!(parts.last(), Some(Part::Text(text)) if text == "\ndone"));
}

#[test]
fn bash_fence_survives_character_by_character_streaming() {
    let text = "` `x`\n```bash\necho 27 > answer.txt\n```\n";
    // stream the fence body one character at a time, the way tokens arrive
    let mut parser = OutputParser::with_grammar(bash_tools(), Grammar::BashFence);
    let mut parts = Vec::new();
    for ch in text.chars() {
        parts.extend(parser.push(&ch.to_string(), false).unwrap());
    }
    parts.extend(parser.push("", true).unwrap());
    let calls: Vec<_> = parts
        .iter()
        .filter_map(|part| match part {
            Part::Tool { name, input, .. } => Some((name.clone(), input.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 1, "expected exactly one tool call: {parts:?}");
    assert_eq!(calls[0].0, "Bash");
    assert_eq!(calls[0].1["command"], json!("echo 27 > answer.txt"));
}

#[test]
fn bash_tag_span_is_also_accepted() {
    let (_, parts) = collect(Grammar::BashFence, &["<bash>\nuname -a\n</bash>"]);
    let tool = parts
        .iter()
        .find_map(|part| match part {
            Part::Tool { input, .. } => Some(input.clone()),
            _ => None,
        })
        .expect("tool part");
    assert_eq!(tool["command"], json!("uname -a"));
}

#[test]
fn qwen_grammar_ignores_fenced_commands() {
    let (_, parts) = collect(Grammar::Qwen, &["```bash\nls\n```"]);
    assert!(parts.iter().all(|part| !matches!(part, Part::Tool { .. })));
}

#[test]
fn bash_fence_without_a_command_tool_is_rejected() {
    let mut parser = OutputParser::with_grammar(vec![], Grammar::BashFence);
    let error = parser.push("```bash\nls\n```", false).err();
    assert!(matches!(parser.push("", true), Err(_)) || error.is_some());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_unicode_stop_never_escapes() {
        let mut s = StopMatcher::new(vec!["停下".into()]);
        assert_eq!(s.push("前停", false), "前");
        assert_eq!(s.push("下后", false), "");
        assert_eq!(s.cutoff, 3);
        assert_eq!(s.push("ignored", true), "");
    }
    #[test]
    fn fragmented_xml_preserves_string_and_typed_arguments() {
        let tools = vec![
            json!({"function":{"name":"write","parameters":{"type":"object","required":["text"],"properties":{"text":{"type":"string"},"count":{"type":"integer"}}}}}),
        ];
        let text = "hello<tool_call>\n<function=write>\n<parameter=text>\n001\n</parameter>\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>";
        let mut p = OutputParser::new(tools);
        let mut parts = Vec::new();
        for c in text.chars() {
            parts.extend(p.push(&c.to_string(), false).unwrap());
        }
        parts.extend(p.push("", true).unwrap());
        assert_eq!(anthropic_blocks(&parts)[0]["text"], "hello");
        assert_eq!(
            anthropic_blocks(&parts)[1]["input"],
            json!({"text":"001","count":2})
        );
    }
    #[test]
    fn invalid_tool_does_not_become_a_successful_call() {
        let mut p = OutputParser::new(vec![]);
        assert!(p
            .push(
                "<tool_call>{\"name\":\"unknown\",\"arguments\":{}}</tool_call>",
                true
            )
            .is_err());
        let mut p = OutputParser::new(vec![]);
        assert!(p.push("<tool_call>{", true).is_err());
    }


    fn claude_tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "Read", "parameters": {
                "type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}}),
            json!({"type": "function", "function": {"name": "Bash", "parameters": {
                "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}),
        ]
    }

    #[test]
    fn auto_grammar_accepts_the_native_block_with_a_lowercase_tool_name() {
        // exactly what LFM3 emitted under Claude Code's declarations (PR 113, raw-on/proxy/0001.json)
        let output = "<tool_call><function=read><parameter=file_path>/workspace/x.py</parameter></function></tool_call>";
        let mut parser = OutputParser::with_grammar(claude_tools(), Grammar::Auto);
        let parts = parser.push(output, true).unwrap();
        let (name, input) = parts
            .iter()
            .find_map(|part| match part {
                Part::Tool { name, input, .. } => Some((name.clone(), input.clone())),
                _ => None,
            })
            .expect("tool part");
        assert_eq!(name, "Read", "the declared spelling wins");
        assert_eq!(input["file_path"], json!("/workspace/x.py"));
    }

    #[test]
    fn auto_grammar_still_accepts_a_fence_under_the_same_parser() {
        let mut parser = OutputParser::with_grammar(claude_tools(), Grammar::Auto);
        let parts = parser.push("ok\n```bash\nls\n```", true).unwrap();
        let (name, input) = parts
            .iter()
            .find_map(|part| match part {
                Part::Tool { name, input, .. } => Some((name.clone(), input.clone())),
                _ => None,
            })
            .expect("tool part");
        assert_eq!(name, "Bash");
        assert_eq!(input["command"], json!("ls"));
    }

    #[test]
    fn default_grammar_keeps_exact_tool_matching() {
        // the shipped Qwen path must not start accepting `read` for `Read`
        let mut parser = OutputParser::with_grammar(claude_tools(), Grammar::Qwen);
        let output = "<tool_call><function=read><parameter=file_path>/x</parameter></function></tool_call>";
        let parts = parser.push(output, true);
        assert!(parts.is_err()
            || parts
                .unwrap()
                .iter()
                .all(|part| !matches!(part, Part::Tool { .. })));
    }

    fn typed_parameter_tools() -> Vec<Value> {
        vec![json!({"function":{"name":"configure","parameters":{
            "type":"object","properties":{
                "options":{"type":"object"},
                "weights":{"type":"array","items":{"type":"number"}},
                "retries":{"type":"integer"},
                "ratio":{"type":"number"},
                "enabled":{"type":"boolean"},
                "empty":{"type":"null"},
                "label":{"type":"string"}
            }
        }}})]
    }

    fn parse_json_label(label: &str, fragmented: bool) {
        let body = json!({"name":"configure","arguments":{"label":label}});
        let output = format!("<tool_call> \n{body}\n</tool_call>");
        let mut parser = OutputParser::new(typed_parameter_tools());
        let mut parts = Vec::new();
        if fragmented {
            for character in output.chars() {
                parts.extend(parser.push(&character.to_string(), false).unwrap());
            }
            parts.extend(parser.push("", true).unwrap());
        } else {
            parts = parser.push(&output, true).unwrap();
        }
        assert_eq!(parts.len(), 1);
        assert_eq!(anthropic_blocks(&parts)[0]["input"], json!({"label":label}));
        assert_eq!(parser.tool_count, 1);
    }

    #[test]
    fn json_tool_string_preserves_literal_closing_marker() {
        parse_json_label("example </tool_call> tag", false);
    }

    #[test]
    fn fragmented_json_tool_strings_preserve_quotes_and_backslashes() {
        for label in [
            "example </tool_call> tag",
            "引用 \"</tool_call>\" 后继续",
            r#"slashes \\\ "</tool_call>" trailing \"#,
        ] {
            parse_json_label(label, true);
        }
    }

    #[test]
    fn json_tool_boundaries_reject_invalid_or_unfinished_calls() {
        for output in [
            r#"<tool_call>{"name":"configure","arguments":{"label":"bad \q escape"}}</tool_call>"#,
            r#"<tool_call>{"name":"configure","arguments":{"label":"missing brace"}</tool_call>"#,
            r#"<tool_call>{"name":"configure","arguments":{"label":"unclosed </tool_call>"#,
            r#"<tool_call>{"name":"configure","arguments":{"label":"complete JSON"}}"#,
        ] {
            for fragmented in [false, true] {
                let mut parser = OutputParser::new(typed_parameter_tools());
                let result = if fragmented {
                    output
                        .chars()
                        .try_for_each(|character| {
                            parser.push(&character.to_string(), false).map(|_| ())
                        })
                        .and_then(|_| parser.push("", true).map(|_| ()))
                } else {
                    parser.push(output, true).map(|_| ())
                };
                assert!(result.is_err(), "Invalid call succeeded: {output}");
                assert_eq!(parser.tool_count, 0);
            }
        }
    }

    #[test]
    fn adjacent_json_and_xml_calls_keep_their_own_string_rules() {
        let output = concat!(
            "<tool_call>{\"name\":\"configure\",\"arguments\":{\"label\":\"JSON </tool_call>\"}}</tool_call>",
            "<tool_call><function=configure><parameter=label>XML \" slash \\\n</parameter></function></tool_call>",
            "<tool_call>{\"name\":\"configure\",\"arguments\":{\"label\":\"last JSON\"}}</tool_call>"
        );
        for fragmented in [false, true] {
            let mut parser = OutputParser::new(typed_parameter_tools());
            let mut parts = Vec::new();
            if fragmented {
                for character in output.chars() {
                    parts.extend(parser.push(&character.to_string(), false).unwrap());
                }
                parts.extend(parser.push("", true).unwrap());
            } else {
                parts = parser.push(output, true).unwrap();
            }
            let blocks = anthropic_blocks(&parts);
            assert_eq!(blocks.len(), 3);
            assert_eq!(blocks[0]["input"]["label"], "JSON </tool_call>");
            assert_eq!(blocks[1]["input"]["label"], "XML \" slash \\");
            assert_eq!(blocks[2]["input"]["label"], "last JSON");
            assert_eq!(parser.tool_count, 3);
        }
    }

    #[test]
    fn tool_objects_reject_integer_literals_that_would_be_rounded() {
        for output in [
            "<tool_call><function=configure><parameter=options>{\"revision\":18446744073709551617}</parameter></function></tool_call>",
            "<tool_call>{\"name\":\"configure\",\"arguments\":{\"options\":{\"revision\":-9223372036854775809}}}</tool_call>",
        ] {
            let mut parser = OutputParser::new(typed_parameter_tools());
            let result = parser.push(output, true);
            assert!(result.is_err(), "Unsafe integer became a tool call: {result:?}");
            assert_eq!(parser.tool_count, 0);
        }
    }

    #[test]
    fn xml_object_parameter_rejects_duplicate_json_keys() {
        for raw in [
            r#"{"mode":1,"mode":2}"#,
            r#"{"nested":{"mode":1,"\u006dode":2}}"#,
        ] {
            let mut parser = OutputParser::new(typed_parameter_tools());
            let output = format!(
                "<tool_call><function=configure><parameter=options>{raw}</parameter></function></tool_call>"
            );
            let error = parser.push(&output, true).unwrap_err();
            assert!(error.contains("duplicate"), "{error}");
            assert_eq!(parser.tool_count, 0);
        }
    }

    #[test]
    fn xml_parameters_preserve_valid_json_types_and_literal_strings() {
        let mut parser = OutputParser::new(typed_parameter_tools());
        let output = concat!(
            "<tool_call><function=configure>",
            "<parameter=options>{\"mode\":2,\"nested\":{\"value\":3}}</parameter>",
            "<parameter=weights>[0.25,1,2.5]</parameter>",
            "<parameter=retries>2</parameter>",
            "<parameter=ratio>0.5</parameter>",
            "<parameter=enabled>false</parameter>",
            "<parameter=empty>null</parameter>",
            "<parameter=label>{\"mode\":1,\"mode\":2}</parameter>",
            "</function></tool_call>"
        );
        let parts = parser.push(output, true).unwrap();
        assert_eq!(
            anthropic_blocks(&parts)[0]["input"],
            json!({
                "options":{"mode":2,"nested":{"value":3}},
                "weights":[0.25,1,2.5],"retries":2,"ratio":0.5,
                "enabled":false,"empty":null,"label":"{\"mode\":1,\"mode\":2}"
            })
        );
        assert_eq!(parser.tool_count, 1);
    }

    #[test]
    fn xml_typed_parameters_propagate_parse_and_type_errors() {
        for (key, raw, expected) in [
            ("options", "{broken}", "invalid JSON"),
            ("weights", "[1,]", "invalid JSON"),
            ("retries", "two", "invalid JSON"),
            ("ratio", "NaN", "invalid JSON"),
            ("enabled", "True", "invalid JSON"),
            ("empty", "nil", "invalid JSON"),
            ("options", "[]", "type object"),
            ("weights", "{}", "type array"),
            ("retries", "1.5", "type integer"),
            ("ratio", "\"0.5\"", "type number"),
            ("enabled", "\"false\"", "type boolean"),
            ("empty", "false", "type null"),
        ] {
            let mut parser = OutputParser::new(typed_parameter_tools());
            let output = format!(
                "<tool_call><function=configure><parameter={key}>{raw}</parameter></function></tool_call>"
            );
            let error = parser.push(&output, true).unwrap_err();
            assert!(error.contains(expected), "{key}={raw}: {error}");
            assert_eq!(parser.tool_count, 0);
        }
    }
}
