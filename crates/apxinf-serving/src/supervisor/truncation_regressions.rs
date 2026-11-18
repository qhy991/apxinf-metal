//! Serving contract for a generation truncated inside a structured block.
//!
//! The LFM3 family opens its `<think>` block as the first token, so a small
//! `max_tokens` ends the generation inside the block. The worker completes
//! normally (`completed` / cause `length`), and the client must receive a normal
//! truncated completion (HTTP 200, `finish_reason=length`, the partial block
//! contents as text) instead of `internal_error: The model output contains an
//! incomplete structured block.`.
//!
//! The allowance cut is the only cause that is reclassified: an `eos` ending or
//! a matched stop sequence inside a block is a malformed emission and keeps
//! failing. A truncated tool-call block never becomes a half-parsed tool call.
//!
//! These tests drive the real coordinator through a scripted worker, so the
//! cause threaded from the terminal frame is what is under test, not a parser
//! unit called directly.
use super::lifecycle_regressions::cpu_service;
use super::*;
use futures_util::FutureExt;

const WORKER_SCRIPT: &str = r#"
import hashlib, json, struct, sys
mode = sys.argv[1]
TOKENS = [3, 4]
THINK = ["<think>We need", " to check", " the log"]
TOOL = ["<tool_call>{\"name\":\"lookup\",\"arguments\":{\"query\":\"x\"}"]
def read():
    line = sys.stdin.readline()
    return json.loads(line) if line else None
def emit(command, kind, **fields):
    print(json.dumps(dict(protocol=command['protocol'], worker_epoch=command['worker_epoch'],
                         kind=kind, **fields)), flush=True)
if mode == 'length_tool':
    chunks = TOOL
elif mode == 'stop_think':
    chunks = ['<think>We need STOP']
else:
    chunks = THINK
while True:
    command = read()
    if command is None: break
    if command['kind'] == 'prepare_input':
        digest = hashlib.sha256(b'apxinf-token-prefix-v1\0' + struct.pack('<Q', len(TOKENS)) +
                              b''.join(struct.pack('<I', token) for token in TOKENS)).hexdigest()
        emit(command, 'prepared_input', command_id=command['command_id'],
             model_revision=command['model_revision'], token_ids=TOKENS,
             effective_input_tokens=len(TOKENS), prompt_digest=digest)
    elif command['kind'] == 'submit':
        assert 'sampling' not in command
        key = dict(request_id=command['request_id'], attempt=command['attempt'])
        emit(command, 'accepted', event_seq=0, **key)
        for index, chunk in enumerate(chunks):
            emit(command, 'tokens', event_seq=index + 1, output_index=index,
                 token_ids=[10], text_delta=chunk, **key)
        cause = 'length' if mode in ('length_think', 'length_tool') else 'eos'
        if mode == 'stop_think':
            control = read()
            assert control['kind'] == 'stop_generation', control['kind']
            emit(control, 'command_result', command_id=control['command_id'], status='accepted')
            cause = 'stop_sequence'
        emit(command, 'terminal', event_seq=len(chunks) + 1, status='completed', cause=cause,
             usage=dict(input_tokens=len(TOKENS), output_tokens=len(chunks)),
             state_result=dict(validity='none', consumed_position=len(TOKENS),
                               history_count=len(TOKENS) + len(chunks)),
             text_delta='',
             metrics=dict(elapsed_ns=20, ttft_ns=10, peak_memory_bytes=2048), **key)
        emit(command, 'resources_released', event_seq=len(chunks) + 2,
             released_lease_ids=command['capacity_lease_ids'], retained_lease_ids=[], **key)
    else:
        raise RuntimeError(command['kind'])
"#;

fn lookup_tool() -> Value {
    json!({"type":"function","function":{"name":"lookup","parameters":{
        "type":"object","required":["query"],
        "properties":{"query":{"type":"string"}}
    }}})
}

async fn truncated_output(mode: &str) -> Vec<Event> {
    let max_tokens = match mode {
        "length_think" => 3,
        "length_tool" => 1,
        _ => 8,
    };
    let epoch = uuid::Uuid::new_v4().to_string();
    let mut child = Command::new("python3")
        .args(["-u", "-c", WORKER_SCRIPT, mode])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let worker = Worker {
        protocol: contracts::PROTOCOL,
        input: child.stdin.take(),
        frames: read_worker_frames(child.stdout.take().unwrap()),
        child,
        _diagnostics: None,
        epoch: epoch.clone(),
        controls: Default::default(),
        sent_commands: 0,
        retired: false,
    };
    let ready = json!({"protocol":contracts::PROTOCOL,"worker_epoch":epoch,
        "model_revision":"1".repeat(64),"capability_revision":"2".repeat(64),
        "eos_token_ids":[2],"capabilities":{"max_context":64}});
    let mut service = cpu_service(ready, Duration::from_secs(2), 2);
    let (completion, stopped) = watch::channel(None);
    Arc::get_mut(&mut service).unwrap().stopped = stopped;
    let mut ticket = service
        .enqueue(Request {
            selection: crate::sampling::Selection::Greedy,
            messages: vec![json!({"role":"user","content":"Check the truncation contract."})],
            tools: vec![lookup_tool()],
            max_tokens,
            stops: if mode == "stop_think" {
                vec!["STOP".into()]
            } else {
                vec![]
            },
            count_only: false,
        })
        .unwrap();
    let task = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        || async { Err::<(Worker, Value), _>("The truncation probe must not rotate.".to_owned()) },
        completion,
    ));
    let result =
        std::panic::AssertUnwindSafe(tokio::time::timeout(Duration::from_secs(3), async {
            let begin = ticket.begin.await.unwrap().unwrap();
            assert_eq!(begin.input_tokens, 2);
            let mut events = Vec::new();
            while let Some(event) = ticket.output.recv().await {
                events.push(event);
            }
            events
        }))
        .catch_unwind()
        .await;
    service.shutdown();
    tokio::time::timeout(Duration::from_secs(3), service.wait_stopped())
        .await
        .unwrap()
        .unwrap();
    task.await.unwrap();
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    result.unwrap().unwrap()
}

fn text_of(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Part(Part::Text(value)) => Some(value.as_str()),
            _ => None,
        })
        .collect()
}

/// `Event` has no `Debug`; this is the diagnostic form used in assertions.
fn describe(events: &[Event]) -> String {
    events
        .iter()
        .map(|event| match event {
            Event::Part(Part::Text(text)) => format!("text({text})"),
            Event::Part(Part::Tool { name, .. }) => format!("tool({name})"),
            Event::Decision(_) => "decision".to_owned(),
            Event::Done {
                cause,
                output_tokens,
                ..
            } => format!("done({cause}, {output_tokens} tokens)"),
            Event::Error(error) => format!("error({}: {})", error.code, error.message),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[tokio::test]
async fn allowance_cut_inside_thinking_surfaces_partial_text_as_length() {
    let events = truncated_output("length_think").await;
    assert!(
        !events.iter().any(|event| matches!(event, Event::Error(_))),
        "an allowance cut must not be reported as a worker fault: {}",
        describe(&events)
    );
    assert_eq!(text_of(&events), "We need to check the log");
    assert_eq!(events.len(), 2, "{}", describe(&events));
    assert!(
        matches!(&events[1], Event::Done { cause, output_tokens: 3, .. } if cause == "length"),
        "{}", describe(&events)
    );
}

#[tokio::test]
async fn allowance_cut_inside_a_tool_call_emits_text_not_a_call() {
    let events = truncated_output("length_tool").await;
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::Part(Part::Tool { .. }))),
        "a truncated tool block must not emit a half-parsed call: {}", describe(&events)
    );
    assert!(
        !events.iter().any(|event| matches!(event, Event::Error(_))),
        "an allowance cut must not be reported as a worker fault: {}", describe(&events)
    );
    assert_eq!(
        text_of(&events),
        "{\"name\":\"lookup\",\"arguments\":{\"query\":\"x\"}"
    );
    assert!(
        matches!(&events[1], Event::Done { cause, .. } if cause == "length"),
        "{}", describe(&events)
    );
}

#[tokio::test]
async fn eos_inside_a_block_stays_an_internal_error() {
    for mode in ["eos_think", "stop_think"] {
        let events = truncated_output(mode).await;
        assert_eq!(events.len(), 1, "{mode}: {}", describe(&events));
        assert!(
            matches!(&events[0], Event::Error(error)
                if error.code == "internal_error"
                && error.message.contains("incomplete structured block")),
            "{mode}: {}", describe(&events)
        );
    }
}
