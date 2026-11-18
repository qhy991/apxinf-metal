#!/usr/bin/env bash
# Drive one engine arm of the LFM3-1.7B engine comparison (doc/engine-comparison-lfm3.md).
#
# One invocation = one arm, one stage, one ABBA segment. The script sends a
# deterministic prompt corpus to an OpenAI-compatible endpoint, keeps the time of
# the first content delta (TTFT) and of the last one (wall), extracts the token
# counts, and writes one raw JSON under
# devlocal/engine-comparison-lfm3/raw/<arm>-<stage>-<ts>.json.
#
# Why one arm per invocation: the campaign discipline (skills/perf-verdict-gate)
# interleaves arms A B C D D C B A inside one quiet window and reports paired
# medians. A driver that runs every arm itself would make the ordering an
# implementation detail of the driver; here the operator owns the order and each
# file records the segment it belongs to.
#
# Portability: stdlib only, tested on CPython 3.9 and 3.14. Every request uses a
# fresh HTTP connection and closes its response: an SSE response is abandoned at
# `[DONE]` (or ollama's `done: true`), and reusing a connection after that raises
# `http.client.ResponseNotReady: Request-sent` on newer Pythons. Transport errors
# are retried once and the retry count is recorded per request.
#
# Measurement surface (see the doc for why):
#   openai-completions  POST /v1/completions  bare completion, no chat template
#                       (llama-server, and ollama's OpenAI compatibility layer)
#   openai-chat         POST /v1/chat/completions  OpenAI chat schema
#                       (the only text surface apxinf-serve publishes; the
#                       family's lane ChatML prefix is included in prompt_tokens)
#   ollama-generate     POST /api/generate   ollama's native schema, used where
#                       its own engine durations are wanted (S3 decomposition)
#
# Timings are measured on the client (first/last SSE content delta), which is the
# only basis available for every arm; engine-reported durations are recorded next
# to them and used by the report for the S3 overhead decomposition. Never
# redesign the corpus between arms: the same corpus seed produces the same text
# for every arm, and the report compares the engines' reported prompt_tokens to
# prove it.
#
# Usage:
#   scripts/bench_engines.sh --arm llamacpp --stage s1 \
#       --endpoint http://127.0.0.1:8081 --model LFM3-1.7B-Q4_K_M \
#       --binary-hash "$(shasum -a 256 llama-server | cut -d' ' -f1)" \
#       --model-hash  "$(shasum -a 256 LFM3-1.7B-Q4_K_M.gguf | cut -d' ' -f1)" \
#       --segment 0 --evidence-log devlocal/.../llama-server.log \
#       --evidence-pattern 'n_past|prompt eval time'
#
#   # apxinf-native: the gateway has no bare-completion route
#   # engine-primary timing basis (buffered-SSE servers; apxinf-serve on the M4)
#   scripts/bench_engines.sh --protocol openai-chat --engine-timings ...
#
#   # ollama with a pinned context (M4: avoids a runner reload per num_ctx)
#   scripts/bench_engines.sh --arm ollama --stage s1 --protocol ollama-generate \
#       --ollama-num-ctx 16384 ...
#
#   scripts/bench_engines.sh --arm apxinf-native --stage s1 --protocol openai-chat \
#       --endpoint http://127.0.0.1:8099 --model LFM3-1.7B \
#       --binary-hash ... --model-hash ... --request-log .../requests.jsonl
#
# Exit codes: 0 recorded and the arm demonstrably generated; 1 recorded but some
# request failed (inspect the raw JSON); 3 recorded but no evidence that a single
# token was generated (a no-op arm is not a result); 64 usage error.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() { sed -n '2,60p' "$0"; }

arm=""
endpoint=""
model=""
stage=""
binary_hash=""
model_hash=""
protocol="openai-completions"
segment="0"
seed="20261118"
corpus_seed="20261118"
corpus_file=""
out_dir="devlocal/engine-comparison-lfm3/raw"
lengths="128,512,2048,8192"
repeats=""
contexts="1024,4096"
gen_lengths="128,512"
count="50"
s3_length="128"
warmup="1"
timeout_s="900"
evidence_log=""
evidence_pattern=""
request_log=""
ollama_num_ctx=""
engine_timings=""
stream=""
quantization=""
prompt_protocol=""
dry_run="0"

need_value() { [[ $# -ge 2 && -n "$2" ]] || { echo "missing value for $1" >&2; exit 64; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm) need_value "$@"; arm="$2"; shift 2 ;;
        --endpoint) need_value "$@"; endpoint="$2"; shift 2 ;;
        --model) need_value "$@"; model="$2"; shift 2 ;;
        --stage) need_value "$@"; stage="$2"; shift 2 ;;
        --binary-hash) need_value "$@"; binary_hash="$2"; shift 2 ;;
        --model-hash) need_value "$@"; model_hash="$2"; shift 2 ;;
        --protocol) need_value "$@"; protocol="$2"; shift 2 ;;
        --segment) need_value "$@"; segment="$2"; shift 2 ;;
        --seed) need_value "$@"; seed="$2"; shift 2 ;;
        --corpus-seed) need_value "$@"; corpus_seed="$2"; shift 2 ;;
        --corpus-file) need_value "$@"; corpus_file="$2"; shift 2 ;;
        --out-dir) need_value "$@"; out_dir="$2"; shift 2 ;;
        --lengths) need_value "$@"; lengths="$2"; shift 2 ;;
        --repeats) need_value "$@"; repeats="$2"; shift 2 ;;
        --contexts) need_value "$@"; contexts="$2"; shift 2 ;;
        --gen-lengths) need_value "$@"; gen_lengths="$2"; shift 2 ;;
        --count) need_value "$@"; count="$2"; shift 2 ;;
        --s3-length) need_value "$@"; s3_length="$2"; shift 2 ;;
        --warmup) need_value "$@"; warmup="$2"; shift 2 ;;
        --timeout) need_value "$@"; timeout_s="$2"; shift 2 ;;
        --evidence-log) need_value "$@"; evidence_log="$2"; shift 2 ;;
        --evidence-pattern) need_value "$@"; evidence_pattern="$2"; shift 2 ;;
        --request-log) need_value "$@"; request_log="$2"; shift 2 ;;
        # M4 adaptation: ollama restarts its llama-server child whenever the
        # per-request num_ctx changes (observed 2026-11-18: one runner start per
        # request), so a context ladder turns prefill into model reload. Pin one
        # num_ctx for every request to keep the arm comparable to llama.cpp's
        # fixed -c. 0/empty keeps the legacy per-case behaviour.
        --ollama-num-ctx) need_value "$@"; ollama_num_ctx="$2"; shift 2 ;;
        # Primary timing basis = the engine's own reported durations. Required when
        # a server buffers its SSE until generation completes (observed on
        # apxinf-serve 2026-11-18: first content delta at wall, decode window ~0),
        # so a client-observed decode window would be degenerate. Applied to both
        # arms of a pair so the table has one basis.
        --engine-timings) engine_timings="1"; shift ;;
        --quantization) need_value "$@"; quantization="$2"; shift 2 ;;
        --prompt-protocol) need_value "$@"; prompt_protocol="$2"; shift 2 ;;
        --stream) stream="1"; shift ;;
        --no-stream) stream="0"; shift ;;
        --no-seed) seed=""; shift ;;
        --dry-run) dry_run="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

[[ -n "$arm" ]] || { echo "--arm is required" >&2; exit 64; }
[[ -n "$endpoint" ]] || { echo "--endpoint is required" >&2; exit 64; }
[[ -n "$model" ]] || { echo "--model is required" >&2; exit 64; }
case "$stage" in s1|s2|s3) ;; *) echo "--stage must be s1, s2 or s3" >&2; exit 64 ;; esac
case "$protocol" in
    openai-completions|openai-chat|ollama-generate) ;;
    *) echo "--protocol must be openai-completions, openai-chat or ollama-generate" >&2; exit 64 ;;
esac

# Hash identity is not optional: a number without an artifact identity cannot be
# re-checked, and this harness exists to make the comparison re-checkable.
for pair in "binary-hash:$binary_hash" "model-hash:$model_hash"; do
    name="${pair%%:*}"; value="${pair#*:}"
    [[ -n "$value" ]] || { echo "--$name is required (sha256 of the artifact under test)" >&2; exit 64; }
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { echo "--$name must be a lowercase sha256 hex digest" >&2; exit 64; }
done

# Same rule as benchmarks/serving: local engines are measured over loopback only.
[[ "$endpoint" =~ ^http://(127\.0\.0\.1|localhost)(:[0-9]+)?$ ]] || {
    echo "--endpoint must be a loopback http URL (http://127.0.0.1:PORT)" >&2; exit 64; }

# Stage defaults. S1/S2 stream so that the client sees TTFT and the decode window;
# S3 is a non-streaming probe of the serving layer, where the engine-reported
# prefill/decode durations are what turn client wall into server overhead.
if [[ -z "$repeats" ]]; then
    case "$stage" in
        s1) repeats="3" ;;
        s2) repeats="2" ;;
        s3) repeats="1" ;;
    esac
fi
if [[ -z "$stream" ]]; then
    case "$stage" in
        s3) stream="0" ;;
        *) stream="1" ;;
    esac
fi
if [[ -z "$prompt_protocol" ]]; then
    case "$protocol" in
        openai-chat) prompt_protocol="lane-chatml" ;;
        *) prompt_protocol="bare" ;;
    esac
fi
if [[ -z "$evidence_pattern" ]]; then
    case "$protocol" in
        openai-completions) evidence_pattern='slot .*(n_past|task)|prompt eval time|eval time|"timings"' ;;
        openai-chat) evidence_pattern='request_settled|"prefill_ms"|prompt_tokens' ;;
        ollama-generate) evidence_pattern='prompt_eval_count|eval_count|"done":true' ;;
    esac
fi
if [[ -n "$corpus_file" ]]; then
    [[ -f "$corpus_file" ]] || { echo "corpus file not found: $corpus_file" >&2; exit 64; }
    corpus_file="$(cd "$(dirname "$corpus_file")" && pwd)/$(basename "$corpus_file")"
fi
if [[ -n "$evidence_log" ]]; then
    [[ -f "$evidence_log" ]] || { echo "evidence log not found: $evidence_log" >&2; exit 64; }
    evidence_log="$(cd "$(dirname "$evidence_log")" && pwd)/$(basename "$evidence_log")"
fi
if [[ -n "$request_log" ]]; then
    [[ -f "$request_log" ]] || { echo "request log not found: $request_log" >&2; exit 64; }
    request_log="$(cd "$(dirname "$request_log")" && pwd)/$(basename "$request_log")"
fi

mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_path="$out_dir/$arm-$stage-$stamp.json"
suffix=1
while [[ -e "$out_path" ]]; do
    suffix=$((suffix + 1))
    out_path="$out_dir/$arm-$stage-$stamp-$suffix.json"
done

export BENCH_REPO_ROOT="$repo_root"
export BENCH_ARM="$arm" BENCH_ENDPOINT="$endpoint" BENCH_MODEL="$model"
export BENCH_STAGE="$stage" BENCH_BINARY_HASH="$binary_hash" BENCH_MODEL_HASH="$model_hash"
export BENCH_PROTOCOL="$protocol" BENCH_SEGMENT="$segment" BENCH_SEED="$seed"
export BENCH_CORPUS_SEED="$corpus_seed" BENCH_CORPUS_FILE="$corpus_file"
export BENCH_OUT="$out_path" BENCH_LENGTHS="$lengths" BENCH_REPEATS="$repeats"
export BENCH_CONTEXTS="$contexts" BENCH_GEN_LENGTHS="$gen_lengths" BENCH_COUNT="$count"
export BENCH_S3_LENGTH="$s3_length" BENCH_WARMUP="$warmup" BENCH_TIMEOUT="$timeout_s"
export BENCH_EVIDENCE_LOG="$evidence_log" BENCH_EVIDENCE_PATTERN="$evidence_pattern"
export BENCH_REQUEST_LOG="$request_log" BENCH_STREAM="$stream" BENCH_DRY_RUN="$dry_run"
export BENCH_OLLAMA_NUM_CTX="$ollama_num_ctx" BENCH_ENGINE_TIMINGS="$engine_timings"
export BENCH_QUANTIZATION="$quantization" BENCH_PROMPT_PROTOCOL="$prompt_protocol"

# The request machinery lives in one embedded Python program: bash cannot time an
# SSE first delta to the millisecond portably (BSD date has no %N) and exact
# v1.1 timings are the point of S1. The heredoc is quoted, so the shell expands
# nothing inside it; every parameter above arrives through the environment.
python3 - <<'PY'
import hashlib, json, os, platform, re, socket, subprocess, sys, time
from datetime import datetime, timezone
from http.client import (HTTPConnection, HTTPException, IncompleteRead,
                         RemoteDisconnected, ResponseNotReady)
from statistics import median

CFG = {key[len("BENCH_"):].lower(): value for key, value in os.environ.items() if key.startswith("BENCH_")}
REPO = CFG["repo_root"]
# The request-log correlation lives in scripts/bench_engines_request_log.py so it
# can be exercised by a fixture test without a host or a model:
#   python3 scripts/test_bench_engines_request_log.py
sys.path.insert(0, os.path.join(REPO, "scripts"))
from bench_engines_request_log import attach_request_log  # noqa: E402

STAGE = CFG["stage"]
PROTOCOL = CFG["protocol"]
STREAM = CFG["stream"] == "1"
OUT = CFG["out"]
DRY = CFG["dry_run"] == "1"
SEED = int(CFG["seed"]) if CFG["seed"] else None
CORPUS_SEED = int(CFG["corpus_seed"])
MAX_WAIT = float(CFG["timeout"])

SCHEMA = "apxinf-engine-bench-arm-v1"


def fail(message, code=64):
    print(f"bench_engines: {message}", file=sys.stderr)
    sys.exit(code)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_ints(value):
    return [int(part) for part in value.replace(" ", "").split(",") if part]


# ---------------------------------------------------------------- case plan ---
def build_cases():
    if STAGE == "s1":
        lengths = split_ints(CFG["lengths"])
        repeats = int(CFG["repeats"])
        return [
            {"id": f"s1-{length}-r{repeat}", "target_tokens": length, "max_tokens": 8, "round": repeat}
            for repeat in range(repeats)
            for length in lengths
        ]
    if STAGE == "s2":
        contexts = split_ints(CFG["contexts"])
        gens = split_ints(CFG["gen_lengths"])
        repeats = int(CFG["repeats"])
        return [
            {"id": f"s2-c{context}-g{gen}-r{repeat}", "target_tokens": context,
             "max_tokens": gen, "context": context, "round": repeat}
            for repeat in range(repeats)
            for context in contexts
            for gen in gens
        ]
    count = int(CFG["count"])
    length = int(CFG["s3_length"])
    return [
        {"id": f"s3-{index:03d}", "target_tokens": length, "max_tokens": 8, "round": 0}
        for index in range(count)
    ]


# ------------------------------------------------------------ prompt corpus ---
# A single seed and a fixed word list make the text identical for every arm. The
# nominal length is in tokens, the generator targets it with a fixed chars/token
# ratio; the engines' reported prompt_tokens are recorded and compared by the
# report. Stage 2 can freeze an exact token-count corpus with --corpus-file
# (built with llama-server /tokenize); the harness then sends byte-identical text
# that is guaranteed to be the same length for every arm.
WORDS = """the of and to in a is that for it as was with be by on not he this are or from at which but have an
they you were her all she there would their we him been has when who will more no if out so said what up its
about into than them can only other new some could time these two may then do first any my now such like our
over man me even most made after also did many before must through back years where much your way well down
should because each just those people how too little state good very make world still own see men work long
get here between both life being under never day same another know while last might us great old year off
come since against go came right used take three states himself few house use during without again place
around however home small found thought went say part once general high upon school every don although got
united left number course war until always away something fact quite""".split()


def lcg(seed):
    state = seed & 0xFFFFFFFFFFFFFFFF

    def step():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return state >> 33

    return step


def generate_prompt(case, chars_per_token=3.85):
    target_chars = max(24, int(round(case["target_tokens"] * chars_per_token)))
    case_seed = int(hashlib.sha256(f"{CORPUS_SEED}:{case['id']}".encode()).hexdigest()[:16], 16)
    step = lcg(case_seed)
    words = []
    length = 0
    while length < target_chars:
        word = WORDS[step() % len(WORDS)]
        words.append(word)
        length += len(word) + 1
    return " ".join(words)


def load_corpus(cases):
    if not CFG["corpus_file"]:
        prompts = {case["id"]: generate_prompt(case) for case in cases}
        return prompts, {"source": "generated", "seed": CORPUS_SEED, "chars_per_token": 3.85}
    document = json.load(open(CFG["corpus_file"]))
    entries = {entry["case"]: entry for entry in document.get("cases", [])}
    prompts = {}
    for case in cases:
        entry = entries.get(case["id"])
        if entry is None:
            fail(f"corpus file {CFG['corpus_file']} has no case {case['id']!r}", 64)
        if entry.get("target_tokens") not in (None, case["target_tokens"]):
            fail(f"corpus case {case['id']!r} targets {entry.get('target_tokens')} tokens, "
                 f"the stage plan asks for {case['target_tokens']}", 64)
        prompts[case["id"]] = entry["prompt"]
    return prompts, {"source": f"file:{CFG['corpus_file']}", "seed": document.get("seed", CORPUS_SEED)}


# ------------------------------------------------------------------- host ---
def run(argv, timeout=15):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return result.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def host_state():
    state = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "started_at": utc_now(),
    }
    if sys.platform == "darwin":
        rows = []
        for line in run(["ps", "-Ao", "pid,pcpu,etime,comm"]).splitlines()[1:]:
            parts = line.split(None, 3)
            if len(parts) != 4:
                continue
            try:
                rows.append({"pid": parts[0], "pcpu": float(parts[1]), "etime": parts[2], "comm": parts[3]})
            except ValueError:
                continue
        rows.sort(key=lambda row: row["pcpu"], reverse=True)
        state["top_cpu"] = rows[:5]
        for line in run(["vm_stat"]).splitlines():
            if "Swapouts" in line:
                state["swapouts"] = line.strip()
        state["thermal"] = run(["pmset", "-g", "therm"]).strip() or None
        state["disk_free"] = run(["df", "-h", "/"]).splitlines()[-1] if run(["df", "-h", "/"]) else None
    return state


def artifact_identity():
    identity = {
        "binary_sha256": CFG["binary_hash"],
        "model_sha256": CFG["model_hash"],
        "quantization": CFG["quantization"] or None,
        "protocol": PROTOCOL,
        "prompt_protocol": CFG["prompt_protocol"] or None,
    }
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15)
        identity["commit"] = commit.stdout.strip() or None
        branch = subprocess.run(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, timeout=15)
        identity["branch"] = branch.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        identity["commit"] = None
    identity["apxinf_env"] = {
        key: value for key, value in sorted(os.environ.items()) if key.startswith("APXINF_")
    }
    return identity


# ------------------------------------------------------------------ HTTP ---
def parse_endpoint(endpoint):
    match = re.match(r"http://([^:/]+)(?::(\d+))?$", endpoint)
    if not match:
        fail(f"unsupported endpoint {endpoint!r}", 64)
    return match.group(1), int(match.group(2) or 80)


HOST, PORT = parse_endpoint(CFG["endpoint"])

# One connection per request. An SSE response is abandoned as soon as `[DONE]`
# (or ollama's `done: true`) arrives, so reusing the connection would leave an
# unread response behind and the next request raises
# `http.client.ResponseNotReady: Request-sent` (observed on CPython 3.14 against
# llama-server). Fresh connections are <1 ms on loopback and keep every arm on
# exactly the same client behaviour. Transport errors are retried once.
TRANSPORT_ERRORS = (HTTPException, OSError, IncompleteRead, RemoteDisconnected,
                    ResponseNotReady)


def new_connection():
    return HTTPConnection(HOST, PORT, timeout=MAX_WAIT)


def perform(path, body, streaming):
    attempt = 0
    while True:
        connection = new_connection()
        try:
            if streaming:
                return consume_stream(connection, path, body)
            return consume_plain(connection, path, body)
        except TRANSPORT_ERRORS as error:
            attempt += 1
            if attempt > 1:
                return {"status": f"error:{type(error).__name__}:{error}",
                        "transport_retry": attempt - 1, "wall_ms": None}
        finally:
            try:
                connection.close()
            except OSError:
                pass


def payload_for(case, prompt):
    if PROTOCOL == "openai-completions":
        body = {
            "model": CFG["model"],
            "prompt": prompt,
            "max_tokens": case["max_tokens"],
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": STREAM,
        }
        if SEED is not None:
            body["seed"] = SEED
        if STREAM:
            body["stream_options"] = {"include_usage": True}
        return "/v1/completions", body
    if PROTOCOL == "openai-chat":
        body = {
            "model": CFG["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": case["max_tokens"],
            "temperature": 0.0,
            "stream": STREAM,
        }
        if SEED is not None:
            body["seed"] = SEED
        if STREAM:
            body["stream_options"] = {"include_usage": True}
        return "/v1/chat/completions", body
    body = {
        "model": CFG["model"],
        "prompt": prompt,
        # `raw` keeps ollama from applying the model's chat template: S1-S3 are
        # bare completion, and the reported prompt_tokens must be the corpus
        # tokens, not corpus + template.
        "raw": True,
        "stream": STREAM,
        "options": {
            "temperature": 0.0,
            "num_predict": case["max_tokens"],
            # Pinned when --ollama-num-ctx is given: a changing num_ctx makes ollama
            # reload its runner (and the reload lands inside TTFT), so the context
            # ladder must not double as a model-reload ladder.
            "num_ctx": int(CFG["ollama_num_ctx"]) if CFG.get("ollama_num_ctx")
            else max(case.get("context", 0), case["target_tokens"] + case["max_tokens"] + 16),
        },
    }
    if SEED is not None:
        body["options"]["seed"] = SEED
    return "/api/generate", body


def content_delta(protocol, event):
    if protocol == "openai-completions":
        choices = event.get("choices") or []
        return bool(choices and (choices[0].get("text") or ""))
    if protocol == "openai-chat":
        choices = event.get("choices") or []
        return bool(choices and ((choices[0].get("delta") or {}).get("content") or ""))
    return bool(event.get("response"))


def consume_stream(connection, path, body):
    """Send a streaming request and time the first/last content delta."""
    raw = json.dumps(body).encode()
    started = time.perf_counter()
    connection.request("POST", path, raw, {"content-type": "application/json", "accept": "text/event-stream"})
    response = connection.getresponse()
    headers_at = time.perf_counter()
    try:
        if response.status != 200:
            detail = response.read()[:2000].decode("utf-8", "replace")
            return {"status": f"error:http_{response.status}", "http_status": response.status,
                    "error": detail, "wall_ms": (time.perf_counter() - started) * 1000.0}
        first_at = None
        last_at = None
        events = []
        while True:
            try:
                line = response.readline()
            except IncompleteRead:
                break
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            if PROTOCOL == "ollama-generate":
                if not text.startswith("{"):
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    continue
                events.append(event)
                if first_at is None and content_delta(PROTOCOL, event):
                    first_at = time.perf_counter()
                if content_delta(PROTOCOL, event):
                    last_at = time.perf_counter()
                if event.get("done"):
                    break
                continue
            if text.startswith("event:"):
                continue
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if first_at is None and content_delta(PROTOCOL, event):
                first_at = time.perf_counter()
            if content_delta(PROTOCOL, event):
                last_at = time.perf_counter()
    finally:
        try:
            response.close()
        except (OSError, HTTPException):
            pass
    closed_at = time.perf_counter()
    timing = {
        "client_ttft_ms": (first_at - started) * 1000.0 if first_at else None,
        "client_decode_ms": (last_at - first_at) * 1000.0 if (first_at and last_at) else None,
        "client_wall_ms": (last_at - started) * 1000.0 if last_at else (closed_at - started) * 1000.0,
        "client_headers_ms": (headers_at - started) * 1000.0,
        "client_close_ms": (closed_at - last_at) * 1000.0 if last_at else None,
    }
    return {"status": "ok", "http_status": 200, "timing": timing, "events": events}


def consume_plain(connection, path, body):
    raw = json.dumps(body).encode()
    started = time.perf_counter()
    connection.request("POST", path, raw, {"content-type": "application/json"})
    response = connection.getresponse()
    headers_at = time.perf_counter()
    try:
        data = response.read()
    finally:
        try:
            response.close()
        except (OSError, HTTPException):
            pass
    finished = time.perf_counter()
    if response.status != 200:
        return {"status": f"error:http_{response.status}", "http_status": response.status,
                "error": data[:2000].decode("utf-8", "replace"),
                "wall_ms": (finished - started) * 1000.0}
    try:
        document = json.loads(data)
    except json.JSONDecodeError as error:
        return {"status": f"error:bad_json:{error}", "http_status": response.status,
                "error": data[:2000].decode("utf-8", "replace"),
                "wall_ms": (finished - started) * 1000.0}
    timing = {
        "client_wall_ms": (finished - started) * 1000.0,
        "client_headers_ms": (headers_at - started) * 1000.0,
    }
    return {"status": "ok", "http_status": 200, "timing": timing, "document": document}


def engine_timings_from_response(document, protocol):
    """Engine-reported durations, when the surface exposes them."""
    if protocol == "ollama-generate":
        if "prompt_eval_duration" not in document:
            return {}
        return {
            "engine_prefill_ms": document.get("prompt_eval_duration", 0) / 1e6,
            "engine_decode_ms": document.get("eval_duration", 0) / 1e6,
            "engine_decode_tokens": document.get("eval_count"),
            "engine_prompt_tokens": document.get("prompt_eval_count"),
            "engine_source": "ollama.api.generate",
        }
    timings = document.get("timings") or {}
    if timings:
        return {
            "engine_prefill_ms": timings.get("prompt_ms"),
            "engine_decode_ms": timings.get("predicted_ms"),
            "engine_decode_tokens": timings.get("predicted_n"),
            "engine_prompt_tokens": timings.get("prompt_n"),
            "engine_source": "llama.cpp.timings",
        }
    usage = document.get("usage") or {}
    if usage:
        return {
            "engine_prompt_tokens": usage.get("prompt_tokens"),
            "engine_completion_tokens": usage.get("completion_tokens"),
        }
    return {}


def usage_from_stream(events):
    """Token counts from the terminal SSE frames (openai: usage; ollama: counts)."""
    usage = {}
    for event in reversed(events):
        if event.get("usage"):
            usage = event["usage"]
            break
    if usage:
        return {"prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}
    if PROTOCOL == "ollama-generate":
        for event in reversed(events):
            if event.get("done"):
                return {"prompt_tokens": event.get("prompt_eval_count"),
                        "completion_tokens": event.get("eval_count")}
    deltas = sum(1 for event in events if content_delta(PROTOCOL, event))
    return {"prompt_tokens": None, "completion_tokens": deltas or None}


def response_id(document, events):
    if document:
        return document.get("id")
    for event in events:
        if event.get("id"):
            return event["id"]
    return None


# ---------------------------------------------------------------- evidence ---
def log_evidence(path, pattern):
    if not path or not pattern:
        return {"path": path or None, "pattern": pattern or None, "matched": False, "line": None}
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 262144))
            text = handle.read().decode("utf-8", "replace")
    except OSError as error:
        return {"path": path, "pattern": pattern, "matched": False, "line": None, "error": str(error)}
    regex = re.compile(pattern)
    line = None
    for candidate in text.splitlines():
        if regex.search(candidate):
            line = candidate
    return {"path": path, "pattern": pattern, "matched": line is not None,
            "line": (line[:500] if line else None)}


def tokenize_fallback(connection, prompts, records):
    """Fill missing prompt token counts from llama-server's /tokenize.

    Some llama.cpp builds omit usage in streamed OpenAI responses; the tokenizer
    is the same one used for the request, so its count is the engine's count.
    """
    cache = {}
    for record in records:
        if record.get("prompt_tokens") is not None:
            continue
        prompt = prompts.get(record["case"])
        if prompt is None:
            continue
        if prompt not in cache:
            try:
                status, document = post_json(connection, "/tokenize", {"content": prompt})
                cache[prompt] = len(document.get("tokens") or []) if status == 200 else None
            except OSError:
                cache[prompt] = None
        if cache[prompt]:
            record["prompt_tokens"] = cache[prompt]
            record["prompt_tokens_source"] = "llama.cpp./tokenize"


def post_json(connection, path, body):
    connection.request("POST", path, json.dumps(body).encode(), {"content-type": "application/json"})
    response = connection.getresponse()
    data = response.read()
    return response.status, (json.loads(data) if data else {})


# -------------------------------------------------------------------- main ---
def main():
    cases = build_cases()
    prompts, corpus_info = load_corpus(cases)
    corpus_sha = hashlib.sha256(
        json.dumps([[case["id"], prompts[case["id"]]] for case in cases], sort_keys=True).encode()
    ).hexdigest()

    if DRY:
        print(json.dumps({
            "arm": CFG["arm"], "stage": STAGE, "protocol": PROTOCOL, "stream": STREAM,
            "cases": [{"id": case["id"], "target_tokens": case["target_tokens"],
                       "max_tokens": case["max_tokens"]} for case in cases],
            "corpus": {**corpus_info, "sha256": corpus_sha},
            "prompt_preview": {case["id"]: prompts[case["id"]][:120] for case in cases[:2]},
        }, indent=2))
        return 0

    identity = artifact_identity()
    host_before = host_state()
    started_at = utc_now()

    records = []
    failures = 0

    def one(case):
        nonlocal failures
        prompt = prompts[case["id"]]
        path, body = payload_for(case, prompt)
        started = time.perf_counter()
        outcome = perform(path, body, STREAM)
        if outcome.get("wall_ms") is None and outcome.get("status", "").startswith("error"):
            return {"case": case["id"], "target_tokens": case["target_tokens"],
                    "max_tokens": case["max_tokens"], "status": outcome["status"],
                    "transport_retry": outcome.get("transport_retry"),
                    "wall_ms": (time.perf_counter() - started) * 1000.0}
        if outcome.get("transport_retry"):
            retries = outcome["transport_retry"]
        else:
            retries = 0
        record = {
            "case": case["id"],
            "target_tokens": case["target_tokens"],
            "max_tokens": case["max_tokens"],
            "status": outcome["status"],
            "http_status": outcome.get("http_status"),
        }
        if retries:
            record["transport_retry"] = retries
        record.update(outcome.get("timing") or {})
        if outcome["status"] != "ok":
            record["error"] = outcome.get("error")
            return record
        if STREAM:
            events = outcome["events"]
            usage = usage_from_stream(events)
            record.update(usage)
            # Some servers repeat engine timings on the terminal frame; keep the
            # last one that carries them (ollama puts durations on its done frame,
            # llama-server on its final timings frame).
            for event in reversed(events):
                if event.get("timings"):
                    record.update(engine_timings_from_response({"timings": event["timings"]}, PROTOCOL))
                    break
            if PROTOCOL == "ollama-generate":
                for event in reversed(events):
                    if event.get("done"):
                        record.update(engine_timings_from_response(event, PROTOCOL))
                        record["finish_reason"] = event.get("done_reason")
                        break
            if record.get("completion_tokens") is None and usage.get("completion_tokens"):
                record["completion_tokens"] = usage["completion_tokens"]
            record["response_id"] = response_id(None, events)
        else:
            document = outcome["document"]
            record.update(engine_timings_from_response(document, PROTOCOL))
            usage = document.get("usage") or {}
            if PROTOCOL == "ollama-generate":
                record["prompt_tokens"] = document.get("prompt_eval_count")
                record["completion_tokens"] = document.get("eval_count")
            else:
                record["prompt_tokens"] = (document.get("usage") or {}).get("prompt_tokens") \
                    or record.get("engine_prompt_tokens")
                record["completion_tokens"] = usage.get("completion_tokens") \
                    or record.get("engine_completion_tokens")
            record["response_id"] = document.get("id")
            record["finish_reason"] = (
                (document.get("choices") or [{}])[0].get("finish_reason")
                if PROTOCOL != "ollama-generate" else document.get("done_reason")
            )
        # Wall on a client basis: last content delta for streams, full body read
        # otherwise. The engine-reported duration never replaces it.
        if record.get("client_wall_ms") is None:
            record["client_wall_ms"] = record.pop("wall_ms", None)
        record["ttft_ms"] = record.get("client_ttft_ms")
        if record["ttft_ms"] is None and record.get("engine_prefill_ms") is not None:
            record["ttft_ms"] = record["engine_prefill_ms"]
            record["ttft_basis"] = "engine"
        if record.get("client_ttft_ms") is not None:
            record["ttft_basis"] = "client"
        if record.get("client_decode_ms") is not None:
            tokens = (record.get("completion_tokens") or 1) - 1
            tokens = max(tokens, 1)
            record["decode_ms"] = record["client_decode_ms"]
            record["decode_tokens"] = tokens
            record["decode_basis"] = "client"
        elif record.get("engine_decode_ms") is not None:
            record["decode_ms"] = record["engine_decode_ms"]
            record["decode_tokens"] = record.get("engine_decode_tokens") or record.get("completion_tokens") or 1
            record["decode_basis"] = "engine"
        if record.get("decode_ms"):
            record["decode_ms_per_token"] = record["decode_ms"] / record["decode_tokens"]
        if record.get("ttft_ms"):
            prompt_tokens = record.get("prompt_tokens")
            if prompt_tokens:
                record["prefill_tok_s"] = prompt_tokens / (record["ttft_ms"] / 1000.0)
        if record.get("decode_ms_per_token"):
            record["decode_tok_s"] = 1000.0 / record["decode_ms_per_token"]
        # Evidence: the engine really generated, not just accepted the request.
        record["evidence"] = json.dumps({
            "response_id": record.get("response_id"),
            "completion_tokens": record.get("completion_tokens"),
            "engine_source": record.get("engine_source"),
            "finish_reason": record.get("finish_reason"),
        }, ensure_ascii=False)[:400]
        return record

    for _ in range(int(CFG["warmup"])):
        warm_case = {"id": "__warmup__", "target_tokens": 128, "max_tokens": 8}
        prompts["__warmup__"] = generate_prompt(warm_case)
        warm = one(warm_case)
        warm["warmup"] = True
        records.append(warm)
        if warm["status"] != "ok":
            failures += 1
    for case in cases:
        record = one(case)
        if record["status"] != "ok":
            failures += 1
        records.append(record)

    samples = [record for record in records if not record.get("warmup")]
    recorded_at = utc_now()
    ledger_join = {"enabled": bool(CFG["request_log"]), "matched": 0,
                   "by_request_id": 0, "by_order": 0}
    if CFG["request_log"]:
        ledger_join["matched"] = attach_request_log(
            samples, CFG["request_log"],
            window=(started_at, recorded_at), n_warmup=int(CFG["warmup"]))
        ledger_join["by_request_id"] = sum(1 for r in samples if r.get("engine_ledger_join") == "request_id")
        ledger_join["by_order"] = sum(1 for r in samples if r.get("engine_ledger_join") == "ordered")

    # Engine-primary timing basis (see --engine-timings). The client-observed
    # values are kept under *_observed so the file still shows what the socket
    # delivered, but they no longer decide the primary metric.
    if CFG.get("engine_timings") == "1":
        for record in samples:
            engine_prefill = record.get("engine_prefill_ms")
            engine_decode = record.get("engine_decode_ms")
            if engine_prefill is None and engine_decode is None:
                continue
            record["client_ttft_ms_observed"] = record.get("client_ttft_ms")
            record["client_decode_ms_observed"] = record.get("client_decode_ms")
            record.pop("client_ttft_ms", None)
            record.pop("client_decode_ms", None)
            if engine_prefill is not None:
                record["ttft_ms"] = engine_prefill
                record["ttft_basis"] = "engine"
                if record.get("prompt_tokens"):
                    record["prefill_tok_s"] = record["prompt_tokens"] / (engine_prefill / 1000.0)
            if engine_decode is not None:
                tokens = record.get("completion_tokens") or record.get("engine_decode_tokens") or 1
                tokens = max(tokens - 1, 1)
                record["decode_ms"] = engine_decode
                record["decode_tokens"] = tokens
                record["decode_ms_per_token"] = engine_decode / tokens
                record["decode_tok_s"] = 1000.0 / record["decode_ms_per_token"]
                record["decode_basis"] = "engine"

    # Fallback prompt counts for streamed responses that omitted usage.
    if any(record.get("prompt_tokens") is None and record.get("status") == "ok" for record in samples) \
            and STAGE in ("s1", "s2", "s3"):
        try:
            probe = HTTPConnection(HOST, PORT, timeout=MAX_WAIT)
            tokenize_fallback(probe, prompts, samples)
            probe.close()
        except OSError:
            pass

    host_after = host_state()
    generated = sum(1 for record in samples
                    if record.get("status") == "ok" and (record.get("completion_tokens") or 0) > 0)
    evidence = log_evidence(CFG["evidence_log"], CFG["evidence_pattern"])
    response_evidence = next(
        (record["evidence"] for record in samples if record.get("status") == "ok"
         and (record.get("completion_tokens") or 0) > 0), None)
    engaged = generated > 0 or evidence["matched"]

    document = {
        "schema": SCHEMA,
        "arm": CFG["arm"],
        "stage": STAGE,
        "protocol": PROTOCOL,
        "prompt_protocol": CFG["prompt_protocol"] or None,
        "endpoint": CFG["endpoint"],
        "model": CFG["model"],
        "started_at": started_at,
        "finished_at": recorded_at,
        "artifact": identity,
        "host": {"before": host_before, "after": host_after},
        "corpus": {**corpus_info, "sha256": corpus_sha, "case_count": len(cases)},
        "params": {
            "temperature": 0.0,
            "seed": SEED,
            "stream": STREAM,
            "segment": int(CFG["segment"]),
            "warmup_requests": int(CFG["warmup"]),
            "timeout_s": MAX_WAIT,
        },
        "cases": cases,
        "timing_primary": "engine" if CFG.get("engine_timings") == "1" else "client-first",
        "request_log": {"path": CFG["request_log"] or None, **ledger_join},
        "requests": records,
        "evidence": {
            "engaged": engaged,
            "generated_requests": generated,
            "response": response_evidence,
            "log": evidence,
        },
        "failures": failures,
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"arm={CFG['arm']} stage={STAGE} protocol={PROTOCOL} stream={int(STREAM)} segment={CFG['segment']}")
    print(f"cases={len(cases)} ok={generated} failed={failures} engaged={'yes' if engaged else 'NO'} "
          f"corpus_sha256={corpus_sha[:16]}")
    print(f"wrote {OUT}")
    if not engaged:
        print("no evidence that the arm generated a token: this is a no-op arm, not a result", file=sys.stderr)
        return 3
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
PY

status=$?
if [[ $status -eq 0 ]]; then
    echo "arm recorded: $out_path"
elif [[ $status -eq 3 ]]; then
    echo "arm NOT recorded as a result: no generation evidence in $out_path" >&2
fi
exit "$status"
