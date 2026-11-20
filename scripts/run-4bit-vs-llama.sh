#!/bin/zsh
# Lead 11:05 same-window 3-arm window: native-4bit vs native-bf16 vs llama.cpp Q4_K_M.
# Frozen by doc/lfm3-4bit-vs-llama-q4.md (Gate + Decision). One-shot; run on M4
# under both Metal locks. No crates/ changes, no gate/threshold changes.
#
# Order A B C C B A A B C (9 segments = 3 per arm); S2 cells ctx1024/4096,
# gen128; engine basis (--engine-timings); per-segment corpus seed
# 20261130+seg; A/A <=5% per arm per cell (RUNNER §4 — window voided, no
# median-fixing). Analysis: scripts/analyze-4bit-vs-llama.py (no heredoc).
#
# Usage: scripts/run-4bit-vs-llama.sh [run_dir]   (default ~/lfm3-dig-run)
set -uo pipefail
export PATH=$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export MLX_ROOT=$HOME/mlx-sdk
export DYLD_LIBRARY_PATH=$MLX_ROOT/lib
RUN=${1:-$HOME/lfm3-dig-run}
REPO=$RUN/wt
RES=$RUN/window-4bit/vs-llama
CLOG=$RES/vs-llama.log
RAW=$RES/raw
LOGS=$RES/logs
mkdir -p "$RAW" "$LOGS" "$RES/prev"

if grep -q "PHASE all_done\|PHASE aborted" "$CLOG" 2>/dev/null; then
  mv "$CLOG" "$RES/vs-llama-prev.log"
fi

echo "PHASE locks_start $(date +%FT%T)" >> "$CLOG"
python3 $RUN/hold_locks.py > $RES/locks.log 2>&1 </dev/null &
LOCKER=$!
GW_PID=""; LLAMA_PID=""; CAFF_PID=""
cleanup() {
  [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null
  [ -n "$LLAMA_PID" ] && kill "$LLAMA_PID" 2>/dev/null
  [ -n "$CAFF_PID" ] && kill "$CAFF_PID" 2>/dev/null
  [ -n "$LOCKER" ] && kill "$LOCKER" 2>/dev/null
}
trap cleanup EXIT INT TERM
deadline=$(( $(date +%s) + 21600 ))
while true; do
  if grep -q "^LOCKS_ACQUIRED" $RES/locks.log 2>/dev/null; then break; fi
  if ! kill -0 $LOCKER 2>/dev/null; then
    echo "LOCKER_DIED" >> "$CLOG"; cat $RES/locks.log >> "$CLOG"; exit 2
  fi
  if (( $(date +%s) > deadline )); then
    echo "LOCK_WAIT_DEADLINE_6H" >> "$CLOG"; exit 3
  fi
  sleep 5
done
echo "LOCKS_HELD_OK $(cat $RES/locks.log) at $(date +%FT%T)" >> "$CLOG"

pmset displaysleepnow 2>/dev/null || true
caffeinate -si & CAFF_PID=$!

cd "$REPO" || exit 4
SERVE=$REPO/target/release/apxinf-serve
WORKER=$REPO/target/release/apxinf-native-worker
MODEL4=$RUN/4bit
SNAPSHOT=$HOME/lfm3-spec/weights-local/LFM3-1.7B
LLAMA_BIN=$HOME/lfm3bench/engines/llama.cpp/build/bin/llama-server
LLAMA_Q4=$HOME/lfm3bench/models/LFM3-1.7B-Q4_K_M.gguf
LLAMA_PORT=8081

# ---- G3: effect proof part 1: binary freshness (same as run-g1prime.sh) ----
NEWEST_SRC=$(find "$REPO/crates" \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "$WORKER" 2>/dev/null | head -1)
if [ ! -x "$SERVE" ] || [ -n "$NEWEST_SRC" ]; then
  echo "G3: source newer than worker ($NEWEST_SRC) — rebuild" >> "$CLOG"
  cargo build --release -p apxinf-model -p apxinf-serving --features native-minicpm5 > "$RES/build.log" 2>&1
  echo "BUILD exit=$? at $(date +%FT%T)" >> "$CLOG"
fi
NEWEST_SRC=$(find "$REPO/crates" \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "$WORKER" 2>/dev/null | head -1)
if [ -n "$NEWEST_SRC" ]; then
  echo "G3 INVALID (empty arm): worker still older than source $NEWEST_SRC" >> "$CLOG"
  echo "PHASE aborted_g3_stale" >> "$CLOG"; exit 101
fi
[ -x "$SERVE" ] || { echo "G3 INVALID: apxinf-serve missing" >> "$CLOG"; echo "PHASE aborted_g3" >> "$CLOG"; exit 101; }
[ -x "$LLAMA_BIN" ] || { echo "G3 INVALID: llama-server missing at $LLAMA_BIN" >> "$CLOG"; echo "PHASE aborted_g3" >> "$CLOG"; exit 101; }
[ -f "$LLAMA_Q4" ] || { echo "G3 INVALID: Q4_K_M gguf missing at $LLAMA_Q4" >> "$CLOG"; echo "PHASE aborted_g3" >> "$CLOG"; exit 101; }

APX_SHA=$(shasum -a 256 "$SERVE" | cut -d' ' -f1)
LLAMA_SHA=$(shasum -a 256 "$LLAMA_BIN" | cut -d' ' -f1)
LLAMA_MSHA=$(shasum -a 256 "$LLAMA_Q4" | cut -d' ' -f1)
echo "identity apxinf-serve=$APX_SHA worker=$(shasum -a 256 $WORKER | cut -d' ' -f1) llama-server=$LLAMA_SHA q4_gguf=$LLAMA_MSHA at $(date +%FT%T)" >> "$CLOG"

# ---- stale-file hygiene (round-64): preserve prior window, do not delete ----
mv "$RAW"/*.json(N) "$RES/prev/" 2>/dev/null || true
mv "$LOGS"/*.log "$LOGS"/*.jsonl(N) "$RES/prev/" 2>/dev/null || true
: > "$RES/vs-llama-requests.jsonl"
: > "$RES/vs-llama-llama-requests.jsonl"
: > "$RES/vs-llama-diag.jsonl"

start_gw() { # tag model_root [worker_env...]
  local tag=$1 root=$2; shift 2
  local out="$LOGS/gw-$tag-stdout.log"
  : > "$out"
  : > "$RES/vs-llama-requests.jsonl"
  : > "$RES/vs-llama-diag.jsonl"
  "$SERVE" --model "$root" --model-id LFM3-1.7B --provider native --model-name metal_lfm3 \
    --native-worker "$WORKER" --metal-library "$MLX_ROOT/lib/libmlx.dylib" \
    --host 127.0.0.1 --port 8099 --max-context 8192 --max-output-tokens 512 \
    --host-pressure-policy disabled \
    --request-log "$RES/vs-llama-requests.jsonl" \
    --diagnostics-file "$RES/vs-llama-diag.jsonl" "$@" \
    > "$out" 2>&1 &
  GW_PID=$!
  for _ in $(seq 1 150); do
    if curl -sf http://127.0.0.1:8099/readyz >/dev/null 2>&1; then
      echo "apxinf gateway up tag=$tag pid=$GW_PID model=$root env=$* at $(date +%FT%T)" >> "$CLOG"
      return 0
    fi
    sleep 2
  done
  echo "apxinf gateway READY_TIMEOUT tag=$tag" >> "$CLOG"; return 1
}
stop_gw() { [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null; wait "$GW_PID" 2>/dev/null; GW_PID=""; }

check_banner() { # tag expect_true
  local tag=$1 expect=$2
  local out="$LOGS/gw-$tag-stdout.log"
  local banner census
  banner=$(python3 "$RUN/extract-gw-grep.py" "$out" 'decode_schedule=' 2>/dev/null | head -1)
  census=$(python3 "$RUN/extract-gw-grep.py" "$out" 'quantized_triples' 2>/dev/null | head -1)
  echo "banner[$tag]: $banner" >> "$CLOG"
  echo "census[$tag]: $census" >> "$CLOG"
  if ! echo "$banner" | grep -q "native_4bit=$expect"; then
    echo "G3 INVALID (empty arm): banner missing native_4bit=$expect" >> "$CLOG"
    return 1
  fi
  if [ "$expect" = "true" ] && ! echo "$census" | grep -q "quantized_triples=154/154"; then
    echo "G3 INVALID (empty arm): quantized census missing" >> "$CLOG"
    return 1
  fi
  return 0
}

start_llama() { # one llama-server resident for its 3 segments (PR-127 pattern)
  local out="$LOGS/llama-server-C.log"
  : > "$out"
  "$LLAMA_BIN" -m "$LLAMA_Q4" -c 16384 -ngl 99 --host 127.0.0.1 --port "$LLAMA_PORT" \
    > "$out" 2>&1 &
  LLAMA_PID=$!
  echo "llama-server up pid=$LLAMA_PID port=$LLAMA_PORT at $(date +%FT%T)" >> "$CLOG"
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "llama-server READY_TIMEOUT" >> "$CLOG"; return 1
}
stop_llama() { [ -n "$LLAMA_PID" ] && kill "$LLAMA_PID" 2>/dev/null; wait "$LLAMA_PID" 2>/dev/null; LLAMA_PID=""; }

bench_seg() { # arm label seg tag model_root expect_4bit binary_hash model_hash quant [worker_env...]
  local arm=$1 seg=$2 tag=$3 root=$4 expect=$5 bsha=$6 msha=$7 quant=$8; shift 8
  start_gw "$tag" "$root" "$@" || { stop_gw; exit 5; }
  check_banner "$tag" "$expect" || { echo "G3_FAIL tag=$tag" >> "$CLOG"; stop_gw; exit 101; }
  echo "seg start arm=$arm seg=$seg tag=$tag at $(date +%FT%T)" >> "$CLOG"
  scripts/bench_engines.sh --arm "$arm" --stage s2 --segment "$seg" \
    --protocol openai-chat --endpoint http://127.0.0.1:8099 --model LFM3-1.7B \
    --binary-hash "$bsha" --model-hash "$msha" --quantization "$quant" \
    --request-log "$RES/vs-llama-requests.jsonl" \
    --evidence-log "$LOGS/gw-$tag-stdout.log" --evidence-pattern 'request_settled|"prefill_ms"|prompt_tokens' \
    --engine-timings --gen-lengths 128 --repeats 2 --corpus-seed $((20261130 + seg)) \
    --contexts 1024,4096 --out-dir "$RAW" > "$LOGS/bench-$arm-seg$seg.log" 2>&1
  local st=$?
  echo "seg arm=$arm seg=$seg exit=$st at $(date +%FT%T)" >> "$CLOG"
  stop_gw
  return $st
}

bench_llama_seg() { # seg
  local seg=$1
  echo "seg start arm=llamacpp-C seg=$seg at $(date +%FT%T)" >> "$CLOG"
  scripts/bench_engines.sh --arm llamacpp-C --stage s2 --segment "$seg" \
    --protocol openai-completions --endpoint "http://127.0.0.1:$LLAMA_PORT" --model LFM3-1.7B-Q4_K_M \
    --binary-hash "$LLAMA_SHA" --model-hash "$LLAMA_MSHA" --quantization gguf-q4_k_m \
    --request-log "$RES/vs-llama-llama-requests.jsonl" \
    --evidence-log "$LOGS/llama-server-C.log" --evidence-pattern 'n_past|prompt eval time|eval time|"timings"' \
    --engine-timings --gen-lengths 128 --repeats 2 --corpus-seed $((20261130 + seg)) \
    --contexts 1024,4096 --out-dir "$RAW" > "$LOGS/bench-llamacpp-C-seg$seg.log" 2>&1
  local st=$?
  echo "seg arm=llamacpp-C seg=$seg exit=$st at $(date +%FT%T)" >> "$CLOG"
  return $st
}

F4_SHA=$(shasum -a 256 "$MODEL4/model.safetensors" | cut -d' ' -f1)
FB_SHA=$(shasum -a 256 "$SNAPSHOT/model.safetensors" | cut -d' ' -f1)

echo "PHASE window_start $(date +%FT%T)" >> "$CLOG"
start_llama || { stop_llama; echo "PHASE aborted_llama_ready" >> "$CLOG"; exit 5; }

# ---- frozen order A B C C B A A B C; seg index is frozen identity ----
bench_seg native-4bit-A 0 A-0 "$MODEL4" true  "$APX_SHA" "$F4_SHA" "4bit-p4-af-g64-f16" --worker-env APXINF_LFM3_4BIT=1 || exit 5
bench_seg native-bf16-B 1 B-1 "$SNAPSHOT" false "$APX_SHA" "$FB_SHA" "bf16" || exit 5
bench_llama_seg 2 || exit 5
bench_llama_seg 5 || exit 5
bench_seg native-bf16-B 4 B-4 "$SNAPSHOT" false "$APX_SHA" "$FB_SHA" "bf16" || exit 5
bench_seg native-4bit-A 3 A-3 "$MODEL4" true  "$APX_SHA" "$F4_SHA" "4bit-p4-af-g64-f16" --worker-env APXINF_LFM3_4BIT=1 || exit 5
bench_seg native-4bit-A 6 A-6 "$MODEL4" true  "$APX_SHA" "$F4_SHA" "4bit-p4-af-g64-f16" --worker-env APXINF_LFM3_4BIT=1 || exit 5
bench_seg native-bf16-B 7 B-7 "$SNAPSHOT" false "$APX_SHA" "$FB_SHA" "bf16" || exit 5
bench_llama_seg 8 || exit 5

stop_llama
echo "PHASE segments_done $(date +%FT%T)" >> "$CLOG"

# ---- analysis: standalone file, no heredoc (round-55 lesson) ----
python3 "$REPO/scripts/analyze-4bit-vs-llama.py" "$RAW" >> "$CLOG" 2>&1
AN=$?
echo "ANALYZE_EXIT=$AN at $(date +%FT%T)" >> "$CLOG"
if [ $AN -eq 2 ]; then
  echo "PHASE aborted_analysis_invalid" >> "$CLOG"
  exit 102
fi
echo "PHASE all_done $(date +%FT%T)" >> "$CLOG"
exit 0
