#!/bin/zsh
# QUEUE #8 G1' quality gate (pre-registered doc/lfm3-native-4bit.md @ f10cae5f):
# solved-rate comparison native-4bit vs native-bf16 on the frozen
# 6-task code suite (MBPP rows 0..4 + HumanEval/80), via the audited
# code_loop.py action template (model emits code; harness writes files
# and runs pristine tests). ABBA to keep the served arm time-ordered
# like the G4-speed window; each arm: 2 gateways, each replaying all 6
# tasks once (segment = gateway instance).
#
# Gates: G3 effect proof (banner native_4bit, census 154/154, binary
# freshness) — same helper as run-g4speed.sh; the gate itself is G1':
# solved-rate delta <= 3pp PASS, 3-10pp PARTIAL candidate, >10pp STOP.
# A/A here = solved-count consistency across the 2 segments per arm;
# a disagreement is reported and the segment split is the tiebreaker
# (each request is independent; no timing median is formed from it).
#
# Runs under both Metal locks, fcntl, in order, waited never stolen
# (RUNNER 3). Usage: run-g1prime.sh [run_dir]  (default ~/lfm3-dig-run)
set -uo pipefail
export PATH=$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export MLX_ROOT=$HOME/mlx-sdk
export DYLD_LIBRARY_PATH=$MLX_ROOT/lib
RUN=${1:-$HOME/lfm3-dig-run}
REPO=$RUN/wt
RES=$RUN/window-4bit/g1prime
mkdir -p "$RES"
CLOG=$RES/g1prime.log
if grep -q "PHASE all_done\|PHASE aborted" "$CLOG" 2>/dev/null; then
  mv "$CLOG" "$RES/g1prime-prev.log"
fi

SUITE=$RES/suite.json
if [ ! -f "$SUITE" ]; then
  echo "building suite (deterministic, offline) at $(date +%FT%T)" >> "$CLOG"
  python3 $RUN/build-g1prime-suite.py \
    --mbpp-cache $HOME/ag2-datasets-cache/google-research-datasets__mbpp__sanitized__test.json \
    --ag2-suite $HOME/ag-run/suite/AG2-suite-v1.json \
    --out "$SUITE" >> "$CLOG" 2>&1 || { echo "suite build failed"; exit 4; }
fi
sha=$(shasum -a 256 "$SUITE" | cut -d' ' -f1)
echo "suite sha256=$sha tasks=6" >> "$CLOG"

echo "PHASE locks_start $(date +%FT%T)" >> "$CLOG"
python3 $RUN/hold_locks.py > $RES/locks.log 2>&1 </dev/null &
LOCKER=$!
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

cd "$REPO" || exit 4
SERVE=$REPO/target/release/apxinf-serve
WORKER=$REPO/target/release/apxinf-native-worker
MODEL4=$RUN/4bit
SNAPSHOT=$HOME/lfm3-spec/weights-local/LFM3-1.7B
F4_SHA=$(shasum -a 256 "$MODEL4/model.safetensors" | cut -d' ' -f1)
FB_SHA=$(shasum -a 256 "$SNAPSHOT/model.safetensors" | cut -d' ' -f1)

# ---- G3: effect proof part 1: binary freshness (same as run-g4speed) ----
NEWEST_SRC=$(find "$REPO/crates" \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "$WORKER" 2>/dev/null | head -1)
if [ ! -x "$SERVE" ] || [ -n "$NEWEST_SRC" ]; then
  echo "G3: source newer than worker ($NEWEST_SRC) — rebuild" >> "$CLOG"
  cargo build --release -p apxinf-model \
    -p apxinf-serving --features native-minicpm5 > "$RES/build.log" 2>&1
  echo "BUILD exit=$? at $(date +%FT%T)" >> "$CLOG"
fi
NEWEST_SRC=$(find "$REPO/crates" \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "$WORKER" 2>/dev/null | head -1)
if [ -n "$NEWEST_SRC" ]; then
  echo "G3 INVALID (empty arm): worker still older than source $NEWEST_SRC" >> "$CLOG"
  echo "PHASE aborted_g3_stale" >> "$CLOG"
  exit 101
fi
APX_SHA=$(shasum -a 256 "$SERVE" | cut -d' ' -f1)
echo "identity apxinf-serve=$APX_SHA worker=$(shasum -a 256 $WORKER | cut -d' ' -f1) model4=$F4_SHA modelbf16=$FB_SHA" >> "$CLOG"

GW_PID=""; CAFF_PID=""
cleanup() {
  [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null
  [ -n "$CAFF_PID" ] && kill "$CAFF_PID" 2>/dev/null
  kill $LOCKER 2>/dev/null
}
trap "cleanup" EXIT INT TERM

pmset displaysleepnow 2>/dev/null || true
caffeinate -si & CAFF_PID=$!

start_gw() { # tag model_root [worker_env...]
  local tag=$1 root=$2; shift 2
  local out="$RES/apxinf-$tag-stdout.log"
  : > "$out"
  : > "$RES/apxinf-$tag-requests.jsonl"   # fresh ledger for the ordered join
  : > "$RES/apxinf-$tag-diag.jsonl"
  "$SERVE" --model "$root" --model-id LFM3-1.7B --provider native --model-name metal_lfm3 \
    --native-worker "$WORKER" --metal-library "$MLX_ROOT/lib/libmlx.dylib" \
    --host 127.0.0.1 --port 8099 --max-context 8192 --max-output-tokens 512 \
    --host-pressure-policy disabled \
    --request-log "$RES/apxinf-$tag-requests.jsonl" \
    --diagnostics-file "$RES/apxinf-$tag-diag.jsonl" "$@" \
    > "$out" 2>&1 &
  GW_PID=$!
  for _ in $(seq 1 150); do
    if curl -sf http://127.0.0.1:8099/readyz >/dev/null 2>&1; then
      echo "gw_up tag=$tag pid=$GW_PID model=$root env=$* at $(date +%FT%T)" >> "$CLOG"
      return 0
    fi
    sleep 2
  done
  echo "gw_READY_TIMEOUT tag=$tag" >> "$CLOG"; return 1
}
stop_gw() { [ -n "$GW_PID" ] && kill "$GW_PID" 2>/dev/null; wait "$GW_PID" 2>/dev/null; GW_PID=""; }

check_banner() { # tag expect_true
  local tag=$1 expect=$2
  local out="$RES/apxinf-$tag-stdout.log"
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

seg_arm() { # arm seg tag model_root expect_4bit [worker_env...]
  local arm=$1 seg=$2 tag=$3 root=$4 expect=$5; shift 5
  start_gw "$tag" "$root" "$@" || { stop_gw; exit 5; }
  check_banner "$tag" "$expect" || { echo "G3_FAIL tag=$tag" >> "$CLOG"; stop_gw; exit 101; }
  echo "seg start arm=$arm seg=$seg at $(date +%FT%T)" >> "$CLOG"
  python3 $RUN/agents-code-loop.py \
    --suite "$SUITE" \
    --gateway http://127.0.0.1:8099 --model LFM3-1.7B \
    --request-log "$RES/apxinf-$tag-requests.jsonl" \
    --out "$RES/$arm-seg$seg.json" \
    --retries 3 --max-tokens 1024 --timeout 300 >> "$RES/$arm-seg$seg.log" 2>&1
  echo "seg arm=$arm seg=$seg exit=$? at $(date +%FT%T)" >> "$CLOG"
  stop_gw
}

# ---- ABBA, 2 segments per arm ----
# A = native-4bit (APXINF_LFM3_4BIT=1), B = native-bf16 (switch unset)
echo "PHASE abba_start $(date +%FT%T)" >> "$CLOG"
seg_arm native-4bit-A 0 A-0 "$MODEL4" true  --worker-env APXINF_LFM3_4BIT=1
seg_arm native-bf16-B 1 B-1 "$SNAPSHOT" false
seg_arm native-bf16-B 2 B-2 "$SNAPSHOT" false
seg_arm native-4bit-A 3 A-3 "$MODEL4" true  --worker-env APXINF_LFM3_4BIT=1
echo "PHASE abba_done $(date +%FT%T)" >> "$CLOG"

# ---- G1' analysis (standalone file, no inline heredoc: round-55 bug 3) ----
python3 $RUN/analyze-g1prime.py "$RES" >> "$CLOG" 2>&1
echo "G1PRIME_EXIT=$?" >> "$CLOG"
echo "PHASE all_done $(date +%FT%T)" >> "$CLOG"
