#!/bin/zsh
# Issue #134 ANE+GPU heterogeneous measurements (Lead 11:12 + Lead 13:38 M4
# split). Frozen by doc/ane-gpu-heterogeneous-measurements.md (Gate + Decision).
# Run on M4 under both Metal locks. No crates/ runtime changes, no gate changes.
#
# W1 lock window: A0 streaming probe -> T1 llama.cpp batch eta -> T2 GPU/ANE/
#   GPU+ANE concurrency.
# W2 lock window: T4 ANE small-shape matmul (+ placement facts) -> T3 shared
#   event round-trip. Marked BLOCKED (exit 103) when the probe binary is not
#   built or unreachable — no data is invented for a blocked measurement.
#
# The probe binary is ane_probe (crates/apxinf-model/examples). Build it
# lock-free before running:
#   SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
#   MLX_ROOT=$HOME/mlx-sdk \
#   cargo build --release -p apxinf-model --features coreml --example ane_probe
#
# Usage: scripts/run-ane-measurements.sh [w1|w2] [run_dir] [probe_bin]
#   default window=w1, run_dir=$HOME/lfm3-dig-run, probe_bin=$RUN/ane-wt/target/release/examples/ane_probe
#
# Driver status (driver rc is the window verdict, not the presence of files —
# round-108/110/111 lesson):
#   0   WINDOW COMPLETE: every probe arm ok and every cell A/A <= 5%
#   101 G3 FAILED (effect proof / binary freshness / tool missing)
#   102 WINDOW INVALID (one or more probe arm failed, or A/A > 5%)
#   103 WINDOW BLOCKED (probe binary not built / not executable)
#   104 WINDOW INVALID (llama server failed to become ready, or T1 request error)
set -uo pipefail
export PATH=$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export MLX_ROOT=$HOME/mlx-sdk
export DYLD_LIBRARY_PATH=$MLX_ROOT/lib

WINDOW=${1:-w1}
RUN=${2:-$HOME/lfm3-dig-run}
REPO=${REPO:-$RUN/ane-wt}
RES=$RUN/ane
RAW=$RES/raw
LOGS=$RES/logs
PROBE=${3:-$REPO/target/release/examples/ane_probe}
CLOG=$RES/ane-$WINDOW.log
LLAMA_BIN=$HOME/lfm3bench/engines/llama.cpp/build/bin/llama-server
LLAMA_Q4=$HOME/lfm3bench/models/LFM3-1.7B-Q4_K_M.gguf
# Frozen T1 grid from doc/ane-gpu-heterogeneous-measurements.md.
T1_BATCHES=(1 4 8 16)
# Frozen T4 grid from the same doc.
T4_CELLS=(1x2048 4x2048 8x2048 16x2048 32x2048 64x2048
          1x4096 4x4096 8x4096 16x4096 32x4096 64x4096)
T4_REPS=5
T4_BATCH=8
T2_REPS=5
T3_REPS=12
T3_ROUNDS=48
A0_REPS=20

mkdir -p "$RAW" "$LOGS" "$RES/prev"
if grep -q "PHASE all_done\|PHASE aborted" "$CLOG" 2>/dev/null; then
  mv "$CLOG" "$RES/ane-$WINDOW-prev.log"
fi

g3_fail() { # msg, exit code
  echo "G3 INVALID: $1" >> "$CLOG"
  echo "PHASE aborted_g3" >> "$CLOG"
  exit $2
}
g3_fail_101() { g3_fail "$1" 101; }
g3_fail_104() { g3_fail "$1" 104; }

echo "PHASE locks_start $(date +%FT%T)" >> "$CLOG"
python3 $RUN/hold_locks.py > $RES/locks-$WINDOW.log 2>&1 </dev/null &
LOCKER=$!
LLAMA_PID=""; CAFF_PID=""
cleanup() {
  [ -n "$LLAMA_PID" ] && kill "$LLAMA_PID" 2>/dev/null
  [ -n "$CAFF_PID" ] && kill "$CAFF_PID" 2>/dev/null
  [ -n "$LOCKER" ] && kill "$LOCKER" 2>/dev/null
}
trap cleanup EXIT INT TERM
deadline=$(( $(date +%s) + 21600 ))
while true; do
  if grep -q "^LOCKS_ACQUIRED" $RES/locks-$WINDOW.log 2>/dev/null; then break; fi
  if ! kill -0 $LOCKER 2>/dev/null; then
    echo "LOCKER_DIED" >> "$CLOG"; cat $RES/locks-$WINDOW.log >> "$CLOG"; exit 2
  fi
  if (( $(date +%s) > deadline )); then
    echo "LOCK_WAIT_DEADLINE_6H" >> "$CLOG"; exit 3
  fi
  sleep 5
done
echo "LOCKS_HELD_OK $(cat $RES/locks-$WINDOW.log) at $(date +%FT%T)" >> "$CLOG"

pmset displaysleepnow 2>/dev/null || true
caffeinate -si & CAFF_PID=$!

# ---- G3: tool and binary gates ----
[ -x "$LLAMA_BIN" ] || g3_fail_101 "llama-server missing at $LLAMA_BIN"
[ -f "$LLAMA_Q4" ] || g3_fail_101 "Q4_K_M gguf missing at $LLAMA_Q4"
[ -x "$PROBE" ] || {
  echo "G3 BLOCKED: ane_probe not executable at $PROBE (build lock-free first)" >> "$CLOG"
  echo "PHASE aborted_blocked_probe" >> "$CLOG"
  exit 103
}
LLAMA_SHA=$(shasum -a 256 "$LLAMA_BIN" | cut -d' ' -f1)
LLAMA_MSHA=$(shasum -a 256 "$LLAMA_Q4" | cut -d' ' -f1)
PROBE_SHA=$(shasum -a 256 "$PROBE" | cut -d' ' -f1)
HW=$(sysctl -n machdep.cpu.brand_string)
[[ "$HW" == *M4* ]] || g3_fail_101 "machine gate: expected Apple M4, got $HW"
{
  echo "machine: $HW"
  echo "probe_bin: $PROBE sha=$PROBE_SHA"
  echo "llama_bin: $LLAMA_BIN sha=$LLAMA_SHA"
  echo "q4_gguf: $LLAMA_Q4 sha=$LLAMA_MSHA"
  date +%FT%T
} > "$RES/hw-$WINDOW.txt"
echo "identity hw=$HW probe=$PROBE_SHA llama=$LLAMA_SHA q4_gguf=$LLAMA_MSHA at $(date +%FT%T)" >> "$CLOG"

# ---- stale-file hygiene (round-64): preserve prior window, do not delete ----
mkdir -p "$RES/prev-$WINDOW"
mv "$RAW"/*.json(N) "$RES/prev-$WINDOW/" 2>/dev/null || true
mv "$LOGS"/*(N) "$RES/prev-$WINDOW/" 2>/dev/null || true
: > "$RES/ane-$WINDOW-probe.jsonl"
: > "$RES/ane-$WINDOW-llama-requests.jsonl"
: > "$RES/ane-$WINDOW-diag.jsonl"

# probe_arm <arm_name> <probe args...> writes one JSON object per line to
# ane-<window>-probe.jsonl. The driver status is read from the probe exit
# code (0 = probe arm ok), never from the presence of the output file.
probe_arm() {
  local arm=$1; shift
  local out="$RAW/$arm.json"
  echo "probe arm=$arm start at $(date +%FT%T)" >> "$CLOG"
  "$PROBE" "$@" --out "$out" >> "$RES/ane-$WINDOW-probe.jsonl" 2>> "$LOGS/$arm.stderr.log"
  local rc=$?
  echo "probe arm=$arm rc=$rc at $(date +%FT%T)" >> "$CLOG"
  return $rc
}

start_llama() { # one llama-server resident for its T1 batch cells
  local out="$LOGS/llama-server.log"
  : > "$out"
  "$LLAMA_BIN" -m "$LLAMA_Q4" -c 16384 -ngl 99 --host 127.0.0.1 --port 8081 \
    > "$out" 2>&1 &
  LLAMA_PID=$!
  echo "llama-server up pid=$LLAMA_PID at $(date +%FT%T)" >> "$CLOG"
  for _ in $(seq 1 90); do
    if curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "llama-server READY_TIMEOUT" >> "$CLOG"; return 1
}
stop_llama() { [ -n "$LLAMA_PID" ] && kill "$LLAMA_PID" 2>/dev/null; wait "$LLAMA_PID" 2>/dev/null; LLAMA_PID=""; }

t1_cell() { # batch B — llama.cpp parallel batch decode, engine basis
  local b=$1
  local req_log="$RES/ane-$WINDOW-llama-requests.jsonl"
  local log="$LOGS/t1-B$b.log"
  : > "$req_log"
  echo "T1 start B=$b at $(date +%FT%T)" >> "$CLOG"
  REPO="$REPO" scripts/bench_engines.sh --arm "t1-B$b" --stage s2 --segment "$b" \
    --protocol openai-completions --endpoint http://127.0.0.1:8081 --model LFM3-1.7B-Q4_K_M \
    --binary-hash "$LLAMA_SHA" --model-hash "$LLAMA_MSHA" --quantization gguf-q4_k_m \
    --request-log "$req_log" \
    --evidence-log "$LOGS/llama-server.log" --evidence-pattern 'n_past|prompt eval time|eval time|"timings"' \
    --engine-timings --gen-lengths 128 --repeats 3 --corpus-seed $((20261200 + b)) \
    --contexts 1024 --out-dir "$RAW" > "$log" 2>&1
  local st=$?
  echo "T1 end B=$b rc=$st at $(date +%FT%T)" >> "$CLOG"
  return $st
}

w1_status=0
w2_status=0

case "$WINDOW" in
  w1)
    # ---- A0: same-window streaming probe (bytes + bandwidth denominator) ----
    probe_arm A0-stream --gpu --reps "$A0_REPS" || w1_status=1
    # ---- T1: llama.cpp batch decode, engine basis, 4 batch values ----
    start_llama || { stop_llama; g3_fail_104 "llama-server did not become ready"; }
    for b in "${T1_BATCHES[@]}"; do
      t1_cell "$b" || w1_status=1
    done
    stop_llama
    # ---- T2: GPU / ANE / GPU+ANE concurrency ----
    probe_arm T2-gpu --gpu --reps "$T2_REPS" || w1_status=1
    probe_arm T2-ane --ane-bw --reps "$T2_REPS" || w1_status=1
    probe_arm T2-both --both --reps "$T2_REPS" || w1_status=1
    ;;
  w2)
    # ---- T4 first (as frozen), with placement facts for effect proof ----
    for cell in "${T4_CELLS[@]}"; do
      probe_arm "T4-$cell-placement" placement --cell "$cell" || w2_status=1
      probe_arm "T4-$cell" matmul --cell "$cell" --reps "$T4_REPS" --batch "$T4_BATCH" || w2_status=1
    done
    # ---- T3: shared-event round-trip x 48 rounds ----
    probe_arm T3-event event --reps "$T3_REPS" --rounds "$T3_ROUNDS" || w2_status=1
    ;;
  *)
    echo "unknown window $WINDOW (want w1|w2)" >> "$CLOG"
    echo "PHASE aborted_usage" >> "$CLOG"
    exit 101
    ;;
esac

# ---- analysis: standalone file, no heredoc (round-55 lesson) ----
python3 "$REPO/scripts/analyze-ane-measurements.py" "$RES" --window "$WINDOW" > "$RES/ane-$WINDOW-analysis.txt" 2>&1
AN=$?
echo "ANALYZE_EXIT=$AN at $(date +%FT%T)" >> "$CLOG"
cat "$RES/ane-$WINDOW-analysis.txt" >> "$CLOG"

# The analysis exit code is the final window verdict. 0 = every probe arm ok
# and every cell A/A <= 5%; 102 = one or more arms failed or A/A > 5%.
if [ "$AN" -eq 103 ]; then
  echo "PHASE aborted_blocked" >> "$CLOG"
  exit 103
fi
if [ "$AN" -ne 0 ]; then
  echo "OUTCOME: INVALID_PARTIAL rc=$AN" >> "$CLOG"
  echo "PHASE aborted_analysis_invalid" >> "$CLOG"
  exit 102
fi
echo "OUTCOME: COMPLETE" >> "$CLOG"
echo "PHASE all_done $(date +%FT%T)" >> "$CLOG"
exit 0
