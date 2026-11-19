#!/bin/zsh
# QUEUE #8 G1' quality gate (pre-registered doc/lfm3-native-4bit.md @ f10cae5f):
# solved-rate comparison native-4bit vs native-bf16 on the frozen
# 6-task code suite (MBPP rows 0..4 + HumanEval/80), via the audited
# code_loop.py action template (model emits code; harness writes files
# and runs pristine tests). ABBA to keep the served arm time-ordered
# like the G4-speed window; each arm: --segments gateway instances, each
# replaying all tasks once (segment = gateway instance). Round-64 default:
# 3 segments per arm on the 20-task suite (Lead 02:18: >=20 tasks, >=3
# segments per arm).
#
# Gates: G3 effect proof (banner native_4bit, census 154/154, binary
# freshness) — same helper as run-g4speed.sh; the gate itself is G1':
# solved-rate delta <= 3pp PASS, 3-10pp PARTIAL candidate, >10pp STOP.
# A/A here = solved-count consistency across the per-arm segments;
# a disagreement is reported and the segment split is the tiebreaker
# (each request is independent; no timing median is formed from it).
#
# Runs under both Metal locks, fcntl, in order, waited never stolen
# (RUNNER 3). Usage: run-g1prime.sh [run_dir] [segments_per_arm]
#   (defaults: ~/lfm3-dig-run, 3; pass 2 to reproduce the n=6 ABBA shape)
set -uo pipefail
export PATH=$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export MLX_ROOT=$HOME/mlx-sdk
export DYLD_LIBRARY_PATH=$MLX_ROOT/lib
RUN=${1:-$HOME/lfm3-dig-run}
SEGS_PER_ARM=${2:-3}
REPO=$RUN/wt
RES=$RUN/window-4bit/g1prime
if [[ "$SEGS_PER_ARM" != (2|3|4) ]]; then
  echo "segments_per_arm must be 2, 3 or 4 (got $SEGS_PER_ARM)"; exit 4
fi
mkdir -p "$RES"
CLOG=$RES/g1prime.log
if grep -q "PHASE all_done\|PHASE aborted" "$CLOG" 2>/dev/null; then
  mv "$CLOG" "$RES/g1prime-prev.log"
fi

SUITE=$RES/suite.json
SUITE_K=$(( SEGS_PER_ARM == 2 ? 5 : 19 ))   # 6 tasks / 2 segs (v1); 20 tasks / >=3 segs (v2, Lead 02:18)
echo "building suite (deterministic, offline, k=$SUITE_K) at $(date +%FT%T)" >> "$CLOG"
python3 $RUN/build-g1prime-suite.py --force \
    --k $SUITE_K \
    --mbpp-cache $HOME/ag2-datasets-cache/google-research-datasets__mbpp__sanitized__test.json \
    --ag2-suite $HOME/ag-run/suite/AG2-suite-v1.json \
    --out "$SUITE" >> "$CLOG" 2>&1 || { echo "suite build failed"; exit 4; }
sha=$(shasum -a 256 "$SUITE" | cut -d' ' -f1)
ntasks=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['tasks']))" "$SUITE")
echo "suite sha256=$sha tasks=$ntasks k=$SUITE_K" >> "$CLOG"
if (( ntasks < 20 )) && (( SEGS_PER_ARM >= 3 )); then
  echo "G1' INVALID: >=3 segments requires the 20-task suite, got $ntasks" >> "$CLOG"
  echo "PHASE aborted_suite_size" >> "$CLOG"; exit 101
fi

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
    --retries 3 --max-tokens 480 --timeout 300 >> "$RES/$arm-seg$seg.log" 2>&1
  echo "seg arm=$arm seg=$seg exit=$? at $(date +%FT%T)" >> "$CLOG"
  stop_gw
}

# ---- ABBA, $SEGS_PER_ARM segments per arm ----
# A = native-4bit (APXINF_LFM3_4BIT=1), B = native-bf16 (switch unset).
# For 3+ segments use A B A B ... (interleaved), ending with A when the
# total is odd: time-order is still balanced and no arm gets all its
# segments in one block.
echo "PHASE abba_start segments_per_arm=$SEGS_PER_ARM at $(date +%FT%T)" >> "$CLOG"
total=$(( SEGS_PER_ARM * 2 ))
# Preserve (do not delete) any previous run's segment results: stale
# native-*-seg*.json from a prior window would otherwise be picked up by
# analyze-g1prime.py's file discovery and poison the verdict. Round-64.
mkdir -p "$RES/prev"
mv "$RES"/native-*-seg*.json(N) "$RES"/native-*-seg*.log(N) \
   "$RES"/g1prime-verdict.json(N) "$RES"/apxinf-*-stdout.log(N) \
   "$RES"/apxinf-*-requests.jsonl(N) "$RES"/apxinf-*-diag.jsonl(N) \
   "$RES/prev/" 2>/dev/null || true
for (( i = 0; i < total; i++ )); do
  seg=$(( i / 2 ))
  if (( i % 2 == 0 )); then
    seg_arm native-4bit-A $seg A-$seg "$MODEL4" true  --worker-env APXINF_LFM3_4BIT=1
  else
    seg_arm native-bf16-B $seg B-$seg "$SNAPSHOT" false
  fi
done
echo "PHASE abba_done $(date +%FT%T)" >> "$CLOG"

# ---- G1' analysis (standalone file, no inline heredoc: round-55 bug 3) ----
python3 $RUN/analyze-g1prime.py "$RES" >> "$CLOG" 2>&1
echo "G1PRIME_EXIT=$?" >> "$CLOG"
echo "PHASE all_done $(date +%FT%T)" >> "$CLOG"
