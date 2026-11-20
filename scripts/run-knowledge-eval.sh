#!/bin/zsh
# QUEUE #8 knowledge/reasoning eval harness (prereg
# doc/lfm3-4bit-knowledge-eval.md; Lead 10:52 scale revision).
#
# Default shape = Lead 10:52 smoke test: MMLU 10 + GSM8K 10, greedy
# temp=0 max_tokens=64, native-4bit (A, APXINF_LFM3_4BIT=1) vs
# native-bf16 (B, switch unset), 1 segment per arm, arms interleaved
# A then B to avoid drift (Lead: no multi-segment ABBA at this scale).
# The full prereg shape stays reachable via args, e.g.
#   scripts/run-knowledge-eval.sh ~/lfm3-dig-run 2 500 200
# (ABBA, 2 segments per arm, all frozen samples).
#
# Subset pick: fixed seeds (below), sorted idx, recorded per segment
# json + analyzer receipt. Engine basis = gateway --request-log joined
# by response request id (ordered next-unclaimed fallback, the
# code-loop.py join); no client-side timing is used.
# G3 effect proof (banner native_4bit, census 154/154, binary freshness)
# copied from scripts/run-g1prime.sh. Sample sha gate (doc §5) runs
# BEFORE any lock is taken.
# Locks: fcntl, in order, waited never stolen (RUNNER 3).
# Usage: scripts/run-knowledge-eval.sh [run_dir] [segs_per_arm] [n_mmlu] [n_gsm8k]
#   defaults: ~/lfm3-dig-run, 1, 10, 10
set -uo pipefail
export PATH=$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export MLX_ROOT=$HOME/mlx-sdk
export DYLD_LIBRARY_PATH=$MLX_ROOT/lib
RUN=${1:-$HOME/lfm3-dig-run}
SEGS_PER_ARM=${2:-1}
N_MMLU=${3:-10}
N_GSM8K=${4:-10}
MMLU_SUBSET_SEED=20261122
GSM8K_SUBSET_SEED=20261123
REPO=$RUN/wt
RES=$RUN/window-4bit/knowledge
SAMPLES=$RUN/samples
if [[ "$SEGS_PER_ARM" != (1|2|3|4) ]]; then
  echo "segs_per_arm must be 1, 2, 3 or 4 (got $SEGS_PER_ARM)"; exit 4
fi
if (( N_MMLU < 1 || N_MMLU > 500 || N_GSM8K < 1 || N_GSM8K > 200 )); then
  echo "n_mmlu must be 1..500 and n_gsm8k 1..200 (got $N_MMLU $N_GSM8K)"; exit 4
fi
mkdir -p "$RES"
CLOG=$RES/knowledge.log
if grep -q "PHASE all_done\|PHASE aborted" "$CLOG" 2>/dev/null; then
  mv "$CLOG" "$RES/knowledge-prev.log"
fi

# ---- sample sha gate (doc §5 frozen values; G3-class provenance) ----
echo "PHASE samples_check start $(date +%FT%T)" >> "$CLOG"
SAMPLE_SPECS=(
  "mmlu-500.jsonl dbb9d807cf276b722e0f9ab40998878a8675e2fff1b7696c965dda58515d8eda"
  "gsm8k-200.jsonl 1ce79a9f4d7bf1b280fe0de308bb12e67d3cb015c316fb6ed737965850f98519"
  "mmlu-500.ids.json 011104644a9f092eb62c941af44c9da683ba344bba2112e311ecdbc12f707fa6"
  "gsm8k-200.ids.json 88243d86aaa248c74e2ce386c3e5a29f1b0511620254dca0d8c172016a4b9da0"
)
for spec in $SAMPLE_SPECS; do
  f=${spec%% *}; want=${spec##* }
  if [[ ! -f "$SAMPLES/$f" ]]; then
    echo "SAMPLE_MISSING $SAMPLES/$f" >> "$CLOG"
    echo "PHASE aborted_samples" >> "$CLOG"; exit 101
  fi
  got=$(shasum -a 256 "$SAMPLES/$f" | cut -d' ' -f1)
  if [[ "$got" != "$want" ]]; then
    echo "SAMPLE_SHA_MISMATCH $f got=$got want=$want" >> "$CLOG"
    echo "PHASE aborted_samples" >> "$CLOG"; exit 101
  fi
done
echo "samples sha ok (4/4) at $(date +%FT%T)" >> "$CLOG"

# ---- dual Metal locks, fcntl, in order, waited never stolen ----
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

# ---- G3: effect proof part 1: binary freshness (same as run-g1prime) ----
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
echo "harness $(basename $0) args=run_dir=$RUN segs_per_arm=$SEGS_PER_ARM n_mmlu=$N_MMLU n_gsm8k=$N_GSM8K mmlu_subset_seed=$MMLU_SUBSET_SEED gsm8k_subset_seed=$GSM8K_SUBSET_SEED commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) at $(date +%FT%T)" >> "$CLOG"

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
  python3 "$RUN/scripts/analyze-knowledge-eval.py" segment \
    --res "$RES" --arm "$arm" --seg "$seg" \
    --gateway http://127.0.0.1:8099 --model LFM3-1.7B \
    --samples "$SAMPLES" --n-mmlu $N_MMLU --n-gsm8k $N_GSM8K \
    --mmlu-seed $MMLU_SUBSET_SEED --gsm8k-seed $GSM8K_SUBSET_SEED \
    --request-log "$RES/apxinf-$tag-requests.jsonl" \
    --repo "$REPO" --out "$RES/$arm-seg$seg.json" \
    --max-tokens 64 --timeout 180 >> "$RES/$arm-seg$seg.log" 2>&1
  echo "seg arm=$arm seg=$seg exit=$? at $(date +%FT%T)" >> "$CLOG"
  stop_gw
}

# ---- ABBA order, $SEGS_PER_ARM segments per arm ----
# A = native-4bit (APXINF_LFM3_4BIT=1), B = native-bf16 (switch unset).
# Smoke default (1 seg/arm) still alternates A then B per Lead 10:52.
echo "PHASE abba_start segments_per_arm=$SEGS_PER_ARM n_mmlu=$N_MMLU n_gsm8k=$N_GSM8K at $(date +%FT%T)" >> "$CLOG"
total=$(( SEGS_PER_ARM * 2 ))
# Preserve (do not delete) prior results: stale files would poison the
# analyzer's file discovery (round-64 lesson from run-g1prime.sh).
mkdir -p "$RES/prev"
mv "$RES"/native-*-seg*.json(N) "$RES"/native-*-seg*.log(N) \
   "$RES"/knowledge-verdict.json(N) "$RES"/apxinf-*-stdout.log(N) \
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

# ---- analysis (standalone file, no inline heredoc: round-55 lesson) ----
python3 "$RUN/scripts/analyze-knowledge-eval.py" analyze "$RES" >> "$CLOG" 2>&1
echo "KNOWLEDGE_EXIT=$?" >> "$CLOG"
echo "PHASE all_done $(date +%FT%T)" >> "$CLOG"
