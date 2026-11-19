#!/bin/zsh
# QUEUE #8 G4-speed window (doc/lfm3-native-4bit.md, Lead product gate).
# One-shot ABBA harness; run on M4 under the two Metal locks.
# Analysis block must stay a pure-python heredoc (no shell-expanded chars)
# — round-55's deployed copy had a `floaint` typo that SyntaxError'd; the
# fixed block below is what round-58 verified.
set -u
export PATH=$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export MLX_ROOT=$HOME/mlx-sdk
export DYLD_LIBRARY_PATH=$MLX_ROOT/lib
RUN=${1:-$HOME/lfm3-dig-run}

CLOG=${CLOG:-$HOME/lfm3-dig-run/window-4bit/g4speed.log}
RES=$HOME/lfm3-dig-run/window-4bit
RAW=$RES/raw
LOGS=$RES/logs
mkdir -p "$RAW" "$LOGS"
if grep -q "PHASE all_done\|PHASE aborted" "$CLOG" 2>/dev/null; then
  mv "$CLOG" "$RES/g4speed-prev.log"
fi
echo "PHASE locks_start $(date +%FT%T)" >> "$CLOG"
python3 $RUN/hold_locks.py > $RES/locks-g4.log 2>&1 </dev/null &
LOCKPID=$!
trap 'kill $LOCKPID 2>/dev/null' EXIT INT TERM
wait $LOCKPID || { echo "LOCK_FAILED"; exit 103; }
echo "PHASE locks_held $(date +%FT%T)" >> "$CLOG"

pmset displaysleepnow
caffeinate -si &>/dev/null &

MODEL4=${MODEL4:-$HOME/lfm3-dig-run/4bit}
SNAPSHOT=${SNAPSHOT:-$HOME/lfm3-spec/weights-local/LFM3-1.7B}
APX_SHA=${APX_SHA:-7f11e798}
F4_SHA=${F4_SHA:-96b021b3}
FB_SHA=${FB_SHA:-135922ca}

bench_seg() { # arm segment corpus_seed tag model_root expect_4bit [worker_env...]
  local arm=$1 seg=$2 seed=$3 tag=$4 root=$5 expect=$6; shift 6
  start_gw "$tag" "$root" "$@" || return 5
  check_banner "$tag" "$expect" || { echo "G3_FAIL tag=$tag" >> "$CLOG"; stop_gw; exit 101; }
  scripts/bench_engines.sh --arm "$arm" --stage s2 --segment "$seg" \
    --protocol openai-chat --endpoint http://127.0.0.1:8099 --model LFM3-1.7B \
    --binary-hash "$APX_SHA" \
    --model-hash "$([ "$expect" = true ] && echo $F4_SHA || echo $FB_SHA)" \
    --request-log "$RES/apxinf-$tag-requests.jsonl" \
    --evidence-log "$LOGS/apxinf-$tag-diag.jsonl" --evidence-pattern 'request_settled' \
    --engine-timings --gen-lengths 128 --repeats 2 --corpus-seed "$seed" \
    --contexts 1024,4096 --out-dir "$RAW" > "$LOGS/bench-$arm-$seg.log" 2>&1
  echo "bench arm=$arm seg=$seg tag=$tag exit=$? at $(date +%FT%T)" >> "$CLOG"
  stop_gw
}

start_gw() { # tag model_root [worker_env...]
  local tag=$1 root=$2; shift 2
  apxinf-serve --model "$root" --host 127.0.0.1 --port 8099 \
    --max-output-tokens 512 "$@" \
    > "$LOGS/gw-$tag.log" 2>&1 &
  GW=$!
  echo "gw_up tag=$tag pid=$GW model=$root env=$* at $(date +%FT%T)" >> "$CLOG"
  for i in {1..60}; do
    curl -sf http://127.0.0.1:8099/health >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "gw timeout tag=$tag" >> "$CLOG"
  return 1
}
stop_gw() { kill $GW 2>/dev/null; wait $GW 2>/dev/null; }

check_banner() { # tag expect_4bit
  local tag=$1 expect=$2
  local out="$LOGS/gw-$tag.log"
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


# ---- ABBA, order A B B A A B (doc 4); 3 segments per arm ----
# A = native-4bit (APXINF_LFM3_4BIT=1, model 4bit), B = native-bf16 (switch unset)
seedA=20261120; seedB=20261121
echo "PHASE abba_start $(date +%FT%T)" >> "$CLOG"
bench_seg native-4bit-A 0 $seedA A-0 "$MODEL4" true  --worker-env APXINF_LFM3_4BIT=1 || exit 5
bench_seg native-bf16-B 1 $seedB B-1 "$SNAPSHOT" false || exit 5
bench_seg native-bf16-B 2 $seedB B-2 "$SNAPSHOT" false || exit 5
bench_seg native-4bit-A 3 $seedA A-3 "$MODEL4" true  --worker-env APXINF_LFM3_4BIT=1 || exit 5
bench_seg native-4bit-A 4 $seedA A-4 "$MODEL4" true  --worker-env APXINF_LFM3_4BIT=1 || exit 5
bench_seg native-bf16-B 5 $seedB B-5 "$SNAPSHOT" false || exit 5
echo "PHASE abba_done $(date +%FT%T)" >> "$CLOG"

# ---- PAIRED + A/A (G5) + speedup ----
python3 - <<'PY' >> "$CLOG" 2>&1
# schema: per-arm raw JSON -> params.segment -> requests[] with case
# ("s2-c1024-g128-r0"), decode_ms_per_token, warmup, status.
# segment median per (ctx, gen_len); cell = median of 3 segment medians.
import json, glob, os, statistics, re, sys
raw = os.path.expanduser("~/lfm3-dig-run/window-4bit/raw")
def load(arm, segs):
    out = {}
    for f in sorted(glob.glob(raw + "/" + arm + "-s2-*.json")):
        d = json.load(open(f))
        seg = d.get("params", {}).get("segment")
        if seg not in segs: continue
        per = {}
        for r in d.get("requests", []):
            if r.get("warmup") or r.get("status") != "ok": continue
            m = re.match(r"s2-c(\d+)-g(\d+)-r\d+", r.get("case") or "")
            if not m or r.get("decode_ms_per_token") is None: continue
            per.setdefault((int(m.group(1)), int(m.group(2))), []).append(
                float(r["decode_ms_per_token"]))
        for k, vals in per.items():
            out.setdefault(k, []).append(statistics.median(vals))
    return out
A = load("native-4bit-A", [0, 3, 4]); B = load("native-bf16-B", [1, 2, 5])
print("PAIRED " + json.dumps({"A": {str(k): [round(v,4) for v in x] for k, x in A.items()},
                                "B": {str(k): [round(v,4) for v in x] for k, x in B.items()}}))
bad = 0
for k in sorted(set(A) & set(B)):
    if not A[k] or not B[k]:
        print("CELL %s MISSING segments A=%d B=%d" % (k, len(A[k]), len(B[k]))); bad = 1; continue
    a = statistics.median(A[k]); b = statistics.median(B[k])
    aa = max((max(A[k]) - min(A[k])) / a * 100, (max(B[k]) - min(B[k])) / b * 100)
    # llama Q4 certified baseline (doc 1) for the <= comparison only
    llama = {1024: 12.39, 4096: 12.71}.get(k[0], None)
    speedup = b / a
    print("CELL %s A=%.4f B=%.4f speedup=%.2fx A/A_worst=%.2f%% segs_A=%d segs_B=%d"
          % (k, a, b, speedup, aa, len(A[k]), len(B[k])))
    print("  vs_llamaQ4=%s (%s ms/tok certified)" % (
        "<= baseline" if llama and a <= llama else "ABOVE baseline/unknown",
        llama))
    if aa > 5.0:
        print("G5 FAIL: A/A %.2f%% > 5%% — window INVALID" % aa); bad = 2
print("G5_RESULT %s" % ("FAIL_INVALID" if bad == 2 else ("PASS" if bad == 0 else "INCOMPLETE")))
sys.exit(bad)
PY
G5=$?
echo "G5_EXIT=$G5 at $(date +%FT%T)" >> "$CLOG"
if [ $G5 -eq 2 ]; then
  echo "PHASE aborted_g5_aa" >> "$CLOG"
  exit 102
fi
echo "PHASE all_done $(date +%FT%T)" >> "$CLOG"
exit 0
