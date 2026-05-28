#!/usr/bin/env bash
# Sweep request-rate against a fake-prefill (decode-only) SGLang server,
# observe how TPOT grows. Workload: 64K input @ ~90% prefix hit, 512 output,
# single-turn, 100 requests per rate point.
#
# Prereqs:
#   - This script lives next to bench_multi_turn.py (git clone of this repo).
#   - SGLang decode server is already running with
#       --disaggregation-mode decode --disaggregation-transfer-backend fake
#   - ShareGPT V3 JSON downloaded locally.
#
# Output:
#   - $SUMMARY_CSV : one row per rate (use this to plot TPOT vs rate)
#   - $LOG         : full stdout/stderr per run
#   - $DUMP_DIR    : workload snapshot (so all rates see identical prompts)

# NOTE: deliberately NOT using `set -e` here.
# bench_multi_turn.py returns 1 if ANY request failed, even just one timeout.
# We want the sweep to keep going through all rates and dump everything to CSV
# so we can see exactly where things break, instead of bailing early.
set -uo pipefail

# ====== EDIT THESE ======
HOST=localhost
PORT=30000
MODEL=/data/GLM-5-FP8/
TOKENIZER=$MODEL
SHAREGPT_PATH=/workspace/infrawaves/zq/ShareGPT_V3_unfiltered_cleaned_split.json
SCRIPT=./bench_multi_turn.py

NUM_GROUPS=1
PROMPTS_PER_GROUP=100   # 100 requests per rate point (num_turns=1)
NUM_TURNS=1
SYS_LEN=57600           # 90% of 64K → shared prefix, hits cache after 1st req
Q_LEN=6400              # 10% of 64K → unique per request, always misses
OUT_LEN=512
SEED=42

EXTRA_BODY='{"bootstrap_host": "2.2.2.2", "bootstrap_room": 0}'

DUMP_DIR=./dump_64k_90pct
SUMMARY_CSV=./tpot_sweep.csv
LOG=./bench.log

# rates: 0.2,0.4,...,1.0 then 1.5,2.0,...,5.0
RATES=(0.2 0.4 0.6 0.8 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0)
# =========================

: > "$LOG"
echo "Sweep start: $(date)" | tee -a "$LOG"
echo "Rates: ${RATES[*]}"   | tee -a "$LOG"

first=1
for r in "${RATES[@]}"; do
  echo "=== request-rate=$r ===" | tee -a "$LOG"

  if [[ $first -eq 1 ]]; then
    # First run: build the dataset from ShareGPT, dump it, also primes
    # the prefix cache for subsequent runs.
    python3 "$SCRIPT" \
      --backend sglang --host "$HOST" --port "$PORT" \
      --model "$MODEL" --tokenizer "$TOKENIZER" \
      --dataset-mode sharegpt --sharegpt-path "$SHAREGPT_PATH" \
      --num-groups $NUM_GROUPS --prompts-per-group $PROMPTS_PER_GROUP \
      --num-turns $NUM_TURNS \
      --system-prompt-len $SYS_LEN --question-len $Q_LEN --output-len $OUT_LEN \
      --seed $SEED \
      --request-rate "$r" --max-concurrency 256 \
      --dump-prompts-dir "$DUMP_DIR" --dump-full-content \
      --extra-request-body "$EXTRA_BODY" \
      --case-name "r${r}" --summary-csv "$SUMMARY_CSV" \
      >> "$LOG" 2>&1 || echo "!! rate=$r returned non-zero (some requests likely failed)" | tee -a "$LOG"
    first=0
  else
    # Subsequent runs: reload the exact same prompts so all rates are
    # comparable, only the arrival rate changes.
    python3 "$SCRIPT" \
      --backend sglang --host "$HOST" --port "$PORT" \
      --model "$MODEL" --tokenizer "$TOKENIZER" \
      --load-dataset "$DUMP_DIR/content.jsonl" \
      --request-rate "$r" --max-concurrency 256 \
      --extra-request-body "$EXTRA_BODY" \
      --case-name "r${r}" --summary-csv "$SUMMARY_CSV" \
      >> "$LOG" 2>&1 || echo "!! rate=$r returned non-zero (some requests likely failed)" | tee -a "$LOG"
  fi

  echo "=== done rate=$r ===" | tee -a "$LOG"
done

echo "All sweeps done: $(date)" | tee -a "$LOG"
echo "Summary CSV: $SUMMARY_CSV"
echo "Full log:    $LOG"
