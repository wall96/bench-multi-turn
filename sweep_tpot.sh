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
MODEL=/gpfs/models/huggingface.co/deepseek-ai/DeepSeek-V3.2
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

# ====== Server-side metrics (optional, bypasses prom server) ======
# When set, snapshots /metrics on each listed server before+after every
# rate point and computes histogram_quantile exactly like grafana does
# server-side. Output goes to a separate CSV so the existing
# client-side CSV is untouched.
# Empty = disabled.
SERVER_METRICS_URLS=""   # e.g. "http://10.51.10.32:30000/metrics,http://10.51.10.33:30000/metrics"
SERVER_METRICS_CSV="./server_metrics.csv"
SERVER_METRICS_LABEL_FILTERS=()  # add as: ("--label-filter" "model_name=DeepSeek-V3.2-4decode-only" ...)
SERVER_METRICS_POST_SLEEP=5
# =========================

: > "$LOG"
echo "Sweep start: $(date)" | tee -a "$LOG"
echo "Rates: ${RATES[*]}"   | tee -a "$LOG"

# Build the bench command for one rate point. Echoed as a single string
# so we can hand it to either python3 directly or to server_metrics.py wrap.
build_cmd() {
  local rate="$1" first="$2"
  if [[ $first -eq 1 ]]; then
    echo python3 "$SCRIPT" \
      --backend sglang --host "$HOST" --port "$PORT" \
      --model "$MODEL" --tokenizer "$TOKENIZER" \
      --dataset-mode sharegpt --sharegpt-path "$SHAREGPT_PATH" \
      --num-groups "$NUM_GROUPS" --prompts-per-group "$PROMPTS_PER_GROUP" \
      --num-turns "$NUM_TURNS" \
      --system-prompt-len "$SYS_LEN" --question-len "$Q_LEN" --output-len "$OUT_LEN" \
      --seed "$SEED" \
      --request-rate "$rate" --max-concurrency 256 \
      --dump-prompts-dir "$DUMP_DIR" --dump-full-content \
      --extra-request-body "$EXTRA_BODY" \
      --case-name "r${rate}" --summary-csv "$SUMMARY_CSV"
  else
    echo python3 "$SCRIPT" \
      --backend sglang --host "$HOST" --port "$PORT" \
      --model "$MODEL" --tokenizer "$TOKENIZER" \
      --load-dataset "$DUMP_DIR/content.jsonl" \
      --request-rate "$rate" --max-concurrency 256 \
      --extra-request-body "$EXTRA_BODY" \
      --case-name "r${rate}" --summary-csv "$SUMMARY_CSV"
  fi
}

first=1
for r in "${RATES[@]}"; do
  echo "=== request-rate=$r ===" | tee -a "$LOG"

  echo "flushing cache..." | tee -a "$LOG"
  curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >> "$LOG" 2>&1 \
    || echo "!! flush_cache failed for rate=$r" | tee -a "$LOG"

  bench_cmd=$(build_cmd "$r" "$first")

  if [[ -n "$SERVER_METRICS_URLS" ]]; then
    # Wrap the bench run with server-side metric snapshotting.
    # ITL / TTFT / E2E percentiles get appended to SERVER_METRICS_CSV.
    python3 ./server_metrics.py wrap \
      --metrics-urls "$SERVER_METRICS_URLS" \
      --metric ITL=sglang:inter_token_latency_seconds \
      --metric TTFT=sglang:time_to_first_token_seconds \
      --metric E2E=sglang:e2e_request_latency_seconds \
      --quantiles 0.5,0.9,0.99 \
      --case-name "r${r}" \
      --summary-csv "$SERVER_METRICS_CSV" \
      --post-sleep "$SERVER_METRICS_POST_SLEEP" \
      "${SERVER_METRICS_LABEL_FILTERS[@]}" \
      -- bash -c "$bench_cmd" \
      >> "$LOG" 2>&1 \
      || echo "!! rate=$r returned non-zero (some requests likely failed)" | tee -a "$LOG"
  else
    eval "$bench_cmd" >> "$LOG" 2>&1 \
      || echo "!! rate=$r returned non-zero (some requests likely failed)" | tee -a "$LOG"
  fi

  if [[ $first -eq 1 ]]; then first=0; fi

  echo "=== done rate=$r ===" | tee -a "$LOG"
  sleep 15
done

echo "All sweeps done: $(date)" | tee -a "$LOG"
echo "Summary CSV (client):  $SUMMARY_CSV"
[[ -n "$SERVER_METRICS_URLS" ]] && echo "Summary CSV (server):  $SERVER_METRICS_CSV"
echo "Full log:              $LOG"
