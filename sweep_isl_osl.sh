#!/usr/bin/env bash
# Matrix sweep: 4 (ISL, OSL) profiles × 4 concurrency levels = 16 runs.
# Closed-loop testing (no rate limit, only max-concurrency throttles).
# Each profile reuses one dumped dataset across all concurrency levels so
# the four points within a profile are directly comparable.
#
# Profiles:
#   small   ISL=1024   OSL=256
#   medium  ISL=4096   OSL=512
#   large   ISL=16384  OSL=1024
#   xlarge  ISL=65536  OSL=1024
#
# Concurrencies: 4, 8, 16, 24
#
# Workload split: 90% system prompt (group-shared, hits cache after 1st req)
#                 10% question (unique per request)
#                 single turn

# Don't bail on a single bad rate point -- collect everything and show what failed.
set -uo pipefail

# ====== EDIT THESE ======
HOST=localhost
PORT=30000
MODEL=/data/GLM-5-FP8/
TOKENIZER=$MODEL
SHAREGPT_PATH=/workspace/infrawaves/zq/ShareGPT_V3_unfiltered_cleaned_split.json
SCRIPT=./bench_multi_turn.py

NUM_GROUPS=1
PROMPTS_PER_GROUP=50    # 50 requests per (profile, concurrency) cell
NUM_TURNS=1
SEED=42

# Decode-only / fake-prefill server. Set to '' if your server is normal.
EXTRA_BODY='{"bootstrap_host": "2.2.2.2", "bootstrap_room": 0}'

DUMP_BASE=./dump_isl_osl
SUMMARY_CSV=./isl_osl_sweep.csv
LOG=./isl_osl_sweep.log

# name  ISL    OSL
PROFILES=(
  "small  1024   256"
  "medium 4096   512"
  "large  16384  1024"
  "xlarge 65536  1024"
)

CONCURRENCIES=(4 8 16 24)

# Cooldown between cells: flush KV / radix cache, sleep, then next run.
# Avoids one cell's residual cache helping the next one.
COOLDOWN_SEC=10
FLUSH_URL="http://$HOST:$PORT/flush_cache"
# =========================

cooldown() {
  # Run BEFORE each cell so every test starts from a clean cache state.
  echo ">>> flush_cache + sleep ${COOLDOWN_SEC}s before next cell" | tee -a "$LOG"
  curl -s -X POST "$FLUSH_URL" >> "$LOG" 2>&1 \
    || echo "!! flush_cache failed (server may not expose /flush_cache)" | tee -a "$LOG"
  sleep "$COOLDOWN_SEC"
}

: > "$LOG"
echo "Sweep start: $(date)" | tee -a "$LOG"

for prof in "${PROFILES[@]}"; do
  read -r name isl osl <<< "$prof"

  # 90 / 10 split: shared prefix vs per-request question
  sys_len=$(( isl * 9 / 10 ))
  q_len=$(( isl - sys_len ))

  dump_dir="$DUMP_BASE/$name"

  echo ""                                                                | tee -a "$LOG"
  echo "================ profile=$name ISL=$isl OSL=$osl (sys=$sys_len q=$q_len) ================" \
                                                                          | tee -a "$LOG"

  first=1
  for conc in "${CONCURRENCIES[@]}"; do
    case_name="${name}_c${conc}"
    echo "=== $case_name ===" | tee -a "$LOG"
    cooldown

    if [[ $first -eq 1 ]]; then
      # First concurrency for this profile: build prompts from ShareGPT, dump them.
      python3 "$SCRIPT" \
        --backend sglang --host "$HOST" --port "$PORT" \
        --model "$MODEL" --tokenizer "$TOKENIZER" \
        --dataset-mode sharegpt --sharegpt-path "$SHAREGPT_PATH" \
        --num-groups $NUM_GROUPS --prompts-per-group $PROMPTS_PER_GROUP \
        --num-turns $NUM_TURNS \
        --system-prompt-len $sys_len --question-len $q_len --output-len $osl \
        --seed $SEED \
        --max-concurrency $conc \
        --dump-prompts-dir "$dump_dir" --dump-full-content \
        --extra-request-body "$EXTRA_BODY" \
        --case-name "$case_name" --summary-csv "$SUMMARY_CSV" \
        >> "$LOG" 2>&1 \
        || echo "!! $case_name returned non-zero" | tee -a "$LOG"
      first=0
    else
      # Subsequent concurrencies for the same profile: reuse the dumped prompts.
      python3 "$SCRIPT" \
        --backend sglang --host "$HOST" --port "$PORT" \
        --model "$MODEL" --tokenizer "$TOKENIZER" \
        --load-dataset "$dump_dir/content.jsonl" \
        --max-concurrency $conc \
        --extra-request-body "$EXTRA_BODY" \
        --case-name "$case_name" --summary-csv "$SUMMARY_CSV" \
        >> "$LOG" 2>&1 \
        || echo "!! $case_name returned non-zero" | tee -a "$LOG"
    fi

    echo "=== done $case_name ===" | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "All sweeps done: $(date)" | tee -a "$LOG"
echo "Summary CSV: $SUMMARY_CSV"
echo "Full log:    $LOG"
