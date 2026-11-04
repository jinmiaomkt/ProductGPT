#!/usr/bin/env bash
# ============================================================================
# Batch 12: one token per product (R28)
#
# WHY
# ---
# Level 6 gives 118 products only 44 tokens: every 3-star weapon shares an id,
# as do the 4-stars and the standard 5-stars. R27 showed that resolution costs
# real accuracy: correcting the product ids (R26) cut the obtained stream's
# entropy from 1.050 to 0.601 nats and every model lost about 0.01-0.02 nats
# with it. Level 7 gives each product its own token and its own embedding row
# (ids 13-130, vocab 134); entropy rises to 3.236 nats.
#
# Arms mirror batch 11 exactly, same seeds, so each is a paired comparison with
# its level-6 twin:
#   gru        plain GRU baseline          b11 0.8892 +/- 0.0172
#   tf_alibi   transformer + recency       b11 0.9083 +/- 0.0055
#   gru_cross  GRU encoder + inventory     b11 0.8970 +/- 0.0075
#
# The data (perproduct/clean_list_int_wide4_simple7_IPT.json) collapses exactly
# onto the level-6 file, so nothing but token resolution differs.
#
# USAGE
#   bash scripts/submit_batch12.sh
#   DRY_RUN=1 bash scripts/submit_batch12.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7"

declare -A ARMS=(
  [gru]="ARCH=gru"
  [tf_alibi]="ALIBI=1"
  [gru_cross]="ENCODER=gru"
)
ORDER=(gru tf_alibi gru_cross)

n=0
for seed in "${SEEDS[@]}"; do
  for name in "${ORDER[@]}"; do
    v="${COMMON},${ARMS[$name]},SEED=${seed},TAG=b12_${name}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b12_%s_s%s\n" "${jid%%.*}" "$name" "$seed"
    fi
    n=$((n + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
