#!/usr/bin/env bash
# ============================================================================
# Batch 8: additive inventory slots on both backbones (R23)
#
# WHY
# ---
# The gen-5 inventory path adds nothing over a plain GRU (-0.009). Three
# specification problems, measured 15 Sep (EXPERIMENTS.md R22-R24): the
# inventory GRU is not additive and updates on every empty row (37.9% of rows);
# the satiation attention spans every obtained token (median 643 for 13
# distinct products); and items obtained before the 1,024-row window are
# invisible. INVENTORY=slots replaces both with one slot per product, counted
# over the full history and lagged to t-1, and keeps the SINGLE attention step,
# so this batch isolates the representation from depth.
#
# ARMS (S=1024, VAL_MODE=late, EPOCHS=40, seeds 1 2 3; customer embedding off)
#   tf_slots    transformer + ALiBi + slots   vs b4_tf_alibi (0.886)
#   gru_slots   GRU encoder + slots           vs b4_gru_cross (0.889) and b2_gru (0.880)
#
# Verified before submission: scripts/test_inventory_slots.py --data (additive,
# lagged, causal on both encoders, eval = train, window + pre-window counts =
# full history on 200 customers). Peak memory at S=1024 on the laptop:
# 5.1 GB (tokens) -> 0.33 GB (slots).
#
# USAGE
#   bash scripts/submit_batch8.sh
#   DRY_RUN=1 bash scripts/submit_batch8.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,INVENTORY=slots"

declare -A ARMS=(
  [tf_slots]="ALIBI=1"
  [gru_slots]="ENCODER=gru"
)
ORDER=(tf_slots gru_slots)

n=0
for seed in "${SEEDS[@]}"; do
  for name in "${ORDER[@]}"; do
    v="${COMMON},${ARMS[$name]},SEED=${seed},TAG=b8_${name}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b8_%s_s%s\n" "${jid%%.*}" "$name" "$seed"
    fi
    n=$((n + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
