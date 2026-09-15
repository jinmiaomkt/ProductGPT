#!/usr/bin/env bash
# ============================================================================
# Batch 9: deep satiation on additive slots (R24)
#
# WHY
# ---
# In the original model, how far the current limited-time offer is already
# satisfied by the inventory gets ONE attention step, and the sequence model
# after it never re-reads the inventory -- so depth never reaches the concept.
# SAT_LAYERS=L stacks L blocks of [offers attend to each other (competition
# between concurrent banners) -> offers attend to inventory slots ->
# feed-forward], pre-norm residual, and keeps one representation per offer slot.
#
# ARMS (S=1024, VAL_MODE=late, EPOCHS=40, seeds 1 2 3; INVENTORY=slots)
#   tf_sat2    transformer + ALiBi, 2 satiation blocks   vs b8_tf_slots
#   tf_sat4    transformer + ALiBi, 4 satiation blocks   vs b8_tf_slots
#   gru_sat2   GRU encoder, 2 satiation blocks           vs b8_gru_slots
#
# Submitted alongside batch 8 so the queue does not idle; it is read against
# batch 8, its matched single-step control.
#
# USAGE
#   bash scripts/submit_batch9.sh
#   DRY_RUN=1 bash scripts/submit_batch9.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,INVENTORY=slots"

declare -A ARMS=(
  [tf_sat2]="ALIBI=1,SAT_LAYERS=2"
  [tf_sat4]="ALIBI=1,SAT_LAYERS=4"
  [gru_sat2]="ENCODER=gru,SAT_LAYERS=2"
)
ORDER=(tf_sat2 tf_sat4 gru_sat2)

n=0
for seed in "${SEEDS[@]}"; do
  for name in "${ORDER[@]}"; do
    v="${COMMON},${ARMS[$name]},SEED=${seed},TAG=b9_${name}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b9_%s_s%s\n" "${jid%%.*}" "$name" "$seed"
    fi
    n=$((n + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
