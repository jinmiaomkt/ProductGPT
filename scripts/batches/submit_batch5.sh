#!/usr/bin/env bash
# ============================================================================
# Batch 5: is recency better measured in HOURS than in events? (R20)
#
# WHY
# ---
# R18 showed the transformer's whole deficit was the absence of any recency
# signal: a zero-parameter ordinal bias (ALiBi) moved the holdout optimum from
# epoch 3 to 19 and closed a 0.13-nat gap to the GRU. But the event index is a
# poor ruler here. Ten pulls four minutes apart and a three-week silence are
# both "one event ago", and R9 measured how extreme that is -- 46.5% of true
# gaps round to zero while the longest reaches 2,401 hours.
#
# The IPT field already carries the hours, so the same bias can be built on
# elapsed time: penalty = s_h * log1p(t_i - t_j), with a learnable per-head
# decay (log1p compresses a 0-2,401h range into 0-7.8, so the fixed ALiBi
# schedule would be far too gentle; the rate is learned instead, initialised
# at 16x the ALiBi values).
#
# ARMS (S=1024, USER_EMB=0, VAL_MODE=late, EPOCHS=40, seeds 1 2 3)
#   tb_time        transformer + time-based recency bias
#   tb_time_do55   the same with dropout 0.55, the best regulariser in batch 2
#
# The ORDINAL comparison is batch 4's tf_alibi and tf_alibi_do55 at the same
# three seeds, so this is a like-for-like swap of the ruler only.
#
# USAGE
#   bash scripts/submit_batch5.sh
#   DRY_RUN=1 bash scripts/submit_batch5.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40"

declare -A ARMS=(
  [tb_time]="TIME_BIAS=time"
  [tb_time_do55]="TIME_BIAS=time,DROPOUT=0.55"
)
ORDER=(tb_time tb_time_do55)

n=0
for name in "${ORDER[@]}"; do
  for seed in "${SEEDS[@]}"; do
    v="${COMMON},${ARMS[$name]},SEED=${seed},TAG=b5_${name}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b5_%s_s%s\n" "${jid%%.*}" "$name" "$seed"
    fi
    n=$((n + 1))
  done
done
echo "--- $n runs submitted"
