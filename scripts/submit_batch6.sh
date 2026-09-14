#!/usr/bin/env bash
# ============================================================================
# Batch 6: recency in HOURS, with the clock lagged one row (R21)
#
# WHY
# ---
# Batch 5 (R20) scored 0.706 -- and was void. Row t's clock included IPT_t,
# the gap ENDING at row t, and since rows are created by outcomes (a draw makes
# a row, a quiet day makes an inserted NotBuy at the 24-hour mark) that gap
# carries 0.40 nats about row t's own label. scripts/ipt_leak_check.py
# measures the channel; scripts/test_time_bias_causality.py is the formal test.
#
# time_bias_lag_ipt=True is now the model default: row t's clock stops at row
# t-1. So TIME_BIAS=time alone gives the honest arm -- nothing else changes.
#
# ARMS (S=1024, USER_EMB=0, VAL_MODE=late, EPOCHS=40, seeds 1 2 3)
#   tb_lag   transformer + lagged time-based recency bias
#
# Comparison: batch 4's tf_alibi (ordinal, 0.886) and batch 2's gru (0.880)
# at the same seeds.
#
# USAGE
#   bash scripts/submit_batch6.sh
#   DRY_RUN=1 bash scripts/submit_batch6.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40"

n=0
for seed in "${SEEDS[@]}"; do
  v="${COMMON},TIME_BIAS=time,SEED=${seed},TAG=b6_tb_lag_s${seed}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
  else
    jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
    printf "%-8s b6_tb_lag_s%s\n" "${jid%%.*}" "$seed"
  fi
  n=$((n + 1))
done
echo "--- $n runs submitted"
