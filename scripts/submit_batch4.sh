#!/usr/bin/env bash
# ============================================================================
# Batch 4: seed the batch-3 winners and combine them (R19).
#
# WHY
# ---
# Batch 3 (R18) found the mechanism behind the transformer's early holdout
# peak: it was NOT campaign memorisation through product identity (tf_noid
# did not move the optimum) but the ABSENCE OF ANY RECENCY SIGNAL in the
# attention stack. With ALiBi the optimum moved from epoch 3 to 19 and the
# transformer tied the GRU (0.888 vs 0.880 +/- 0.008). The offer-inventory
# cross-attention earned ~0.03-0.04 nats on both encoders.
#
# A tie is not an advantage. These runs ask whether combining the recency
# bias with the regularisation that helped the transformer (dropout 0.55,
# +0.02 in batch 2) produces a model that beats the GRU by more than seed
# noise (~0.009 nats), and they replicate the single-seed winners.
#
# ARMS (all S=1024, USER_EMB=0, VAL_MODE=late, EPOCHS=40, seeds 1 2 3)
#   tf_alibi          transformer + ALiBi                          (replicate)
#   tf_alibi_do25     + dropout 0.25
#   tf_alibi_do55     + dropout 0.55
#   tf_alibi_noid     + products by attributes only
#   gru_cross         GRU encoder inside gen-5, cross-attention on (replicate)
#
# USAGE
#   bash scripts/submit_batch4.sh
#   DRY_RUN=1 bash scripts/submit_batch4.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40"

declare -A ARMS=(
  [tf_alibi]="ALIBI=1"
  [tf_alibi_do25]="ALIBI=1,DROPOUT=0.25"
  [tf_alibi_do55]="ALIBI=1,DROPOUT=0.55"
  [tf_alibi_noid]="ALIBI=1,PROD_ID=0"
  [gru_cross]="ENCODER=gru"
)
ORDER=(tf_alibi_do55 tf_alibi gru_cross tf_alibi_do25 tf_alibi_noid)

n=0
for name in "${ORDER[@]}"; do
  for seed in "${SEEDS[@]}"; do
    v="${COMMON},${ARMS[$name]},SEED=${seed},TAG=b4_${name}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b4_%s_s%s\n" "${jid%%.*}" "$name" "$seed"
    fi
    n=$((n + 1))
  done
done
echo "--- $n runs submitted"
