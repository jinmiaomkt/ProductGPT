#!/usr/bin/env bash
# ============================================================================
# Batch 32: completes the reviewer's index-vs-attributes comparison.
#
#   identity + attributes   the standard model            (b29_hl2, run)
#   identity only           attributes zeroed             (b26_idonly, run)
#   ATTRIBUTES ONLY         PROD_ID=0, no identity        <- this batch
#
# The economic reading (Jin, in the manuscript source): if the seller only ever
# revives old products, identity suffices; if the seller introduces new ones,
# the model must work from attributes. This cell measures what that costs.
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,CAMP_FE=1"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
n=0
for seed in 1 2 3 4 5; do
  v="${COMMON},SEED=${seed},${FROZEN},KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2,PROD_ID=0,TAG=b32_attronly_s${seed}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "qsub -v $v"; else
    jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"; printf "%-8s b32_attronly_s%s\n" "${jid%%.*}" "$seed"; fi
  n=$((n+1))
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n listed" || echo "--- $n submitted"
