#!/usr/bin/env bash
# ============================================================================
# Batch 31: is the half-life optimum INTERIOR? Validation selected 2 occasions,
# the shortest value tested in batch 29, so the curve may be at its boundary.
# Extending to 0.5 and 1 settles whether holdings decay over a couple of
# occasions or are effectively contemporaneous -- a different marketing claim.
# Frozen, 5 seeds; combines with b29's 2, 5, 10 into one profile on VALIDATION.
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,CAMP_FE=1"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
n=0
for hl in 0.5 1; do
  for seed in 1 2 3 4 5; do
    tag="b31_hl$(echo $hl | tr -d '.')_s${seed}"
    v="${COMMON},SEED=${seed},${FROZEN},KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=${hl},TAG=${tag}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "qsub -v $v"; else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "$tag"; fi
    n=$((n+1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n listed" || echo "--- $n submitted"
