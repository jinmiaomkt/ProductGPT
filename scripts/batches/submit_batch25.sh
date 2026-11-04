#!/usr/bin/env bash
# ============================================================================
# Batch 25: the half-life as a PROFILE LIKELIHOOD, and fresh seeds for the
# leading hybrid design.
#
# A. Batch 24 showed the half-life never moves under gradient descent (starts
#    of 10/30/100 return 9.9/29.8/99.7) but that the FIT depends on its value
#    (holdout 0.8845 / 0.8914 / 0.8946). So the gradient is too small to move a
#    parameter of that scale while the likelihood still has a slope. The grid
#    IS the estimator; DECAY_FREEZE makes that explicit. Values bracket the
#    apparent optimum below 10.
# B. R32's recurrence-then-attention design leads on one seed (0.8789). Three
#    fresh seeds against the incumbent, the same discipline that deflated four
#    earlier leads.
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,CAMP_FE=1"
STOCK="INVENTORY=slots,SAT_LAYERS=2"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,${STOCK},D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
n=0
go () { if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "qsub -v $1"; else
  jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; fi; n=$((n+1)); }

for hl in 2 5 10 20 40; do
  for seed in 1 2 3; do
    go "${COMMON},SEED=${seed},${FROZEN},DECAY=exp,TIER=1,DECAY_INIT=${hl},DECAY_FREEZE=1,TAG=b25_prof${hl}_s${seed}"
  done
done
for seed in 11 12 13; do
  go "${COMMON},SEED=${seed},ENCODER=gru_attn,FUSE=seq_ra,ALIBI=1,${STOCK},D_MODEL=256,D_FF=768,N_LAYERS=2,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1,TAG=b25_seqra_d256N2_s${seed}"
  go "${COMMON},SEED=${seed},ENCODER=gru_attn,FUSE=seq_ar,ALIBI=1,${STOCK},D_MODEL=128,D_FF=384,N_LAYERS=4,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1,TAG=b25_seqar_d128N4_s${seed}"
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n listed" || echo "--- $n submitted"
