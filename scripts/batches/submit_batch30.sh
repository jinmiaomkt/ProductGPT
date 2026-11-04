#!/usr/bin/env bash
# ============================================================================
# Batch 30: HETEROGENEITY -- the binding constraint, per R34 s7 and the
# brand-choice study E2b (unmodelled taste variation collapses the attribute
# recovery from a lift of 21.7 to 0.03, and biases satiation by 2x).
#
# Every reported run so far has had user embeddings OFF. Three arms on the
# frozen R33 configuration, 5 seeds each:
#
#   H1  USER_EMB=1              a per-customer embedding
#   H2  MIX_HEADS=8             the Lu & Kannan per-customer mixture over
#                               output projections (already in the codebase, R7)
#   H3  both
#
# The control is b29_hl2 (same configuration, no heterogeneity), already run.
#
# THE POINT OF THE 2x2: a per-customer parameter cannot transfer to customers
# the model has never seen -- out-of-sample customers share the unknown-user
# mean. So the in-sample and out-of-sample holdout cells should diverge, and
# how much they diverge is the measurement we want. Selection on validation.
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="MAX_EVENTS=1024,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,CAMP_FE=1"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2"
n=0
go () { if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "qsub -v $1"; else
  jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; fi; n=$((n+1)); }
for seed in 1 2 3 4 5; do
  go "${COMMON},USER_EMB=1,SEED=${seed},${FROZEN},${STOCK},TAG=b30_useremb_s${seed}"
  go "${COMMON},USER_EMB=0,MIX_HEADS=8,SEED=${seed},${FROZEN},${STOCK},TAG=b30_mixhead_s${seed}"
  go "${COMMON},USER_EMB=1,MIX_HEADS=8,SEED=${seed},${FROZEN},${STOCK},TAG=b30_both_s${seed}"
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n listed" || echo "--- $n submitted"
