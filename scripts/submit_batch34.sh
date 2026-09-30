#!/usr/bin/env bash
# Batch 34 -- the matched arm WITHOUT campaign fixed effects.
# Paired with batch 33's half-life-2 cell, this is the 20-vs-21 test that was
# never actually run: does controlling for campaign-level demand shocks change
# the fit, and does it move the decay?
set -euo pipefail
COMMON="PROFILE=hpcc,MAX_EVENTS=1024,EPOCHS=40,BATCH_SIZE=4,VOCAB_LEVEL=7,VAL_MODE=late,VAL_FROM=27"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2"
n=0
go() { jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
for seed in 1 2 3 4 5; do
  go "${COMMON},SEED=${seed},${FROZEN},${STOCK},CAMP_FE=0,TAG=b34_nocampfe_hl2_s${seed}"
done
echo "submitted $n runs"
