#!/usr/bin/env bash
# Batch 37 -- the capacity-matched duplet, closing a confound in batch 36.
#
# Severing the obtained stream shrinks the fusion input from 6*d_model to
# 2*d_model, which costs 111,360 parameters: 1,482,236 -> 1,370,876, or -7.5%.
# So in batch 36 a duplet LOSS is not attributable -- it could be capacity
# rather than information. (A duplet TIE needs no correction: a smaller model
# matching a larger one is stronger evidence, not weaker.)
#
# d_ff 288 -> 360 restores the count to 1,502,852, +1.4% against the triplet,
# the closest available match. Same severance, same information set, capacity
# back.
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="PROFILE=hpcc,MAX_EVENTS=1024,EPOCHS=40,BATCH_SIZE=4,VOCAB_LEVEL=7,VAL_MODE=late,VAL_FROM=27"
BASE="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2,CAMP_FE=1"
n=0
go() { jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
for seed in 1 2 3 4 5; do
  go "${COMMON},SEED=${seed},${BASE},D_FF=360,${STOCK},NO_OBTAINED=1,TAG=b37_duplet_capmatch_s${seed}"
done
echo "submitted $n runs"
