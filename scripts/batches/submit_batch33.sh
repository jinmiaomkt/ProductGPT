#!/usr/bin/env bash
# Batch 33 -- the campaign-FE re-run.
#
# Every run before 30 Sep 2026 that claimed --camp-fe 1 computed the same
# function as one without: collate_multistream never copied the campaign id into
# the batch, so camp_bias received no gradient (proved by its weights being
# bit-identical across configurations sharing a seed). That voided the 20-vs-21
# and 22-vs-23 pairs and left the half-life profile without the control R34 s6
# requires. The collate fix is in; this re-establishes the profile WITH the
# control, and batch 34 supplies the matched arm without it.
set -euo pipefail
COMMON="PROFILE=hpcc,MAX_EVENTS=1024,EPOCHS=40,BATCH_SIZE=4,VOCAB_LEVEL=7,VAL_MODE=late,VAL_FROM=27"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1"
n=0
go() { jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
for hl in 1 2 5; do
  for seed in 1 2 3 4 5; do
    go "${COMMON},SEED=${seed},${FROZEN},${STOCK},DECAY_INIT=${hl},CAMP_FE=1,TAG=b33_campfe_hl${hl}_s${seed}"
  done
done
echo "submitted $n runs"
