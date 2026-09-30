#!/usr/bin/env bash
# Batch 35 -- R36d: scheduled sampling as a remedy for the R36c failure.
#
# R36c: teacher-forced samples are INSIDE the null band (0.307) while free
# running is far outside (1.793) and abandons the majority class (NotBuy
# 0.503 -> 0.183). The per-step distribution is right; the feedback loop is not.
# Scheduled sampling exposes training to the prefixes the model will visit.
#
# Selection on VALIDATION as always. A variant is adopted only if it improves
# the SEQUENCE discrepancy without costing more than 0.02 nats of one-step fit;
# scheduled sampling is expected to cost a little one-step accuracy, and the
# question is whether it buys generative validity.
#
# NOTE the implementation corrupts the DECISION stream only; the obtained
# stream stays real. If this fails, the inventory channel is the binding one.
set -euo pipefail
COMMON="PROFILE=hpcc,MAX_EVENTS=1024,EPOCHS=40,BATCH_SIZE=4,VOCAB_LEVEL=7,VAL_MODE=late,VAL_FROM=27"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2,CAMP_FE=1"
n=0
go() { jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
for p in 0.15 0.35; do
  for seed in 1 2 3; do
    go "${COMMON},SEED=${seed},${FROZEN},${STOCK},SCHED_SAMPLING=${p},SS_RAMP=10,TAG=b35_ss${p/./}_s${seed}"
  done
done
echo "submitted $n runs"
