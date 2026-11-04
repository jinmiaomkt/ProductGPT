#!/usr/bin/env bash
# Batch 39 -- R43: the gen-6 streaming hybrid against the rev-2 gen-5 incumbent.
#
# PRE-REGISTERED in EXPERIMENTS.md (R43). Do NOT submit until step 0 (the rollout
# probe, scripts/r43_probe_cpu.pbs) has passed and Jin has approved the entry.
#
# Every arm runs from ~/ProductGPT/work_gen6, so both families train on the same
# commit. Hyperparameters are frozen at the b36 triplet's for every arm (no
# tuning on either side); gen 6's trainer defaults reproduce them.
#
#   A  b39_g5rev2   gen-5 stack at model_rev 2, last 1,024 occasions   (control)
#   B  b39_g6full   gen-6 code, window and chunk unlimited: the gen-5 function
#                   on FULL histories (test T3 shows it equals gen 5 exactly)
#   C  b39_g6w384   gen 6 as proposed: window 384, chunks of 512
#   D  b39_g6w128   window sensitivity, three seeds, no decision attached
#
# A->B isolates reading full histories; B->C isolates the streaming
# approximation (window plus chunked training). 18 runs, seed-major so complete
# seed pairs land first.
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
[[ "$(basename "$PWD")" == "work_gen6" ]] || { echo "submit from ~/ProductGPT/work_gen6"; exit 2; }
COMMON5="PROFILE=hpcc,MAX_EVENTS=1024,EPOCHS=40,BATCH_SIZE=4,VOCAB_LEVEL=7,VAL_MODE=late,VAL_FROM=27"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2,CAMP_FE=1"
n=0
go5() { jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
go6() { jid="$(qsub -v "$1" scripts/gen6_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "$1"; n=$((n+1)); }
for seed in 1 2 3 4 5; do
  go5 "${COMMON5},SEED=${seed},${FROZEN},${STOCK},TAG=b39_g5rev2_s${seed}"
  go6 "SEED=${seed},WINDOW=0,CHUNK=2048,TAG=b39_g6full"
  go6 "SEED=${seed},WINDOW=384,CHUNK=512,TAG=b39_g6w384"
done
for seed in 1 2 3; do
  go6 "SEED=${seed},WINDOW=128,CHUNK=512,TAG=b39_g6w128"
done
echo "submitted $n runs"
