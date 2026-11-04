#!/usr/bin/env bash
# Batch 38 -- R38 / gen 6: PITY DEPTH as a sixth stream.
#
# The player sees their pity counter; it is on screen. The model sees only which
# products arrived, and would have to infer the counter by learning rarity and
# the reset rule from 118 unsupervised product tokens. Lineage: the
# goal-gradient hypothesis (Kivetz, Urminsky & Zheng 2006) and sunk cost --
# players push to 90 BECAUSE they are at 75.
#
# It is also the state variable whose absence plausibly causes R36c's generative
# failure: free running collapses NotBuy 0.503 -> 0.183 and spirals into buying,
# and pity resets after a hit, so a model conditioning on it has a reason to
# stop.
#
# Verified before submission (scripts/test_pity_stream.py, 7 checks): the
# vectorised cumsum/cummax equals a brute-force replay to 1e-7; future rows move
# neither the feature nor the logits at t; and no pity cell determines the label
# (max cell share 0.548 against a 0.313 marginal majority) -- informative, not a
# leak.
#
# Arms, level 7, five seeds each. The sixth stream ADDS 38,016 parameters
# (1,482,236 -> 1,520,252, +2.6%), so a win could be capacity. Shrinking the
# pity model overshoots -- d_ff 240 lands at -4.9% and 216 at -8.7%, both worse
# matches than leaving it alone -- so the CONTROL gets the extra capacity
# instead: d_ff 306 gives 1,522,142, within 0.12% of the pity arm.
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="PROFILE=hpcc,MAX_EVENTS=1024,EPOCHS=40,BATCH_SIZE=4,VOCAB_LEVEL=7,VAL_MODE=late,VAL_FROM=27"
BASE="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2,CAMP_FE=1"
n=0
go() { jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
for seed in 1 2 3 4 5; do
  go "${COMMON},SEED=${seed},${BASE},D_FF=288,${STOCK},PITY_STREAM=1,TAG=b38_pity_s${seed}"
  go "${COMMON},SEED=${seed},${BASE},D_FF=306,${STOCK},TAG=b38_ctrl_capmatch_s${seed}"
done
echo "submitted $n runs"
