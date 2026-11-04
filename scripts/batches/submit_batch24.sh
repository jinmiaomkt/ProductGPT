#!/usr/bin/env bash
# ============================================================================
# Batch 24: two diagnostics, not comparisons.
#
# A. IS THE HALF-LIFE IDENTIFIED ON REAL DATA? The corrected ladder returns a
#    half-life of 29.8-29.9 from a start of 30.0 -- a 0.7% move with almost no
#    spread across seeds, while the duplicate weights moved 20%. Either the
#    likelihood is flat in the decay (plausible: stage 4 put the whole stock
#    path at 0.0137 nats) or the estimate is anchored at its prior. Starting
#    from 10 and 100 settles it: if the estimates converge, the data identify
#    it; if each stays near its start, they do not and no half-life should be
#    reported. R34 said the decay IS identified -- but its simulated satiation
#    was far stronger than what the real data appear to carry.
#
# B. IS THE BLOCK DESIGN BAD, OR IS MY IMPLEMENTATION? fuse=block scores
#    0.98-0.99 against 0.88 for everything else -- too large a gap to accept
#    without a check. With BLOCK_LEN at the full sequence there is exactly one
#    block, so it should behave like the sequential design. If it does, small
#    blocks are genuinely bad; if it does not, the bug is mine.
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,CAMP_FE=1"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
n=0
go () {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "qsub -v $1"; else
    jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; fi
  n=$((n + 1))
}
for init in 10 30 100; do
  for seed in 1 2 3; do
    go "${COMMON},SEED=${seed},${FROZEN},DECAY=exp,TIER=1,DECAY_INIT=${init},TAG=b24_hl${init}_s${seed}"
  done
done
BASE="ENCODER=gru_attn,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
for bl in 128 1024; do
  go "${COMMON},SEED=1,${BASE},FUSE=block,BLOCK_LEN=${bl},TAG=b24_block_len${bl}"
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n listed" || echo "--- $n submitted"
