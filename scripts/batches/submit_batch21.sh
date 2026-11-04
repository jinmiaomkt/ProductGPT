#!/usr/bin/env bash
# ============================================================================
# Batch 21: R33b -- the same ladder WITH campaign fixed effects.
#
# R34 s6 measured what happens without them: a campaign-level demand shock moves
# in lockstep with "time since the product left the assortment", and the decay
# absorbs the calendar. True half-life 30 was estimated at 59 (shock sd 0.3) and
# 147 (sd 0.6); with campaign FE it came back at 29.5-30.4 in both.
#
# Batch 20 is the same ladder without them, so the pair is also a direct test of
# that simulation prediction on real data: if b20's half-life is much longer
# than b21's, the calendar was being absorbed here too.
#
# The ladder itself:
#
# R34 decided what belongs here. The decay half-life and the duplicate weights
# are identified under our offer rotation; a free product-product kernel is
# not. So the ladder climbs through the identified parts first, and the free
# kernel appears ONLY as an upper bound for a specification test -- if it
# cannot beat the attribute kernel out of sample, there is nothing in the data
# beyond the restriction, and we report that rather than a picture of it.
#
#   L0  counts                         the stage-4 frozen hybrid (already run)
#   L1  + learned decay                Guadagni-Little smoothing, half-life free
#   L2  + duplicate weights            depth-of-holding, g(1) > g(2) > g(3)
#   L3  + attribute kernel             McAlister satiation, a few coefficients
#   L4  + free QKV kernel              upper bound only, NOT interpreted
#
# Architecture is FROZEN at the stage-4 hybrid (b15_hyb_c14); only the stock
# path moves, so every arm is comparable. 5 arms x 5 seeds = 25 runs.
# Adoption follows the pre-registered rule: a rung is adopted only if it beats
# the rung above by 0.02 on VALIDATION.
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,CAMP_FE=1"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"

declare -A RUNGS=(
  [L1_decay]="DECAY=exp"
  [L2_decay_tier]="DECAY=exp,TIER=1"
  [L3_attr]="KERNEL=attr,DECAY=exp,TIER=1"
  [L4_learned]="KERNEL=learned,DECAY=exp,TIER=1"
  [L2b_tier]="TIER=1"
)
ORDER=(L1_decay L2b_tier L2_decay_tier L3_attr L4_learned)

n=0
for rung in "${ORDER[@]}"; do
  for seed in 1 2 3 4 5; do
    v="${COMMON},SEED=${seed},${FROZEN},${RUNGS[$rung]},TAG=b21_${rung}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b21_%s_s%s\n" "${jid%%.*}" "$rung" "$seed"
    fi
    n=$((n + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
