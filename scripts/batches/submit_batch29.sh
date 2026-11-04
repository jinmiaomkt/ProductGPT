#!/usr/bin/env bash
# ============================================================================
# Batch 29: the R33 model with the half-life chosen PROPERLY.
#
# Batch 25's profile was read off the holdout cell, which is selection on the
# holdout -- against the R30 protocol, and it makes batch 28's headline
# optimistic. This redoes it cleanly:
#
#   * three half-lives, FROZEN, each at 5 seeds, with the corrected attribute
#     similarity and campaign fixed effects;
#   * the winner is chosen on CAMPAIGN-27 VALIDATION
#     (scripts/summarize_search.py --prefix b29_ --group);
#   * only then is the holdout read, for the winner alone.
#
# The grid brackets the validation optimum from batch 25 (2, with 5 and 10
# inside one seed sd), and is run here WITH the kernel present, which batch 25
# was not.
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,CAMP_FE=1"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1"
n=0
for hl in 2 5 10; do
  for seed in 1 2 3 4 5; do
    v="${COMMON},SEED=${seed},${FROZEN},${STOCK},DECAY_INIT=${hl},TAG=b29_hl${hl}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "qsub -v $v"; else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"; printf "%-8s b29_hl%s_s%s\n" "${jid%%.*}" "$hl" "$seed"; fi
    n=$((n + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n listed" || echo "--- $n submitted"
