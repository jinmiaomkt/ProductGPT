#!/usr/bin/env bash
# ============================================================================
# Batch 19: R32 -- the three hybrid designs we had specified but never built.
#
#   seq_ra  GRU over the sequence, attention over its hidden states
#   seq_ar  attention first, a GRU reads out the contextualised sequence
#   block   attention within a window, recurrence across windows (O(S*K))
#
# Equal budget to what the incumbent hybrid got, at a third of the scale:
# 4 capacity cells + 8 hyperparameter draws per design = 12 x 3 = 36 runs,
# seed 1, level 7, per-product counts as the stock path. The incumbent
# (FUSE=stack) is not re-run: its frontier is b15_hyb_c14 / b14_hyb_d96_N6.
#
# Ranked on VALIDATION (scripts/summarize_search.py --prefix b19_). A design
# replaces the incumbent only by the 0.02 rule, then on three fresh seeds.
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7,N_HEADS=8,SEED=1"
STOCK="INVENTORY=slots,SAT_LAYERS=2"
BASE="ENCODER=gru_attn,ALIBI=1,${STOCK}"
# the frozen hybrid's training hyperparameters (b15_hyb_c14)
FROZEN="LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"

CAPS=("96 6" "128 4" "176 2" "256 2")
# 8 draws spanning the pre-registered stage-2 ranges
HPS=(
  "LR=1.2e-04,DROPOUT=0.05,WD=0,GRAD_ACCUM=2,WARMUP=0.02"
  "LR=2.0e-04,DROPOUT=0.2,WD=0.1,GRAD_ACCUM=8,WARMUP=0.1"
  "LR=3.3e-04,DROPOUT=0.1,WD=0.01,GRAD_ACCUM=4,WARMUP=0.05"
  "LR=4.6e-04,DROPOUT=0.3,WD=0.05,GRAD_ACCUM=2,WARMUP=0.05"
  "LR=5.6e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.05"
  "LR=7.2e-04,DROPOUT=0.05,WD=0.1,GRAD_ACCUM=4,WARMUP=0.1"
  "LR=9.5e-04,DROPOUT=0.2,WD=0,GRAD_ACCUM=8,WARMUP=0.02"
  "LR=1.15e-03,DROPOUT=0.3,WD=0.01,GRAD_ACCUM=4,WARMUP=0.1"
)

n=0
submit () {   # $1 = variable string, $2 = tag
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "qsub -v $1,TAG=$2 scripts/gen5_train_hpcc.pbs"
  else
    jid="$(qsub -v "$1,TAG=$2" scripts/gen5_train_hpcc.pbs)"
    printf "%-8s %s\n" "${jid%%.*}" "$2"
  fi
  n=$((n + 1))
}

for fuse in seq_ra seq_ar block; do
  extra=""
  [[ "$fuse" == "block" ]] && extra=",BLOCK_LEN=64"
  for cap in "${CAPS[@]}"; do
    set -- $cap
    d=$1; N=$2
    submit "${COMMON},${BASE},FUSE=${fuse}${extra},D_MODEL=${d},D_FF=$((3 * d)),N_LAYERS=${N},${FROZEN}" \
           "b19_${fuse}_d${d}_N${N}"
  done
  k=0
  for hp in "${HPS[@]}"; do
    submit "${COMMON},${BASE},FUSE=${fuse}${extra},D_MODEL=96,D_FF=288,N_LAYERS=6,${hp}" \
           "b19_${fuse}_c$(printf '%02d' $k)"
    k=$((k + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
