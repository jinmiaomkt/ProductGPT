#!/usr/bin/env bash
# ============================================================================
# Batch 14: capacity frontier, stage 1 of the tuning programme (R30)
#
# WHY
# ---
# Every architecture comparison so far sits at ONE point on each family's
# capacity curve, inherited from a gen-4 sweep on different data. R29 showed
# what that costs: a single extra width on the GRU (d=176) moved it 0.0148 at
# level 7 and overturned R28's headline. This stage measures the curve for all
# four families so later comparisons are frontier-to-frontier.
#
# DESIGN  (level 7; per-product counts as the stock path; 1 seed)
#   families   gru       plain GRU baseline
#              gru_enc   gen-5 recurrent encoder
#              tf        transformer + ALiBi recency
#              hyb       GRU and attention interleaved
#   width      d_model 96, 128, 176, 256   (d_ff = 3 x d_model, 8 heads)
#   depth      N = 2, 4, 6
#   = 4 x 4 x 3 = 48 runs, seed 1 only
#
# Per R30 P1, these are ranked on CAMPAIGN-27 VALIDATION with
#   python3 scripts/summarize_search.py --prefix b14_
# The holdout stays closed until stage 4.
#
# USAGE
#   bash scripts/submit_batch14.sh
#   DRY_RUN=1 bash scripts/submit_batch14.sh | tail -3
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEED=1
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,VOCAB_LEVEL=7"
STOCK="INVENTORY=slots,SAT_LAYERS=2"

declare -A FAM=(
  [gru]="ARCH=gru"
  [gru_enc]="ENCODER=gru,${STOCK}"
  [tf]="ALIBI=1,${STOCK}"
  [hyb]="ENCODER=gru_attn,FUSE=stack,ALIBI=1,${STOCK}"
)
ORDER=(gru gru_enc tf hyb)
WIDTHS=(96 128 176 256)
DEPTHS=(2 4 6)

n=0
for d in "${WIDTHS[@]}"; do
  for N in "${DEPTHS[@]}"; do
    for fam in "${ORDER[@]}"; do
      v="${COMMON},${FAM[$fam]},D_MODEL=${d},D_FF=$((3 * d)),N_HEADS=8,N_LAYERS=${N},SEED=${SEED},TAG=b14_${fam}_d${d}_N${N}"
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
      else
        jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
        printf "%-8s b14_%s_d%s_N%s\n" "${jid%%.*}" "$fam" "$d" "$N"
      fi
      n=$((n + 1))
    done
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
