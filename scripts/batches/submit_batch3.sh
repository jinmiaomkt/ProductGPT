#!/usr/bin/env bash
# ============================================================================
# Batch 3: component ablation + recency bias + the memorisation test (R18).
#
# WHY
# ---
# R17: a GRU on the SAME feature pipeline beats the gen-5 transformer by
# 0.12-0.15 nats with 45% fewer parameters, and the transformer's holdout
# optimum sits at epoch 1-3 while the GRU's is at 24+. The hypothesis in R18
# is that the attention stack learns campaign IDENTITY through the offer
# stream (product-id embeddings are a campaign fingerprint), which fits the
# calibration period and is useless on unseen holdout banners.
#
# These are SCREENING runs: one seed each, holdout tracked every epoch so the
# position of the optimum is visible, not only its value. Winners get three
# seeds afterwards. All select on late validation (R16).
#
# ARMS (all S=1024, USER_EMB=0, VAL_MODE=late, TRACK_HOLDOUT=1, EPOCHS=40)
#   tf_nocross   transformer encoder, offer-inventory cross-attention OFF
#   gru_cross    GRU encoder inside gen-5, cross-attention ON   ("GRU + cross-attn")
#   gru_nocross  GRU encoder inside gen-5, cross-attention OFF  (baseline + inventory GRU)
#   tf_alibi     transformer + ALiBi recency bias
#   tf_noid      transformer, products by attributes only      (the mechanism test)
#   gru_noid     standalone GRU baseline, products by attributes only
#
# With the existing b2_tf_noemb (transformer, cross ON) and b2_gru (baseline)
# from batch 2 these complete the 2x2 {encoder} x {cross-attention}.
#
# USAGE
#   bash scripts/submit_batch3.sh
#   DRY_RUN=1 bash scripts/submit_batch3.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,TRACK_HOLDOUT=1,EPOCHS=40,SEED=1"

declare -A ARMS=(
  [tf_nocross]="CROSS_ATTN=0"
  [gru_cross]="ENCODER=gru"
  [gru_nocross]="ENCODER=gru,CROSS_ATTN=0"
  [tf_alibi]="ALIBI=1"
  [tf_noid]="PROD_ID=0"
  [gru_noid]="ARCH=gru,PROD_ID=0"
)
ORDER=(tf_noid gru_noid tf_alibi gru_cross tf_nocross gru_nocross)   # mechanism test first

n=0
for name in "${ORDER[@]}"; do
  v="${COMMON},${ARMS[$name]},TAG=b3_${name}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
  else
    jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
    printf "%-8s b3_%s\n" "${jid%%.*}" "$name"
  fi
  n=$((n + 1))
done
echo "--- $n runs submitted"
