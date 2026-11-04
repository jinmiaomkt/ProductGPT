#!/usr/bin/env bash
# ============================================================================
# Batch 7: is the model deep or wide enough? (R22)
#
# WHY
# ---
# Every run so far used one capacity: d_model=128, 4 layers, 8 heads, d_ff=384.
# The behavioural mechanisms are arguably deep -- how far the need for the
# current limited-time offer is already met by the inventory is an interaction
# of offer attributes, owned attributes and counts -- so capacity is an
# untested explanation for the transformer only tying the GRU.
#
# Recurrent arms at matched depth and width make sure any gain is credited to
# capacity, not to one architecture happening to like it.
#
# ARMS (S=1024, VAL_MODE=late, EPOCHS=40, seeds 1 2 3; customer embedding off)
#   Transformer + ordinal recency bias (ALiBi)
#     tf_N2        2 layers,  d_model 128
#     tf_N8        8 layers,  d_model 128
#     tf_d256      4 layers,  d_model 256
#     tf_N8_d256   8 layers,  d_model 256
#   Plain GRU
#     gru_N2       2 layers,  d_model 128
#     gru_N8       8 layers,  d_model 128
#     gru_d256     4 layers,  d_model 256
#
# The 4-layer, 128-wide references already exist at the same seeds:
# b4_tf_alibi (0.886) and b2_gru (0.880).
#
# MEMORY: the reference peaks at 20.2 GB of the L40S's 44 GB, dominated by the
# offer-inventory cross-attention, which does not grow with width or depth.
# tf_N8_d256 is the arm most likely to run out; it fails fast if so.
#
# USAGE
#   bash scripts/submit_batch7.sh
#   DRY_RUN=1 bash scripts/submit_batch7.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40"
W256="D_MODEL=256,N_HEADS=8,D_FF=768"

declare -A ARMS=(
  [tf_N2]="ALIBI=1,N_LAYERS=2"
  [tf_N8]="ALIBI=1,N_LAYERS=8"
  [tf_d256]="ALIBI=1,${W256}"
  [tf_N8_d256]="ALIBI=1,N_LAYERS=8,${W256}"
  [gru_N2]="ARCH=gru,N_LAYERS=2"
  [gru_N8]="ARCH=gru,N_LAYERS=8"
  [gru_d256]="ARCH=gru,${W256}"
)
# Seed 1 of every arm first, so a partial batch is already a complete screen.
ORDER=(tf_N2 tf_N8 tf_d256 tf_N8_d256 gru_N2 gru_N8 gru_d256)

n=0
for seed in "${SEEDS[@]}"; do
  for name in "${ORDER[@]}"; do
    v="${COMMON},${ARMS[$name]},SEED=${seed},TAG=b7_${name}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b7_%s_s%s\n" "${jid%%.*}" "$name" "$seed"
    fi
    n=$((n + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
