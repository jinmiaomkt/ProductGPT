#!/usr/bin/env bash
# ============================================================================
# Batch 11: core replication on the CORRECTED obtained-products stream (R27)
#
# WHY
# ---
# R26 found that every gen-5 run to date read an obtained-products stream in
# which 45 of 122 products carried the wrong id (version-2 codes decoded with
# the version-6 table). The file was regenerated and verified on Sep 17 2026.
# Decisions, offers and timing are unchanged, so this batch reruns only the
# configurations whose conclusions could depend on product identity: the
# architecture comparison and the value of reading the inventory.
#
# ARMS (S=1024, VAL_MODE=late, EPOCHS=40, USER_EMB=0, seeds 1 2 3)
#   gru              plain GRU baseline (no inventory GRU, no offer-inventory
#                    attention)                          pre-fix 0.8802 (b2)
#   gru_cross        gen-5 GRU encoder + inventory GRU + offer-inventory
#                    attention                           pre-fix 0.8889 (b4)
#   gru_nocross      same without the offer-inventory attention
#                                                        pre-fix 0.9137 (b3, 1 seed)
#   tf_alibi         transformer + ordinal recency + attention
#                                                        pre-fix 0.8860 (b4)
#   tf_alibi_nocross same without the offer-inventory attention (new)
#
# Decision rules are pre-registered in EXPERIMENTS.md, R27.
#
# USAGE
#   bash scripts/submit_batch11.sh
#   DRY_RUN=1 bash scripts/submit_batch11.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40"

declare -A ARMS=(
  [gru]="ARCH=gru"
  [gru_cross]="ENCODER=gru"
  [gru_nocross]="ENCODER=gru,CROSS_ATTN=0"
  [tf_alibi]="ALIBI=1"
  [tf_alibi_nocross]="ALIBI=1,CROSS_ATTN=0"
)
# Seed 1 of every arm first, so a full first pass lands before any replication.
ORDER=(gru tf_alibi gru_cross gru_nocross tf_alibi_nocross)

n=0
for seed in "${SEEDS[@]}"; do
  for name in "${ORDER[@]}"; do
    v="${COMMON},${ARMS[$name]},SEED=${seed},TAG=b11_${name}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b11_%s_s%s\n" "${jid%%.*}" "$name" "$seed"
    fi
    n=$((n + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
