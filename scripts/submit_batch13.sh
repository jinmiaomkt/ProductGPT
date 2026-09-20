#!/usr/bin/env bash
# ============================================================================
# Batch 13: the full factorial (R29)
#
# WHY
# ---
# R28 found a crossover: with 44 pooled product tokens the GRU wins, with 118
# per-product tokens the transformer wins. That came from three arms at one
# vocabulary each. This batch crosses the three design dimensions properly so
# the interaction can be read instead of inferred, and adds the capacity
# control a reviewer will ask for first.
#
# DESIGN  (3 seeds each; 24 configurations; 72 runs)
#
#   vocabulary            6 = 118 products share 44 tokens
#                         7 = one token per product (ids 13-130)
#
#   decision-level        gru       plain GRU baseline, d=128 (no stock path)
#   memory                gruw      plain GRU at d=176 -> 1.25M params, matched
#                                   to the transformer's 1.22M  [CAPACITY CONTROL]
#                         gru_enc   gen-5 recurrent encoder
#                         tf        transformer + ALiBi ordinal recency
#                         tf_norec  transformer, no recency prior
#                         hyb       GRU and attention interleaved (stack)
#
#   stock-level           nostock   no offer-inventory attention
#   memory                tokens    inventory GRU + offer-inventory attention
#                         slots2    additive per-product counts + 2 satiation
#                                   layers  (at level 7 these are REAL per-product
#                                   counts over 118 products, not 44 pools)
#
#   Crossed: {gru_enc, tf, hyb} x {nostock, tokens, slots2} x {6,7}   = 18
#   Plus:    {gru, gruw} x {6,7}                                      =  4
#   Plus:    tf_norec x tokens x {6,7}                                =  2
#
# Seed-major order: every configuration runs at seed 1 before any seed 2, so a
# complete first pass lands in about a third of the total time.
#
# USAGE
#   bash scripts/submit_batch13.sh
#   DRY_RUN=1 bash scripts/submit_batch13.sh | head -30
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40"

# decision-level memory -> flags
declare -A DEC=(
  [gru]="ARCH=gru"
  [gruw]="ARCH=gru,D_MODEL=176,D_FF=528"
  [gru_enc]="ENCODER=gru"
  [tf]="ALIBI=1"
  [tf_norec]=""
  [hyb]="ENCODER=gru_attn,FUSE=stack,ALIBI=1"
)
# stock-level memory -> flags
declare -A STOCK=(
  [nostock]="CROSS_ATTN=0"
  [tokens]=""
  [slots2]="INVENTORY=slots,SAT_LAYERS=2"
)

CROSSED_DEC=(gru_enc tf hyb)
CROSSED_STOCK=(nostock tokens slots2)

emit() {  # $1 = tag, $2 = flags, $3 = seed
  local v="${COMMON},${2:+$2,}SEED=${3},TAG=b13_${1}_s${3}"
  v="${v//,,/,}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
  else
    jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
    printf "%-8s b13_%s_s%s\n" "${jid%%.*}" "$1" "$3"
  fi
  n=$((n + 1))
}

n=0
for seed in "${SEEDS[@]}"; do
  for level in 6 7; do
    lvl=""; [[ "$level" == "7" ]] && lvl="VOCAB_LEVEL=7"
    # baselines and the capacity control
    for d in gru gruw; do
      emit "v${level}_${d}" "${DEC[$d]}${lvl:+,$lvl}" "$seed"
    done
    # the 3 x 3 crossing
    for d in "${CROSSED_DEC[@]}"; do
      for s in "${CROSSED_STOCK[@]}"; do
        flags="${DEC[$d]}"
        [[ -n "${STOCK[$s]}" ]] && flags="${flags:+$flags,}${STOCK[$s]}"
        [[ -n "$lvl" ]] && flags="${flags:+$flags,}$lvl"
        emit "v${level}_${d}_${s}" "$flags" "$seed"
      done
    done
    # does product identity substitute for the recency prior?
    emit "v${level}_tf_norec" "${DEC[tf_norec]}${lvl:+$lvl}" "$seed"
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
