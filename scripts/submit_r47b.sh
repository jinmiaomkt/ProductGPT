#!/usr/bin/env bash
# R47 parts B1-B3 -- the half-life profile (pre-registered in EXPERIMENTS.md R47).
#
#   bash scripts/submit_r47b.sh b12           # B1 and B2: 42 runs
#   bash scripts/submit_r47b.sh b3 <HL>       # B3 at B2's best half-life HL
#                                            #   (a number, or "inf"): 12 runs
#
# Runs on R44's FROZEN gen-6 configuration ($FROZEN, written at R44 stage 3) and
# refuses without it. Submit from ~/ProductGPT/work_r47 (branch r47-forgetting).
# The half-life is frozen at each grid value (DECAY_FREEZE=1); "inf" is no
# decay (DECAY=none). The rest of the stock stays as in the incumbent: attribute
# kernel, duplicate tiers, campaign control.
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
FROZEN="${FROZEN:-$HOME/ProductGPT/r44_frozen_g6.env}"
part="${1:?usage: submit_r47b.sh b12 | b3 <HL>}"
[[ "$(basename "$PWD")" == "work_r47" ]] || { echo "submit from ~/ProductGPT/work_r47"; exit 2; }
[[ -s "$FROZEN" ]] || { echo "no frozen R44 gen-6 configuration at $FROZEN"; exit 3; }
HP="$(tr -d ' \n' < "$FROZEN")"
[[ "$HP" == *WINDOW=* && "$HP" == *LR=* ]] || { echo "unexpected content in $FROZEN: $HP"; exit 4; }
echo "frozen R44 gen-6 configuration: $HP"
n=0
go() { jid="$(qsub -v "$1" scripts/gen6_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
hl_vars() { if [[ "$1" == "inf" ]]; then echo "DECAY=none"; else echo "DECAY=exp,DECAY_INIT=$1,DECAY_FREEZE=1"; fi; }
COMMON="BATCH_SIZE=4,N_HEADS=8,CHUNK=512,KERNEL=attr,TIER=1,CAMP_FE=1,${HP}"
case "$part" in
  b12)
    for seed in 1 2 3; do
      for hl in 1 2 5 10 30 100 inf; do
        go "SEED=${seed},${COMMON},$(hl_vars "$hl"),SLOT_STATE=full,TAG=r47b1_hl${hl}"
        go "SEED=${seed},${COMMON},$(hl_vars "$hl"),SLOT_STATE=stock_only,TAG=r47b2_hl${hl}"
      done
    done ;;
  b3)
    hl="${2:?b3 needs B2's best half-life}"
    for seed in 1 2 3; do
      for d in 1 0.5 0.1 0; do
        go "SEED=${seed},${COMMON},$(hl_vars "$hl"),SLOT_STATE=stock_only,CAMP_DECAY=${d},TAG=r47b3_hl${hl}_d${d}"
      done
    done ;;
  *) echo "unknown part $part"; exit 5 ;;
esac
echo "submitted $n runs"
