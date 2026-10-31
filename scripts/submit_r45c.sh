#!/usr/bin/env bash
# R45 part C -- the real left-censored players: 10 gen-6 runs, then one
# evaluation that waits for all of them.
#
# Pre-registered in EXPERIMENTS.md (R45 part C), 31 Oct 2026. Runs on R44's
# FROZEN gen-6 configuration, read from $FROZEN. Submit from
# ~/ProductGPT/work_r45c, after the gate (scripts/r45c_tests_cpu.pbs) passes.
#
# Arms, seeds 1-5, both with the 4,417 in-sample censored players training:
#   I0  no flag
#   I   the censoring flag (part A's rule)
# Arm F is R44's stage-4 gen-6 runs (b39_g6w384); the evaluation scores it on
# the censored cells with the same partition.
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
FROZEN="${FROZEN:-$HOME/ProductGPT/r44_frozen_g6.env}"
EXTRA=perproduct/censored_list_int_wide4_simple7_IPT.json
[[ "$(basename "$PWD")" == "work_r45c" ]] || { echo "submit from ~/ProductGPT/work_r45c"; exit 2; }
[[ -s "$FROZEN" ]] || { echo "no frozen R44 gen-6 configuration at $FROZEN"; exit 3; }
[[ -s "$HOME/ProductGPT/data/$EXTRA" ]] || { echo "no $EXTRA under ~/ProductGPT/data"; exit 4; }
HP="$(tr -d ' \n' < "$FROZEN")"
[[ "$HP" == *WINDOW=* && "$HP" == *LR=* ]] || { echo "unexpected content in $FROZEN: $HP"; exit 5; }
echo "frozen R44 gen-6 configuration: $HP"
ids=()
go() { jid="$(qsub -v "$1" scripts/gen6_train_hpcc.pbs)"; ids+=("${jid%%.*}"); printf "%-8s %s\n" "${jid%%.*}" "$1"; }
for seed in 1 2 3 4 5; do
  base="SEED=${seed},BATCH_SIZE=4,N_HEADS=8,CHUNK=512,${HP},EXTRA_DATA=${EXTRA},EXTRA_TRAIN=1"
  go "${base},CENSOR_FLAG=0,TAG=r45c_I0"
  go "${base},CENSOR_FLAG=1,TAG=r45c_I"
done
dep="$(IFS=:; echo "${ids[*]}")"
ev="$(qsub -W depend=afterok:${dep} -v TAGS=b39_g6w384:r45c_I0:r45c_I,SEEDS=1:2:3:4:5 scripts/r45c_eval.pbs)"
echo "submitted ${#ids[@]} runs; evaluation ${ev%%.*} waits for all of them"
