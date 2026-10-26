#!/usr/bin/env bash
# R45 part A -- censored training and testing, 39 gen-6 runs.
#
# Pre-registered in EXPERIMENTS.md (R45), approved by Jin 27 Oct 2026. Runs on
# R44's FROZEN gen-6 configuration, read from $FROZEN (one line of qsub -v
# variables written when R44 stage 3 freezes gen 6, e.g.
#   WINDOW=384,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1
# ). Refuses to run without it. Submit from ~/ProductGPT/work_r45 (branch
# r45-censoring), which carries the chop options.
#
# Arms x calendar depth k in {1,2,4}, seeds 1-3, all with CHOP_WHERE=both:
#   L  CHOP_STATE=labels        volume control
#   T  CHOP_STATE=true          missing context only
#   Z  CHOP_STATE=zero          a real left-censored customer
#   ZF CHOP_STATE=zero + flag   the candidate fix
# plus the count cross-check: Z at 128 occasions (matches calendar k=2).
# Arm F (no chop) is R44's stage-4 gen-6 runs on seeds 1-3.
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
FROZEN="${FROZEN:-$HOME/ProductGPT/r44_frozen_g6.env}"
[[ "$(basename "$PWD")" == "work_r45" ]] || { echo "submit from ~/ProductGPT/work_r45"; exit 2; }
[[ -s "$FROZEN" ]] || { echo "no frozen R44 gen-6 configuration at $FROZEN; R44 stage 3 has not frozen it"; exit 3; }
HP="$(tr -d ' \n' < "$FROZEN")"
[[ "$HP" == *WINDOW=* && "$HP" == *LR=* ]] || { echo "unexpected content in $FROZEN: $HP"; exit 4; }
echo "frozen R44 gen-6 configuration: $HP"
n=0
go() { jid="$(qsub -v "$1" scripts/gen6_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
for seed in 1 2 3; do
  for k in 1 2 4; do
    base="SEED=${seed},BATCH_SIZE=4,N_HEADS=8,CHUNK=512,${HP},CHOP=calendar,CHOP_DEPTH=${k},CHOP_WHERE=both"
    go "${base},CHOP_STATE=labels,TAG=r45a_L_k${k}"
    go "${base},CHOP_STATE=true,TAG=r45a_T_k${k}"
    go "${base},CHOP_STATE=zero,TAG=r45a_Z_k${k}"
    go "${base},CHOP_STATE=zero,CENSOR_FLAG=1,TAG=r45a_ZF_k${k}"
  done
  go "SEED=${seed},BATCH_SIZE=4,N_HEADS=8,CHUNK=512,${HP},CHOP=count,CHOP_DEPTH=128,CHOP_WHERE=both,CHOP_STATE=zero,TAG=r45a_Zcount_d128"
done
echo "submitted $n runs"
