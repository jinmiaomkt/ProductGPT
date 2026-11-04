#!/usr/bin/env bash
# Batch 40 -- R44 stage 2: the R30 stage-2 random search for gen 5 (rev-2
# incumbent) and gen 6, 24 shared configurations each, seed 1, 48 runs.
#
# PRE-REGISTERED in EXPERIMENTS.md (R44). Do NOT submit before Jin has approved
# the entry. Submit from ~/ProductGPT/work_gen6 so both families run one commit.
# Selection reads validation only: python3 scripts/summarize_search.py --prefix b40_
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
[[ "$(basename "$PWD")" == "work_gen6" ]] || { echo "submit from ~/ProductGPT/work_gen6"; exit 2; }
n=0
while read -r fam vars; do
  case "$fam" in
    G5) pbs=scripts/gen5_train_hpcc.pbs ;;
    G6) pbs=scripts/gen6_train_hpcc.pbs ;;
    *) echo "bad line: $fam $vars"; exit 3 ;;
  esac
  jid="$(qsub -v "$vars" "$pbs")"
  printf "%-8s %s %s\n" "${jid%%.*}" "$fam" "${vars##*TAG=}"
  n=$((n+1))
done < <(python3 scripts/r44_sample.py)
echo "submitted $n runs"
