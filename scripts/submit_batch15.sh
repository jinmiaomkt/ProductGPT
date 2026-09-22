#!/usr/bin/env bash
# ============================================================================
# Batch 15: R30 stage 2, random hyperparameter search around each family's
# stage-1 frontier. 24 configs x 4 families = 96 runs, level 7, slots2 stock
# path, seed 1. The plan is generated from stage-1 VALIDATION results by
# scripts/r30_stage2_sample.py (reproducible RNG); see EXPERIMENTS.md R30.
#
# Rank afterwards on validation only:
#   python3 scripts/summarize_search.py --prefix b15_ --top 20
#
# USAGE
#   DRY_RUN=1 bash scripts/submit_batch15.sh
#   bash scripts/submit_batch15.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

plan="$(python3 scripts/r30_stage2_sample.py)"
n=0
while IFS= read -r v; do
  [[ -z "$v" ]] && continue
  tag="${v##*TAG=}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
  else
    jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
    printf "%-8s %s\n" "${jid%%.*}" "$tag"
  fi
  n=$((n + 1))
done <<< "$plan"
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
