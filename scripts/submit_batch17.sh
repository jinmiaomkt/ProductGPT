#!/usr/bin/env bash
# ============================================================================
# Batch 16: R30 stage 3 -- confirmation. Top 3 configurations per family by
# validation across stages 1 and 2, each re-run on three FRESH seeds (11,12,13)
# = 36 runs. Ranked by validation MEAN over seeds; the holdout opens at stage 4.
#
#   DRY_RUN=1 bash scripts/submit_batch17.sh | tail -5
#   bash scripts/submit_batch17.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

plan="$(python3 scripts/r30_stage4_select.py)"
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
