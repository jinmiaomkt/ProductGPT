#!/usr/bin/env bash
# ============================================================================
# Batch 2: honest comparisons, baselines, and the dropout sweep.
#
# PRECONDITION -- do not run this until batch 1 has passed its gate.
# Every run here selects its checkpoint on LATE-calibration validation
# (VAL_MODE=late). That is only honest if late validation has been shown to
# pick an epoch near the holdout optimum (jobs 40852/40853; read them with
# scripts/drift_diagnostic.py). Under the old customer-based validation, R13
# showed selection costing up to 0.155 nats; running 24 jobs on a selection
# rule that has not been checked would reproduce that at scale.
#
# WHAT IT SUBMITS (24 runs, S=1024, VAL_MODE=late, default patience)
#   Step 3  transformer   noemb | emb | mix8 | noemb dropout .25   x 3 seeds
#   Step 4a baselines     gru | lstm (no user embedding)            x 3 seeds
#   Step 4b dropout sweep noemb dropout .40 | .55                   x 3 seeds
#                         (0.10 and 0.25 already come from step 3)
#
# Seeds vary initialisation and data order only. The customer partition is
# fixed by split_seed=33, so all 24 runs share one partition and the spread
# across seeds measures training noise.
#
# Seed 1 of every configuration also runs with TRACK_HOLDOUT=1, so selection
# can be checked per configuration, not just for the two architectures the
# gate covered. Tracking never affects selection; it costs ~35% more time.
#
# USAGE (on OMEGA, from the repo root):
#   DRY_RUN=1 bash scripts/submit_batch2.sh     # print the qsub lines only
#   bash scripts/submit_batch2.sh               # submit
# ============================================================================
set -euo pipefail

export PATH="$PATH:/opt/pbs/bin"
DRY_RUN="${DRY_RUN:-0}"
SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,VAL_MODE=late,EPOCHS=40"

# name -> extra qsub variables
declare -a NAMES=(tf_noemb tf_emb tf_mix8 tf_noemb_do25 gru lstm tf_noemb_do40 tf_noemb_do55)
declare -A VARS=(
  [tf_noemb]="USER_EMB=0"
  [tf_emb]=""
  [tf_mix8]="USER_EMB=0,MIX_HEADS=8"
  [tf_noemb_do25]="USER_EMB=0,DROPOUT=0.25"
  [gru]="ARCH=gru,USER_EMB=0"
  [lstm]="ARCH=lstm,USER_EMB=0"
  [tf_noemb_do40]="USER_EMB=0,DROPOUT=0.40"
  [tf_noemb_do55]="USER_EMB=0,DROPOUT=0.55"
)

n=0
for name in "${NAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    v="${COMMON},SEED=${seed},TAG=b2_${name}_s${seed}"
    [[ -n "${VARS[$name]}" ]] && v="${v},${VARS[$name]}"
    [[ "$seed" == "1" ]] && v="${v},TRACK_HOLDOUT=1"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      echo "${jid%%.*}  b2_${name}_s${seed}"
    fi
    n=$((n + 1))
  done
done
if [[ "$DRY_RUN" == "1" ]]; then
  echo "--- $n runs (dry run: nothing submitted)"
else
  echo "--- $n runs submitted"
fi
