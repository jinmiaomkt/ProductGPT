#!/usr/bin/env bash
# Batch 42 -- R44 stage 4: the frozen gen-5 configuration (c06) on seeds 2-5 (4 runs).
#
# Stage 4 is each family's frozen configuration on seeds 1-5. The other six
# runs already exist and are reused (EXPERIMENTS.md R44, stage 4):
#   gen 5 c06 seed 1     b40_g5_c06_s1     (stage 2)
#   gen 6 R43 seeds 1-5  b39_g6w384_s1..5  (R43 step 1; the pre-registered reuse)
# The qsub string is re-drawn from r44_sample.py (same RNG, same configuration)
# with only SEED and TAG changed. Read with scripts/r44_stage4.py, once.
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
[[ "$(basename "$PWD")" == "work_gen6" ]] || { echo "submit from ~/ProductGPT/work_gen6"; exit 2; }
vars="$(python3 scripts/r44_sample.py | grep -E '^G5 .*TAG=b40_g5_c06_s1$' | cut -d' ' -f2-)"
vars="${vars%$'\r'}"
[[ -n "$vars" ]] || { echo "c06 not found in r44_sample.py output"; exit 3; }
n=0
for seed in 2 3 4 5; do
  v="${vars/SEED=1,/SEED=${seed},}"
  v="${v/TAG=b40_g5_c06_s1/TAG=b42_g5_c06_s${seed}}"
  jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
  printf "%-8s %s\n" "${jid%%.*}" "${v##*TAG=}"; n=$((n+1))
done
echo "submitted $n runs"
