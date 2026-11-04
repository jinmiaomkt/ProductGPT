#!/usr/bin/env bash
# Batch 41 -- R44 stage 3: confirmation on fresh seeds 11, 12, 13 (18 runs).
#
# Candidates (stage 2's validation ranking, approved by Jin 27 Oct 2026):
#   gen 5: c06, c23, c07                      (stage-2 draws, scripts/r44_sample.py)
#   gen 6: c23, the R43 configuration, c02    (R43's = the trainer's frozen defaults,
#                                              window 384)
# The stage-2 qsub strings are re-drawn from r44_sample.py (same RNG, same
# configurations) with only SEED and TAG changed. Freeze each family by its
# validation mean over these seeds: python3 scripts/summarize_search.py --prefix b41_ --group
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
[[ "$(basename "$PWD")" == "work_gen6" ]] || { echo "submit from ~/ProductGPT/work_gen6"; exit 2; }
G5_KEEP="c06 c23 c07"
G6_KEEP="c23 c02"
n=0
go() { jid="$(qsub -v "$2" "$1")"; printf "%-8s %s\n" "${jid%%.*}" "${2##*TAG=}"; n=$((n+1)); }
draws="$(python3 scripts/r44_sample.py)"
for seed in 11 12 13; do
  while read -r fam vars; do
    vars="${vars%$'\r'}"
    cfg="${vars##*TAG=b40_g?_}"; cfg="${cfg%%_s1}"
    if [[ "$fam" == "G5" && " $G5_KEEP " == *" $cfg "* ]]; then
      v="${vars/SEED=1,/SEED=${seed},}"
      go scripts/gen5_train_hpcc.pbs "${v/TAG=b40_g5_${cfg}_s1/TAG=b41_g5_${cfg}_s${seed}}"
    elif [[ "$fam" == "G6" && " $G6_KEEP " == *" $cfg "* ]]; then
      v="${vars/SEED=1,/SEED=${seed},}"
      go scripts/gen6_train_hpcc.pbs "${v/TAG=b40_g6_${cfg}/TAG=b41_g6_${cfg}}"
    fi
  done <<< "$draws"
  go scripts/gen6_train_hpcc.pbs "SEED=${seed},BATCH_SIZE=4,N_HEADS=8,CHUNK=512,WINDOW=384,TAG=b41_g6_r43"
done
echo "submitted $n runs"
