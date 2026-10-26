#!/bin/bash
# ============================================================================
# R46 part A on HPCC CPUs, as one dependency chain (EXPERIMENTS.md R46):
#
#   export (G1, features, G2)          short, 8 CPUs
#     -> stage 1 array, 12 fits        short, 16 CPUs each, in parallel
#        -> stage 2 array, 20 R6 fits  long,  16 CPUs each, in parallel
#           -> compare                 short, 4 CPUs
#
# Run from the R46 worktree:  cd ~/ProductGPT/work_r46 && bash scripts/submit_r46a.sh
# SMOKE=1 runs the whole chain on the first 300 customers into runs/r46_smoke
# (G1 is not checked there), to prove the wiring before the real chain.
# Stage 3 (SEs of the selected R6 fit) is submitted by hand after compare:
#     qsub -v STAGE=3,TAG=R6_K<k>_s<start> -q long -l walltime=24:00:00 scripts/r46_fit_cpu.pbs
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PBSBIN=/opt/pbs/bin
if [ "${SMOKE:-0}" = "1" ]; then
    R46=/storage/home/jinmiao/ProductGPT/runs/r46_smoke
    E=$($PBSBIN/qsub -v R46=$R46,SMOKE=1 scripts/r46_smoke_cpu.pbs)
    echo "smoke chain: $E"
    exit 0
fi
R46=/storage/home/jinmiao/ProductGPT/runs/r46
E=$($PBSBIN/qsub -v R46=$R46 scripts/r46_export_cpu.pbs)
S1=$($PBSBIN/qsub -J 0-11 -v STAGE=1,R46=$R46 -W depend=afterok:$E scripts/r46_fit_cpu.pbs)
S2=$($PBSBIN/qsub -J 0-19 -v STAGE=2,R46=$R46 -q long -l select=1:ncpus=16:mem=128GB \
     -l walltime=48:00:00 -W depend=afterok:$S1 scripts/r46_fit_cpu.pbs)
C=$($PBSBIN/qsub -v R46=$R46 -W depend=afterany:$S2 scripts/r46_compare_cpu.pbs)
echo "export $E"
echo "stage 1 (M0..R5) $S1"
echo "stage 2 (R6 x 20) $S2"
echo "compare $C"
