#!/bin/bash
# ============================================================================
# R46 part B on HPCC CPUs, as one dependency chain (EXPERIMENTS.md R46 part B):
#
#   rollout array, 19 elements     short, 4 CPUs each, in parallel
#     -> compare                   short, 4 CPUs
#
# Run from the R46 worktree, after the smoke (qsub scripts/r46b_smoke_cpu.pbs)
# has been read:   cd ~/ProductGPT/work_r46 && bash scripts/submit_r46b.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PBSBIN=/opt/pbs/bin
A=$($PBSBIN/qsub -J 0-18 scripts/r46b_rollout_cpu.pbs)
C=$($PBSBIN/qsub -W depend=afterany:$A scripts/r46b_compare_cpu.pbs)
echo "rollout array (reference + 9 rungs x 2 modes) $A"
echo "compare $C"
