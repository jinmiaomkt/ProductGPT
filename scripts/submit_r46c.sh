#!/bin/bash
# ============================================================================
# R46 part C on HPCC CPUs, as one dependency chain (EXPERIMENTS.md R46 part C):
#
#   C1 parametric bootstrap, 40 elements   \
#   C2 customer bootstrap, 50 elements      >  at once, short, 8 CPUs each
#   H6 fits, 5 starts                      /
#     -> C3 Keane, 80 elements             after the H6 fits (afterok)
#     -> read-out                          after everything (afterany)
#
# Run from the R46 worktree, after the tests and the smoke have been read:
#     cd ~/ProductGPT/work_r46 && bash scripts/submit_r46c.sh
# The smoke:  qsub -v STAGE=TEST scripts/r46c_cpu.pbs, then
#             qsub -J 0-3 -v STAGE=SMOKE1 ..., then -J 0-3 -v STAGE=SMOKE2 ...,
#             then qsub -v STAGE=CMP,SMOKE=1 scripts/r46c_cpu.pbs
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PBSBIN=/opt/pbs/bin
J1=$($PBSBIN/qsub -J 0-39 -v STAGE=C1 scripts/r46c_cpu.pbs)
J2=$($PBSBIN/qsub -J 0-49 -v STAGE=C2 scripts/r46c_cpu.pbs)
JH=$($PBSBIN/qsub -J 0-4 -v STAGE=H6 scripts/r46c_cpu.pbs)
J3=$($PBSBIN/qsub -J 0-79 -v STAGE=C3 -W depend=afterok:$JH scripts/r46c_cpu.pbs)
JC=$($PBSBIN/qsub -v STAGE=CMP -W depend=afterany:$J1:$J2:$J3 scripts/r46c_cpu.pbs)
echo "C1 parametric bootstrap (R6 0-19, R4 20-39) $J1"
echo "C2 customer bootstrap $J2"
echo "H6 fits $JH"
echo "C3 Keane (after H6) $J3"
echo "read-out $JC"
