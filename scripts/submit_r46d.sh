#!/bin/bash
# ============================================================================
# R46 part D on HPCC CPUs, as one dependency chain (EXPERIMENTS.md R46 part D):
#
#   REALB  real R6B, 2 starts            short  \
#   REALK  real K = 9 / 12, 6 starts     long    >  at once, 8 CPUs each
#   NULL1  20 fresh null (b) tables      short  /
#     -> NULL2B  R6B on each table, 40   short  after NULL1 (afterok)
#     -> NULL2K  K = 9 / 12, 120         long   after NULL1 (afterok)
#     -> read-out                               after everything (afterany)
#
# Run from the R46 worktree, after the tests and the smoke have been read:
#     cd ~/ProductGPT/work_r46 && bash scripts/submit_r46d.sh [K_MEM]
# K_MEM is the memory per K > 6 element (default 24GB), sized from the smoke.
# The smoke:  qsub -v STAGE=TEST scripts/r46d_cpu.pbs, then
#             qsub -J 0-2 -v STAGE=SMOKE1 ..., then -J 0-1 -v STAGE=SMOKE2 ...,
#             then qsub -v STAGE=CMP,SMOKE=1 scripts/r46d_cpu.pbs
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PBSBIN=/opt/pbs/bin
K_MEM="${1:-24GB}"
LONG=(-q long -l walltime=24:00:00 -l select=1:ncpus=8:mem=$K_MEM)
JB=$($PBSBIN/qsub -J 0-1 -v STAGE=REALB scripts/r46d_cpu.pbs)
JK=$($PBSBIN/qsub -J 0-5 -v STAGE=REALK "${LONG[@]}" scripts/r46d_cpu.pbs)
J1=$($PBSBIN/qsub -J 0-19 -v STAGE=NULL1 scripts/r46d_cpu.pbs)
J2B=$($PBSBIN/qsub -J 0-39 -v STAGE=NULL2B -W depend=afterok:$J1 scripts/r46d_cpu.pbs)
J2K=$($PBSBIN/qsub -J 0-119 -v STAGE=NULL2K -W depend=afterok:$J1 "${LONG[@]}" scripts/r46d_cpu.pbs)
JC=$($PBSBIN/qsub -v STAGE=CMP -W depend=afterany:$JB:$JK:$J2B:$J2K scripts/r46d_cpu.pbs)
echo "REALB  real R6B starts 0-1 $JB"
echo "REALK  real K = 9 / 12 (long, $K_MEM) $JK"
echo "NULL1  20 fresh null (b) tables $J1"
echo "NULL2B R6B on each table (after NULL1) $J2B"
echo "NULL2K K = 9 / 12 on each table (after NULL1, long) $J2K"
echo "read-out $JC"
