#!/usr/bin/env bash
# Batch 36 -- R37: does the INVENTORY SATIATION stream earn its place?
#
# The b18 arm recorded as "no stock path at all" was CROSS_ATTN=0 only. In the
# model `pieces = [z_x, z_o, z_y, h_H]` when z_sat is None, so the per-event
# acquisition embedding AND the inventory GRU both survived: it removed one of
# two inventory routes. Every b18 arm had an inventory stream, so that table
# ranks REPRESENTATIONS and says nothing about whether the stream is needed.
#
# This is the gen-5 DUPLET, the ablation gen 4 ran as triplet-vs-duplet:
# --no-obtained drops every term derived from the obtained stream (z_o, h_H and,
# under slots, z_sat / z_inv / z_stock), leaving the offer and the PREVIOUS
# DECISION. That separates SATIATION (what you hold, cumulatively) from STATE
# DEPENDENCE (whether the last occasion was a purchase), which is the comparison
# the question is actually about.
#
# Verified before submission: scrambling the obtained stream moves the logits by
# 1.87 (slots) / 1.68 (tokens) with the flag off, and by EXACTLY 0.00 with it on,
# while the previous-decision stream still moves them by ~0.4. The stream is
# severed and state dependence is intact.
#
# Arms, at vocabulary level 7 (118 per-product tokens), 5 seeds each:
#   b36_triplet  the frozen config -- control
#   b36_duplet   --no-obtained
# Read on VALIDATION as always.
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"
COMMON="PROFILE=hpcc,MAX_EVENTS=1024,EPOCHS=40,BATCH_SIZE=4,VOCAB_LEVEL=7,VAL_MODE=late,VAL_FROM=27"
FROZEN="ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,SAT_LAYERS=2,D_MODEL=96,D_FF=288,N_LAYERS=6,LR=2.19e-04,DROPOUT=0.1,WD=0.05,GRAD_ACCUM=2,WARMUP=0.1"
STOCK="KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2,CAMP_FE=1"
n=0
go() { jid="$(qsub -v "$1" scripts/gen5_train_hpcc.pbs)"; printf "%-8s %s\n" "${jid%%.*}" "${1##*TAG=}"; n=$((n+1)); }
for seed in 1 2 3 4 5; do
  go "${COMMON},SEED=${seed},${FROZEN},${STOCK},TAG=b36_triplet_s${seed}"
  go "${COMMON},SEED=${seed},${FROZEN},${STOCK},NO_OBTAINED=1,TAG=b36_duplet_s${seed}"
done
echo "submitted $n runs"
