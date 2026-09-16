#!/usr/bin/env bash
# ============================================================================
# Batch 10: recurrence + attention over past occasions (R25)
#
# WHY
# ---
# Across 44 configurations no transformer has beaten the best recurrent model,
# and the honest summary is a tie. The pre-2017 architecture (Bahdanau 2015;
# Luong 2015) combined the two rather than replacing one with the other, and
# that combination has never been tested here at the SEQUENCE level: --encoder
# gru only swaps the stack, keeping the offer-inventory cross-attention.
#
# The two memories do different jobs. Recurrence carries a decay prior for
# free -- recent occasions weigh more, which is why the GRU wins (R18, R22).
# Attention retrieves a specific past occasion by content regardless of
# distance, which is why it failed alone and why a fixed decay bias rescued it.
#
# ARMS (S=1024, VAL_MODE=late, EPOCHS=40, seeds 1 2 3; inventory = tokens)
#   hyb_gate        GRU and attention over the same event representations,
#                   blended by a learned gate; attention keeps ALiBi
#   hyb_stack       GRU and attention interleaved layer by layer
#   hyb_gate_norec  hyb_gate with NO recency bias on the attention half:
#                   does recurrence already supply recency?
#
# Controls at the same seeds: b4_gru_cross (GRU encoder, 0.8889) and
# b4_tf_alibi (attention encoder, 0.8860), both inside gen 5; the plain GRU
# baseline (b2_gru, 0.8802) is the overall benchmark.
#
# The gated arms log the mean gate weight on the recurrent branch each epoch
# (1.0 = pure recurrence, 0.0 = pure retrieval) -- a reportable diagnostic,
# never used for selection.
#
# USAGE
#   bash scripts/submit_batch10.sh
#   DRY_RUN=1 bash scripts/submit_batch10.sh
# ============================================================================
set -euo pipefail
export PATH="$PATH:/opt/pbs/bin"

SEEDS=(1 2 3)
COMMON="MAX_EVENTS=1024,USER_EMB=0,VAL_MODE=late,EPOCHS=40,ENCODER=gru_attn"

declare -A ARMS=(
  [hyb_gate]="FUSE=gate,ALIBI=1"
  [hyb_stack]="FUSE=stack,ALIBI=1"
  [hyb_gate_norec]="FUSE=gate"
)
ORDER=(hyb_gate hyb_stack hyb_gate_norec)

n=0
for seed in "${SEEDS[@]}"; do
  for name in "${ORDER[@]}"; do
    v="${COMMON},${ARMS[$name]},SEED=${seed},TAG=b10_${name}_s${seed}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "qsub -v $v scripts/gen5_train_hpcc.pbs"
    else
      jid="$(qsub -v "$v" scripts/gen5_train_hpcc.pbs)"
      printf "%-8s b10_%s_s%s\n" "${jid%%.*}" "$name" "$seed"
    fi
    n=$((n + 1))
  done
done
[[ "${DRY_RUN:-0}" == "1" ]] && echo "--- $n runs listed (dry run)" || echo "--- $n runs submitted"
