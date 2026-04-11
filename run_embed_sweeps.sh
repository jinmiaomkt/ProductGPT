#!/bin/bash
# run_embed_sweeps.sh
# Runs GRU_embed then LSTM_embed sequentially on the same GPU.
# Usage: screen -S sweeps
#        bash run_embed_sweeps.sh

echo "=========================================="
echo "Starting GRU_embed sweep: $(date)"
echo "=========================================="
python3 hP_tuning_GRU_embed.py 2>&1 | tee gru_embed.log

echo ""
echo "=========================================="
echo "GRU_embed finished: $(date)"
echo "Starting LSTM_embed sweep: $(date)"
echo "=========================================="
python3 hP_tuning_LSTM_embed.py 2>&1 | tee lstm_embed.log

echo ""
echo "=========================================="
echo "All sweeps finished: $(date)"
echo "=========================================="