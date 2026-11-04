#!/bin/bash
# run_mixture_flash_tune.sh
#
# Full workflow for FlashAttention + Mixture-Head hyperparameter search.
#
# REQUIREMENTS:
#   - A10G GPU (g5.2xlarge) for FlashAttention
#   - Files on EC2: model4_mixture_flash.py, train4_mixture_flash_aws.py,
#                   ray_tune4_mixture_flash.py, select_best_mixture_flash.py
#
# USAGE:
#   1. Upload code to EC2:
#      scp -i ProductGPT_key.pem model4_mixture_flash.py train4_mixture_flash_aws.py \
#          ray_tune4_mixture_flash.py select_best_mixture_flash.py \
#          ec2-user@<EC2_IP>:/home/ec2-user/ProductGPT/
#
#   2. SSH into EC2:
#      ssh -i ProductGPT_key.pem ec2-user@<EC2_IP>
#
#   3. Start in a screen session:
#      screen -S mixture_tune
#      cd /home/ec2-user/ProductGPT
#      bash run_mixture_flash_tune.sh
#
#   4. Detach: Ctrl+A, D
#   5. Reattach: screen -r mixture_tune
#
# MONITORING:
#   # From another terminal:
#   screen -r mixture_tune                    # see live output
#   python3 select_best_mixture_flash.py      # ranked table of all trials
#   nvidia-smi                                # GPU usage

set -e

echo "=========================================="
echo "Mixture Flash HP Tuning: $(date)"
echo "=========================================="

# Kill any leftover ray processes
ray stop 2>/dev/null || true
sleep 2

# Run the sweep (Stage A: 100 trials on 30% data + Stage B: retrain best on 100%)
python3 ray_tune4_mixture_flash.py 2>&1 | tee mixture_flash_tune.log

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="

# Print final rankings
python3 select_best_mixture_flash.py