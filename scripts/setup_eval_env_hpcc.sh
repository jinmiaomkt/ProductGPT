#!/bin/bash
set -euo pipefail

module load python/3.11.4
module load cuda/12.5

mkdir -p /storage/home/jinmiao/ProductGPT/venvs
python -m venv /storage/home/jinmiao/ProductGPT/venvs/productgpt-eval
source /storage/home/jinmiao/ProductGPT/venvs/productgpt-eval/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements_eval_hpcc.txt

python -c "import torch; print('torch:', torch.__version__); print('built for CUDA:', torch.version.cuda)"
echo "Environment ready: /storage/home/jinmiao/ProductGPT/venvs/productgpt-eval"
