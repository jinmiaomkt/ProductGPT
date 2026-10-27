#!/bin/bash
# ============================================================================
# R48 one-off setup, run on the OMEGA LOGIN node (it has internet; the compute
# nodes are not assumed to).  Creates a separate venv so the eval venv that
# other jobs use is never modified, and downloads Qwen3-8B (Apache-2.0,
# ungated: no Hugging Face token is needed).
#
#     cd ~/ProductGPT/work_r48 && bash scripts/r48_setup_login.sh
# ============================================================================
set -euo pipefail
module purge
module load python/3.11.4

ROOT=/storage/home/jinmiao/ProductGPT
V=$ROOT/venvs/productgpt-llm
M=$ROOT/models/Qwen3-8B

[ -d "$V" ] || python -m venv "$V"
source "$V/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# transformers pinned to the version the laptop tests passed with
pip install "transformers==4.57.6" accelerate numpy pandas openpyxl
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

mkdir -p "$M"
python -c "from huggingface_hub import snapshot_download as d; print('model in', d('Qwen/Qwen3-8B', local_dir='$M'))"
du -sh "$M"
