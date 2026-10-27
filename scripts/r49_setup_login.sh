#!/bin/bash
# ============================================================================
# R49 one-off setup, run on the OMEGA LOGIN node (it has internet; the compute
# nodes are not assumed to).
#   - venv productgpt-llm5: transformers 5.21.0, the first major version with the
#     Qwen3.5/3.6 architecture (qwen3_5).  The R48 venv productgpt-llm is not
#     touched; arm A (Qwen3-32B) runs in it, exactly as R48 did.
#   - weights, bf16, both Apache-2.0 and ungated (no Hugging Face token):
#       Qwen/Qwen3-32B   ~66 GB  -> models/Qwen3-32B
#       Qwen/Qwen3.6-27B ~56 GB  -> models/Qwen3.6-27B (includes its vision tower,
#                                   which the text-only scorer never loads)
#     The commit of each download is written to <model>/REVISION.txt.
#
#     cd ~/ProductGPT/work_r48 && nohup bash scripts/r49_setup_login.sh \
#         > ~/ProductGPT/logs/r49_setup.log 2>&1 &
# ============================================================================
set -euo pipefail
module purge
module load python/3.11.4

ROOT=/storage/home/jinmiao/ProductGPT
V=$ROOT/venvs/productgpt-llm5
echo "R49 setup started $(date) on $(hostname)"

[ -d "$V" ] || python -m venv "$V"
source "$V/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==5.21.0" accelerate numpy pandas openpyxl
python -c "import torch, transformers, accelerate; print('torch', torch.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__)"

for REPO in Qwen/Qwen3-32B Qwen/Qwen3.6-27B; do
    M=$ROOT/models/${REPO#Qwen/}
    mkdir -p "$M"
    python -c "
import sys
from huggingface_hub import HfApi, snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
sha = HfApi().model_info(repo).sha
snapshot_download(repo, local_dir=dest, revision=sha)
open(dest + '/REVISION.txt', 'w').write(repo + ' ' + sha + '\n')
print(repo, 'at', sha, 'in', dest)
" "$REPO" "$M"
    du -sh "$M"
done
sha256sum $ROOT/models/Qwen3-8B/tokenizer.json $ROOT/models/Qwen3-32B/tokenizer.json
echo "R49 setup finished $(date)"
