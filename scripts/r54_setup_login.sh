#!/bin/bash
# ============================================================================
# R54 one-off setup, run on the OMEGA LOGIN node (it has internet; the compute
# nodes are not assumed to).
#   - venv productgpt-llm70: as productgpt-llm5 (torch 2.6.0, transformers 5.21.0)
#     plus peft and bitsandbytes, for Centaur and its base.  productgpt-llm5 is not
#     touched; arms A and B run in it, as R49 arm B and R53 did.
#   - weights at the commits the R54 plan names, each written to REVISION.txt:
#       meta-models/Muse-Glimmer-30B                59.6 GB  bf16
#       google/gemma-4-31B-it                       62.6 GB  bf16
#       unsloth/Meta-Llama-3.1-70B-bnb-4bit         39.5 GB  4-bit, Centaur's base
#       marcelbinz/Llama-3.1-Centaur-70B-adapter     0.4 GB  LoRA
#
#     cd ~/ProductGPT/work_r54 && setsid nohup bash -l scripts/r54_setup_login.sh \
#         > ~/ProductGPT/logs/r54_setup.log 2>&1 < /dev/null &
# ============================================================================
set -euo pipefail
module purge
module load python/3.11.4

ROOT=/storage/home/jinmiao/ProductGPT
V=$ROOT/venvs/productgpt-llm70
echo "R54 setup started $(date) on $(hostname)"

[ -d "$V" ] || python -m venv "$V"
source "$V/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers==5.21.0" accelerate numpy pandas openpyxl peft bitsandbytes
python -c "import torch, transformers, accelerate, peft, bitsandbytes as b; print('torch', torch.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__, 'peft', peft.__version__, 'bitsandbytes', b.__version__)"

fetch() {
    local repo=$1 sha=$2 M=$ROOT/models/${1#*/}
    mkdir -p "$M"
    python -c "
import sys
from huggingface_hub import snapshot_download
repo, sha, dest = sys.argv[1:4]
snapshot_download(repo, local_dir=dest, revision=sha)
open(dest + '/REVISION.txt', 'w').write(repo + ' ' + sha + '\n')
print(repo, 'at', sha, 'in', dest)
" "$repo" "$sha" "$M"
    du -sh "$M"
}
fetch marcelbinz/Llama-3.1-Centaur-70B-adapter 159600db8be99dc183c289923148dfd96cbd8e07
fetch unsloth/Meta-Llama-3.1-70B-bnb-4bit a009b8db2439814febe725486a5ed388f12a8744
fetch meta-models/Muse-Glimmer-30B cec94fcfd6512f2cfe708138c32533b346ea4f69
fetch google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475
echo "R54 setup finished $(date)"
