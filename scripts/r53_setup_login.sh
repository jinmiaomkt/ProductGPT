#!/bin/bash
# ============================================================================
# R53 one-off setup, run on the OMEGA LOGIN node (it has internet; the compute
# nodes are not assumed to).  The venv productgpt-llm5 is R49's and is not touched.
#   Qwen/Qwen3.8-27B ~56 GB, bf16, Apache-2.0, ungated -> models/Qwen3.8-27B
#   (includes its vision tower, which the text-only scorer never loads)
# The commit is pinned to the one the R53 plan names and written to REVISION.txt.
#
#     cd ~/ProductGPT/work_r48 && nohup bash -l scripts/r53_setup_login.sh \
#         > ~/ProductGPT/logs/r53_setup.log 2>&1 &
# ============================================================================
set -euo pipefail
module purge
module load python/3.11.4

ROOT=/storage/home/jinmiao/ProductGPT
REPO=Qwen/Qwen3.8-27B
SHA=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
M=$ROOT/models/Qwen3.8-27B
echo "R53 setup started $(date) on $(hostname)"
source "$ROOT/venvs/productgpt-llm5/bin/activate"
mkdir -p "$M"
python -c "
import sys
from huggingface_hub import snapshot_download
repo, sha, dest = sys.argv[1:4]
snapshot_download(repo, local_dir=dest, revision=sha)
open(dest + '/REVISION.txt', 'w').write(repo + ' ' + sha + '\n')
print(repo, 'at', sha, 'in', dest)
" "$REPO" "$SHA" "$M"
du -sh "$M"
sha256sum "$ROOT/models/Qwen3.6-27B/tokenizer.json" "$M/tokenizer.json"
echo "R53 setup finished $(date)"
