#!/usr/bin/env bash
# env.sh - set up the ProductGPT import path (Linux / SMU HPCC / macOS)
#
# WHY THIS EXISTS
# ---------------
# The scripts in this repo import each other by plain module name, e.g.
#     from config4 import get_config
#     from model4_decoderonly_feature_performer import build_transformer
#
# That worked when every file sat in one flat folder. Now that files are
# grouped into gen*/ and tooling folders, Python needs to be told where to
# look. Putting every code folder on PYTHONPATH restores exactly the same
# import namespace as before, so no Python file had to be edited.
#
# USAGE (from the repo root, once per shell session):
#     source ./env.sh
#
# Then run scripts normally, e.g.
#     python gen4_full_productgpt/train4_decoderonly_flash_feature_aws.py
#
# In a PBS job script, source it after activating the venv:
#     source "$PBS_O_WORKDIR/env.sh"

_productgpt_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_productgpt_dirs=(
    "gen0_encoder_decoder"
    "gen12_encdec_lto"
    "gen2_lp_duplet"
    "gen4_full_productgpt"
    "gen5_multistream"
    "baselines"
    "evaluation"
    "tuning"
    "crossval"
    "analysis"
    "vendor/transformer_xl"
)

for _d in "${_productgpt_dirs[@]}"; do
    PYTHONPATH="${_productgpt_root}/${_d}${PYTHONPATH:+:${PYTHONPATH}}"
done

# The repo root itself, so "import shared.layers" resolves.
PYTHONPATH="${_productgpt_root}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONPATH

echo "PYTHONPATH set for ProductGPT (${#_productgpt_dirs[@]} folders)."
echo "Repo root: ${_productgpt_root}"

# This script only sets PYTHONPATH. The data location comes from
# PRODUCTGPT_DATA, which the PBS job scripts export themselves but an
# interactive shell does not. Say so here rather than letting it surface as a
# traceback from paths.py three calls deep.
if [ -n "${PRODUCTGPT_DATA:-}" ]; then
    echo "PRODUCTGPT_DATA: ${PRODUCTGPT_DATA}"
else
    echo "WARNING: PRODUCTGPT_DATA is not set - get_config() will fail."
    echo "  On SMU HPCC:  export PRODUCTGPT_DATA=/storage/home/\$USER/ProductGPT/data"
    echo "  Add that line to ~/.bashrc to make it permanent."
fi

unset _d _productgpt_dirs _productgpt_root
