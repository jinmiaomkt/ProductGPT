# env.ps1 - set up the ProductGPT import path (Windows / PowerShell)
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
# USAGE (from the repo root, once per terminal session):
#     . .\env.ps1
#
# The leading dot matters: it runs the script in your current shell so the
# variable sticks around. Then run scripts normally, e.g.
#     python gen4_full_productgpt\train4_decoderonly_flash_feature_aws.py

$repo = $PSScriptRoot

$codeDirs = @(
    "gen0_encoder_decoder",
    "gen12_encdec_lto",
    "gen2_lp_duplet",
    "gen4_full_productgpt",
    "gen5_multistream",
    "baselines",
    "evaluation",
    "tuning",
    "crossval",
    "analysis",
    "vendor\transformer_xl"
)

$paths = $codeDirs | ForEach-Object { Join-Path $repo $_ }

# The repo root itself, so "import shared.layers" resolves.
$paths = @($repo) + $paths
$env:PYTHONPATH = ($paths -join ";") + $(if ($env:PYTHONPATH) { ";" + $env:PYTHONPATH } else { "" })

Write-Host "PYTHONPATH set for ProductGPT ($($codeDirs.Count) folders)."
Write-Host "Repo root: $repo"
