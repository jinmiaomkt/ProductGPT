@echo off
REM env.bat - set up the ProductGPT import path (Windows / cmd.exe)
REM
REM This is the cmd.exe counterpart of env.ps1. Use whichever matches your
REM shell -- you can tell them apart by the prompt:
REM     C:\...>            cmd.exe        -> use  env.bat
REM     PS C:\...>         PowerShell     -> use  . .\env.ps1
REM
REM WHY THIS EXISTS
REM ---------------
REM Scripts import each other by plain module name (from config4 import ...).
REM That worked when every file sat in one flat folder. Now that files are
REM grouped into gen*/ and tooling folders, Python needs to be told where to
REM look. Putting every code folder on PYTHONPATH restores the same import
REM namespace as before, so no Python file had to be edited.
REM
REM USAGE (from the repo root, once per terminal session):
REM     env.bat
REM
REM Then run scripts normally, e.g.
REM     python gen4_full_productgpt\train4_decoderonly_flash_feature_aws.py
REM
REM NOTE: this only sets PYTHONPATH. The data location comes from
REM PRODUCTGPT_DATA, which should already be set permanently via:
REM     setx PRODUCTGPT_DATA "C:\Users\jinmiao\ResearchData\productgpt"
REM (setx affects NEW terminals only, not the one you run it in.)

set "REPO=%~dp0"
REM strip the trailing backslash that %~dp0 always includes
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"

REM The repo root itself comes first, so "import paths" and
REM "import shared.layers" resolve.
set "PYTHONPATH=%REPO%"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\gen0_encoder_decoder"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\gen12_encdec_lto"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\gen2_lp_duplet"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\gen4_full_productgpt"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\gen5_multistream"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\baselines"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\evaluation"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\tuning"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\crossval"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\analysis"
set "PYTHONPATH=%PYTHONPATH%;%REPO%\vendor\transformer_xl"

echo PYTHONPATH set for ProductGPT (11 folders + repo root).
echo Repo root: %REPO%
if defined PRODUCTGPT_DATA (
    echo PRODUCTGPT_DATA: %PRODUCTGPT_DATA%
) else (
    echo WARNING: PRODUCTGPT_DATA is not set - get_config^(^) will fail.
    echo   Fix with: setx PRODUCTGPT_DATA "C:\Users\jinmiao\ResearchData\productgpt"
    echo   then open a NEW terminal.
)
