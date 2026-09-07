"""
paths.py - single source of truth for where ProductGPT data and outputs live.

WHY THIS EXISTS
----------------
CLAUDE.md hard rule: "Code must read the data location ONLY from
PRODUCTGPT_DATA env var; never hard-code either path (the OneDrive path
contains spaces and is machine-specific)."

Before this file, ~61 scripts hard-coded /home/ec2-user/data/... . That
worked only on the original AWS EC2 box. This module resolves the data
directory from the PRODUCTGPT_DATA environment variable instead, so the
exact same code runs on this Windows laptop and on SMU HPCC unchanged -
only the environment variable differs between machines.

USAGE
-----
    from paths import data_file, product_feature_xlsx

    train_path = data_file("clean_list_int_wide4_simple6_FeatureBasedTrain.json")
    feat_path  = product_feature_xlsx()

Set PRODUCTGPT_DATA once per machine:
    Windows (persists across sessions): setx PRODUCTGPT_DATA "C:\\Users\\jinmiao\\ResearchData\\productgpt"
    HPCC (in the .pbs job script):      export PRODUCTGPT_DATA=/storage/home/jinmiao/ProductGPT/data

This module never reads the CONTENTS of any data file - only checks that
paths exist. Per CLAUDE.md, data contents must never be read, printed, or
copied into code, comments, or responses.
"""
from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────── repo root ───────────────────────────
# This file sits at the repo root (C:\Users\jinmiao\ProductGPT\ProductGPT),
# so its own location is a reliable anchor regardless of the caller's cwd.
REPO_ROOT = Path(__file__).resolve().parent


class ProductGPTPathError(RuntimeError):
    """Raised when PRODUCTGPT_DATA is unset, or an expected file/dir is missing."""


# ─────────────────────────── data directory ───────────────────────────
def data_dir() -> Path:
    """
    The working copy of the training data, resolved ONLY from the
    PRODUCTGPT_DATA environment variable - never hard-coded here.

    Raises ProductGPTPathError with setup instructions if the variable
    is unset or does not point at a real directory, rather than letting
    callers hit a confusing FileNotFoundError three layers into training.
    """
    raw = os.environ.get("PRODUCTGPT_DATA")
    if not raw:
        raise ProductGPTPathError(
            "PRODUCTGPT_DATA is not set.\n"
            "  Windows (persists across sessions):\n"
            '    setx PRODUCTGPT_DATA "C:\\Users\\jinmiao\\ResearchData\\productgpt"\n'
            "    (then open a NEW terminal - setx does not affect the current one)\n"
            "  Linux / SMU HPCC (in your .pbs script or shell profile):\n"
            "    export PRODUCTGPT_DATA=/storage/home/jinmiao/ProductGPT/data"
        )
    p = Path(raw)
    if not p.is_dir():
        raise ProductGPTPathError(
            f"PRODUCTGPT_DATA is set to {raw!r} but that is not an existing directory. "
            "Check for typos, or that the drive/mount is available."
        )
    return p


def data_file(name: str) -> Path:
    """
    Resolve a single file inside PRODUCTGPT_DATA by name.

    Raises ProductGPTPathError naming the exact expected path if the file
    is missing, so a missing-data problem is caught at startup, not deep
    inside a training loop.
    """
    p = data_dir() / name
    if not p.exists():
        raise ProductGPTPathError(
            f"Expected data file not found: {p}\n"
            f"(PRODUCTGPT_DATA={data_dir()})\n"
            "Copy it from the OneDrive master archive's Data\\ folder if it's missing."
        )
    return p


# ───────────────── named accessors for files configs expect ─────────────────
# One function per file so a typo'd filename is a one-line fix, not a
# repo-wide grep. Add to this list as new data revisions appear.

def wide4_simple6_full() -> Path:
    """Full 30-campaign set. Used as test/infer input by config2 and config4."""
    return data_file("clean_list_int_wide4_simple6.json")


def wide4_simple6_train_feature() -> Path:
    """Feature-model train split. Used by config2 and config4."""
    return data_file("clean_list_int_wide4_simple6_FeatureBasedTrain.json")


def wide4_simple6_train_index() -> Path:
    """Index-model train split (gen4 index variant)."""
    return data_file("clean_list_int_wide4_simple6_IndexBasedTrain.json")


def wide4_simple4_train_index() -> Path:
    """Older preprocessing revision. Used by config0."""
    return data_file("clean_list_int_wide4_simple4_IndexBasedTrain.json")


def wide12_simple3() -> Path:
    """Used by config12."""
    return data_file("clean_list_int_wide12_simple3.json")


def wide12_full() -> Path:
    """
    Used by config12git. Different filename than config12's
    wide12_simple3() - the two gen-12 configs disagree on which
    preprocessing revision they train on; this is a pre-existing split,
    not something introduced here.
    """
    return data_file("clean_list_int_wide12.json")


def product_feature_xlsx() -> Path:
    """
    The 34-column product attribute table every `feature` model needs.
    Passed to shared.features.load_feature_tensor(product_feature_xlsx()).
    """
    return data_file("SelectedFigureWeaponEmbeddingIndex.xlsx")


# ─────────────────────────── output directory ───────────────────────────
# CLAUDE.md hard rule: "Never write outputs (checkpoints, results, logs)
# into the OneDrive folder or the data directory." Outputs default to the
# repo's own checkpoints/ and results/ folders, which .gitignore excludes.
# Override with PRODUCTGPT_OUTPUT if you want them elsewhere (e.g. a
# scratch disk on HPCC) - this one is optional, unlike PRODUCTGPT_DATA.
def output_dir() -> Path:
    raw = os.environ.get("PRODUCTGPT_OUTPUT")
    p = Path(raw) if raw else (REPO_ROOT / "checkpoints")
    p.mkdir(parents=True, exist_ok=True)
    return p
