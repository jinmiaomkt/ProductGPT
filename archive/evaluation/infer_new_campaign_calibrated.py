#!/usr/bin/env python3
"""
generate_campaign28_calibrated.py

Modified from the original inference script to support probability calibration.

Changes vs original:
  1. New Calibrator class hierarchy (TemperatureCalibrator, PlattCalibrator, VectorCalibrator)
  2. load_calibrator() factory function
  3. sample_decision_from_logits() now accepts an optional calibrator
  4. Both generate_campaign28_step1 and step2 accept and pass through the calibrator
  5. New CLI args: --calibrator_ckpt, --calibrator_type
  6. Diagnostic output on calibrator load

Usage examples:

  # First, inspect your calibrator to find out its type:
  python3 inspect_calibrator.py --ckpt /path/to/calibrator.pt

  # Then run with calibration:
  python3 generate_campaign28_calibrated.py \\
    --step 2 \\
    --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \\
    --ckpt /tmp/FullProductGPT_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0.pt \\
    --calibrator_ckpt /tmp/calibrator_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0.pt \\
    --calibrator_type temperature \\
    --feat_xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \\
    --out /home/ec2-user/outputs/campaign28_calibrated.jsonl \\
    --lto28 22 30 45 50 \\
    --seed_base 42 --repeat 5 --first
"""
import argparse
import gzip
from html import parser
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TextIO

from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config4 import get_config
from model4_decoderonly_feature_performer import build_transformer

import copy

AI_RATE = 15


# =========================================================================
# CALIBRATOR CLASSES  (NEW)
# =========================================================================

class BaseCalibrator(ABC):
    """Abstract base class for probability calibrators."""

    @abstractmethod
    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply calibration to raw logits.

        Args:
            logits: shape (num_classes,) — raw logits for the decision classes.

        Returns:
            calibrated_logits: same shape — logits after calibration transform.
        """
        ...

    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convenience: calibrate logits then softmax.
        Override this if your calibrator works directly on probabilities.
        """
        return F.softmax(self.calibrate_logits(logits), dim=-1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class TemperatureCalibrator(BaseCalibrator):
    """
    Temperature scaling: calibrated_logits = logits / T

    The most common calibration method (Guo et al., 2017).
    A single learned scalar T > 0.
    """

    def __init__(self, temperature: float):
        self.temperature = float(temperature)
        assert self.temperature > 0, f"Temperature must be positive, got {self.temperature}"

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def __repr__(self):
        return f"TemperatureCalibrator(T={self.temperature:.6f})"


class PlattCalibrator(BaseCalibrator):
    """
    Platt scaling (per-class): calibrated_logits = weight * logits + bias

    weight: shape (num_classes,) or scalar
    bias:   shape (num_classes,) or scalar
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.weight = weight.float().detach()
        self.bias = bias.float().detach()

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.weight.to(logits.device) * logits + self.bias.to(logits.device)

    def __repr__(self):
        return f"PlattCalibrator(weight={self.weight}, bias={self.bias})"


class VectorCalibrator(BaseCalibrator):
    """
    Vector scaling: calibrated_logits = diag(scale) @ logits + shift

    scale: shape (num_classes,)
    shift: shape (num_classes,)
    """

    def __init__(self, scale: torch.Tensor, shift: torch.Tensor):
        self.scale = scale.float().detach()
        self.shift = shift.float().detach()

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale.to(logits.device) * logits + self.shift.to(logits.device)

    def __repr__(self):
        return f"VectorCalibrator(scale={self.scale}, shift={self.shift})"


class IdentityCalibrator(BaseCalibrator):
    """No-op calibrator (pass-through). Used when no calibrator is loaded."""

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def __repr__(self):
        return "IdentityCalibrator(no calibration)"


# =========================================================================
# CALIBRATOR LOADING  (NEW)
# =========================================================================

def load_calibrator(
    ckpt_path: str,
    calibrator_type: str = "auto",
    device: torch.device = torch.device("cpu"),
) -> BaseCalibrator:
    """
    Load a calibrator from a checkpoint file.

    Args:
        ckpt_path:       Path to calibrator .pt file.
        calibrator_type: One of "temperature", "platt", "vector", "auto".
                         "auto" tries to detect the type from checkpoint keys.
        device:          Device (used for tensor placement).

    Returns:
        A BaseCalibrator instance.
    """
    state = torch.load(ckpt_path, map_location=device)

    # ---- Unwrap if nested in 'model_state_dict' ----
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict):
        sd = state
    elif hasattr(state, "state_dict"):
        # It's an nn.Module saved directly (torch.save(module, path))
        sd = state.state_dict()
    else:
        raise ValueError(f"Cannot parse calibrator checkpoint: top-level type is {type(state)}")

    # ---- Normalize keys to lowercase for matching ----
    sd_lower = {k.lower(): v for k, v in sd.items()}

    # ---- Auto-detect type ----
    if calibrator_type == "auto":
        if "temperature" in sd_lower or "temp" in sd_lower:
            calibrator_type = "temperature"
        elif "weight" in sd_lower and "bias" in sd_lower:
            calibrator_type = "platt"
        elif "scale" in sd_lower and ("shift" in sd_lower or "bias" in sd_lower):
            calibrator_type = "vector"
        else:
            # Last resort: check if there's a single scalar parameter
            scalars = {k: v for k, v in sd.items() if torch.is_tensor(v) and v.numel() == 1}
            if len(scalars) == 1:
                calibrator_type = "temperature"
                key = list(scalars.keys())[0]
                sd_lower["temperature"] = scalars[key]
            else:
                raise ValueError(
                    f"Cannot auto-detect calibrator type from keys: {list(sd.keys())}. "
                    f"Run inspect_calibrator.py and specify --calibrator_type explicitly."
                )

    # ---- Construct calibrator ----
    if calibrator_type == "temperature":
        # Look for the temperature value in various common key names
        t_val = None
        for key in ["temperature", "temp", "t"]:
            if key in sd_lower:
                v = sd_lower[key]
                if torch.is_tensor(v):
                    t_val = v.item()
                else:
                    t_val = float(v)
                break

        # If not found by name, check if there's exactly one scalar parameter
        if t_val is None:
            scalars = {k: v for k, v in sd.items() if torch.is_tensor(v) and v.numel() == 1}
            if len(scalars) == 1:
                t_val = list(scalars.values())[0].item()

        if t_val is None:
            raise ValueError(
                f"Temperature calibrator requested but no temperature value found. "
                f"Checkpoint keys: {list(sd.keys())}"
            )

        cal = TemperatureCalibrator(temperature=t_val)

    elif calibrator_type == "platt":
        weight = sd_lower.get("weight") or sd_lower.get("linear.weight") or sd_lower.get("a")
        bias = sd_lower.get("bias") or sd_lower.get("linear.bias") or sd_lower.get("b")
        if weight is None or bias is None:
            raise ValueError(
                f"Platt calibrator requested but weight/bias not found. "
                f"Checkpoint keys: {list(sd.keys())}"
            )
        cal = PlattCalibrator(weight=weight.squeeze(), bias=bias.squeeze())

    elif calibrator_type == "vector":
        scale = sd_lower.get("scale") or sd_lower.get("weight")
        shift = sd_lower.get("shift") or sd_lower.get("bias")
        if scale is None or shift is None:
            raise ValueError(
                f"Vector calibrator requested but scale/shift not found. "
                f"Checkpoint keys: {list(sd.keys())}"
            )
        cal = VectorCalibrator(scale=scale.squeeze(), shift=shift.squeeze())

    else:
        raise ValueError(f"Unknown calibrator_type: {calibrator_type}")

    print(f"[CALIBRATOR] Loaded: {cal}")
    return cal


# =========================================================================
# ORIGINAL CODE BELOW — with calibrator threaded through
# =========================================================================

@dataclass
class TraceCfg:
    enabled: bool = False
    max_steps: int = 200
    fout: Optional[TextIO] = None

def trim_to_ctx(seq: List[int], max_len: int, ai_rate: int = AI_RATE) -> List[int]:
    if max_len <= 0:
        return seq
    if len(seq) <= max_len:
        return seq
    tail = seq[-max_len:]
    rem = len(tail) % ai_rate
    if rem != 0:
        tail = tail[rem:]
    return tail

def _json_fallback(o):
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return str(o)

def _trace_emit(rec: dict, do_print: bool = True, fout: Optional[TextIO] = None):
    line = json.dumps(rec, ensure_ascii=False, default=_json_fallback)
    if do_print:
        print(line, flush=True)
    if fout is not None:
        fout.write(line + "\n")
        fout.flush()


# ----------------------------
# Constants
# ----------------------------
SOS_DEC_ID = 10
EOS_PROD_ID_LOCAL = 57
SOS_PROD_ID_LOCAL = 58

BUY1_DECS  = {1, 3, 5, 7}
BUY10_DECS = {2, 4, 6, 8}
NOTBUY_DEC = 9

SPECIAL_PROD_TOKS = {EOS_PROD_ID_LOCAL, SOS_PROD_ID_LOCAL}

@dataclass
class GenValidationCfg:
    eos_prod_id: int = EOS_PROD_ID_LOCAL
    sos_prod_id: int = SOS_PROD_ID_LOCAL
    require_lto4_match: bool = True
    strict_buy10_full: bool = True
    buy10_missing_is_error: bool = True
    strict_buy1_tail_zero: bool = True


def iter_blocks_from_seq(seq_tokens: List[int], ai_rate: int = AI_RATE):
    if len(seq_tokens) % ai_rate != 0:
        raise ValueError(f"seq length {len(seq_tokens)} not divisible by {ai_rate}")
    n_blocks = len(seq_tokens) // ai_rate
    for b in range(n_blocks):
        block = seq_tokens[b*ai_rate:(b+1)*ai_rate]
        lto4 = block[0:4]
        out10 = block[4:14]
        dec1 = int(block[14])
        yield b, lto4, out10, dec1


def _is_valid_real_outcome(tok: int, cfg: GenValidationCfg) -> bool:
    if tok == 0:
        return False
    if tok in (cfg.eos_prod_id, cfg.sos_prod_id):
        return False
    return True


def validate_seq_campaign28(
    seq_campaign28: List[int],
    expected_lto4: Optional[List[int]],
    cfg: GenValidationCfg,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if len(seq_campaign28) == 0:
        errors.append("seq_campaign28 is empty.")
        return {"ok": False, "n_blocks": 0, "errors": errors, "warnings": warnings}

    if len(seq_campaign28) % AI_RATE != 0:
        errors.append(f"seq_campaign28 length {len(seq_campaign28)} not divisible by {AI_RATE}.")
        return {"ok": False, "n_blocks": 0, "errors": errors, "warnings": warnings}

    n_blocks = len(seq_campaign28) // AI_RATE

    for b, lto4, out10, dec1 in iter_blocks_from_seq(seq_campaign28, AI_RATE):
        if cfg.require_lto4_match and expected_lto4 is not None:
            if list(lto4) != list(expected_lto4):
                errors.append(f"b={b}: LTO4 mismatch. got={lto4} expected={expected_lto4}")

        if cfg.sos_prod_id in out10:
            errors.append(f"b={b}: SOS_PROD_ID={cfg.sos_prod_id} appears in OUT10: {out10}")

        if dec1 == NOTBUY_DEC:
            expected = [cfg.eos_prod_id] + [0]*9
            if list(out10) != expected:
                errors.append(f"b={b}: DEC1==9 but OUT10 invalid. got={out10} expected={expected}")
            if any(tok == cfg.eos_prod_id for tok in out10[1:]):
                errors.append(f"b={b}: DEC1==9 but EOS appears after OUT10[0]: {out10}")
        else:
            if cfg.eos_prod_id in out10:
                errors.append(f"b={b}: DEC1={dec1} but EOS_PROD_ID appears in OUT10: {out10}")

        if dec1 in BUY1_DECS:
            if not _is_valid_real_outcome(int(out10[0]), cfg):
                errors.append(f"b={b}: buy-1 DEC1={dec1} but OUT10[0] not valid: {out10[0]}")
            tail = out10[1:]
            if cfg.strict_buy1_tail_zero and any(int(x) != 0 for x in tail):
                errors.append(f"b={b}: buy-1 DEC1={dec1} but OUT10[1:] nonzero: out10={out10}")
            if int(out10[0]) in SPECIAL_PROD_TOKS:
                errors.append(f"b={b}: buy-1 DEC1={dec1} but OUT10[0] is special: {out10[0]}")

        if dec1 in BUY10_DECS:
            realized = [int(x) for x in out10 if _is_valid_real_outcome(int(x), cfg)]
            num_realized = len(realized)
            if cfg.strict_buy10_full:
                if num_realized != 10:
                    msg = f"b={b}: buy-10 DEC1={dec1} realized={num_realized}/10. out10={out10}"
                    if cfg.buy10_missing_is_error:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
            else:
                if num_realized < 10:
                    warnings.append(f"b={b}: buy-10 DEC1={dec1} only {num_realized}/10. out10={out10}")
            if any(int(x) in SPECIAL_PROD_TOKS for x in out10):
                errors.append(f"b={b}: buy-10 DEC1={dec1} special token(s) in OUT10: {out10}")

        if dec1 not in BUY1_DECS and dec1 not in BUY10_DECS and dec1 != NOTBUY_DEC:
            errors.append(f"b={b}: unknown DEC1={dec1}")

    ok = (len(errors) == 0)
    print("\n=== GENERATED SEQ VALIDATION ===")
    print(f"Blocks checked: {n_blocks}")
    print(f"OK: {ok}")
    print(f"Errors: {len(errors)}  Warnings: {len(warnings)}")
    if errors:
        print("\n[ERRORS]")
        for e in errors[:50]:
            print("  " + e)
    if warnings:
        print("\n[WARNINGS]")
        for w in warnings[:50]:
            print("  " + w)
    return {"ok": ok, "n_blocks": n_blocks, "errors": errors, "warnings": warnings}


def extract_block_decs(history_tokens):
    n_blocks = len(history_tokens) // AI_RATE
    return [int(history_tokens[i*AI_RATE + 14]) for i in range(n_blocks)]

def maybe_append_missing_terminal_block(
    history_tokens: list[int],
    decision_tokens: list[int],
    terminal_prod_tok: int = EOS_PROD_ID_LOCAL,
) -> tuple[list[int], bool]:
    if len(history_tokens) % AI_RATE != 0:
        raise ValueError("history_tokens length must be divisible by 15")
    if not decision_tokens:
        return history_tokens, False

    block_decs = extract_block_decs(history_tokens)
    if block_decs and block_decs[0] == SOS_DEC_ID:
        block_decs_no_sos = block_decs[1:]
    else:
        block_decs_no_sos = block_decs

    if (len(decision_tokens) == len(block_decs_no_sos) + 1 and
        decision_tokens[:len(block_decs_no_sos)] == block_decs_no_sos):
        last_lto4 = history_tokens[-AI_RATE : -AI_RATE + 4]
        new_block = list(last_lto4) + [int(terminal_prod_tok)] + [0]*9 + [int(decision_tokens[-1])]
        return history_tokens + new_block, True

    if decision_tokens[-1] == 9 and (not block_decs_no_sos or block_decs_no_sos[-1] != 9):
        last_lto4 = history_tokens[-AI_RATE : -AI_RATE + 4]
        new_block = list(last_lto4) + [int(terminal_prod_tok)] + [0]*9 + [9]
        return history_tokens + new_block, True

    return history_tokens, False

DECISION_IDS = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)

DECISION_META = {
    1: ("Buy 1 Regular",   "regular",  1),
    2: ("Buy 10 Regular",  "regular", 10),
    3: ("Buy 1 Figure-A",  "figure_a", 1),
    4: ("Buy 10 Figure-A", "figure_a",10),
    5: ("Buy 1 Figure-B",  "figure_b", 1),
    6: ("Buy 10 Figure-B", "figure_b",10),
    7: ("Buy 1 Weapon",    "weapon",   1),
    8: ("Buy 10 Weapon",   "weapon",  10),
    9: ("Not Buy",         "none",     0),
}

TOK_3STAR_WEAPON = 13
TOK_4STAR_FIGURE = 14
TOK_5STAR_REG_FIGURE = 15
TOK_4STAR_WEAPON = 16
TOK_5STAR_REG_WEAPON = 17
PAD_ID = 0
EOS_PROD_ID = 57
SOS_PROD_ID = 58

DEC_BUY_1_REG = 1
DEC_BUY_10_REG = 2
DEC_BUY_1_FIG_A = 3
DEC_BUY_10_FIG_A = 4
DEC_BUY_1_FIG_B = 5
DEC_BUY_10_FIG_B = 6
DEC_BUY_1_WEP = 7
DEC_BUY_10_WEP = 8
DEC_NOT_BUY = 9


# ----------------------------
# IO helpers
# ----------------------------
def _iter_rows(path: str) -> Iterable[Dict[str, Any]]:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        first = f.read(1)
        if first == "":
            return
        f.seek(0)
        if first == "[":
            data = json.load(f)
            for row in data:
                yield row
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def parse_int_sequence(val: Any, na_to: int = 0) -> List[int]:
    NA_STRINGS = {"NA", "Na", "NaN", "nan", "None", "null", ""}

    def tok_to_int(t: Any) -> int:
        if t is None:
            return na_to
        if isinstance(t, (int, np.integer)):
            return int(t)
        if isinstance(t, float):
            if np.isnan(t):
                return na_to
            return int(t)
        if isinstance(t, str):
            s = t.strip()
            if s in NA_STRINGS:
                return na_to
            try:
                return int(s)
            except ValueError:
                try:
                    return int(float(s))
                except Exception:
                    return na_to
        return na_to

    tokens: List[int] = []
    if isinstance(val, list):
        if len(val) == 1 and isinstance(val[0], str):
            parts = val[0].strip().split()
            return [tok_to_int(p) for p in parts]
        for x in val:
            if isinstance(x, str):
                parts = x.strip().split()
                for p in parts:
                    tokens.append(tok_to_int(p))
            else:
                tokens.append(tok_to_int(x))
        return tokens
    if isinstance(val, str):
        parts = val.strip().split()
        return [tok_to_int(p) for p in parts]
    raise TypeError(f"Unsupported sequence type: {type(val)}")


def extract_last_outcomes_from_history(history_tokens: List[int], ai_rate: int = AI_RATE) -> List[int]:
    assert len(history_tokens) >= ai_rate and len(history_tokens) % ai_rate == 0
    last_block = history_tokens[-ai_rate:]
    return last_block[4:14]

def iter_blocks(history_tokens: List[int], ai_rate: int = AI_RATE):
    assert len(history_tokens) % ai_rate == 0
    n_blocks = len(history_tokens) // ai_rate
    for b in range(n_blocks):
        block = history_tokens[b*ai_rate:(b+1)*ai_rate]
        lto4 = block[0:4]
        out10 = block[4:14]
        dec1 = int(block[14])
        yield b, lto4, out10, dec1

def format_triplet(lto4: List[int], out10: List[int], dec1: int) -> str:
    return f"[{' '.join(map(str, lto4))}] [{' '.join(map(str, out10))}] [{dec1}]"

def extract_dec1_sequence_from_aggregate(history_tokens: List[int]) -> List[int]:
    return [dec1 for _, _, _, dec1 in iter_blocks(history_tokens)]


# ----------------------------
# Model sampling (decision) — MODIFIED FOR CALIBRATION
# ----------------------------
@torch.no_grad()
def sample_decision_from_logits(
    logits_last_pos: torch.Tensor,
    decision_ids: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
    calibrator: Optional[BaseCalibrator] = None,    # <-- NEW PARAMETER
) -> int:
    """
    Sample a decision from the model's logits at the decision position.

    The pipeline is:
        1. Extract logits for the 9 decision token IDs
        2. Apply user temperature scaling (--temperature CLI arg)
        3. Apply calibrator transform (learned calibration)
        4. Softmax -> sample or argmax

    Note: the user --temperature and the calibrator are SEPARATE concerns.
    - User temperature controls generation diversity (like in any LLM sampling).
    - Calibrator corrects the model's probability estimates to be better calibrated.

    If you want ONLY calibration (no extra temperature manipulation), set --temperature 1.0.
    """
    ids = decision_ids.to(logits_last_pos.device)
    dec_logits = logits_last_pos[ids]  # (9,)

    # Step 1: User temperature (generation diversity control)
    if temperature != 1.0:
        dec_logits = dec_logits / max(temperature, 1e-8)

    # Step 2: Calibration transform (learned probability correction)
    if calibrator is not None:
        dec_logits = calibrator.calibrate_logits(dec_logits)

    probs = F.softmax(dec_logits, dim=-1)

    if greedy:
        k = torch.argmax(probs).item()
    else:
        k = torch.multinomial(probs, 1).item()

    return int(ids[k].item())


# ----------------------------
# Step 1: fixed outcomes — MODIFIED FOR CALIBRATION
# ----------------------------
@torch.no_grad()
def generate_campaign28_step1_fixed_outcomes(
    model,
    history_tokens: List[int],
    lto28_tokens: List[int],
    fixed_outcomes_after_step0: List[int],
    device: torch.device,
    init_prev_dec: Optional[int] = None,
    max_seq_len_ckpt: int = 1024,
    max_steps28: int = 500,
    stop_decision: int = 9,
    temperature: float = 1.0,
    greedy: bool = False,
    trace: Optional[TraceCfg] = None,
    calibrator: Optional[BaseCalibrator] = None,    # <-- NEW
) -> Dict[str, Any]:
    assert len(lto28_tokens) == 4
    assert len(fixed_outcomes_after_step0) == 10
    assert len(history_tokens) >= AI_RATE and len(history_tokens) % AI_RATE == 0

    model.eval()
    outcomes_step0 = extract_last_outcomes_from_history(history_tokens)

    if init_prev_dec is None:
        init_prev_dec = int(history_tokens[-1])

    prev_dec = int(init_prev_dec)
    seq_full = list(history_tokens)
    seq_c28: List[int] = []
    decisions28: List[int] = []
    stopped = False
    stop_step: Optional[int] = None

    for t in range(max_steps28):
        out10 = outcomes_step0 if t == 0 else fixed_outcomes_after_step0
        block = list(lto28_tokens) + list(out10) + [prev_dec]

        if trace is not None and trace.enabled:
            print(format_triplet(lto28_tokens, out10, prev_dec), flush=True)

        seq_full.extend(block)
        seq_c28.extend(block)
        seq_full = trim_to_ctx(seq_full, max_seq_len_ckpt, AI_RATE)

        x = torch.tensor(seq_full, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        logits_at_decpos = logits[0, -1, :]

        dec = sample_decision_from_logits(
            logits_at_decpos, DECISION_IDS,
            temperature=temperature, greedy=greedy,
            calibrator=calibrator,                   # <-- PASSED THROUGH
        )
        decisions28.append(dec)

        if dec == stop_decision:
            stopped = True
            stop_step = t
            break
        prev_dec = dec

    return {
        "seq_campaign28": seq_c28,
        "seq_full": seq_full,
        "decisions28": decisions28,
        "stopped": stopped,
        "stop_step": stop_step,
    }


# ----------------------------
# Step 2: gacha simulation (banners, pity, etc.)
# ----------------------------
def decision_to_banner_and_pulls(dec: int) -> Tuple[str, int]:
    if dec == DEC_BUY_1_REG:    return "regular", 1
    if dec == DEC_BUY_10_REG:   return "regular", 10
    if dec == DEC_BUY_1_FIG_A:  return "figure_a", 1
    if dec == DEC_BUY_10_FIG_A: return "figure_a", 10
    if dec == DEC_BUY_1_FIG_B:  return "figure_b", 1
    if dec == DEC_BUY_10_FIG_B: return "figure_b", 10
    if dec == DEC_BUY_1_WEP:    return "weapon", 1
    if dec == DEC_BUY_10_WEP:   return "weapon", 10
    if dec == DEC_NOT_BUY:      return "none", 0
    return "unknown", 0


@dataclass
class BannerCfg:
    name: str
    base_p5: float
    hard_pity5: int
    soft_pity5: int
    base_p4: float
    hard_pity4: int
    soft_pity4: int
    featured_prob_on_5: float
    has_featured: bool


@dataclass
class BannerState:
    pity5: int = 0
    pity4: int = 0
    guarantee_featured_5: bool = False
    guarantee_featured_4: bool = False
    fate_points: int = 0
    epitomized_target: int = 0


def make_banner_cfgs() -> Dict[str, BannerCfg]:
    return {
        "regular": BannerCfg(
            name="regular",
            base_p5=0.006, hard_pity5=90, soft_pity5=74,
            base_p4=0.051, hard_pity4=10, soft_pity4=9,
            featured_prob_on_5=0.0, has_featured=False,
        ),
        "figure_a": BannerCfg(
            name="figure_a",
            base_p5=0.006, hard_pity5=90, soft_pity5=74,
            base_p4=0.051, hard_pity4=10, soft_pity4=9,
            featured_prob_on_5=0.5, has_featured=True,
        ),
        "figure_b": BannerCfg(
            name="figure_b",
            base_p5=0.006, hard_pity5=90, soft_pity5=74,
            base_p4=0.051, hard_pity4=10, soft_pity4=9,
            featured_prob_on_5=0.5, has_featured=True,
        ),
        "weapon": BannerCfg(
            name="weapon",
            base_p5=0.007, hard_pity5=80, soft_pity5=63,
            base_p4=0.060, hard_pity4=10, soft_pity4=9,
            featured_prob_on_5=0.75, has_featured=True,
        ),
    }


cfgs = make_banner_cfgs()


def _pity_adjusted_prob(base: float, pity: int, soft: int, hard: int) -> float:
    i = pity + 1
    if i >= hard:
        return 1.0
    if i < soft:
        return base
    inc = (i - soft + 1) * (base * 10.0)
    return float(min(1.0, base + inc))


def sample_rarity_one_pull(cfg: BannerCfg, st: BannerState, rng: np.random.Generator) -> int:
    p5 = _pity_adjusted_prob(cfg.base_p5, st.pity5, cfg.soft_pity5, cfg.hard_pity5)
    p4 = _pity_adjusted_prob(cfg.base_p4, st.pity4, cfg.soft_pity4, cfg.hard_pity4)
    u = rng.random()
    if u < p5:
        return 5
    u2 = rng.random()
    if u2 < p4:
        return 4
    return 3


def sample_outcome_token(
    banner: str, rarity: int, lto4: List[int],
    st: BannerState, rng: np.random.Generator,
    use_epitomized: bool = False,
) -> int:
    figA, figB, wep1, wep2 = lto4

    if rarity == 3:
        return TOK_3STAR_WEAPON
    if rarity == 4:
        return TOK_4STAR_FIGURE if rng.random() < 0.5 else TOK_4STAR_WEAPON

    # rarity == 5
    if banner == "regular":
        return TOK_5STAR_REG_FIGURE if rng.random() < 0.5 else TOK_5STAR_REG_WEAPON

    if banner in ("figure_a", "figure_b"):
        featured = figA if banner == "figure_a" else figB
        if featured == 0:
            return TOK_5STAR_REG_FIGURE
        if st.guarantee_featured_5:
            st.guarantee_featured_5 = False
            return int(featured)
        if rng.random() < cfgs[banner].featured_prob_on_5:
            return int(featured)
        else:
            st.guarantee_featured_5 = True
            return TOK_5STAR_REG_FIGURE

    if banner == "weapon":
        featured_pool = [t for t in [wep1, wep2] if t != 0]
        if not featured_pool:
            return TOK_5STAR_REG_WEAPON

        def pick_featured_weapon() -> int:
            if use_epitomized and st.epitomized_target in featured_pool and st.fate_points >= 2:
                return int(st.epitomized_target)
            return int(rng.choice(featured_pool))

        if st.guarantee_featured_5:
            tok = pick_featured_weapon()
            st.guarantee_featured_5 = False
        else:
            if rng.random() < cfgs["weapon"].featured_prob_on_5:
                tok = pick_featured_weapon()
            else:
                st.guarantee_featured_5 = True
                tok = TOK_5STAR_REG_WEAPON

        if use_epitomized and st.epitomized_target in featured_pool:
            if tok == st.epitomized_target:
                st.fate_points = 0
            else:
                st.fate_points = min(2, st.fate_points + 1)
        return tok

    return TOK_3STAR_WEAPON


def update_pity_after_pull(st: BannerState, rarity: int) -> None:
    st.pity5 += 1
    st.pity4 += 1
    if rarity == 5:
        st.pity5 = 0
        st.pity4 = 0
    elif rarity == 4:
        st.pity4 = 0


def simulate_outcomes_for_decision(
    decision: int, lto4: List[int],
    states: Dict[str, BannerState],
    rng: np.random.Generator,
    use_epitomized: bool = False,
) -> Tuple[List[int], Dict[str, Any]]:
    banner, n_pulls = decision_to_banner_and_pulls(decision)
    info = {"banner": banner, "pulls": n_pulls}

    if decision == DEC_NOT_BUY:
        return [EOS_PROD_ID_LOCAL] + [0]*9, info

    out10 = [0] * 10
    if n_pulls == 0 or banner == "none":
        return out10, info
    if banner not in states:
        return out10, info

    st = states[banner]
    cfg = cfgs[banner]

    for k in range(n_pulls):
        rarity = sample_rarity_one_pull(cfg, st, rng)
        tok = sample_outcome_token(
            banner=banner, rarity=rarity, lto4=lto4,
            st=st, rng=rng, use_epitomized=use_epitomized,
        )
        out10[k] = int(tok)
        update_pity_after_pull(st, rarity)

    return out10, info


def infer_states_from_history(history_tokens: List[int]) -> Dict[str, BannerState]:
    assert len(history_tokens) % AI_RATE == 0 and len(history_tokens) >= AI_RATE

    states = {
        "regular": BannerState(),
        "figure_a": BannerState(),
        "figure_b": BannerState(),
        "weapon": BannerState(),
    }

    n_blocks = len(history_tokens) // AI_RATE
    for b in range(n_blocks):
        block = history_tokens[b * AI_RATE:(b + 1) * AI_RATE]
        lto4 = block[0:4]
        out10 = block[4:14]
        prev_dec = int(block[14])

        banner, n_pulls = decision_to_banner_and_pulls(prev_dec)
        if banner not in states or n_pulls == 0:
            continue

        st = states[banner]
        cfg = cfgs[banner]

        for k in range(n_pulls):
            tok = int(out10[k])
            if tok == 0 or tok in {EOS_PROD_ID, SOS_PROD_ID}:
                continue

            if tok in (TOK_5STAR_REG_FIGURE, TOK_5STAR_REG_WEAPON) or (18 <= tok <= 56):
                rarity = 5
            elif tok in (TOK_4STAR_FIGURE, TOK_4STAR_WEAPON):
                rarity = 4
            else:
                rarity = 3

            if rarity == 5 and cfg.has_featured:
                figA, figB, wep1, wep2 = lto4
                if banner in ("figure_a", "figure_b"):
                    featured = figA if banner == "figure_a" else figB
                    if featured != 0 and tok != featured:
                        st.guarantee_featured_5 = True
                    else:
                        st.guarantee_featured_5 = False
                elif banner == "weapon":
                    featured_pool = [t for t in [wep1, wep2] if t != 0]
                    if featured_pool and tok not in featured_pool:
                        st.guarantee_featured_5 = True
                    else:
                        st.guarantee_featured_5 = False

            update_pity_after_pull(st, rarity)

    return states


# ----------------------------
# Step 2: simulated outcomes — MODIFIED FOR CALIBRATION
# ----------------------------
@torch.no_grad()
def generate_campaign28_step2_simulated_outcomes(
    model,
    history_tokens: List[int],
    lto28_tokens: List[int],
    device: torch.device,
    init_prev_dec: Optional[int],
    max_steps28: int,
    stop_decision: int,
    temperature: float,
    greedy: bool,
    rng: np.random.Generator,
    max_ctx_tokens: int = 1024,
    use_epitomized: bool = False,
    epitomized_target: int = 0,
    trace: Optional[TraceCfg] = None,
    trace_enabled: bool = False,
    trace_max_steps: int = 0,
    uid: str = "",
    run_id: int = 0,
    run_seed: Optional[int] = None,
    trace_fout: Optional[TextIO] = None,
    calibrator: Optional[BaseCalibrator] = None,    # <-- NEW
) -> Dict[str, Any]:
    """
    Campaign 28 generation with simulated outcomes (Step 2a+2b).
    Now with optional probability calibration.
    """
    assert len(lto28_tokens) == 4
    assert len(history_tokens) % AI_RATE == 0

    model.eval()
    states = infer_states_from_history(history_tokens)

    if epitomized_target != 0:
        states["weapon"].epitomized_target = int(epitomized_target)

    outcomes_step0 = extract_last_outcomes_from_history(history_tokens)

    if init_prev_dec is None:
        init_prev_dec = int(history_tokens[-1])

    prev_dec = int(init_prev_dec)
    seq_full = list(history_tokens)
    seq_c28: List[int] = []
    decisions28: List[int] = []
    stopped = False
    stop_step: Optional[int] = None

    for t in range(max_steps28):
        if t == 0:
            out10 = outcomes_step0
            state_before = copy.deepcopy(states)
            state_after  = copy.deepcopy(states)
            outcome_source = "history_c27_lastblock"
            banner = DECISION_META.get(prev_dec, ("UNK", "unknown", 0))[1]
            pulls  = DECISION_META.get(prev_dec, ("UNK", "unknown", 0))[2]
        else:
            state_before = copy.deepcopy(states)
            out10, info = simulate_outcomes_for_decision(
                decision=prev_dec, lto4=lto28_tokens,
                states=states, rng=rng,
                use_epitomized=use_epitomized,
            )
            state_after = copy.deepcopy(states)
            outcome_source = "simulated"
            banner = info.get("banner", DECISION_META.get(prev_dec, ("UNK","unknown",0))[1])
            pulls  = info.get("pulls",  DECISION_META.get(prev_dec, ("UNK","unknown",0))[2])

        block = list(lto28_tokens) + list(out10) + [prev_dec]
        seq_full.extend(block)
        seq_c28.extend(block)
        seq_full = trim_to_ctx(seq_full, max_ctx_tokens, AI_RATE)

        x = torch.tensor(seq_full, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        logits_at_decpos = logits[0, -1, :]

        dec = sample_decision_from_logits(
            logits_at_decpos, DECISION_IDS,
            temperature=temperature, greedy=greedy,
            calibrator=calibrator,                   # <-- PASSED THROUGH
        )

        do_trace = bool(trace and trace.enabled and (trace.max_steps == 0 or t < trace.max_steps))
        if do_trace:
            prev_name, _, _ = DECISION_META.get(prev_dec, ("UNK", "unknown", 0))
            dec_name,  _, _ = DECISION_META.get(dec,      ("UNK", "unknown", 0))
            rec = {
                "uid": uid, "step": 2, "run": run_id, "run_seed": run_seed, "t": t,
                "prev_dec": prev_dec, "prev_dec_name": prev_name,
                "banner": banner, "n_pulls": pulls,
                "outcome_source": outcome_source, "out10": out10,
                "state_before": state_before, "state_after": state_after,
                "sampled_dec": dec, "sampled_dec_name": dec_name,
                "calibrated": calibrator is not None and not isinstance(calibrator, IdentityCalibrator),
            }
            _trace_emit(rec, do_print=True, fout=trace_fout)

        decisions28.append(dec)

        if dec == stop_decision:
            stopped = True
            stop_step = t
            break
        prev_dec = dec

    return {
        "seq_campaign28": seq_c28,
        "seq_full": seq_full,
        "decisions28": decisions28,
        "stopped": stopped,
        "stop_step": stop_step,
        "final_states": {k: asdict(v) for k, v in states.items()},
    }


# ----------------------------
# Model / checkpoint helpers
# ----------------------------
def load_feature_tensor(xls_path: Path, feature_cols: List[str], vocab_size_src: int) -> torch.Tensor:
    df = pd.read_excel(xls_path, sheet_name=0)
    feat_dim = len(feature_cols)
    arr = np.zeros((vocab_size_src, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if 0 <= token_id < vocab_size_src:
            arr[token_id] = row[feature_cols].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)


def clean_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def strip_prefix(k: str) -> str:
        return k[7:] if k.startswith("module.") else k
    ignore = ("weight_fake_quant", "activation_post_process")
    return {strip_prefix(k): v for k, v in raw.items() if not any(tok in k for tok in ignore)}


def infer_ckpt_shapes(state_dict: Dict[str, torch.Tensor], fallback_layers: int, fallback_pe: int) -> Tuple[int, int]:
    layer_idxs = []
    for k in state_dict.keys():
        if k.startswith("decoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_idxs.append(int(parts[2]))
    n_layers = (max(layer_idxs) + 1) if layer_idxs else fallback_layers
    if "pos_enc.pe" in state_dict:
        pe_len = int(state_dict["pos_enc.pe"].shape[1])
    else:
        pe_len = fallback_pe
    return n_layers, pe_len


# ----------------------------
# Main — MODIFIED FOR CALIBRATION
# ----------------------------
def main():
    cfg = get_config()
    parser = argparse.ArgumentParser()

    parser.add_argument("--step", type=int, choices=[1, 2], default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed_base", type=int, default=None)

    parser.add_argument("--uid", type=str, default=None)
    parser.add_argument("--first", action="store_true")
    parser.add_argument("--n_users", type=int, default=0)

    parser.add_argument("--data", default=cfg["filepath"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)

    # ---------- CALIBRATOR ARGS (NEW) ----------
    parser.add_argument("--calibrator_ckpt", type=str, default=None,
                        help="Path to calibrator checkpoint .pt file. "
                             "If not provided, no calibration is applied.")
    parser.add_argument("--calibrator_type", type=str, default="auto",
                        choices=["auto", "temperature", "platt", "vector"],
                        help="Type of calibrator. 'auto' tries to detect from checkpoint keys.")
    # -------------------------------------------

    parser.add_argument("--lto28", nargs=4, type=int, required=True, metavar=("a", "b", "c", "d"))
    parser.add_argument("--fixed_outcomes", nargs=10, type=int, default=[0] * 10)
    parser.add_argument("--max_ctx_tokens", type=int, default=1024)

    parser.add_argument("--max_steps28", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--init_prev_dec", type=int, default=None)

    parser.add_argument("--use_epitomized", action="store_true")
    parser.add_argument("--epitomized_target", type=int, default=0)

    parser.add_argument("--feat_xlsx", type=str,
                        default="/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")

    parser.add_argument("--print_tokens", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate_users", type=int, default=50)
    parser.add_argument("--validate_last_k", type=int, default=8)
    parser.add_argument("--validate_trace_states", action="store_true")

    parser.add_argument("--trace_triplets", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace_out", type=str, default=None)
    parser.add_argument("--trace_max_steps", type=int, default=200)

    args = parser.parse_args()

    if args.validate:
        from collections import defaultdict
        # (batch_validate is unchanged — omitted here for brevity but you'd keep it)
        print("Validation mode not shown in this calibrated version — use original script.")
        return

    if args.first:
        args.n_users = 1

    # ---------- LOAD CALIBRATOR (NEW) ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.calibrator_ckpt is not None:
        calibrator = load_calibrator(
            ckpt_path=args.calibrator_ckpt,
            calibrator_type=args.calibrator_type,
            device=device,
        )
        print(f"[CALIBRATOR] Active: {calibrator}")
        print(f"[CALIBRATOR] User temperature: {args.temperature}")
        print(f"[CALIBRATOR] Pipeline: logits -> user_temp -> calibrator -> softmax -> sample")
    else:
        calibrator = IdentityCalibrator()
        print("[CALIBRATOR] None loaded — using raw model logits.")
    # -------------------------------------------

    trace_fout = None
    if args.trace_out is not None:
        Path(args.trace_out).parent.mkdir(parents=True, exist_ok=True)
        trace_fout = open(args.trace_out, "wt")

    trace_cfg = TraceCfg(enabled=bool(args.trace), max_steps=int(args.trace_max_steps), fout=trace_fout)

    FEATURE_COLS = [
        "Rarity", "MaxLife", "MaxOffense", "MaxDefense",
        "WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow",
        "WeaponTypeMagic", "WeaponTypePolearm",
        "EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire",
        "EthnicityThunder", "EthnicityWind",
        "GenderFemale", "GenderMale",
        "CountryRuiYue", "CountryDaoQi", "CountryZhiDong", "CountryMengDe",
        "type_figure",
        "MinimumAttack", "MaximumAttack",
        "MinSpecialEffect", "MaxSpecialEffect",
        "SpecialEffectEfficiency", "SpecialEffectExpertise", "SpecialEffectAttack",
        "SpecialEffectSuper", "SpecialEffectRatio", "SpecialEffectPhysical",
        "SpecialEffectLife",
        "LTO",
    ]

    feature_tensor = load_feature_tensor(Path(args.feat_xlsx), FEATURE_COLS, cfg["vocab_size_src"])

    PAD_ID_LOCAL = 0
    SOS_DEC_ID_L, EOS_DEC_ID_L, UNK_DEC_ID_L = 10, 11, 12
    EOS_PROD_ID_L, SOS_PROD_ID_L, UNK_PROD_ID_L = 57, 58, 59
    SPECIAL_IDS = [PAD_ID_LOCAL, SOS_DEC_ID_L, EOS_DEC_ID_L, UNK_DEC_ID_L, EOS_PROD_ID_L, SOS_PROD_ID_L]

    state = torch.load(args.ckpt, map_location=device)
    if "model_state_dict" in state:
        raw_sd = state["model_state_dict"]
    elif "module" in state:
        raw_sd = state["module"]
    else:
        raw_sd = state
    state_dict = clean_state_dict(raw_sd)

    n_layers_ckpt, max_seq_len_ckpt = infer_ckpt_shapes(
        state_dict, fallback_layers=cfg["N"], fallback_pe=cfg.get("seq_len_tgt", 1024),
    )

    model = build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        d_model=cfg["d_model"],
        n_layers=n_layers_ckpt,
        n_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
        nb_features=cfg["nb_features"],
        max_seq_len=max_seq_len_ckpt,
        kernel_type=cfg["kernel_type"],
        feature_tensor=feature_tensor,
        special_token_ids=SPECIAL_IDS,
    ).to(device).eval()

    model.load_state_dict(state_dict, strict=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out_path.suffix == ".gz" else open

    processed = 0

    with opener(out_path, "wt") as fout:
        for row in _iter_rows(args.data):
            uid = row["uid"][0] if isinstance(row.get("uid"), list) else row.get("uid")

            if args.uid is not None and uid != args.uid:
                continue

            history_tokens = parse_int_sequence(row["AggregateInput"], na_to=0)
            decs = []
            if "Decision" in row:
                decs = parse_int_sequence(row["Decision"])

            terminal_prod_tok = EOS_PROD_ID_LOCAL
            history_tokens, appended = maybe_append_missing_terminal_block(
                history_tokens, decs, terminal_prod_tok=terminal_prod_tok
            )
            if appended:
                print(f"[FIX] uid={uid}: appended terminal block", flush=True)
            if len(history_tokens) % AI_RATE != 0:
                raise ValueError(f"uid={uid}: history length {len(history_tokens)} not divisible by {AI_RATE}")

            init_prev_dec = args.init_prev_dec
            if init_prev_dec is None:
                try:
                    decs = parse_int_sequence(row["Decision"])
                    if len(decs) > 0:
                        init_prev_dec = int(history_tokens[-1])
                except Exception:
                    pass

            repeat = max(1, args.repeat)

            for r in range(repeat):
                run_seed = None
                if args.seed_base is not None:
                    run_seed = args.seed_base + r
                    rng = np.random.default_rng(run_seed)
                    torch.manual_seed(run_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(run_seed)
                else:
                    rng = np.random.default_rng()

                if args.step == 1:
                    out = generate_campaign28_step1_fixed_outcomes(
                        model=model,
                        history_tokens=history_tokens,
                        lto28_tokens=args.lto28,
                        fixed_outcomes_after_step0=args.fixed_outcomes,
                        device=device,
                        init_prev_dec=init_prev_dec,
                        max_steps28=args.max_steps28,
                        stop_decision=9,
                        temperature=args.temperature,
                        greedy=args.greedy,
                        trace=trace_cfg,
                        calibrator=calibrator,           # <-- NEW
                    )
                    payload = {
                        "uid": uid, "step": 1, "run": r, "run_seed": run_seed,
                        "Campaign28_Decisions": out["decisions28"],
                        "stopped": out["stopped"],
                        "stop_step": out["stop_step"],
                        "calibrated": not isinstance(calibrator, IdentityCalibrator),
                    }
                    if args.print_tokens:
                        payload["Campaign28_AggregateInput"] = out["seq_campaign28"]
                        payload["Full_AggregateInput"] = out["seq_full"]

                else:
                    if args.use_epitomized and args.epitomized_target == 0:
                        raise ValueError("--use_epitomized requires --epitomized_target")

                    out = generate_campaign28_step2_simulated_outcomes(
                        model=model,
                        history_tokens=history_tokens,
                        lto28_tokens=args.lto28,
                        device=device,
                        init_prev_dec=init_prev_dec,
                        max_steps28=args.max_steps28,
                        stop_decision=9,
                        temperature=args.temperature,
                        greedy=args.greedy,
                        rng=rng,
                        use_epitomized=args.use_epitomized,
                        epitomized_target=args.epitomized_target,
                        max_ctx_tokens=args.max_ctx_tokens,
                        trace=trace_cfg,
                        uid=uid,
                        run_id=r,
                        run_seed=run_seed,
                        trace_fout=trace_fout,
                        calibrator=calibrator,           # <-- NEW
                    )

                    val_cfg = GenValidationCfg(
                        require_lto4_match=True,
                        strict_buy10_full=True,
                        buy10_missing_is_error=True,
                    )
                    _ = validate_seq_campaign28(
                        seq_campaign28=out["seq_campaign28"],
                        expected_lto4=list(args.lto28),
                        cfg=val_cfg,
                    )

                    payload = {
                        "uid": uid, "step": 2, "run": r, "run_seed": run_seed,
                        "Campaign28_Decisions": out["decisions28"],
                        "stopped": out["stopped"],
                        "stop_step": out["stop_step"],
                        "final_states": out["final_states"],
                        "calibrated": not isinstance(calibrator, IdentityCalibrator),
                    }
                    if args.print_tokens:
                        payload["Campaign28_AggregateInput"] = out["seq_campaign28"]
                        payload["Full_AggregateInput"] = out["seq_full"]

                line = json.dumps(payload)
                fout.write(line + "\n")
                if not args.quiet:
                    print(line, flush=True)

            processed += 1
            if args.uid is not None:
                break
            if args.n_users and processed >= args.n_users:
                break

    if trace_fout is not None:
        trace_fout.close()

    if args.validate_trace_states:
        if args.step != 2:
            raise ValueError("--validate_trace_states requires --step 2")
        if args.trace_out is None:
            raise ValueError("--validate_trace_states requires --trace_out")
        validate_trace_state_consistency(
            trace_jsonl_path=args.trace_out,
            lto4=list(args.lto28),
            eos_prod_id=EOS_PROD_ID_LOCAL,
            sos_prod_id=SOS_PROD_ID_LOCAL,
        )

    if processed == 0:
        raise ValueError("No matching consumer found. Check --uid or input data.")

    if args.quiet:
        print(f"[DONE] Wrote outputs to {args.out}", flush=True)


if __name__ == "__main__":
    main()