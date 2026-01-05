#!/usr/bin/env python3
import argparse
import gzip
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TextIO

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config4 import get_config
from model4_decoderonly_feature_performer import build_transformer  # adjust if needed

import copy

@dataclass
class TraceCfg:
    enabled: bool = False
    max_steps: int = 200   # 0 means no limit
    fout: Optional[TextIO] = None

def _json_fallback(o):
    # dataclasses
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    # numpy scalars / arrays
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    # torch tensors
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    # last resort
    return str(o)

def _trace_emit(rec: dict, do_print: bool = True, fout: Optional[TextIO] = None):
    line = json.dumps(rec, ensure_ascii=False, default=_json_fallback)
    if do_print:
        print(line, flush=True)
    if fout is not None:
        fout.write(line + "\n")
        fout.flush()

# ----------------------------
# Constants / Token meanings
# ----------------------------
AI_RATE = 15
SOS_DEC_ID = 10                 # from your main()
EOS_PROD_ID_LOCAL = 57          # from your main()
SOS_PROD_ID_LOCAL = 58          # from your main()

BUY1_DECS  = {1, 3, 5, 7}
BUY10_DECS = {2, 4, 6, 8}
NOTBUY_DEC = 9

# These are "special" product IDs that should not appear as real outcomes.
SPECIAL_PROD_TOKS = {EOS_PROD_ID_LOCAL, SOS_PROD_ID_LOCAL}

@dataclass
class GenValidationCfg:
    eos_prod_id: int = EOS_PROD_ID_LOCAL
    sos_prod_id: int = SOS_PROD_ID_LOCAL
    require_lto4_match: bool = True
    # If True: buy-10 must have all 10 outcomes nonzero/non-special.
    # If False: allow missing outcomes but report warnings.
    strict_buy10_full: bool = True
    # If True: treat buy-10 missing outcomes as ERROR; else WARNING.
    buy10_missing_is_error: bool = True
    # If True: treat any nonzero tail after buy-1 as ERROR.
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
    # Valid realized outcome token: nonzero and not special.
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
    """
    Validate generated seq_campaign28 blocks for feasibility invariants.
    Returns a summary dict and prints detailed issues.
    """
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

        # (Optional) check LTO4 is constant and matches args.lto28
        if cfg.require_lto4_match and expected_lto4 is not None:
            if list(lto4) != list(expected_lto4):
                errors.append(f"b={b}: LTO4 mismatch. got={lto4} expected={expected_lto4}")

        # Check SOS never appears in OUT10
        if cfg.sos_prod_id in out10:
            errors.append(f"b={b}: SOS_PROD_ID={cfg.sos_prod_id} appears in OUT10: {out10}")

        # Enforce EOS appearance rules:
        # - If dec1==9: OUT10 must be [EOS,0,...,0]
        # - Otherwise: EOS must not appear anywhere in OUT10
        if dec1 == NOTBUY_DEC:
            expected = [cfg.eos_prod_id] + [0]*9
            if list(out10) != expected:
                errors.append(f"b={b}: DEC1==9 but OUT10 invalid. got={out10} expected={expected}")

            # Also ensure EOS appears only at position 0 (redundant if exact match)
            if any(tok == cfg.eos_prod_id for tok in out10[1:]):
                errors.append(f"b={b}: DEC1==9 but EOS appears after OUT10[0]: {out10}")

        else:
            if cfg.eos_prod_id in out10:
                errors.append(f"b={b}: DEC1={dec1} but EOS_PROD_ID appears in OUT10: {out10}")

        # Buy-1 rules
        if dec1 in BUY1_DECS:
            if not _is_valid_real_outcome(int(out10[0]), cfg):
                errors.append(f"b={b}: buy-1 DEC1={dec1} but OUT10[0] not a valid realized token: {out10[0]} out10={out10}")

            tail = out10[1:]
            if cfg.strict_buy1_tail_zero and any(int(x) != 0 for x in tail):
                errors.append(f"b={b}: buy-1 DEC1={dec1} but OUT10[1:] contains nonzero: out10={out10}")

            # Also disallow special tokens as realized in OUT10[0]
            if int(out10[0]) in SPECIAL_PROD_TOKS:
                errors.append(f"b={b}: buy-1 DEC1={dec1} but OUT10[0] is special token {out10[0]} out10={out10}")

        # Buy-10 rules
        if dec1 in BUY10_DECS:
            realized = [int(x) for x in out10 if _is_valid_real_outcome(int(x), cfg)]
            num_realized = len(realized)

            # If strict: require all 10 realized outcomes
            if cfg.strict_buy10_full:
                if num_realized != 10:
                    msg = f"b={b}: buy-10 DEC1={dec1} but realized outcomes={num_realized}/10. out10={out10}"
                    if cfg.buy10_missing_is_error:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
            else:
                # non-strict: allow fewer, but warn if very low
                if num_realized < 10:
                    warnings.append(f"b={b}: buy-10 DEC1={dec1} has only {num_realized}/10 realized outcomes. out10={out10}")

            # Ensure no special tokens appear in OUT10 for buy-10
            if any(int(x) in SPECIAL_PROD_TOKS for x in out10):
                errors.append(f"b={b}: buy-10 DEC1={dec1} but OUT10 contains special token(s): out10={out10}")

        # Not-buy already handled EOS rule; also ensure no other realized tokens
        if dec1 == NOTBUY_DEC:
            # By exact match, only EOS at [0] is allowed and rest zeros.
            pass

        # Unknown decision id
        if dec1 not in BUY1_DECS and dec1 not in BUY10_DECS and dec1 != NOTBUY_DEC:
            errors.append(f"b={b}: unknown DEC1={dec1}")

    ok = (len(errors) == 0)

    # Print concise report
    print("\n=== GENERATED SEQ VALIDATION ===")
    print(f"Blocks checked: {n_blocks}")
    print(f"OK: {ok}")
    print(f"Errors: {len(errors)}  Warnings: {len(warnings)}")

    if errors:
        print("\n[ERRORS]")
        for e in errors[:50]:
            print("  " + e)
        if len(errors) > 50:
            print(f"  ... ({len(errors)-50} more)")

    if warnings:
        print("\n[WARNINGS]")
        for w in warnings[:50]:
            print("  " + w)
        if len(warnings) > 50:
            print(f"  ... ({len(warnings)-50} more)")

    return {"ok": ok, "n_blocks": n_blocks, "errors": errors, "warnings": warnings}

def extract_block_decs(history_tokens):
    n_blocks = len(history_tokens) // AI_RATE
    return [int(history_tokens[i*AI_RATE + 14]) for i in range(n_blocks)]

def maybe_append_missing_terminal_block(
    history_tokens: list[int],
    decision_tokens: list[int],
    terminal_prod_tok: int = EOS_PROD_ID_LOCAL,  # recommended: 57
) -> tuple[list[int], bool]:
    """
    If Decision has one extra terminal decision (commonly 9) not present in AggregateInput,
    append one last [LTO4][OUT10][DEC1] block:
      LTO4 = last block's LTO4
      OUT10 = [terminal_prod_tok, 0, ..., 0]
      DEC1 = decision_tokens[-1]
    """
    if len(history_tokens) % AI_RATE != 0:
        raise ValueError("history_tokens length must be divisible by 15")

    if not decision_tokens:
        return history_tokens, False

    block_decs = extract_block_decs(history_tokens)

    # Many datasets start AggregateInput with a SOS_DEC block; Decision often excludes that.
    if block_decs and block_decs[0] == SOS_DEC_ID:
        block_decs_no_sos = block_decs[1:]
    else:
        block_decs_no_sos = block_decs

    # Case 1 (clean): Decision matches all existing blocks and has exactly one extra at the end.
    if (len(decision_tokens) == len(block_decs_no_sos) + 1 and
        decision_tokens[:len(block_decs_no_sos)] == block_decs_no_sos):
        last_lto4 = history_tokens[-AI_RATE : -AI_RATE + 4]
        new_block = list(last_lto4) + [int(terminal_prod_tok)] + [0]*9 + [int(decision_tokens[-1])]
        return history_tokens + new_block, True

    # Case 2 (pragmatic): the last decision differs; append if Decision ends with 9 but AggregateInput doesn’t.
    # This avoids breaking your inference run on the specific issue you described.
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


# Product token meanings (as you defined)
TOK_3STAR_WEAPON = 13
TOK_4STAR_FIGURE = 14
TOK_5STAR_REG_FIGURE = 15
TOK_4STAR_WEAPON = 16
TOK_5STAR_REG_WEAPON = 17
# 18-37: limited-time 5* figure (pity banner)
# 38-56: limited-time 5* weapon (pity banner)
PAD_ID = 0
EOS_PROD_ID = 57
SOS_PROD_ID = 58

# Decisions (as you defined)
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
    """Supports JSONL(.gz) or a JSON array file."""
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
    """
    Parse AggregateInput-like fields that may contain:
      - a single string
      - a list containing one big string
      - a list of ints/strings
    Robust to NA-like tokens by mapping them to `na_to` (default PAD=0).
    """
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
    """Last block = [LTO4][OUT10][PREV_DEC1]; outcomes are positions 4..13 (0-based)."""
    assert len(history_tokens) >= ai_rate and len(history_tokens) % ai_rate == 0, \
        "history_tokens must be non-empty and a multiple of 15."
    last_block = history_tokens[-ai_rate:]
    return last_block[4:14]  # length 10

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
    # exact requested format: [LTO4] [OUT10] [DEC1]
    return f"[{' '.join(map(str, lto4))}] [{' '.join(map(str, out10))}] [{dec1}]"

def extract_dec1_sequence_from_aggregate(history_tokens: List[int]) -> List[int]:
    return [dec1 for _, _, _, dec1 in iter_blocks(history_tokens)]

def validate_one_user(uid: str, history_tokens: List[int], decision_tokens: List[int],
                      k_last: int = 8, sos_dec_id: int = 10, stop_dec_id: int = 9) -> Dict[str, Any]:
    """
    Returns summary stats and prints a concise tail comparison.
    - Drops leading SOS (10) from AggregateInput DEC1 stream if present.
    - Detects terminal-stop extra decision in `Decision` (common) and reports it.
    - Counts "DEC1==9 but OUT10 has nonzero" within AggregateInput.
    """
    decs_agg = extract_dec1_sequence_from_aggregate(history_tokens)
    has_sos = (len(decs_agg) > 0 and decs_agg[0] == sos_dec_id)
    decs_agg_nos = decs_agg[1:] if has_sos else decs_agg

    # Count not-buy blocks with nonzero outcomes
    notbuy_nonzero = 0
    examples_notbuy = []
    for b, lto4, out10, dec1 in iter_blocks(history_tokens):
        if dec1 == stop_dec_id and any(x != 0 for x in out10):
            notbuy_nonzero += 1
            if len(examples_notbuy) < 3:
                examples_notbuy.append((b, lto4, out10, dec1))

    # Alignment checks
    min_len = min(len(decs_agg_nos), len(decision_tokens))
    prefix_match = (decs_agg_nos[:min_len] == decision_tokens[:min_len])

    terminal_extra = False
    if len(decision_tokens) == len(decs_agg_nos) + 1 and prefix_match:
        terminal_extra = True

    # Print tail diagnostics
    tail_agg = decs_agg_nos[-k_last:] if len(decs_agg_nos) >= k_last else decs_agg_nos
    tail_dec = decision_tokens[-k_last:] if len(decision_tokens) >= k_last else decision_tokens

    print(f"\n[UID] {uid}")
    print(f"  Aggregate blocks: {len(decs_agg)} (has_sos={has_sos}) -> compare len={len(decs_agg_nos)}")
    print(f"  Decision len: {len(decision_tokens)}")
    print(f"  Prefix match up to min_len={min_len}: {prefix_match}")
    print(f"  Terminal extra decision in Decision: {terminal_extra}")
    print(f"  Tail Aggregate DEC1: {tail_agg}")
    print(f"  Tail Decision:       {tail_dec}")
    print(f"  Count DEC1==9 but OUT10 nonzero in AggregateInput: {notbuy_nonzero}")

    if examples_notbuy:
        print("  Examples of [LTO4][OUT10][DEC1==9] with nonzero OUT10:")
        for b, lto4, out10, dec1 in examples_notbuy:
            print(f"    block={b}: {format_triplet(lto4, out10, dec1)}")

    # Also print last k triplets from AggregateInput for manual inspection
    blocks = list(iter_blocks(history_tokens))
    tail_blocks = blocks[-k_last:] if len(blocks) >= k_last else blocks
    print("  Last triplets from AggregateInput:")
    for b, lto4, out10, dec1 in tail_blocks:
        print(f"    b={b}: {format_triplet(lto4, out10, dec1)}")

    return {
        "uid": uid,
        "has_sos": has_sos,
        "len_agg_blocks": len(decs_agg),
        "len_agg_compare": len(decs_agg_nos),
        "len_decision": len(decision_tokens),
        "prefix_match": prefix_match,
        "terminal_extra": terminal_extra,
        "notbuy_nonzero": notbuy_nonzero,
    }

def batch_validate(data_path: str, n_users: int = 50, k_last: int = 8):
    """
    Iterates through dataset and prints per-user tail diagnostics + global summary.
    """
    total = 0
    mismatch = 0
    terminal_extra_cnt = 0
    any_notbuy_nonzero = 0
    sum_notbuy_nonzero = 0

    for row in _iter_rows(data_path):
        uid = row["uid"][0] if isinstance(row.get("uid"), list) else row.get("uid")
        history_tokens = parse_int_sequence(row["AggregateInput"], na_to=0)
        decision_tokens = parse_int_sequence(row["Decision"]) if "Decision" in row else []

        info = validate_one_user(uid, history_tokens, decision_tokens, k_last=k_last)

        total += 1
        if not info["prefix_match"]:
            mismatch += 1
        if info["terminal_extra"]:
            terminal_extra_cnt += 1
        if info["notbuy_nonzero"] > 0:
            any_notbuy_nonzero += 1
            sum_notbuy_nonzero += info["notbuy_nonzero"]

        if n_users and total >= n_users:
            break

    print("\n=== GLOBAL SUMMARY ===")
    print(f"Users checked: {total}")
    print(f"Users with prefix mismatch (Aggregate vs Decision): {mismatch}")
    print(f"Users with terminal-extra decision (Decision len = Aggregate len + 1): {terminal_extra_cnt}")
    print(f"Users with any (DEC1==9 AND OUT10 nonzero) blocks: {any_notbuy_nonzero}")
    print(f"Total such blocks across checked users: {sum_notbuy_nonzero}")

# ----------------------------
# Model sampling (decision)
# ----------------------------
@torch.no_grad()
def sample_decision_from_logits(
    logits_last_pos: torch.Tensor,
    decision_ids: torch.Tensor,
    temperature: float = 1.0,
    greedy: bool = False,
) -> int:
    ids = decision_ids.to(logits_last_pos.device)
    dec_logits = logits_last_pos[ids]  # (9,)

    if temperature != 1.0:
        dec_logits = dec_logits / max(temperature, 1e-8)

    probs = F.softmax(dec_logits, dim=-1)

    if greedy:
        k = torch.argmax(probs).item()
    else:
        k = torch.multinomial(probs, 1).item()

    return int(ids[k].item())


# ----------------------------
# Step 1: fixed outcomes after t=0
# ----------------------------
@torch.no_grad()
def generate_campaign28_step1_fixed_outcomes(
    model,
    history_tokens: List[int],
    lto28_tokens: List[int],
    fixed_outcomes_after_step0: List[int],
    device: torch.device,
    init_prev_dec: Optional[int] = None,
    max_steps28: int = 500,
    stop_decision: int = 9,
    temperature: float = 1.0,
    greedy: bool = False,
    trace: Optional[TraceCfg] = None,   # NEW
) -> Dict[str, Any]:
    assert len(lto28_tokens) == 4, "lto28_tokens must have length 4."
    assert len(fixed_outcomes_after_step0) == 10, "fixed_outcomes_after_step0 must have length 10."
    assert len(history_tokens) >= AI_RATE and len(history_tokens) % AI_RATE == 0, \
        "history_tokens must be non-empty and a multiple of 15."

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

        x = torch.tensor(seq_full, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
        logits = model(x)                                                        # (1,T,V)
        logits_at_decpos = logits[0, -1, :]                                      # (V,)

        dec = sample_decision_from_logits(
            logits_at_decpos, DECISION_IDS, temperature=temperature, greedy=greedy
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
# Step 2: gacha simulation
#   2a: infer counters from history
#   2b: simulate outcomes in campaign 28
# ----------------------------
def decision_to_banner_and_pulls(dec: int) -> Tuple[str, int]:
    """
    Map decision id to (banner_type, num_pulls).
    """
    if dec == DEC_BUY_1_REG:
        return "regular", 1
    if dec == DEC_BUY_10_REG:
        return "regular", 10
    if dec == DEC_BUY_1_FIG_A:
        return "figure_a", 1
    if dec == DEC_BUY_10_FIG_A:
        return "figure_a", 10
    if dec == DEC_BUY_1_FIG_B:
        return "figure_b", 1
    if dec == DEC_BUY_10_FIG_B:
        return "figure_b", 10
    if dec == DEC_BUY_1_WEP:
        return "weapon", 1
    if dec == DEC_BUY_10_WEP:
        return "weapon", 10
    if dec == DEC_NOT_BUY:
        return "none", 0
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
    featured_prob_on_5: float  # conditional on 5* and not guaranteed (character: 0.5, weapon: 0.75, regular: 0.0)
    has_featured: bool


@dataclass
class BannerState:
    pity5: int = 0                 # pulls since last 5*
    pity4: int = 0                 # pulls since last 4* or higher (we reset on 4* or 5*)
    guarantee_featured_5: bool = False
    guarantee_featured_4: bool = False  # not identifiable in your dataset, but kept for completeness
    fate_points: int = 0           # weapon epitomized path (0..2) - optional use
    epitomized_target: int = 0     # token id of selected weapon (must be one of the two featured), 0 means unset

def make_banner_cfgs() -> Dict[str, BannerCfg]:
    """
    Banner parameters consistent with the KQM summary and evidence:
    - 5* soft pity: 74 (character/standard), 63 (weapon)
    - 4* soft pity: 9
    - hard pity: 90 (character/standard), 80 (weapon), 10 (4*)
    - after soft pity, increase by base_rate*10 each pull (capped), hard pity forced.
    """
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


def _pity_adjusted_prob(base: float, pity: int, soft: int, hard: int) -> float:
    """
    pity: pulls since last hit (0 means next pull is 1st in the cycle)
    i = pity + 1 is current pull number in cycle.
    Before soft: base
    At/after soft: base + (i - soft + 1) * (base*10)
    At hard: forced 1.0
    """
    i = pity + 1
    if i >= hard:
        return 1.0
    if i < soft:
        return base
    inc = (i - soft + 1) * (base * 10.0)
    return float(min(1.0, base + inc))


def sample_rarity_one_pull(cfg: BannerCfg, st: BannerState, rng: np.random.Generator) -> int:
    """
    Return rarity category:
      5 -> 5*
      4 -> 4*
      3 -> 3*
    Enforces:
      - 5* hard pity
      - 4* hard pity within 10 pulls for 4* OR higher (we reset pity4 on 4* or 5*)
      - soft pity adjustments for both 5* and 4*
    """
    p5 = _pity_adjusted_prob(cfg.base_p5, st.pity5, cfg.soft_pity5, cfg.hard_pity5)
    # 4* probability applies only if not 5*
    p4 = _pity_adjusted_prob(cfg.base_p4, st.pity4, cfg.soft_pity4, cfg.hard_pity4)

    u = rng.random()
    if u < p5:
        return 5
    # conditional on not 5*
    u2 = rng.random()
    if u2 < p4:
        return 4
    return 3


def sample_outcome_token(
    banner: str,
    rarity: int,
    lto4: List[int],
    st: BannerState,
    rng: np.random.Generator,
    use_epitomized: bool = False,
) -> int:
    """
    Map (banner, rarity) to a product token, and update featured/epitomized logic.
    lto4 = [figA, figB, wep1, wep2] for the CURRENT campaign step.
    """
    figA, figB, wep1, wep2 = lto4

    if rarity == 3:
        return TOK_3STAR_WEAPON

    if rarity == 4:
        # Your tokenization distinguishes 4* figure vs weapon, but not featured vs off-feature.
        # We choose 50/50 by default.
        return TOK_4STAR_FIGURE if rng.random() < 0.5 else TOK_4STAR_WEAPON

    # rarity == 5
    if banner == "regular":
        # no featured; choose between regular 5* figure/weapon (simple)
        return TOK_5STAR_REG_FIGURE if rng.random() < 0.5 else TOK_5STAR_REG_WEAPON

    if banner in ("figure_a", "figure_b"):
        featured = figA if banner == "figure_a" else figB
        # If featured token is 0 (missing/NA), treat as no-featured banner (fallback to regular 5*)
        if featured == 0:
            return TOK_5STAR_REG_FIGURE

        if st.guarantee_featured_5:
            tok = int(featured)
            st.guarantee_featured_5 = False
            return tok

        # 50/50 featured vs off-feature on a 5*
        if rng.random() < cfgs[banner].featured_prob_on_5:
            return int(featured)
        else:
            # off-feature: mapped to your "5-star regular figure"
            st.guarantee_featured_5 = True
            return TOK_5STAR_REG_FIGURE

    if banner == "weapon":
        featured_pool = [t for t in [wep1, wep2] if t != 0]
        if not featured_pool:
            return TOK_5STAR_REG_WEAPON

        def pick_featured_weapon() -> int:
            if use_epitomized and st.epitomized_target in featured_pool and st.fate_points >= 2:
                return int(st.epitomized_target)
            if use_epitomized and st.epitomized_target in featured_pool:
                # default 50/50 between the two featured (matches typical banner)
                return int(rng.choice(featured_pool))
            return int(rng.choice(featured_pool))

        if st.guarantee_featured_5:
            tok = pick_featured_weapon()
            st.guarantee_featured_5 = False
        else:
            # 75/25 featured vs off-feature on a 5*
            if rng.random() < cfgs["weapon"].featured_prob_on_5:
                tok = pick_featured_weapon()
            else:
                st.guarantee_featured_5 = True
                tok = TOK_5STAR_REG_WEAPON

        # Update fate points if epitomized is enabled and target set
        if use_epitomized and st.epitomized_target in featured_pool:
            if tok == st.epitomized_target:
                st.fate_points = 0
            else:
                st.fate_points = min(2, st.fate_points + 1)

        return tok

    return TOK_3STAR_WEAPON


def update_pity_after_pull(st: BannerState, rarity: int) -> None:
    """
    Update pity counters after a single pull.
    In Genshin, a 5* also satisfies the "4* or higher within 10 pulls" condition,
    so we reset pity4 on both 4* and 5*.
    """
    # increment first, then reset on hits
    st.pity5 += 1
    st.pity4 += 1

    if rarity == 5:
        st.pity5 = 0
        st.pity4 = 0
    elif rarity == 4:
        st.pity4 = 0


def simulate_outcomes_for_decision(
    decision: int,
    lto4: List[int],
    states: Dict[str, BannerState],
    rng: np.random.Generator,
    use_epitomized: bool = False,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Produce OUT10 for the decision and an info dict.
    Enforces feasibility:
      - 0 pulls: all zeros
      - 1 pull: exactly 1 nonzero then zeros
      - 10 pulls: up to 10 nonzero
    """
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
            banner=banner,
            rarity=rarity,
            lto4=lto4,
            st=st,
            rng=rng,
            use_epitomized=use_epitomized,
        )
        out10[k] = int(tok)
        update_pity_after_pull(st, rarity)

    return out10, info

def infer_states_from_history(history_tokens: List[int]) -> Dict[str, BannerState]:
    """
    Step 2a: infer pity/guarantee counters from Campaigns 1–27 token history.
    History is blocks of 15:
      [LTO4][OUT10][PREV_DEC]
    OUT10 correspond to PREV_DEC's pull outcomes.
    The LTO4 in *that same block* identifies the featured pool at that time.
    """
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

        # For each realized pull (first n_pulls outcome tokens), update pity and featured guarantee
        for k in range(n_pulls):
            SPECIAL_PROD_TOKS = {EOS_PROD_ID, SOS_PROD_ID}  # or use the *_LOCAL vars
            tok = int(out10[k])
            if tok == 0:
                # if data contains 0 even when supposed to have a pull, treat as missing
                continue
            
            if tok in SPECIAL_PROD_TOKS:
                continue

            # infer rarity from token id
            if tok in (TOK_5STAR_REG_FIGURE, TOK_5STAR_REG_WEAPON) or (18 <= tok <= 56):
                rarity = 5
            elif tok in (TOK_4STAR_FIGURE, TOK_4STAR_WEAPON):
                rarity = 4
            else:
                rarity = 3

            # featured / off-feature logic inference (only meaningful for 5*)
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


def rarity_from_token(tok: int) -> int:
    # Mirror your inference logic:
    # 5* tokens: 15,17,18..56
    # 4*: 14,16
    # else: 3
    if tok in (15, 17) or (18 <= tok <= 56):
        return 5
    if tok in (14, 16):
        return 4
    return 3

def decision_to_banner_and_pulls(dec: int) -> Tuple[str, int]:
    if dec == 1: return "regular", 1
    if dec == 2: return "regular", 10
    if dec == 3: return "figure_a", 1
    if dec == 4: return "figure_a", 10
    if dec == 5: return "figure_b", 1
    if dec == 6: return "figure_b", 10
    if dec == 7: return "weapon", 1
    if dec == 8: return "weapon", 10
    if dec == 9: return "none", 0
    return "unknown", 0

def bannerstate_from_dict(d: Dict[str, Any]) -> BannerState:
    return BannerState(
        pity5=int(d.get("pity5", 0)),
        pity4=int(d.get("pity4", 0)),
        guarantee_featured_5=bool(d.get("guarantee_featured_5", False)),
        guarantee_featured_4=bool(d.get("guarantee_featured_4", False)),
        fate_points=int(d.get("fate_points", 0)),
        epitomized_target=int(d.get("epitomized_target", 0)),
    )

def bannerstate_equal_dict(st: BannerState, d: Dict[str, Any]) -> bool:
    dd = asdict(st)
    # compare only keys you care about (all in BannerState)
    return all(dd[k] == (bool(d[k]) if isinstance(dd[k], bool) else int(d[k])) if k in d else dd[k] == dd[k]
               for k in dd.keys())

def apply_guarantee_inference_state(banner: str, tok: int, lto4: List[int], st: BannerState) -> None:
    # only meaningful for 5*
    if not (tok in (TOK_5STAR_REG_FIGURE, TOK_5STAR_REG_WEAPON) or (18 <= tok <= 56)):
        return

    figA, figB, wep1, wep2 = lto4

    if banner in ("figure_a", "figure_b"):
        featured = figA if banner == "figure_a" else figB
        st.guarantee_featured_5 = (featured != 0 and tok != featured)

    elif banner == "weapon":
        featured_pool = [t for t in [wep1, wep2] if t != 0]
        st.guarantee_featured_5 = (bool(featured_pool) and tok not in featured_pool)

def apply_guarantee_inference(
    banner: str,
    tok: int,
    lto4: List[int],
    st: Dict[str, Any],
) -> None:
    """
    Apply the same *inference-style* guarantee update you used in infer_states_from_history:
      - if banner has featured and tok is 5*:
          - if tok not in featured set -> guarantee_featured_5 = True
          - else -> guarantee_featured_5 = False
    """
    if not (tok in (15, 17) or (18 <= tok <= 56)):
        return  # only meaningful for 5*

    figA, figB, wep1, wep2 = lto4

    if banner in ("figure_a", "figure_b"):
        featured = figA if banner == "figure_a" else figB
        if featured != 0 and tok != featured:
            st["guarantee_featured_5"] = True
        else:
            st["guarantee_featured_5"] = False

    elif banner == "weapon":
        featured_pool = [t for t in [wep1, wep2] if t != 0]
        if featured_pool and tok not in featured_pool:
            st["guarantee_featured_5"] = True
        else:
            st["guarantee_featured_5"] = False

def _state_equal_dict(a: Dict[str, Any], b: Dict[str, Any], keys: List[str]) -> bool:
    """
    Compare two banner-state dicts on selected keys, with type normalization.
    """
    for k in keys:
        av = a.get(k, 0)
        bv = b.get(k, 0)

        # normalize bool/int
        if isinstance(av, bool) or isinstance(bv, bool):
            if bool(av) != bool(bv):
                return False
        else:
            try:
                if int(av) != int(bv):
                    return False
            except Exception:
                if av != bv:
                    return False
    return True

def _rarity_from_token(tok: int) -> int:
    # Mirror your inference/simulator logic:
    # 5* tokens: 15,17,18..56
    # 4*: 14,16
    # else: 3
    if tok in (TOK_5STAR_REG_FIGURE, TOK_5STAR_REG_WEAPON) or (18 <= tok <= 56):
        return 5
    if tok in (TOK_4STAR_FIGURE, TOK_4STAR_WEAPON):
        return 4
    return 3


def _decision_to_banner_and_pulls_trace(dec: int) -> Tuple[str, int]:
    # Local helper to avoid relying on any globally-duplicated definitions.
    if dec == 1: return "regular", 1
    if dec == 2: return "regular", 10
    if dec == 3: return "figure_a", 1
    if dec == 4: return "figure_a", 10
    if dec == 5: return "figure_b", 1
    if dec == 6: return "figure_b", 10
    if dec == 7: return "weapon", 1
    if dec == 8: return "weapon", 10
    if dec == 9: return "none", 0
    return "unknown", 0

def _update_pity_after_pull_dict(st: Dict[str, Any], rarity: int) -> None:
    """
    Dict-based pity update (do NOT name this update_pity_after_pull to avoid collisions).
    Matches your BannerState logic:
      - increment pity5 and pity4 each pull
      - reset pity5 and pity4 on 5*
      - reset pity4 on 4*
    """
    st["pity5"] = int(st.get("pity5", 0)) + 1
    st["pity4"] = int(st.get("pity4", 0)) + 1

    if rarity == 5:
        st["pity5"] = 0
        st["pity4"] = 0
    elif rarity == 4:
        st["pity4"] = 0


def _apply_featured_guarantee_update_dict(
    banner: str,
    tok: int,
    lto4: List[int],
    st: Dict[str, Any],
) -> None:
    """
    Update guarantee_featured_5 based on the *observed* 5* token and current featured pool.

    This is consistent with both:
      - your infer_states_from_history “inference-style” logic, and
      - your simulator’s effect on guarantee_featured_5 (featured -> False; off-feature -> True).

    Only applies on 5* and only for banners that have featured items.
    """
    rarity = _rarity_from_token(tok)
    if rarity != 5:
        return

    figA, figB, wep1, wep2 = [int(x) for x in lto4]

    if banner in ("figure_a", "figure_b"):
        featured = figA if banner == "figure_a" else figB
        if featured != 0 and tok != featured:
            st["guarantee_featured_5"] = True
        else:
            st["guarantee_featured_5"] = False

    elif banner == "weapon":
        featured_pool = [t for t in (wep1, wep2) if t != 0]
        if featured_pool and tok not in featured_pool:
            st["guarantee_featured_5"] = True
        else:
            st["guarantee_featured_5"] = False


def _apply_epitomized_fate_points_update_dict(
    banner: str,
    tok: int,
    lto4: List[int],
    st: Dict[str, Any],
) -> None:
    """
    Mirror the fate-point update you implemented in sample_outcome_token (weapon only):
      if use_epitomized and epitomized_target is in featured_pool:
          if tok == target -> fate_points = 0
          else -> fate_points = min(2, fate_points+1)

    Note: This update is *only* for 5* outcomes on weapon banner.
    """
    if banner != "weapon":
        return

    rarity = _rarity_from_token(tok)
    if rarity != 5:
        return

    figA, figB, wep1, wep2 = [int(x) for x in lto4]
    featured_pool = [t for t in (wep1, wep2) if t != 0]
    target = int(st.get("epitomized_target", 0) or 0)

    if target == 0:
        return
    if target not in featured_pool:
        return

    if tok == target:
        st["fate_points"] = 0
    else:
        st["fate_points"] = min(2, int(st.get("fate_points", 0)) + 1)


def _apply_one_pull_update_for_banner_dict(
    banner: str,
    tok: int,
    lto4: List[int],
    st: Dict[str, Any],
    eos_prod_id: int,
    sos_prod_id: int,
) -> None:
    """
    Apply one realized pull update to *one banner state dict*.
    This is the core “state_after = f(state_before, prev_dec, out10)” validator.
    """
    tok = int(tok)

    # Disallow special tokens as realized outcomes
    if tok in (eos_prod_id, sos_prod_id):
        return
    if tok == 0:
        return

    # 1) featured guarantee update (5* only, featured banners only)
    _apply_featured_guarantee_update_dict(banner, tok, lto4, st)

    # 2) epitomized fate points update (weapon + 5* only, target set)
    _apply_epitomized_fate_points_update_dict(banner, tok, lto4, st)

    # 3) pity update (always)
    rarity = _rarity_from_token(tok)
    _update_pity_after_pull_dict(st, rarity)


def validate_trace_state_consistency(
    trace_jsonl_path: str,
    lto4: List[int],
    eos_prod_id: int = EOS_PROD_ID_LOCAL,
    sos_prod_id: int = SOS_PROD_ID_LOCAL,
    hard_pity5_map: Optional[Dict[str, int]] = None,
    hard_pity4: int = 10,
) -> Dict[str, Any]:
    """
    Validate Step-2 trace JSONL:
      - Cross-record chaining:
          * t monotonic within (uid,run)
          * next.prev_dec == current.sampled_dec, unless stop happened
      - OUT10 rules:
          * SOS never appears in OUT10
          * if prev_dec==9 -> OUT10 == [EOS,0,...,0]
          * else -> EOS must not appear in OUT10
      - State transition rules for t>=1:
          * if prev_dec==9 (or n_pulls==0) => states unchanged
          * else => only the affected banner state changes, and it matches applying
            per-pull updates to state_before[banner] using OUT10[:n_pulls]
      - Feasibility:
          * pity5 < hard_pity5(banner)
          * pity4 < hard_pity4
    """
    if hard_pity5_map is None:
        hard_pity5_map = {"regular": 90, "figure_a": 90, "figure_b": 90, "weapon": 80}

    # group records by (uid, run)
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    total_records = 0
    parse_errors = 0

    with open(trace_jsonl_path, "rt") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                parse_errors += 1
                continue
            if int(rec.get("step", -1)) != 2:
                continue
            uid = rec.get("uid", "")
            run = int(rec.get("run", 0))
            groups[(uid, run)].append(rec)
            total_records += 1

    errors: List[str] = []
    warnings: List[str] = []

    STATE_KEYS = [
        "pity5",
        "pity4",
        "guarantee_featured_5",
        "guarantee_featured_4",
        "fate_points",
        "epitomized_target",
    ]

    for (uid, run), recs in groups.items():
        recs.sort(key=lambda r: int(r.get("t", -1)))

        # ---- Cross-record checks: t monotonic + prev_dec chaining ----
        for i, r in enumerate(recs):
            t = int(r.get("t", -1))
            if t != i:
                warnings.append(
                    f"uid={uid} run={run}: non-contiguous t (expected {i}, got {t}); "
                    f"trace_max_steps may truncate."
                )

        for i in range(len(recs) - 1):
            cur = recs[i]
            nxt = recs[i + 1]

            cur_sampled = int(cur.get("sampled_dec", -1))
            nxt_prev = int(nxt.get("prev_dec", -1))

            # Once you stop (sampled_dec == 9), generation breaks, so there should be no next record.
            if cur_sampled == 9:
                errors.append(
                    f"uid={uid} run={run}: sampled_dec==9 at t={cur.get('t')} "
                    f"but trace continues at t={nxt.get('t')}."
                )
                break

            if nxt_prev != cur_sampled:
                errors.append(
                    f"uid={uid} run={run}: decision chaining mismatch: "
                    f"t={cur.get('t')} sampled_dec={cur_sampled} but t={nxt.get('t')} prev_dec={nxt_prev}"
                )

        # ---- Per-record checks: mechanics ----
        for r in recs:
            t = int(r.get("t", -1))
            prev_dec = int(r.get("prev_dec", -1))
            out10 = [int(x) for x in r.get("out10", [])]

            if len(out10) != 10:
                errors.append(f"uid={uid} run={run} t={t}: out10 length != 10 (got {len(out10)})")
                continue

            # EOS/SOS in OUT10 rules
            if sos_prod_id in out10:
                errors.append(f"uid={uid} run={run} t={t}: SOS appears in OUT10={out10}")

            if prev_dec == 9:
                expected = [eos_prod_id] + [0] * 9
                if out10 != expected:
                    errors.append(
                        f"uid={uid} run={run} t={t}: prev_dec=9 but OUT10 invalid: got={out10} expected={expected}"
                    )
            else:
                if eos_prod_id in out10:
                    errors.append(f"uid={uid} run={run} t={t}: prev_dec={prev_dec} but EOS appears in OUT10={out10}")

            # outcome_source sanity
            src = r.get("outcome_source", "")
            if t == 0 and src != "history_c27_lastblock":
                warnings.append(f"uid={uid} run={run} t=0: outcome_source={src} (expected history_c27_lastblock)")
            if t >= 1 and src != "simulated":
                warnings.append(f"uid={uid} run={run} t={t}: outcome_source={src} (expected simulated)")

            # t==0: you intentionally do not update states; verify state_before == state_after
            if t == 0:
                if r.get("state_before") != r.get("state_after"):
                    errors.append(f"uid={uid} run={run} t=0: state_before != state_after (should be unchanged at t=0)")
                continue

            banner, n_pulls = _decision_to_banner_and_pulls_trace(prev_dec)

            # If prev_dec==9: OUT10 is EOS-only but states must not change.
            if prev_dec == 9 or banner == "none" or n_pulls == 0:
                if r.get("state_before") != r.get("state_after"):
                    errors.append(f"uid={uid} run={run} t={t}: prev_dec={prev_dec} (no-pull) but states changed")
                continue

            sb_all = r.get("state_before", {})
            sa_all = r.get("state_after", {})

            if not isinstance(sb_all, dict) or not isinstance(sa_all, dict):
                errors.append(f"uid={uid} run={run} t={t}: state_before/state_after not dict")
                continue

            if banner not in sb_all or banner not in sa_all:
                errors.append(f"uid={uid} run={run} t={t}: missing banner={banner} in state_before/after")
                continue

            # Unaffected banners should not change
            for bname in sb_all.keys():
                if bname == banner:
                    continue
                if sb_all.get(bname) != sa_all.get(bname):
                    errors.append(
                        f"uid={uid} run={run} t={t}: banner={banner} changed unrelated banner={bname}"
                    )

            # Recompute expected after-state for the affected banner
            st = dict(sb_all[banner])  # copy

            # Apply n_pulls pulls (first n_pulls tokens in out10)
            for k in range(n_pulls):
                tok = int(out10[k])

                if tok == 0:
                    # If you truly never expect missing outcomes inside realized pulls, promote to error.
                    warnings.append(
                        f"uid={uid} run={run} t={t}: tok==0 within realized pulls (k={k}) out10={out10}"
                    )
                    continue

                if tok in (eos_prod_id, sos_prod_id):
                    errors.append(
                        f"uid={uid} run={run} t={t}: special token {tok} within realized pulls out10={out10}"
                    )
                    continue

                _apply_one_pull_update_for_banner_dict(
                    banner=banner,
                    tok=tok,
                    lto4=lto4,
                    st=st,
                    eos_prod_id=eos_prod_id,
                    sos_prod_id=sos_prod_id,
                )

            expected_after = st
            got_after = sa_all[banner]

            if not _state_equal_dict(expected_after, got_after, STATE_KEYS):
                errors.append(
                    f"uid={uid} run={run} t={t}: state mismatch banner={banner}. "
                    f"expected={expected_after} got={got_after}"
                )

            # Feasibility checks
            hard5 = int(hard_pity5_map.get(banner, 90))
            if int(got_after.get("pity5", 0)) >= hard5:
                errors.append(
                    f"uid={uid} run={run} t={t}: pity5 infeasible banner={banner} "
                    f"pity5={got_after.get('pity5')} hard={hard5}"
                )
            if int(got_after.get("pity4", 0)) >= hard_pity4:
                errors.append(
                    f"uid={uid} run={run} t={t}: pity4 infeasible banner={banner} "
                    f"pity4={got_after.get('pity4')} hard={hard_pity4}"
                )

    ok = (len(errors) == 0)

    print("\n=== TRACE STATE CONSISTENCY VALIDATION ===")
    print(f"Trace file: {trace_jsonl_path}")
    print(f"Groups (uid,run): {len(groups)}")
    print(f"Records checked: {total_records}")
    print(f"Parse errors: {parse_errors}")
    print(f"OK: {ok}")
    print(f"Errors: {len(errors)}  Warnings: {len(warnings)}")

    if errors:
        print("\n[ERRORS]")
        for e in errors[:50]:
            print("  " + e)
        if len(errors) > 50:
            print(f"  ... ({len(errors)-50} more)")

    if warnings:
        print("\n[WARNINGS]")
        for w in warnings[:50]:
            print("  " + w)
        if len(warnings) > 50:
            print(f"  ... ({len(warnings)-50} more)")

    return {
        "ok": ok,
        "groups": len(groups),
        "records": total_records,
        "parse_errors": parse_errors,
        "errors": errors,
        "warnings": warnings,
    }

# def _apply_one_pull_update_for_banner(
#     banner: str,
#     tok: int,
#     lto4: List[int],
#     st: Dict[str, Any],
#     use_epitomized: bool,
# ) -> None:
#     # Update guarantee_featured_5 (your inference-style logic)
#     apply_guarantee_inference(banner, tok, lto4, st)

#     # Update pity counters
#     rarity = rarity_from_token(tok)
#     update_pity_after_pull(st, rarity)

#     # Weapon epitomized fate points logic (only if target set)
#     if use_epitomized and banner == "weapon" and rarity == 5:
#         tgt = int(st.get("epitomized_target", 0) or 0)
#         if tgt != 0:
#             if tok == tgt:
#                 st["fate_points"] = 0
#             else:
#                 st["fate_points"] = min(2, int(st.get("fate_points", 0)) + 1)

# def validate_trace_state_consistency(
#     trace_jsonl_path: str,
#     lto4: List[int],
#     eos_prod_id: int = EOS_PROD_ID_LOCAL,
#     sos_prod_id: int = SOS_PROD_ID_LOCAL,
#     hard_pity5_map: Optional[Dict[str, int]] = None,
#     hard_pity4: int = 10,
# ) -> Dict[str, Any]:
#     if hard_pity5_map is None:
#         hard_pity5_map = {"regular": 90, "figure_a": 90, "figure_b": 90, "weapon": 80}

#     # group records by (uid, run)
#     groups = defaultdict(list)
#     total_records = 0
#     parse_errors = 0

#     with open(trace_jsonl_path, "rt") as f:
#         for ln, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rec = json.loads(line)
#             except Exception:
#                 parse_errors += 1
#                 continue
#             if int(rec.get("step", -1)) != 2:
#                 continue
#             uid = rec.get("uid", "")
#             run = int(rec.get("run", 0))
#             t = int(rec.get("t", -1))
#             groups[(uid, run)].append(rec)
#             total_records += 1

#     errors: List[str] = []
#     warnings: List[str] = []

#     STATE_KEYS = ["pity5","pity4","guarantee_featured_5","guarantee_featured_4","fate_points","epitomized_target"]

#     for (uid, run), recs in groups.items():
#         recs.sort(key=lambda r: int(r["t"]))

#         # ---- Cross-record checks: t monotonic + prev_dec chaining ----
#         for i, r in enumerate(recs):
#             t = int(r["t"])
#             if t != i:
#                 warnings.append(f"uid={uid} run={run}: non-contiguous t (expected {i}, got {t}); trace_max_steps may truncate.")

#         for i in range(len(recs) - 1):
#             cur = recs[i]
#             nxt = recs[i + 1]

#             cur_prev = int(cur["prev_dec"])
#             cur_sampled = int(cur["sampled_dec"])
#             nxt_prev = int(nxt["prev_dec"])

#             # Once you stop (sampled_dec == 9), generation breaks, so there should be no next record.
#             if cur_sampled == 9:
#                 errors.append(f"uid={uid} run={run}: sampled_dec==9 at t={cur['t']} but trace continues at t={nxt['t']}.")
#                 break

#             if nxt_prev != cur_sampled:
#                 errors.append(
#                     f"uid={uid} run={run}: decision chaining mismatch: "
#                     f"t={cur['t']} sampled_dec={cur_sampled} but t={nxt['t']} prev_dec={nxt_prev}"
#                 )

#         # ---- Per-record checks: mechanics ----
#         for r in recs:
#             t = int(r["t"])
#             prev_dec = int(r["prev_dec"])
#             out10 = [int(x) for x in r["out10"]]

#             # EOS/SOS in OUT10 rules
#             if sos_prod_id in out10:
#                 errors.append(f"uid={uid} run={run} t={t}: SOS appears in OUT10={out10}")

#             if prev_dec == 9:
#                 expected = [eos_prod_id] + [0]*9
#                 if out10 != expected:
#                     errors.append(f"uid={uid} run={run} t={t}: prev_dec=9 but OUT10 invalid: got={out10} expected={expected}")
#             else:
#                 if eos_prod_id in out10:
#                     errors.append(f"uid={uid} run={run} t={t}: prev_dec={prev_dec} but EOS appears in OUT10={out10}")

#             # outcome_source sanity
#             src = r.get("outcome_source", "")
#             if t == 0 and src != "history_c27_lastblock":
#                 warnings.append(f"uid={uid} run={run} t=0: outcome_source={src} (expected history_c27_lastblock)")
#             if t >= 1 and src != "simulated":
#                 warnings.append(f"uid={uid} run={run} t={t}: outcome_source={src} (expected simulated)")

#             # t==0: you intentionally do not update states; verify state_before == state_after
#             if t == 0:
#                 if r["state_before"] != r["state_after"]:
#                     errors.append(f"uid={uid} run={run} t=0: state_before != state_after (should be unchanged at t=0)")
#                 continue

#             banner, n_pulls = decision_to_banner_and_pulls(prev_dec)

#             # For prev_dec==9, n_pulls=0. Your schema still has OUT10=[EOS,0..]
#             # and states should not change.
#             if prev_dec == 9 or banner == "none" or n_pulls == 0:
#                 if r["state_before"] != r["state_after"]:
#                     errors.append(f"uid={uid} run={run} t={t}: prev_dec=9 but states changed")
#                 continue

#             sb_all = r["state_before"]
#             sa_all = r["state_after"]

#             if banner not in sb_all or banner not in sa_all:
#                 errors.append(f"uid={uid} run={run} t={t}: missing banner={banner} in state_before/after")
#                 continue

#             # Unaffected banners should not change
#             for bname in sb_all.keys():
#                 if bname == banner:
#                     continue
#                 if sb_all[bname] != sa_all[bname]:
#                     errors.append(f"uid={uid} run={run} t={t}: banner={banner} changed state of unrelated banner={bname}")

#             # Recompute expected after-state for the affected banner
#             st = dict(sb_all[banner])
#             use_epitomized = (banner == "weapon" and int(st.get("epitomized_target", 0) or 0) != 0)

#             for k in range(n_pulls):
#                 tok = int(out10[k])
#                 if tok == 0:
#                     # missingness; if you don't expect this, treat as error
#                     warnings.append(f"uid={uid} run={run} t={t}: tok==0 within realized pulls (k={k}) out10={out10}")
#                     continue
#                 if tok in (eos_prod_id, sos_prod_id):
#                     errors.append(f"uid={uid} run={run} t={t}: special token {tok} in realized pulls out10={out10}")
#                     continue

#                 _apply_one_pull_update_for_banner(banner, tok, lto4, st, use_epitomized)

#             expected_after = st
#             got_after = sa_all[banner]

#             if not _state_equal_dict(expected_after, got_after, STATE_KEYS):
#                 errors.append(
#                     f"uid={uid} run={run} t={t}: state mismatch banner={banner}. "
#                     f"expected={expected_after} got={got_after}"
#                 )

#             # Feasibility checks
#             hard5 = int(hard_pity5_map.get(banner, 90))
#             if int(got_after.get("pity5", 0)) >= hard5:
#                 errors.append(f"uid={uid} run={run} t={t}: pity5 infeasible banner={banner} pity5={got_after.get('pity5')} hard={hard5}")
#             if int(got_after.get("pity4", 0)) >= hard_pity4:
#                 errors.append(f"uid={uid} run={run} t={t}: pity4 infeasible banner={banner} pity4={got_after.get('pity4')} hard={hard_pity4}")

#     ok = (len(errors) == 0)

#     print("\n=== TRACE STATE CONSISTENCY VALIDATION ===")
#     print(f"Trace file: {trace_jsonl_path}")
#     print(f"Groups (uid,run): {len(groups)}")
#     print(f"Records checked: {total_records}")
#     print(f"Parse errors: {parse_errors}")
#     print(f"OK: {ok}")
#     print(f"Errors: {len(errors)}  Warnings: {len(warnings)}")

#     if errors:
#         print("\n[ERRORS]")
#         for e in errors[:50]:
#             print("  " + e)
#         if len(errors) > 50:
#             print(f"  ... ({len(errors)-50} more)")

#     if warnings:
#         print("\n[WARNINGS]")
#         for w in warnings[:50]:
#             print("  " + w)
#         if len(warnings) > 50:
#             print(f"  ... ({len(warnings)-50} more)")

#     return {"ok": ok, "groups": len(groups), "records": total_records, "errors": errors, "warnings": warnings}

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
    use_epitomized: bool = False,
    epitomized_target: int = 0,
    trace: Optional[TraceCfg] = None,   
    trace_enabled: bool = False,
    trace_max_steps: int = 0,
    uid: str = "",
    run_id: int = 0,
    run_seed: Optional[int] = None,
    trace_fout: Optional[TextIO] = None,
) -> Dict[str, Any]:
    """
    Campaign 28 generation with simulated outcomes (Step 2a+2b).

    t=0 uses OUT10 from last observed block in campaign 27 (already reflected in inferred states).
    For t>=1, OUT10 is simulated from the previous step's sampled decision.
    """
    assert len(lto28_tokens) == 4
    assert len(history_tokens) % AI_RATE == 0

    model.eval()

    states = infer_states_from_history(history_tokens)

    # set epitomized target (optional)
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
            state_after  = copy.deepcopy(states)  # no update on t=0 if you treat it as observed
            outcome_source = "history_c27_lastblock"
            banner = DECISION_META.get(prev_dec, ("UNK", "unknown", 0))[1]
            pulls  = DECISION_META.get(prev_dec, ("UNK", "unknown", 0))[2]

            # IMPORTANT: do NOT update states here; this outcome is already in history
        else:
            state_before = copy.deepcopy(states)

            out10, info = simulate_outcomes_for_decision(
                decision=prev_dec,
                lto4=lto28_tokens,
                states=states,
                rng=rng,
                use_epitomized=use_epitomized,
            )

            state_after = copy.deepcopy(states)
            outcome_source = "simulated"
            banner = info.get("banner", DECISION_META.get(prev_dec, ("UNK","unknown",0))[1])
            pulls  = info.get("pulls",  DECISION_META.get(prev_dec, ("UNK","unknown",0))[2])

        block = list(lto28_tokens) + list(out10) + [prev_dec]
        seq_full.extend(block)
        seq_c28.extend(block)

        x = torch.tensor(seq_full, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        logits_at_decpos = logits[0, -1, :]

        dec = sample_decision_from_logits(
            logits_at_decpos, DECISION_IDS, temperature=temperature, greedy=greedy
        )

        # Trace print (limit steps if requested)
        do_trace = bool(trace and trace.enabled and (trace.max_steps == 0 or t < trace.max_steps))
        if do_trace:
            prev_name, _, _ = DECISION_META.get(prev_dec, ("UNK", "unknown", 0))
            dec_name,  _, _ = DECISION_META.get(dec,      ("UNK", "unknown", 0))

            rec = {
                "uid": uid,
                "step": 2,
                "run": run_id,
                "run_seed": run_seed,
                "t": t,

                # decision that generated OUT10
                "prev_dec": prev_dec,
                "prev_dec_name": prev_name,
                "banner": banner,
                "n_pulls": pulls,
                "outcome_source": outcome_source,

                # outcomes written into [OUT10]
                "out10": out10,

                # gacha states
                "state_before": state_before,
                "state_after": state_after,

                # decision sampled for next step
                "sampled_dec": dec,
                "sampled_dec_name": dec_name,
            }
            # _trace_emit(rec, do_print=True, fout=trace_fout)
            # _trace_emit(rec, do_print=True, fout=None)
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


# Global banner cfgs (needed in a few helper functions)
cfgs = make_banner_cfgs()


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = get_config()
    parser = argparse.ArgumentParser()

    parser.add_argument("--step", type=int, choices=[1, 2], default=2,
                        help="1: fixed outcomes after t=0; 2: infer counters + simulate outcomes")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat generation multiple times for the selected consumer(s).")
    parser.add_argument("--seed_base", type=int, default=None,
                        help="Base seed; run r uses seed_base + r (useful to demonstrate randomness).")

    parser.add_argument("--uid", type=str, default=None,
                        help="Run inference for a single consumer uid only (exact match).")
    parser.add_argument("--first", action="store_true",
                        help="Run inference for the first consumer only (overrides --n_users).")
    parser.add_argument("--n_users", type=int, default=0,
                        help="Run inference for the first N consumers, then stop. 0 means no limit.")

    parser.add_argument("--data", default=cfg["filepath"], help="JSON/JSONL(.gz) with Campaigns 1–27 sequences")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    parser.add_argument("--out", required=True, help="Output JSONL(.gz) path")

    parser.add_argument("--lto28", nargs=4, type=int, required=True, metavar=("a", "b", "c", "d"),
                        help="Campaign 28 firm action tokens (4 ints): [figA, figB, wep1, wep2]")
    parser.add_argument("--fixed_outcomes", nargs=10, type=int, default=[0] * 10,
                        help="Step=1 only: 10 frozen outcome tokens used for Campaign 28 steps >= 1")

    parser.add_argument("--max_steps28", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--init_prev_dec", type=int, default=None)

    # weapon banner epitomized path (optional)
    parser.add_argument("--use_epitomized", action="store_true",
                        help="Enable epitomized path logic for weapon banner (requires --epitomized_target).")
    parser.add_argument("--epitomized_target", type=int, default=0,
                        help="Selected target weapon token id (must be one of the two weapon LTOs).")

    parser.add_argument("--feat_xlsx", type=str,
                        default="/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")

    # output verbosity
    parser.add_argument("--print_tokens", action="store_true",
                        help="Also output Campaign28_AggregateInput and Full_AggregateInput (very long).")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print JSON lines to stdout (file output still written).")

    parser.add_argument("--validate", action="store_true",
                        help="Validate AggregateInput vs Decision and report anomalies.")
    parser.add_argument("--validate_users", type=int, default=50)
    parser.add_argument("--validate_last_k", type=int, default=8)
    parser.add_argument("--validate_trace_states", action="store_true",
                        help="Validate Step-2 trace file (requires --trace_out and --trace).")

    # trace
    parser.add_argument("--trace_triplets", action="store_true",
                    help="Print generated triplets as [LTO4] [OUT10] [DEC1] line by line.")
    parser.add_argument("--trace", action="store_true",
                        help="Print a per-step trace (decision/outcomes/state) to stdout.")
    parser.add_argument("--trace_out", type=str, default=None,
                        help="Optional path to also write trace JSONL.")
    parser.add_argument("--trace_max_steps", type=int, default=200,
                        help="Max number of steps to trace (to avoid huge logs). 0 = no limit.")

    args = parser.parse_args()

    # trace_enabled = bool(args.trace)
    # trace_max_steps = int(args.trace_max_steps)

    if args.validate:
        batch_validate(args.data, n_users=args.validate_users, k_last=args.validate_last_k)
        return

    if args.first:
        args.n_users = 1

    trace_fout = None
    if args.trace_out is not None:
        Path(args.trace_out).parent.mkdir(parents=True, exist_ok=True)
        trace_fout = open(args.trace_out, "wt")

    trace_cfg = TraceCfg(enabled=bool(args.trace), max_steps=int(args.trace_max_steps), fout=trace_fout)

    # Feature columns (keep exactly aligned with training)
    FEATURE_COLS = [
        "Rarity", "MaxLife", "MaxOffense", "MaxDefense",
        "WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow", "WeaponTypeMagic", "WeaponTypePolearm",
        "EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire", "EthnicityThunder", "EthnicityWind",
        "GenderFemale", "GenderMale",
        "CountryRuiYue", "CountryDaoQi", "CountryZhiDong", "CountryMengDe",
        "type_figure",
        "MinimumAttack", "MaximumAttack",
        "MinSpecialEffect", "MaxSpecialEffect",
        "SpecialEffectEfficiency", "SpecialEffectExpertise", "SpecialEffectAttack", "SpecialEffectSuper",
        "SpecialEffectRatio", "SpecialEffectPhysical", "SpecialEffectLife",
        "LTO",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load feature tensor ----
    feature_tensor = load_feature_tensor(Path(args.feat_xlsx), FEATURE_COLS, cfg["vocab_size_src"])

    # ---- special tokens (match training) ----
    PAD_ID_LOCAL = 0
    SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
    EOS_PROD_ID_LOCAL, SOS_PROD_ID_LOCAL, UNK_PROD_ID = 57, 58, 59
    SPECIAL_IDS = [PAD_ID_LOCAL, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID_LOCAL, SOS_PROD_ID_LOCAL]

    # ---- load checkpoint, infer shapes ----
    state = torch.load(args.ckpt, map_location=device)
    if "model_state_dict" in state:
        raw_sd = state["model_state_dict"]
    elif "module" in state:
        raw_sd = state["module"]
    else:
        raw_sd = state
    state_dict = clean_state_dict(raw_sd)

    n_layers_ckpt, max_seq_len_ckpt = infer_ckpt_shapes(
        state_dict,
        fallback_layers=cfg["N"],
        fallback_pe=cfg.get("seq_len_tgt", 1024),
    )

    # ---- build model to match checkpoint ----
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

    # ---- output file ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out_path.suffix == ".gz" else open

    processed = 0

    with opener(out_path, "wt") as fout:
        for row in _iter_rows(args.data):
            uid = row["uid"][0] if isinstance(row.get("uid"), list) else row.get("uid")

            # selection
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

            # optional: infer init_prev_dec from row["Decision"]
            init_prev_dec = args.init_prev_dec
            if init_prev_dec is None and "Decision" in row:
                try:
                    decs = parse_int_sequence(row["Decision"])
                    if len(decs) > 0:
                        init_prev_dec = int(decs[-1])
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
                    )
                    payload = {
                        "uid": uid,
                        "step": 1,
                        "run": r,
                        "run_seed": run_seed,
                        "Campaign28_Decisions": out["decisions28"],
                        "stopped": out["stopped"],
                        "stop_step": out["stop_step"],
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
                        trace=trace_cfg,
                        uid=uid,
                        run_id=r,
                        run_seed=run_seed,
                        trace_fout=trace_fout,
                    )

                    val_cfg = GenValidationCfg(
                        require_lto4_match=True,
                        strict_buy10_full=True,
                        buy10_missing_is_error=True,
                    )

                    # Validate generated seq_campaign28 in-memory
                    _ = validate_seq_campaign28(
                        seq_campaign28=out["seq_campaign28"],
                        expected_lto4=list(args.lto28),
                        cfg=val_cfg,
                    )

                    payload = {
                        "uid": uid,
                        "step": 2,
                        "run": r,
                        "run_seed": run_seed,
                        "Campaign28_Decisions": out["decisions28"],
                        "stopped": out["stopped"],
                        "stop_step": out["stop_step"],
                        "final_states": out["final_states"],
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

    if args.trace_out is not None and not args.trace:
        raise ValueError("--trace_out requires --trace to be enabled")

    if args.validate_trace_states:
        if args.step != 2:
            raise ValueError("--validate_trace_states requires --step 2")
        if args.trace_out is None:
            raise ValueError("--validate_trace_states requires --trace_out")
        # Ensure trace was actually written
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
