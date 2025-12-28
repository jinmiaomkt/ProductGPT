#!/usr/bin/env python3
import argparse
import gzip
import json
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# -------------------------
# Your project imports
# -------------------------
from config4_index_git import get_config
from model4_decoderonly_feature_performer import build_transformer  # adjust if your module name differs


AI_RATE = 15
DECISION_IDS = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)


def smart_open(path: str):
    if path == "-":
        return nullcontext(torch.sys.stdout)  # not used typically
    if path.endswith(".gz"):
        return gzip.open(path, "wt")
    return open(path, "w")


def _iter_rows(path: str) -> Iterable[Dict[str, Any]]:
    """
    Supports:
      - JSONL (one JSON object per line)
      - JSON array file (a list of objects)
    """
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


def parse_int_sequence(val: Any) -> List[int]:
    """
    Handles:
      - string: "1 2 3"
      - list of ints
      - list with one string element: ["1 2 3"]
    """
    if isinstance(val, list):
        if len(val) == 1 and isinstance(val[0], str):
            return [int(x) for x in val[0].strip().split()]
        # list of ints/strings
        out = []
        for x in val:
            if isinstance(x, (int, np.integer)):
                out.append(int(x))
            elif isinstance(x, str) and x.strip():
                out.extend([int(t) for t in x.strip().split()])
        return out
    if isinstance(val, str):
        return [int(x) for x in val.strip().split()] if val.strip() else []
    raise TypeError(f"Unsupported sequence type: {type(val)}")


def extract_last_outcomes_from_history(history_tokens: List[int], ai_rate: int = AI_RATE) -> List[int]:
    """
    From the last complete 15-token block in history:
      block = [LTO4][OUT10][PREV_DEC1]
    outcomes are positions 4..13 (0-based within block).
    """
    assert len(history_tokens) >= ai_rate and len(history_tokens) % ai_rate == 0, \
        "history_tokens must be non-empty and a multiple of 15."
    last_block = history_tokens[-ai_rate:]
    return last_block[4:14]  # length 10


@torch.no_grad()
def sample_decision_from_logits(
    logits_last_pos: torch.Tensor,      # (V,)
    decision_ids: torch.Tensor,         # (9,)
    temperature: float = 1.0,
    greedy: bool = False,
) -> int:
    device = logits_last_pos.device
    ids = decision_ids.to(device)
    dec_logits = logits_last_pos[ids]  # (9,)

    if temperature != 1.0:
        dec_logits = dec_logits / max(temperature, 1e-8)

    probs = F.softmax(dec_logits, dim=-1)

    if greedy:
        k = torch.argmax(probs).item()
    else:
        k = torch.multinomial(probs, 1).item()

    return int(ids[k].item())


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
) -> Dict[str, Any]:
    assert len(lto28_tokens) == 4, "lto28_tokens must have length 4."
    assert len(fixed_outcomes_after_step0) == 10, "fixed_outcomes_after_step0 must have length 10."
    assert len(history_tokens) >= AI_RATE and len(history_tokens) % AI_RATE == 0, \
        "history_tokens must be non-empty and a multiple of 15."

    model.eval()

    outcomes_step0 = extract_last_outcomes_from_history(history_tokens)

    # Fallback: last slot token in history (previous decision token for the last observed block).
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


def load_feature_tensor(xls_path: Path, feature_cols: List[str], vocab_size_src: int) -> torch.Tensor:
    """
    Returns (vocab_size_src, feat_dim) float32.
    Fills rows for token_id = NewProductIndex6 when within range.
    """
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
    return {strip_prefix(k): v for k, v in raw.items()
            if not any(tok in k for tok in ignore)}


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=cfg["filepath"], help="JSON/JSONL(.gz) with Campaigns 1–27 sequences")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    parser.add_argument("--out", required=True, help="Output JSONL(.gz) path")

    # Campaign 28 inputs
    parser.add_argument("--lto28", nargs=4, type=int, required=True, metavar=("a", "b", "c", "d"),
                        help="Campaign 28 firm action tokens (4 ints)")

    # Step 1 frozen outcomes for steps>=1
    parser.add_argument("--fixed_outcomes", nargs=10, type=int, default=[0]*10,
                        help="10 frozen outcome tokens used for Campaign 28 steps >= 1")

    # generation controls
    parser.add_argument("--max_steps28", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")

    # warm-start previous decision for step0 (recommended if you can supply it)
    parser.add_argument("--init_prev_dec", type=int, default=None,
                        help="Warm-start previous decision token for Campaign 28 step0. "
                             "If omitted, falls back to last token of history.")

    # feature tensor
    parser.add_argument("--feat_xlsx", type=str, default="/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")

    args = parser.parse_args()

    # Feature columns (keep consistent with training)
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

    # Ensure positional encoding can cover history + rollout
    # Your PositionalEncoding can extend dynamically, but we still pass a safe max_seq_len.
    # For safety, set it to at least current window or a generous upper bound.
    max_seq_len = max(cfg.get("seq_len_ai", 0), 20000)

    feature_tensor = load_feature_tensor(Path(args.feat_xlsx), FEATURE_COLS, cfg["vocab_size_src"])

    # Special tokens list should match training; use your training definition.
    # If you used different special IDs, update this list accordingly.
    PAD_ID = 0
    SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
    EOS_PROD_ID, SOS_PROD_ID, UNK_PROD_ID = 57, 58, 59
    SPECIAL_IDS = [PAD_ID, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID, SOS_PROD_ID]

    model = build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        d_model=cfg["d_model"],
        n_layers=cfg["N"],
        n_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=0.0,  # inference
        nb_features=cfg["nb_features"],
        max_seq_len=max_seq_len,
        kernel_type=cfg["kernel_type"],
        feature_tensor=feature_tensor,
        special_token_ids=SPECIAL_IDS,
    ).to(device).eval()

    # Load checkpoint
    state = torch.load(args.ckpt, map_location=device)
    if "model_state_dict" in state:
        raw_sd = state["model_state_dict"]
    elif "module" in state:
        raw_sd = state["module"]
    else:
        raw_sd = state
    model.load_state_dict(clean_state_dict(raw_sd), strict=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Run generation per user
    opener = gzip.open if out_path.suffix == ".gz" else open
    with opener(out_path, "wt") as fout:
        for row in _iter_rows(args.data):
            uid = row["uid"][0] if isinstance(row.get("uid"), list) else row.get("uid")

            history_tokens = parse_int_sequence(row["AggregateInput"])

            # Optional: override init_prev_dec using your "Decision" field if present
            init_prev_dec = args.init_prev_dec
            if init_prev_dec is None and "Decision" in row:
                try:
                    decs = parse_int_sequence(row["Decision"])
                    if len(decs) > 0:
                        init_prev_dec = int(decs[-1])  # last actual decision observed
                except Exception:
                    pass

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
            )

            payload = {
                "uid": uid,
                "Campaign28_AggregateInput": " ".join(map(str, out["seq_campaign28"])),
                "Full_AggregateInput": " ".join(map(str, out["seq_full"])),
                "Campaign28_Decisions": " ".join(map(str, out["decisions28"])),
                "stopped": out["stopped"],
                "stop_step": out["stop_step"],
            }
            fout.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
