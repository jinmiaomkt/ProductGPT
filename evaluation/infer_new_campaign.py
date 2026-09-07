#!/usr/bin/env python3
import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config4 import get_config
from model4_decoderonly_feature_performer import build_transformer  # adjust if needed

AI_RATE = 15
DECISION_IDS = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)

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

    def tok_to_int(t: Any) -> Optional[int]:
        if t is None:
            return na_to
        if isinstance(t, (int, np.integer)):
            return int(t)
        if isinstance(t, float):
            # if somebody wrote numeric NA as float nan
            if np.isnan(t):
                return na_to
            return int(t)
        if isinstance(t, str):
            s = t.strip()
            if s in NA_STRINGS:
                return na_to
            # Sometimes tokens may be like "13.0" if exported poorly
            try:
                return int(s)
            except ValueError:
                try:
                    return int(float(s))
                except Exception:
                    # Last resort: treat as missing
                    return na_to
        # unknown type -> treat as missing
        return na_to

    tokens: List[int] = []

    if isinstance(val, list):
        # common case: ["1 2 3 ..."]
        if len(val) == 1 and isinstance(val[0], str):
            parts = val[0].strip().split()
            for p in parts:
                tokens.append(tok_to_int(p))
            return tokens

        # list of mixed entries
        for x in val:
            if isinstance(x, str):
                # could be "1 2 3" chunk or single token "NA"
                parts = x.strip().split()
                if len(parts) == 0:
                    continue
                for p in parts:
                    tokens.append(tok_to_int(p))
            else:
                tokens.append(tok_to_int(x))
        return tokens

    if isinstance(val, str):
        parts = val.strip().split()
        for p in parts:
            tokens.append(tok_to_int(p))
        return tokens

    raise TypeError(f"Unsupported sequence type: {type(val)}")

def extract_last_outcomes_from_history(history_tokens: List[int], ai_rate: int = AI_RATE) -> List[int]:
    """Last block = [LTO4][OUT10][PREV_DEC1]; outcomes are positions 4..13 (0-based)."""
    assert len(history_tokens) >= ai_rate and len(history_tokens) % ai_rate == 0, \
        "history_tokens must be non-empty and a multiple of 15."
    last_block = history_tokens[-ai_rate:]
    return last_block[4:14]

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


def infer_ckpt_shapes(state_dict: Dict[str, torch.Tensor], fallback_layers: int, fallback_pe: int) -> tuple[int, int]:
    # layers
    layer_idxs = []
    for k in state_dict.keys():
        if k.startswith("decoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_idxs.append(int(parts[2]))
    n_layers = (max(layer_idxs) + 1) if layer_idxs else fallback_layers

    # pe length
    if "pos_enc.pe" in state_dict:
        pe_len = int(state_dict["pos_enc.pe"].shape[1])
    else:
        pe_len = fallback_pe

    return n_layers, pe_len


def main():
    cfg = get_config()

    parser = argparse.ArgumentParser()

    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat generation multiple times for the selected consumer(s).")
    parser.add_argument("--seed_base", type=int, default=None,
                        help="Base seed; run r uses seed_base + r.")

    parser.add_argument("--uid", type=str, default=None, help="Run inference for a single consumer uid only (exact match).")
    parser.add_argument("--first", action="store_true", help="Run inference for the first consumer in the data file.")
    parser.add_argument("--n_users", type=int, default=0, help="Run inference for the first N consumers, then stop. 0 means no limit.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")

    parser.add_argument("--data", default=cfg["filepath"], help="JSON/JSONL(.gz) with Campaigns 1–27 sequences")
    parser.add_argument("--ckpt", required=True, help="Checkpoint .pt path")
    parser.add_argument("--out", required=True, help="Output JSONL(.gz) path")

    parser.add_argument("--lto28", nargs=4, type=int, required=True, metavar=("a", "b", "c", "d"),
                        help="Campaign 28 firm action tokens (4 ints)")
    parser.add_argument("--fixed_outcomes", nargs=10, type=int, default=[0] * 10,
                        help="10 frozen outcome tokens used for Campaign 28 steps >= 1")

    parser.add_argument("--max_steps28", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--init_prev_dec", type=int, default=None)

    parser.add_argument("--feat_xlsx", type=str,
                        default="/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")

    # printing controls
    parser.add_argument("--print", dest="do_print", action="store_true",
                        help="Print each output JSON line to stdout")
    parser.add_argument("--print_tokens", action="store_true",
                        help="Also print generated token streams (very long)")

    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

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
    PAD_ID = 0
    SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
    EOS_PROD_ID, SOS_PROD_ID, UNK_PROD_ID = 57, 58, 59
    SPECIAL_IDS = [PAD_ID, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID, SOS_PROD_ID]

    # ---- load checkpoint first, infer shapes ----
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
        max_seq_len=max_seq_len_ckpt,   # match ckpt; PE can extend dynamically later
        kernel_type=cfg["kernel_type"],
        feature_tensor=feature_tensor,
        special_token_ids=SPECIAL_IDS,
    ).to(device).eval()

    model.load_state_dict(state_dict, strict=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out_path.suffix == ".gz" else open

    # ---- main loop ----
    with opener(out_path, "wt") as fout:

        processed = 0
        for row in _iter_rows(args.data):
            uid = row["uid"][0] if isinstance(row.get("uid"), list) else row.get("uid")

            # ----- selection logic -----
            if args.uid is not None and uid != args.uid:
                continue
            if args.first and processed > 0:
                break
            
            history_tokens = parse_int_sequence(row["AggregateInput"], na_to=0)

            # (optional) strict alignment check
            if len(history_tokens) % AI_RATE != 0:
                raise ValueError(f"uid={uid}: history length {len(history_tokens)} not divisible by {AI_RATE}")

            vmax = cfg["vocab_size_src"] - 1

            bad = [t for t in history_tokens if not (0 <= t <= vmax)]
            if bad:
                raise ValueError(f"Found out-of-range token(s) for uid={uid}: examples={bad[:20]} (vocab max={vmax})")

            if len(history_tokens) % AI_RATE != 0:
                raise ValueError(f"uid={uid}: history length {len(history_tokens)} not divisible by {AI_RATE}")

            init_prev_dec = args.init_prev_dec
            if init_prev_dec is None and "Decision" in row:
                try:
                    decs = parse_int_sequence(row["Decision"])
                    if len(decs) > 0:
                        init_prev_dec = int(decs[-1])
                except Exception:
                    pass

            # out = generate_campaign28_step1_fixed_outcomes(
            #     model=model,
            #     history_tokens=history_tokens,
            #     lto28_tokens=args.lto28,
            #     fixed_outcomes_after_step0=args.fixed_outcomes,
            #     device=device,
            #     init_prev_dec=init_prev_dec,
            #     max_steps28=args.max_steps28,
            #     stop_decision=9,
            #     temperature=args.temperature,
            #     greedy=args.greedy,
            # )

            # payload = {
            #     "uid": uid,
            #     "Campaign28_Decisions": out["decisions28"],
            #     "stopped": out["stopped"],
            #     "stop_step": out["stop_step"],
            # }

            # if args.print_tokens:
            #     payload["Campaign28_AggregateInput"] = out["seq_campaign28"]
            #     payload["Full_AggregateInput"] = out["seq_full"]

            # line = json.dumps(payload)

            # # always write to file
            # fout.write(line + "\n")

            repeat = max(1, args.repeat)

            for r in range(repeat):
                # Per-run seed to demonstrate randomness
                if args.seed_base is not None:
                    run_seed = args.seed_base + r
                    import random
                    random.seed(run_seed)
                    np.random.seed(run_seed)
                    torch.manual_seed(run_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(run_seed)
                else:
                    run_seed = None

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
                    "run": r,
                    "run_seed": run_seed,
                    "Campaign28_Decisions": out["decisions28"],
                    "stopped": out["stopped"],
                    "stop_step": out["stop_step"],
                }

                line = json.dumps(payload)
                fout.write(line + "\n")
                print(line, flush=True)

            # optionally print to screen
            if args.do_print:
                print(line, flush=True)

            processed += 1

                # ----- stopping logic -----
            if args.uid is not None:
                break
            if args.n_users and processed >= args.n_users:
                break
            if args.first:
                break
        if processed == 0:
            raise ValueError("No matching consumer found. Check --uid or input data.")

    # final message (useful when --print is off)
    if not args.do_print:
        print(f"[DONE] Wrote outputs to {args.out}", flush=True)

if __name__ == "__main__":
    main()
