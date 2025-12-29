#!/usr/bin/env python3
import argparse
import gzip
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TextIO

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config4 import get_config
from model4_decoderonly_feature_performer import build_transformer  # adjust if needed

import copy

def _trace_emit(
    rec: dict,
    do_print: bool = True,
    fout: Optional[TextIO] = None,
):
    line = json.dumps(rec, ensure_ascii=False)
    if do_print:
        print(line, flush=True)
    if fout is not None:
        fout.write(line + "\n")
        fout.flush()


# ----------------------------
# Constants / Token meanings
# ----------------------------
AI_RATE = 15
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
) -> List[int]:
    """
    Produce OUT10 for the decision (0/1/10 pulls), enforcing feasibility:
      - 0 pulls: all zeros
      - 1 pull: exactly 1 nonzero then zeros
      - 10 pulls: up to 10 nonzero
    """
    banner, n_pulls = decision_to_banner_and_pulls(decision)
    out10 = [0] * 10

    if n_pulls == 0 or banner == "none":
        return out10

    if banner not in states:
        # unknown banner -> treat as no outcomes
        return out10

    st = states[banner]
    cfg = cfgs[banner]

    for k in range(n_pulls):
        rarity = sample_rarity_one_pull(cfg, st, rng)
        tok = sample_outcome_token(banner, rarity, lto4, st, rng, use_epitomized=use_epitomized)
        out10[k] = int(tok)
        update_pity_after_pull(st, rarity)

    return out10


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
            tok = int(out10[k])
            if tok == 0:
                # if data contains 0 even when supposed to have a pull, treat as missing
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
    # ---- add these ----
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
        if trace_enabled and (trace_max_steps == 0 or t < trace_max_steps):
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

    parser.add_argument("--trace", action="store_true",
                        help="Print a per-step trace (decision/outcomes/state) to stdout.")
    parser.add_argument("--trace_out", type=str, default=None,
                        help="Optional path to also write trace JSONL.")
    parser.add_argument("--trace_max_steps", type=int, default=200,
                        help="Max number of steps to trace (to avoid huge logs). 0 = no limit.")

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

    args = parser.parse_args()

    trace_enabled = bool(args.trace)
    trace_max_steps = int(args.trace_max_steps)

    if args.first:
        args.n_users = 1

    trace_fout = None
    if args.trace_out is not None:
        Path(args.trace_out).parent.mkdir(parents=True, exist_ok=True)
        trace_fout = open(args.trace_out, "wt")

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

                        # ---- add these ----
                        trace_enabled=trace_enabled,
                        trace_max_steps=trace_max_steps,
                        uid=uid,
                        run_id=r,
                        run_seed=run_seed,
                        trace_fout=trace_fout,
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

    if processed == 0:
        raise ValueError("No matching consumer found. Check --uid or input data.")

    if args.quiet:
        print(f"[DONE] Wrote outputs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
