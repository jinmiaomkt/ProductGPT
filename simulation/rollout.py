"""R36a: roll a trained ProductGPT forward on a fixed occasion grid.

WHAT THIS DOES.  Every result up to R35 is one-step-ahead: given a TRUE history,
how well is the next decision predicted.  This rolls the model forward on its
OWN outputs over the holdout campaigns, so the sequences it produces can be
compared to the real ones as sequences.

FIXED GRID, AND WHY.  The model consumes IPT as a recency ruler and never
predicts it; its head emits only the 9-way decision.  It therefore cannot
generate the occasion grid, so the grid is held at its observed values and only
the decisions are generated.  What this tests is the COMPOSITION and DEPENDENCE
STRUCTURE of decisions given when occasions arrive -- not purchase timing or
frequency.  That limitation is real and belongs in the paper; removing it needs
a duration head (R36e), which is a modelling change, not an evaluation change.

Holding the grid is legitimate here for a specific reason: `is_inserted` is
carried in the batch but consumed by neither the model nor the trainer, so
conditioning on the real row grid does not hand the model its label.  Verified
by grep over gen5_multistream/ and shared/; asserted in scripts/test_rollout.py.

WHAT FEEDS BACK.  Two streams, and only two:
    prev_decision[t+1] <- the sampled decision at t
    obtained[t+1]      <- what the environment says that decision yielded
The inventory, duplicate tiers and every satiation term are pure functions of
`obtained`, so those follow automatically.  The offer stream is exogenous and
stays at its real values.

SAMPLING.  Ancestral, at temperature 1, with no top-k and no nucleus. Truncated
decoding changes the distribution being tested, which is the object of the
exercise.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "shared"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

N_CLASSES = 9
NOT_BUY = 9


# --------------------------------------------------------------- model load
def load_checkpoint(ckpt_path, device: str = "cpu", *, vocab_level: Optional[int] = None):
    """Rebuild the trained model from a run's best.pt.

    The checkpoint carries `cfg` and `num_users`, so the architecture is
    reconstructed exactly rather than guessed from flags.
    """
    import config5
    import dataset_multistream as dsm
    import model_multistream_state_space as mm
    from shared.features import load_feature_tensor

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    if vocab_level is not None:
        config5.apply_vocab_level(cfg, vocab_level)
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    mm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"], cfg["unk_prod_id"])

    feat = load_feature_tensor(config5.feature_file_path(cfg),
                               id_column=cfg.get("feature_id_column"),
                               first_prod_id=cfg["first_prod_id"],
                               last_prod_id=cfg["last_prod_id"],
                               max_token_id=cfg["vocab_size_src"] - 1)
    model = mm.build_transformer(
        vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
        max_seq_len=cfg["max_events"], d_model=cfg["d_model"], n_layers=cfg["N"],
        n_heads=cfg["num_heads"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
        feature_tensor=feat, ai_rate=cfg["ai_rate"], num_users=state["num_users"],
        lto_len=cfg["lto_len"], obtained_len=cfg["obtained_len"],
        prev_dec_len=cfg["prev_dec_len"],
        use_user_embedding=cfg["use_user_embedding"],
        num_mix_heads=cfg.get("num_mix_heads", 0),
        encoder=cfg.get("encoder", "transformer"),
        use_offer_inventory_attn=cfg.get("use_offer_inventory_attn", True),
        attn_recency_bias=cfg.get("attn_recency_bias", False),
        attn_time_bias=cfg.get("attn_time_bias", "none"),
        product_id_embed=cfg.get("product_id_embed", True),
        time_bias_lag_ipt=cfg.get("time_bias_lag_ipt", True),
        inventory=cfg.get("inventory", "tokens"), sat_layers=cfg.get("sat_layers", 0),
        kernel=cfg.get("kernel", "none"), decay=cfg.get("decay", "none"),
        decay_init=cfg.get("decay_init", 30.0),
        decay_freeze=cfg.get("decay_freeze", False),
        use_product_features=cfg.get("use_product_features", True),
        tier=cfg.get("tier", False), block_len=cfg.get("block_len", 64),
        n_campaigns=(cfg.get("n_campaigns", 32) if cfg.get("camp_fe") else 0),
        fuse=cfg.get("fuse", "gate"),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model, cfg


# ------------------------------------------------------------------ helpers
def _forward_logits(model, buf: Dict[str, torch.Tensor], upto: int) -> torch.Tensor:
    """Decision logits (B, 9) at row `upto`, from the prefix rows 0..upto."""
    s = slice(0, upto + 1)
    kw = {}
    if buf.get("ipt") is not None:
        kw["ipt"] = buf["ipt"][:, s]
    if buf.get("campaign") is not None and getattr(model, "camp_bias", None) is not None:
        kw["campaign"] = buf["campaign"][:, s]
    out = model(buf["lto"][:, s], buf["obtained"][:, s], buf["prev_decision"][:, s],
                buf.get("user_idx"), **kw)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits[:, upto, 1:1 + N_CLASSES]


def _replicate(batch: Dict[str, torch.Tensor], n_rep: int) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.repeat_interleave(n_rep, dim=0).clone()
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------ rollout
@torch.no_grad()
def rollout(model, batch: Dict[str, torch.Tensor], env, vocab, *, start: int,
            n_rep: int = 1, seed: int = 0, temperature: float = 1.0,
            device: str = "cpu", teacher_forced: bool = False) -> Dict[str, np.ndarray]:
    """Generate decisions for rows >= `start`, feeding the model its own output.

    `teacher_forced=True` keeps the real streams and only samples -- the control
    that isolates how much of any discrepancy is the feedback loop rather than
    the per-step distribution.

    Returns
        decisions  (B*n_rep, S) int64, 0 outside the generated span
        alive      (B*n_rep, S) bool, rows that exist in the real sequence
        obtained   (B*n_rep, S, 10) int64, the acquisitions the environment
                   produced.  The satiation signature -- the functional the
                   stock path exists to get right, and the one H1 turns on --
                   is computed from this, so it has to come back out.
        lto        (B*n_rep, S, 4) int64, the (exogenous) offers, carried so a
                   caller can score functionals without re-deriving them.
    """
    from gacha_env import PityState

    buf = _replicate({k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in batch.items()}, n_rep)
    B, S = buf["label"].shape
    rng = np.random.default_rng(seed)
    alive = (buf["label"] != 0).cpu().numpy()
    gen = np.zeros((B, S), dtype=np.int64)

    # Pity is carried per simulated customer and warmed on the real prefix, so a
    # rollout does not start every banner at zero pity.
    states = [PityState() for _ in range(B)]
    if env is not None and not teacher_forced:
        _warm_pity(states, buf, vocab, env, start)

    offers = buf["lto"].cpu().numpy()
    for t in range(start, S):
        if not alive[:, t].any():
            break
        logits = _forward_logits(model, buf, t).float()
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        u = rng.random((B, 1))
        draw = (probs.cumsum(axis=1) < u).sum(axis=1)
        y = np.clip(draw + 1, 1, N_CLASSES)
        y = np.where(alive[:, t], y, 0)
        gen[:, t] = y

        if t + 1 >= S:
            break
        if teacher_forced:
            continue
        nxt = np.zeros((B, buf["obtained"].shape[2]), dtype=np.int64)
        for b in range(B):
            if not alive[b, t] or y[b] == 0:
                continue
            nxt[b] = env.step(int(y[b]), offers[b, t], states[b], rng)
        buf["obtained"][:, t + 1] = torch.as_tensor(nxt, device=device)
        buf["prev_decision"][:, t + 1] = torch.as_tensor(y, device=device)

    return {"decisions": gen, "alive": alive, "start": start,
            "obtained": buf["obtained"].cpu().numpy(),
            "lto": buf["lto"].cpu().numpy()}


def _warm_pity(states, buf, vocab, env, start: int) -> None:
    """Replay the real prefix so pity counters enter the rollout where they were."""
    from gacha_env import BANNER_OF
    obt = buf["obtained"][:, :start].cpu().numpy()
    prv = buf["prev_decision"][:, :start].cpu().numpy()
    for b in range(obt.shape[0]):
        st = states[b]
        for t in range(obt.shape[1]):
            d = int(prv[t] if prv.ndim == 1 else prv[b, t])
            banner = BANNER_OF.get(d)
            if banner is None:
                continue
            for p in obt[b, t]:
                p = int(p)
                if not (vocab.first_id <= p <= vocab.last_id):
                    continue
                r = vocab.rarity.get(p, 3)
                if r == 5:
                    st.since5[banner] = 0
                    st.since4[banner] = 0
                elif r == 4:
                    st.since5[banner] += 1
                    st.since4[banner] = 0
                else:
                    st.since5[banner] += 1
                    st.since4[banner] += 1


def first_row_of_campaign(campaign: torch.Tensor, first_holdout: int) -> np.ndarray:
    """Per sequence, the index of the first row in campaign >= first_holdout."""
    c = campaign.cpu().numpy()
    S = c.shape[1]
    hit = c >= first_holdout
    idx = np.where(hit.any(axis=1), hit.argmax(axis=1), S)
    return idx
