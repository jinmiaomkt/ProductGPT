"""R36b: the generators ProductGPT has to beat as a SEQUENCE model.

Two, deliberately weak, and both fitted on calibration campaigns only:

  iid      each decision drawn from the empirical marginal, independent across
           rows.  Matches every share by construction and must fail everything
           that depends on order.  The sanity floor: a metric that cannot
           separate this from real data is not measuring dependence.

  markov1  a first-order chain on the 9 decisions.  Captures persistence and
           the NotBuy run structure with 81 numbers and no covariates.  This is
           the bar a deep sequence model actually has to clear, and stating it
           is what stops "the rollout looks plausible" from being the finding.

Both run through the SAME acquisition environment and the same fixed grid as
the model rollout, so a difference between them is a difference in the decision
process and nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "simulation", REPO / "gen5_multistream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

N_CLASSES = 9


# ─────────────────────────────── fitting ────────────────────────────────
def fit_marginal(items: Sequence[dict], *, holdout_from: int) -> np.ndarray:
    """P(decision), estimated on calibration rows only.  Returns (9,)."""
    c = np.zeros(N_CLASSES, dtype=np.float64)
    for it in items:
        lab = np.asarray(it["label"])
        camp = np.asarray(it["campaign"]) if it.get("campaign") is not None else None
        S = int((lab != 0).sum())
        for t in range(S):
            if camp is not None and int(camp[t]) >= holdout_from:
                continue
            y = int(lab[t])
            if 1 <= y <= N_CLASSES:
                c[y - 1] += 1
    if c.sum() <= 0:
        raise RuntimeError("fit_marginal saw no calibration rows")
    return c / c.sum()


def fit_markov(items: Sequence[dict], *, holdout_from: int,
               alpha: float = 1.0) -> np.ndarray:
    """P(y_t | y_{t-1}) on calibration rows.  Returns (10, 9); row 0 = no history.

    Laplace-smoothed so a transition never seen in calibration cannot make a
    holdout rollout impossible.
    """
    c = np.full((N_CLASSES + 1, N_CLASSES), alpha, dtype=np.float64)
    for it in items:
        lab = np.asarray(it["label"])
        camp = np.asarray(it["campaign"]) if it.get("campaign") is not None else None
        S = int((lab != 0).sum())
        for t in range(S):
            if camp is not None and int(camp[t]) >= holdout_from:
                continue
            y = int(lab[t])
            if not (1 <= y <= N_CLASSES):
                continue
            prev = int(lab[t - 1]) if t > 0 else 0
            if not (0 <= prev <= N_CLASSES):
                prev = 0
            c[prev, y - 1] += 1
    return c / c.sum(axis=1, keepdims=True)


# ────────────────────────────── simulation ──────────────────────────────
def simulate(batch: Dict[str, torch.Tensor], env, vocab, sampler: Callable, *,
             start: int, n_rep: int = 1, seed: int = 0) -> Dict[str, np.ndarray]:
    """Run `sampler` on the fixed grid, through the real environment.

    `sampler(prev, t, rng)` takes the previous decision per sequence (0 where
    there is none) and returns the next decision per sequence.
    """
    from gacha_env import PityState
    from rollout import _replicate, _warm_pity

    buf = _replicate(batch, n_rep)
    B, S = buf["label"].shape
    rng = np.random.default_rng(seed)
    alive = (buf["label"] != 0).cpu().numpy()
    gen = np.zeros((B, S), dtype=np.int64)
    obtained = buf["obtained"].cpu().numpy().copy()
    offers = buf["lto"].cpu().numpy()

    states = [PityState() for _ in range(B)]
    if env is not None:
        _warm_pity(states, buf, vocab, env, start)

    prev = buf["prev_decision"].cpu().numpy()[:, start].copy()
    for t in range(start, S):
        if not alive[:, t].any():
            break
        y = np.asarray(sampler(prev, t, rng), dtype=np.int64)
        y = np.where(alive[:, t], np.clip(y, 1, N_CLASSES), 0)
        gen[:, t] = y
        if t + 1 >= S:
            break
        for b in range(B):
            if alive[b, t] and y[b] > 0:
                obtained[b, t + 1] = env.step(int(y[b]), offers[b, t], states[b], rng)
            else:
                obtained[b, t + 1] = 0
        prev = y
    return {"decisions": gen, "alive": alive, "start": start,
            "obtained": obtained, "lto": offers}


def iid_sampler(probs: np.ndarray) -> Callable:
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()

    def _s(prev, t, rng):
        return rng.choice(N_CLASSES, size=prev.shape[0], p=p) + 1
    return _s


def markov_sampler(P: np.ndarray) -> Callable:
    Pm = np.asarray(P, dtype=np.float64)
    cum = Pm.cumsum(axis=1)

    def _s(prev, t, rng):
        idx = np.clip(prev, 0, N_CLASSES).astype(np.int64)
        u = rng.random((prev.shape[0], 1))
        return (cum[idx] < u).sum(axis=1) + 1
    return _s


def real_sequences(batch: Dict[str, torch.Tensor], *, start: int) -> Dict[str, np.ndarray]:
    """The observed sequences, in the shape the metrics expect."""
    lab = batch["label"].cpu().numpy()
    return {"decisions": lab.copy(), "alive": lab != 0, "start": start,
            "obtained": batch["obtained"].cpu().numpy(),
            "lto": batch["lto"].cpu().numpy()}
