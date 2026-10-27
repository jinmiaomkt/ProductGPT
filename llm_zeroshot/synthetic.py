"""Synthetic products and customers for the R48/R49 tests and the GPU self-check.

No data: random draws shaped like the R46 occasion table.  Moved here from
scripts/test_r48_zeroshot.py so llm_zeroshot.score --selfcheck can build the
same customers on the GPU with the real model.
"""
from __future__ import annotations

import numpy as np


def synthetic_products():
    rng = np.random.default_rng(0)
    p = {k: np.zeros(60, np.int64) for k in ("rarity", "fig", "lto", "female")}
    p["element"] = np.full(60, -1, np.int64)
    p["wtype"] = np.full(60, -1, np.int64)
    for i in range(13, 57):
        p["rarity"][i] = rng.choice([3, 4, 5])
        p["fig"][i] = int(i < 40)
        p["lto"][i] = int(rng.random() < 0.5)
        p["female"][i] = int(rng.random() < 0.5)
        p["element"][i] = rng.integers(0, 7) if i < 40 else -1
        p["wtype"][i] = rng.integers(0, 5)
    return p


def synthetic_customer(S, seed):
    rng = np.random.default_rng(seed)
    lab = rng.integers(1, 10, S)
    prev = np.concatenate([[rng.integers(1, 10)], lab[:-1]])
    lto = np.zeros((S, 4), np.int64)
    for start in range(0, S, 15):                       # a new campaign every 15 rows
        lto[start:start + 15] = [rng.integers(13, 40), rng.choice([0, rng.integers(13, 40)]),
                                 rng.integers(40, 57), rng.integers(40, 57)]
    obt = np.zeros((S, 10), np.int64)
    for t in range(1, S):
        k = {1: 1, 2: 10, 3: 1, 4: 10, 5: 1, 6: 10, 7: 1, 8: 10, 9: 0}[int(lab[t - 1])]
        obt[t, :k] = rng.choice(list(range(13, 57)) + [59], size=k)
    inv = np.zeros(44)
    inv[[0, 5, 5 - 0]] = [1, 2, 2]
    return lab, prev, lto, obt, inv


def perturb_from(lab, obt, t):
    """Change decision t and everything after it (decisions and obtained items)."""
    lab2, obt2 = lab.copy(), obt.copy()
    lab2[t:] = (lab2[t:] % 9) + 1
    obt2[t + 1:] = np.where(obt2[t + 1:] > 0, 59, 0)
    return lab2, obt2


def perturb_decisions_from(lab, t):
    """Change decision t and every later decision only.  Each answer is one letter
    token, so the transcript keeps its length: every tensor keeps its shape."""
    lab2 = lab.copy()
    lab2[t:] = (lab2[t:] % 9) + 1
    return lab2
