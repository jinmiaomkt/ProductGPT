"""The acquisition environment: what a decision yields.

R36a.  A rollout is not just the model.  The model emits a decision in 1..9;
something outside the model has to turn that decision into the `obtained`
block the next row shows, because that block is what drives the inventory and
hence every satiation term the model conditions on.

TWO PROCESSES, TWO SOURCES OF TRUTH
-----------------------------------
*Rarity* (3*/4*/5*) is institutional: the published rates plus pity.  The data
agrees with them closely enough to use them as given -- measured effective
rates on the calibration period are p5 = 0.0125 (figure banners) and 0.0173
(weapon), against published effective rates of roughly 1.6% and 1.85%, with
p4 = 0.13 against a published 13%.

*Identity* (which product) is NOT reproducible from the published rules on this
data.  Under the real 50/50 plus its guarantee, roughly 75% of 5* pulls on a
character banner should be the featured character.  Measured here: 43.0% on
figure banners and 40.1% on weapon banners, with about a quarter landing on
limited products that were not on offer at that row -- which the real banner
mechanics forbid.  The event-level attribution of acquisitions to decisions in
this dataset is imperfect (a Buy10 fills ten slots only 72% of the time and one
slot 23% of the time), and no amount of correct gacha logic will reproduce
that.  So the identity process is CALIBRATED on the calibration campaigns.

Both are switchable.  `identity="published"` gives the textbook 50/50 and is
kept because R35's counterfactual argument -- that the rules are institutional
facts rather than estimates -- applies to the rarity process and to any
counterfactual that changes the calendar rather than the attribution.  Report
both when a policy number depends on it.

Nothing here is estimated on the holdout.  `fit` refuses campaigns at or after
the holdout boundary.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Decision -> (banner, nominal pulls).  Matches rev_vec = [1,10,1,10,1,10,1,10,0].
BANNER_OF: Dict[int, str] = {1: "reg", 2: "reg", 3: "figA", 4: "figA",
                             5: "figB", 6: "figB", 7: "wep", 8: "wep"}
NOMINAL_PULLS: Dict[int, int] = {1: 1, 2: 10, 3: 1, 4: 10,
                                 5: 1, 6: 10, 7: 1, 8: 10, 9: 0}
NOT_BUY = 9
OBTAINED_LEN = 10

# Which lto slot carries the featured product for each banner.
# Verified: slot 0 is always a 5* figure, slot 1 a second 5* figure in 44.7% of
# campaigns (no Figure-B banner otherwise), slots 2-3 always 5* weapons, and
# exactly one offer tuple occurs per campaign.
FEATURED_SLOTS: Dict[str, Tuple[int, ...]] = {
    "reg": (), "figA": (0,), "figB": (1,), "wep": (2, 3)}


@dataclass
class BannerRules:
    """Published rates.  Soft pity raises the rate linearly; hard pity forces."""
    base_p5: float
    soft_p5: int
    hard_p5: int
    base_p4: float
    soft_p4: int
    hard_p4: int
    featured_prob: float          # P(featured | 5*, not guaranteed)


PUBLISHED: Dict[str, BannerRules] = {
    "reg":  BannerRules(0.006, 74, 90, 0.051, 9, 10, 0.0),
    "figA": BannerRules(0.006, 74, 90, 0.051, 9, 10, 0.5),
    "figB": BannerRules(0.006, 74, 90, 0.051, 9, 10, 0.5),
    "wep":  BannerRules(0.007, 63, 80, 0.060, 9, 10, 0.75),
}


def _pity_prob(base: float, since: int, soft: int, hard: int) -> float:
    """`since` = pulls since the last hit; the pull about to happen is since+1."""
    i = since + 1
    if i >= hard:
        return 1.0
    if i < soft:
        return base
    return float(min(1.0, base + (i - soft + 1) * base * 10.0))


@dataclass
class PityState:
    """Per-banner pity counters for one simulated customer."""
    since5: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    since4: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    guaranteed: Dict[str, bool] = field(default_factory=lambda: defaultdict(bool))

    def copy(self) -> "PityState":
        return PityState(defaultdict(int, self.since5),
                         defaultdict(int, self.since4),
                         defaultdict(bool, self.guaranteed))


class AcquisitionEnv:
    """Decision + offer -> an `obtained` row of length 10.

    Call `fit` once on the calibration period, then `step` per decision.
    """

    def __init__(self, vocab: "ProductTable", *, rarity: str = "pity",
                 identity: str = "empirical"):
        if rarity not in ("pity", "marginal"):
            raise ValueError(f"rarity must be 'pity' or 'marginal', got {rarity!r}")
        if identity not in ("empirical", "published"):
            raise ValueError(f"identity must be 'empirical' or 'published', got {identity!r}")
        self.vocab = vocab
        self.rarity_mode = rarity
        self.identity_mode = identity
        # filled by fit()
        self.p_nslots: Dict[int, np.ndarray] = {}
        self.p_rarity: Dict[str, np.ndarray] = {}
        self.p_class: Dict[Tuple[str, int], np.ndarray] = {}
        self.pool_p: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._fitted = False

    # ---------------------------------------------------------------- fit
    def fit(self, items: Sequence[dict], *, holdout_from: int) -> "AcquisitionEnv":
        """Calibrate on rows strictly before campaign `holdout_from`.

        `items` are TransformerDataset entries (tensors); only calibration rows
        are read, so nothing here can leak the holdout.
        """
        nslots: Dict[int, Counter] = defaultdict(Counter)
        rar: Dict[str, Counter] = defaultdict(Counter)
        cls: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
        within: Dict[Tuple[int, int, int], Counter] = defaultdict(Counter)

        for it in items:
            lto = np.asarray(it["lto"])
            obt = np.asarray(it["obtained"])
            prv = np.asarray(it["prev_decision"])
            lab = np.asarray(it["label"])
            camp = np.asarray(it["campaign"]) if it.get("campaign") is not None else None
            S = int((lab != 0).sum())
            for t in range(1, S):
                if camp is not None and int(camp[t]) >= holdout_from:
                    continue
                d = int(prv[t])
                if d < 1 or d > 9:
                    continue
                row = obt[t]
                valid = self.vocab.in_range(row)
                nslots[d][int(valid.sum())] += 1
                if d == NOT_BUY:
                    continue
                banner = BANNER_OF[d]
                offer = [int(x) for x in lto[t - 1]]
                featured = {offer[s] for s in FEATURED_SLOTS[banner] if offer[s] > 0}
                for p in row[valid]:
                    p = int(p)
                    r = self.vocab.rarity[p]
                    rar[banner][r] += 1
                    # Condition on the banner ACTUALLY having a featured slot
                    # this campaign: step() only consults p_class when the offer
                    # is non-empty, so this must be the same conditional law.
                    # Slot 1 is empty in 55% of campaigns (no Figure-B banner),
                    # which is where the two estimands come apart.
                    if r == 5 and featured:
                        if p in featured:
                            k = "featured"
                        elif self.vocab.lto[p] == 0:
                            k = "standard"
                        else:
                            k = "other_lto"
                        cls[(banner, r)][k] += 1
                    within[(r, self.vocab.figure[p], self.vocab.lto[p])][p] += 1

        if not nslots:
            raise RuntimeError("fit() saw no calibration rows -- check holdout_from "
                               "and that the batch carries a campaign field")

        self.p_nslots = {d: _to_prob(c, OBTAINED_LEN + 1) for d, c in nslots.items()}
        self.p_rarity = {b: _to_prob({r: n for r, n in c.items()}, 6) for b, c in rar.items()}
        self.p_class = {}
        for key, c in cls.items():
            tot = sum(c.values())
            self.p_class[key] = np.array([c["featured"], c["standard"], c["other_lto"]],
                                         dtype=np.float64) / max(tot, 1)
        self.pool_p = {}
        for key, c in within.items():
            ids = np.array(sorted(c), dtype=np.int64)
            w = np.array([c[int(i)] for i in ids], dtype=np.float64)
            self.pool_p[key] = (ids, w / w.sum())
        self._fitted = True
        return self

    # --------------------------------------------------------------- step
    def step(self, decision: int, offer: Sequence[int], state: PityState,
             rng: np.random.Generator) -> np.ndarray:
        """One decision -> an (OBTAINED_LEN,) int64 row, zero-padded."""
        if not self._fitted:
            raise RuntimeError("call fit() before step()")
        out = np.zeros(OBTAINED_LEN, dtype=np.int64)
        d = int(decision)
        if d == NOT_BUY or d not in BANNER_OF:
            return out

        n = int(rng.choice(len(self.p_nslots[d]), p=self.p_nslots[d]))
        if n == 0:
            return out
        banner = BANNER_OF[d]
        offer = [int(x) for x in offer]
        for j in range(min(n, OBTAINED_LEN)):
            r = self._draw_rarity(banner, state, rng)
            out[j] = self._draw_identity(banner, r, offer, state, rng)
        return out

    # ------------------------------------------------------------ internals
    def _draw_rarity(self, banner: str, st: PityState, rng: np.random.Generator) -> int:
        if self.rarity_mode == "marginal":
            p = self.p_rarity.get(banner)
            if p is None:
                return 3
            return int(rng.choice(len(p), p=p))
        cfg = PUBLISHED[banner]
        p5 = _pity_prob(cfg.base_p5, st.since5[banner], cfg.soft_p5, cfg.hard_p5)
        if rng.random() < p5:
            st.since5[banner] = 0
            st.since4[banner] = 0
            return 5
        st.since5[banner] += 1
        p4 = _pity_prob(cfg.base_p4, st.since4[banner], cfg.soft_p4, cfg.hard_p4)
        if rng.random() < p4:
            st.since4[banner] = 0
            return 4
        st.since4[banner] += 1
        return 3

    def _draw_identity(self, banner: str, rarity: int, offer: List[int],
                       st: PityState, rng: np.random.Generator) -> int:
        featured = [offer[s] for s in FEATURED_SLOTS[banner]
                    if s < len(offer) and offer[s] > 0]
        if rarity == 5 and featured:
            if self.identity_mode == "published":
                cfg = PUBLISHED[banner]
                if st.guaranteed[banner] or rng.random() < cfg.featured_prob:
                    st.guaranteed[banner] = False
                    return int(rng.choice(featured))
                st.guaranteed[banner] = True
                return self._from_pool(5, self.vocab.figure_of_banner(banner), 0, rng,
                                       exclude=featured)
            probs = self.p_class.get((banner, 5))
            if probs is not None and probs.sum() > 0:
                k = int(rng.choice(3, p=probs))
                if k == 0:
                    return int(rng.choice(featured))
                is_fig = self.vocab.figure_of_banner(banner)
                return self._from_pool(5, is_fig, 0 if k == 1 else 1, rng,
                                       exclude=featured)
        is_fig = self.vocab.figure_of_banner(banner) if rarity == 5 else None
        return self._from_pool(rarity, is_fig, None, rng, exclude=featured)

    def _from_pool(self, rarity: int, is_fig: Optional[int], lto: Optional[int],
                   rng: np.random.Generator, exclude: Sequence[int] = ()) -> int:
        """Draw a product by calibrated frequency from the matching pool.

        Falls back by relaxing lto, then the figure/weapon split, so a pool that
        happens to be empty in calibration never crashes a rollout.
        """
        for key in self.vocab.pool_keys(rarity, is_fig, lto):
            hit = self.pool_p.get(key)
            if hit is None:
                continue
            ids, w = hit
            if exclude:
                keep = ~np.isin(ids, np.asarray(exclude, dtype=np.int64))
                if keep.any():
                    ids, w = ids[keep], w[keep]
                    w = w / w.sum()
            return int(rng.choice(ids, p=w))
        return 0


class ProductTable:
    """Rarity, figure/weapon and limited flags, read from the level-7 vocab."""

    def __init__(self, ids: np.ndarray, rarity: np.ndarray, figure: np.ndarray,
                 lto: np.ndarray, first_id: int, last_id: int):
        self.first_id, self.last_id = int(first_id), int(last_id)
        self.rarity = dict(zip(ids.tolist(), rarity.tolist()))
        self.figure = dict(zip(ids.tolist(), figure.tolist()))
        self.lto = dict(zip(ids.tolist(), lto.tolist()))

    @classmethod
    def from_xlsx(cls, path, id_col: str = "NewProductIndex7") -> "ProductTable":
        import pandas as pd
        df = pd.read_excel(path)
        ids = df[id_col].astype(int).to_numpy()
        return cls(ids, df["Rarity"].astype(int).to_numpy(),
                   df["type_figure"].astype(int).to_numpy(),
                   df["LTO"].astype(int).to_numpy(), ids.min(), ids.max())

    def in_range(self, arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        return (a >= self.first_id) & (a <= self.last_id)

    @staticmethod
    def figure_of_banner(banner: str) -> int:
        return 0 if banner == "wep" else 1

    @staticmethod
    def pool_keys(rarity: int, is_fig: Optional[int], lto: Optional[int]):
        """Most specific pool first, then progressively relaxed."""
        figs = [is_fig] if is_fig is not None else [1, 0]
        ltos = [lto] if lto is not None else [0, 1]
        for f in figs:
            for l in ltos:
                yield (rarity, f, l)
        for f in (1, 0):
            for l in (0, 1):
                yield (rarity, f, l)


def _to_prob(counter, size: int) -> np.ndarray:
    p = np.zeros(size, dtype=np.float64)
    for k, v in dict(counter).items():
        if 0 <= int(k) < size:
            p[int(k)] = float(v)
    s = p.sum()
    if s <= 0:
        p[0] = 1.0
        return p
    return p / s
