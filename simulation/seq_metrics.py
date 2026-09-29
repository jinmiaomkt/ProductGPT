"""R36b: scoring generated sequences AS SEQUENCES.

Three layers, plus the floor that makes any of them mean anything.

L1  per-customer functionals the model was NOT fit to -- spend, incidence, run
    lengths, switching, autocorrelation, escalation, and the satiation
    signature.  The last one is the functional the stock path exists to get
    right and the one H1 turns on: P(buy on a banner | already holding k copies
    of what that banner features).

L2  distributional distance between the set of real sequences and the set of
    generated ones, in L1 space: energy distance, MMD, and a classifier
    two-sample test.  The classifier's feature importances name the failure
    mode, which is usually worth more than the AUC.

L3  per-customer calibration: where the real value falls among that customer's
    M replicates.  A uniform rank histogram is calibrated; U-shaped is
    under-dispersed, which is the classic failure of a one-step-trained model
    rolled out.  The variogram score is the dependence-sensitive complement --
    the energy score has known weak power against misspecified dependence, and
    dependence is exactly what is at stake here.

THE NULL BAND.  A discrepancy number is meaningless without knowing what two
samples of REAL data score against each other.  Every comparison here is made
at a common sample size against a split-half null computed the same way, so
"indistinguishable" means "inside the band real data produces against itself".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REV = np.array([0, 1, 10, 1, 10, 1, 10, 1, 10, 0], dtype=np.float64)  # index by decision
BANNER_ID = np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, -1], dtype=np.int64)  # 0 reg,1 A,2 B,3 wep
N_BANNERS = 4
BUY = np.array([False, True, True, True, True, True, True, True, True, False])
IS_TEN = np.array([False, False, True, False, True, False, True, False, True, False])
# lto slots that carry each banner's featured product; 'reg' has none
FEATURED_SLOTS: Dict[int, Tuple[int, ...]] = {0: (), 1: (0,), 2: (1,), 3: (2, 3)}
SAT_BUCKETS = 4  # k = 0, 1, 2, 3+


# ────────────────────────────── L1 functionals ──────────────────────────────
def functional_names() -> List[str]:
    names = ["spend", "incidence"]
    names += [f"share_d{d}" for d in range(1, 10)]
    names += ["longest_notbuy_run", "mean_buy_run", "max_buy_run",
              "switch_rate", "spend_hhi"]
    names += [f"acf_lag{k}" for k in range(1, 6)]
    names += ["p_buy10_given_buy10", "p_buy_given_buy"]
    names += [f"sat_k{k}" for k in range(SAT_BUCKETS)]
    return names


def functionals(decisions: np.ndarray, alive: np.ndarray, obtained: np.ndarray,
                lto: np.ndarray, *, start: int, first_prod_id: int = 13,
                last_prod_id: int = 130) -> np.ndarray:
    """(B, F) per-sequence features over the generated span [start, S).

    `obtained` must be the stream that ACCOMPANIES `decisions` -- the real one
    when scoring real sequences, the environment's when scoring generated ones.
    Holdings before the decision at row t include obtained[t], because that row
    carries the result of the decision at t-1 and is known when t is decided.
    """
    B, S = decisions.shape
    P = last_prod_id - first_prod_id + 1
    span = np.zeros((B, S), dtype=bool)
    span[:, start:] = True
    live = span & alive & (decisions > 0)
    n_live = live.sum(axis=1).astype(np.float64)
    safe_n = np.maximum(n_live, 1.0)

    d = np.where(live, decisions, 0)
    buy = BUY[d] & live
    spend = REV[d].sum(axis=1)
    n_buy = buy.sum(axis=1).astype(np.float64)

    feats = [spend, n_buy / safe_n]
    for k in range(1, 10):
        feats.append(((d == k) & live).sum(axis=1) / safe_n)

    feats.append(_longest_run(live & (d == 9)))
    mean_run, max_run = _run_stats(buy & live)
    feats += [mean_run, max_run]

    # switching among purchase rows only
    bid = np.where(buy, BANNER_ID[d], -1)
    feats.append(_switch_rate(bid))

    # spend concentration across the four banners
    sp = np.zeros((B, N_BANNERS))
    for b in range(N_BANNERS):
        sp[:, b] = np.where(buy & (BANNER_ID[d] == b), REV[d], 0.0).sum(axis=1)
    tot = np.maximum(sp.sum(axis=1, keepdims=True), 1e-9)
    feats.append(((sp / tot) ** 2).sum(axis=1))

    for lag in range(1, 6):
        feats.append(_autocorr(buy.astype(np.float64), live, lag))

    feats.append(_cond_rate(IS_TEN[d] & buy, IS_TEN[d] & buy, live))
    feats.append(_cond_rate(buy, buy, live))

    feats += list(_satiation_signature(d, live, obtained, lto,
                                       first_prod_id, last_prod_id, P))
    return np.vstack([np.asarray(f, dtype=np.float64) for f in feats]).T


def _longest_run(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[0])
    cur = np.zeros(mask.shape[0])
    for t in range(mask.shape[1]):
        cur = np.where(mask[:, t], cur + 1, 0.0)
        out = np.maximum(out, cur)
    return out


def _run_stats(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    B, S = mask.shape
    cur = np.zeros(B)
    total, count, mx = np.zeros(B), np.zeros(B), np.zeros(B)
    for t in range(S):
        on = mask[:, t]
        ended = (~on) & (cur > 0)
        total += np.where(ended, cur, 0.0)
        count += ended.astype(float)
        mx = np.maximum(mx, cur)
        cur = np.where(on, cur + 1, 0.0)
    total += np.where(cur > 0, cur, 0.0)
    count += (cur > 0).astype(float)
    mx = np.maximum(mx, cur)
    return total / np.maximum(count, 1.0), mx


def _switch_rate(bid: np.ndarray) -> np.ndarray:
    B, S = bid.shape
    prev = np.full(B, -1)
    sw, n = np.zeros(B), np.zeros(B)
    for t in range(S):
        cur = bid[:, t]
        hit = cur >= 0
        has_prev = hit & (prev >= 0)
        sw += (has_prev & (cur != prev)).astype(float)
        n += has_prev.astype(float)
        prev = np.where(hit, cur, prev)
    return sw / np.maximum(n, 1.0)


def _autocorr(x: np.ndarray, live: np.ndarray, lag: int) -> np.ndarray:
    a, b = x[:, :-lag], x[:, lag:]
    m = live[:, :-lag] & live[:, lag:]
    n = m.sum(axis=1).astype(np.float64)
    ok = n > 2
    ma = np.where(m, a, 0.0).sum(axis=1) / np.maximum(n, 1.0)
    mb = np.where(m, b, 0.0).sum(axis=1) / np.maximum(n, 1.0)
    da, db = np.where(m, a - ma[:, None], 0.0), np.where(m, b - mb[:, None], 0.0)
    num = (da * db).sum(axis=1)
    den = np.sqrt((da ** 2).sum(axis=1) * (db ** 2).sum(axis=1))
    out = np.where((den > 1e-12) & ok, num / np.maximum(den, 1e-12), np.nan)
    return out


def _cond_rate(event: np.ndarray, given: np.ndarray, live: np.ndarray) -> np.ndarray:
    """P(event at t | given at t-1), over consecutive live rows."""
    g = given[:, :-1] & live[:, :-1]
    e = event[:, 1:] & live[:, 1:]
    n = g.sum(axis=1).astype(np.float64)
    return np.where(n > 0, (g & e).sum(axis=1) / np.maximum(n, 1.0), np.nan)


def _satiation_signature(d: np.ndarray, live: np.ndarray, obtained: np.ndarray,
                         lto: np.ndarray, first_id: int, last_id: int,
                         P: int) -> List[np.ndarray]:
    """P(buy on banner b at t | already holding k copies of b's featured item).

    Pooled over the three banners that have a featured product; the regular
    banner has none, so it is excluded.
    """
    B, S = d.shape
    counts = np.zeros((B, P), dtype=np.int32)
    hit = np.zeros((B, SAT_BUCKETS)), np.zeros((B, SAT_BUCKETS))
    num, den = hit
    rows = np.arange(B)
    for t in range(S):
        ob = obtained[:, t]
        valid = (ob >= first_id) & (ob <= last_id)
        idx = np.clip(ob - first_id, 0, P - 1)
        np.add.at(counts, (rows[:, None].repeat(ob.shape[1], 1)[valid], idx[valid]), 1)
        if not live[:, t].any():
            continue
        for b, slots in FEATURED_SLOTS.items():
            if not slots:
                continue
            k = np.zeros(B, dtype=np.int64)
            present = np.zeros(B, dtype=bool)
            for s in slots:
                pid = lto[:, t, s]
                ok = (pid >= first_id) & (pid <= last_id)
                present |= ok
                k += np.where(ok, counts[rows, np.clip(pid - first_id, 0, P - 1)], 0)
            bucket = np.clip(k, 0, SAT_BUCKETS - 1)
            elig = live[:, t] & present
            bought = elig & (BANNER_ID[d[:, t]] == b)
            np.add.at(den, (rows[elig], bucket[elig]), 1.0)
            np.add.at(num, (rows[bought], bucket[bought]), 1.0)
    return [np.where(den[:, k] > 0, num[:, k] / np.maximum(den[:, k], 1.0), np.nan)
            for k in range(SAT_BUCKETS)]


# ─────────────────────────── standardisation ────────────────────────────
@dataclass
class Scaler:
    """Standardise with REAL-set statistics, then impute missing to the real mean.

    Both sets get identical treatment, so an undefined functional (no Buy10, so
    no escalation rate) cannot by itself create a discrepancy.
    """
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        return cls(np.nanmean(X, axis=0), np.nanstd(X, axis=0))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / np.where(self.sd > 1e-12, self.sd, 1.0)
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


# ───────────────────────────── L2 distances ─────────────────────────────
def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """2 E|X-Y| - E|X-X'| - E|Y-Y'|; zero iff the distributions agree.

    The within-sample terms leave the diagonal out, so this is the UNBIASED
    estimator and it is SIGNED: under the null it scatters around zero and is
    negative about half the time.  That is the point -- it makes the split-half
    null band a proper reference distribution rather than something pinned at
    zero.  Do not "fix" a negative value.
    """
    return float(2 * _mean_dist(X, Y) - _mean_dist(X, X, same=True)
                 - _mean_dist(Y, Y, same=True))


def _mean_dist(A: np.ndarray, B: np.ndarray, same: bool = False) -> float:
    D = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0))
    if same:
        n = A.shape[0]
        if n < 2:
            return 0.0
        return float((D.sum() - np.trace(D)) / (n * (n - 1)))
    return float(D.mean())


def mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    Z = np.vstack([X, Y])
    if gamma is None:
        d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-9)
    kxx = _k(X, X, gamma, True)
    kyy = _k(Y, Y, gamma, True)
    kxy = _k(X, Y, gamma, False)
    return float(kxx + kyy - 2 * kxy)


def _k(A, B, gamma, same):
    K = np.exp(-gamma * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    if same:
        n = A.shape[0]
        if n < 2:
            return 0.0
        return float((K.sum() - np.trace(K)) / (n * (n - 1)))
    return float(K.mean())


def c2st(X: np.ndarray, Y: np.ndarray, *, seed: int = 0, n_splits: int = 5
         ) -> Dict[str, object]:
    """Classifier two-sample test: AUC for telling real from generated.

    0.5 means indistinguishable.  `importance` ranks the functionals the
    classifier leaned on, which is the diagnosis.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    Z = np.vstack([X, Y])
    y = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, imps = [], []
    for tr, te in skf.split(Z, y):
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
        clf.fit(Z[tr], y[tr])
        p = clf.predict_proba(Z[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        r = permutation_importance(clf, Z[te], y[te], n_repeats=5,
                                   random_state=seed, scoring="roc_auc")
        imps.append(r.importances_mean)
    a = np.asarray(aucs)
    return {"auc": float(a.mean()),
            "auc_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "ci95": (float(a.mean() - 1.96 * a.std(ddof=1) / np.sqrt(len(a))),
                     float(a.mean() + 1.96 * a.std(ddof=1) / np.sqrt(len(a))))
            if len(a) > 1 else (float(a.mean()), float(a.mean())),
            "importance": np.asarray(imps).mean(axis=0)}


# ──────────────────────────── the null band ─────────────────────────────
def null_band(real: np.ndarray, *, n_boot: int = 200, seed: int = 0,
              stat=energy_distance) -> np.ndarray:
    """Split-half real-vs-real: what the statistic scores on data that agrees."""
    rng = np.random.default_rng(seed)
    n = real.shape[0]
    half = n // 2
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.permutation(n)
        out[i] = stat(real[idx[:half]], real[idx[half:2 * half]])
    return out


def compare(real: np.ndarray, sim: np.ndarray, *, n_boot: int = 200, seed: int = 0,
            stat=energy_distance) -> Dict[str, float]:
    """Observed statistic against the split-half null, AT A MATCHED SAMPLE SIZE.

    Energy distance and MMD both depend on n, so the observed value is computed
    on half-sized samples too; otherwise the comparison to the null is rigged.
    """
    rng = np.random.default_rng(seed)
    n = real.shape[0]
    half = max(n // 2, 2)
    null = null_band(real, n_boot=n_boot, seed=seed, stat=stat)
    obs = np.empty(n_boot)
    for i in range(n_boot):
        ri = rng.permutation(n)[:half]
        si = rng.permutation(sim.shape[0])[:half]
        obs[i] = stat(real[ri], sim[si])
    o = float(obs.mean())
    return {"stat": o, "null_mean": float(null.mean()),
            "null_q95": float(np.quantile(null, 0.95)),
            "p_value": float((null >= o).mean()),
            "inside_band": bool(o <= np.quantile(null, 0.95)),
            "z": float((o - null.mean()) / max(null.std(ddof=1), 1e-12))}


# ───────────────────────── L3 calibration / PIT ─────────────────────────
def rank_histogram(real_T: np.ndarray, sim_T: np.ndarray, *, seed: int = 0
                   ) -> Dict[str, object]:
    """Where the real value falls among that customer's M replicates.

    real_T (n,), sim_T (n, M).  Ties broken at random, which is what makes the
    histogram uniform under a calibrated generator with discrete functionals.
    """
    rng = np.random.default_rng(seed)
    n, M = sim_T.shape
    below = (sim_T < real_T[:, None]).sum(1)
    equal = (sim_T == real_T[:, None]).sum(1)
    r = below + (rng.random((n,)) * (equal + 1)).astype(int)
    ok = np.isfinite(real_T) & np.isfinite(sim_T).all(1)
    r = r[ok]
    if r.size == 0:
        return {"ranks": r, "uniform_p": float("nan"), "shape": "undefined",
                "n": 0, "bins": np.zeros(1)}
    nb = min(M + 1, 20)
    bins = np.bincount(np.minimum((r * nb) // (M + 1), nb - 1), minlength=nb)
    exp = r.size / nb
    chi2 = float(((bins - exp) ** 2 / exp).sum())
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, nb - 1))
    edge = bins[0] + bins[-1]
    mid = bins[nb // 2 - 1:nb // 2 + 1].sum()
    shape = ("uniform" if p > 0.05 else
             "under-dispersed" if edge > 2.5 * exp else
             "over-dispersed" if mid > 2.5 * exp else "biased")
    return {"ranks": r, "uniform_p": p, "shape": shape, "n": int(r.size),
            "bins": bins}


def variogram_score(real: np.ndarray, sim: np.ndarray, p: float = 0.5) -> float:
    """Dependence-sensitive score: matches pairwise spread, not just margins.

    real (n, F), sim (n, M, F).  Lower is better.  The complement to energy
    distance, which has weak power against misspecified dependence.
    """
    n, M, F = sim.shape
    tot = 0.0
    for i in range(F):
        for j in range(i + 1, F):
            obs = np.abs(real[:, i] - real[:, j]) ** p
            ens = (np.abs(sim[:, :, i] - sim[:, :, j]) ** p).mean(axis=1)
            good = np.isfinite(obs) & np.isfinite(ens)
            if good.any():
                tot += float(((obs[good] - ens[good]) ** 2).mean())
    return tot / max(F * (F - 1) / 2, 1)
