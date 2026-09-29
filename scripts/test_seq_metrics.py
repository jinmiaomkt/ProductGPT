"""Correctness checks for the R36b sequence metrics and baseline generators.

Run:  python scripts/test_seq_metrics.py

  A. FUNCTIONALS   hand-computed sequences with known answers, including the
                   satiation signature's bucketing and the NaN convention.
  B. DISTANCES     energy / MMD / C2ST behave on samples whose answer is known.
  C. NULL BAND     agreeing data lands inside the band, shifted data does not.
  D. CALIBRATION   a calibrated ensemble gives a uniform rank histogram; a
                   too-narrow one is reported as under-dispersed.
  E. BASELINES     the samplers reproduce the laws they were fitted to.
  F. END TO END    the pipeline separates the i.i.d. baseline from real
                   sequences.  This is the check that matters: i.i.d. destroys
                   all dependence, so a metric that cannot see it is not
                   measuring what R36 is about.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
import seq_metrics as sm  # noqa: E402
from baselines import (fit_marginal, fit_markov, iid_sampler,  # noqa: E402
                       markov_sampler, real_sequences, simulate)
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


NAMES = sm.functional_names()
IDX = {n: i for i, n in enumerate(NAMES)}


def feats_of(dec, obt=None, lto=None, start=0):
    dec = np.asarray(dec, dtype=np.int64)
    if dec.ndim == 1:
        dec = dec[None, :]
    B, S = dec.shape
    obt = np.zeros((B, S, 10), dtype=np.int64) if obt is None else np.asarray(obt)
    lto = np.zeros((B, S, 4), dtype=np.int64) if lto is None else np.asarray(lto)
    return sm.functionals(dec, dec > 0, obt, lto, start=start)


# ───────────────────────────── A. functionals ─────────────────────────────
print("A. functionals on hand-computed sequences")
# 3 = Buy1 FigA (rev 1), 4 = Buy10 FigA (rev 10), 9 = NotBuy
seq = [3, 9, 9, 9, 4, 4, 7]
f = feats_of(seq)[0]
check("spend", f[IDX["spend"]] == 1 + 10 + 10 + 1, f"{f[IDX['spend']]}")
check("incidence", abs(f[IDX["incidence"]] - 4 / 7) < 1e-9)
check("share of NotBuy", abs(f[IDX["share_d9"]] - 3 / 7) < 1e-9)
check("longest NotBuy run", f[IDX["longest_notbuy_run"]] == 3)
check("max buy run", f[IDX["max_buy_run"]] == 3, f"{f[IDX['max_buy_run']]}")
check("mean buy run", abs(f[IDX["mean_buy_run"]] - 2.0) < 1e-9, f"{f[IDX['mean_buy_run']]}")
# purchases: FigA, FigA, FigA, Wep -> 1 switch out of 3 consecutive pairs
check("switch rate", abs(f[IDX["switch_rate"]] - 1 / 3) < 1e-9, f"{f[IDX['switch_rate']]}")
# spend 22: FigA 21, Wep 1 -> hhi = (21/22)^2 + (1/22)^2
check("spend HHI", abs(f[IDX["spend_hhi"]] - ((21 / 22) ** 2 + (1 / 22) ** 2)) < 1e-9)
# P(Buy10 at t | Buy10 at t-1): pairs with Buy10 first = (4,4),(4,7) -> 1/2
check("escalation P(Buy10|Buy10)", abs(f[IDX["p_buy10_given_buy10"]] - 0.5) < 1e-9,
      f"{f[IDX['p_buy10_given_buy10']]}")

f2 = feats_of([9, 9, 9, 9])[0]
check("undefined escalation is NaN, not 0", np.isnan(f2[IDX["p_buy10_given_buy10"]]))
check("all-NotBuy gives zero spend", f2[IDX["spend"]] == 0)

# satiation: one banner, featured product 13, holdings grow via `obtained`
S = 6
lto = np.zeros((1, S, 4), dtype=np.int64)
lto[0, :, 0] = 13                      # slot 0 -> banner figA featured = 13
obt = np.zeros((1, S, 10), dtype=np.int64)
obt[0, 2, 0] = 13                      # a copy arrives, known from row 2 on
obt[0, 4, 0] = 13                      # a second copy, known from row 4 on
dec = np.array([[3, 9, 3, 9, 9, 9]])   # buy FigA at rows 0 and 2 only
g = sm.functionals(dec, dec > 0, obt, lto, start=0)[0]
# k=0 at rows 0,1 -> bought at row 0 -> 1/2 ; k=1 at rows 2,3 -> bought at 2 -> 1/2
check("satiation bucket k=0", abs(g[IDX["sat_k0"]] - 0.5) < 1e-9, f"{g[IDX['sat_k0']]}")
check("satiation bucket k=1", abs(g[IDX["sat_k1"]] - 0.5) < 1e-9, f"{g[IDX['sat_k1']]}")
check("satiation bucket k=2", abs(g[IDX["sat_k2"]] - 0.0) < 1e-9, f"{g[IDX['sat_k2']]}")
check("satiation bucket k=3+ undefined here", np.isnan(g[IDX["sat_k3"]]))

# ───────────────────────────── B. distances ──────────────────────────────
print("\nB. distances")
rng = np.random.default_rng(0)
A = rng.normal(size=(150, 6))
Bm = rng.normal(size=(150, 6))
C = rng.normal(size=(150, 6)) + 2.0
eaa, eac = sm.energy_distance(A, Bm), sm.energy_distance(A, C)
check("energy distance ~0 for same law", abs(eaa) < 0.15, f"{eaa:.4f}")
check("energy distance large for shifted law", eac > 3.0, f"{eac:.4f}")
check("energy distance is ordered", eac > eaa)
maa, mac = sm.mmd_rbf(A, Bm), sm.mmd_rbf(A, C)
check("MMD ~0 for same law", abs(maa) < 0.02, f"{maa:.4f}")
check("MMD large for shifted law", mac > maa * 5 + 0.05, f"{mac:.4f}")
r_same = sm.c2st(A, Bm, seed=0)
r_diff = sm.c2st(A, C, seed=0)
check("C2ST AUC ~0.5 for same law", abs(r_same["auc"] - 0.5) < 0.12, f"{r_same['auc']:.3f}")
check("C2ST AUC ~1 for shifted law", r_diff["auc"] > 0.95, f"{r_diff['auc']:.3f}")

# ───────────────────────────── C. null band ──────────────────────────────
print("\nC. null band")
real = rng.normal(size=(200, 5))
same = rng.normal(size=(200, 5))
shift = rng.normal(size=(200, 5)) + 0.6
c_same = sm.compare(real, same, n_boot=60, seed=1)
c_shift = sm.compare(real, shift, n_boot=60, seed=1)
check("agreeing data lands inside the band", c_same["inside_band"],
      f"stat={c_same['stat']:.4f} q95={c_same['null_q95']:.4f} p={c_same['p_value']:.2f}")
check("shifted data falls outside the band", not c_shift["inside_band"],
      f"stat={c_shift['stat']:.4f} q95={c_shift['null_q95']:.4f} z={c_shift['z']:.1f}")
# The within-sample terms leave the diagonal out, which makes the energy
# statistic UNBIASED and therefore signed: under the null it fluctuates around
# zero and is negative about half the time.  The population quantity is
# non-negative; this estimator of it is not, and that is what makes it usable
# as a null band.  Assert the band is centred near zero, not that it is positive.
check("null band is finite and centred near zero",
      np.isfinite(c_same["null_mean"]) and abs(c_same["null_mean"]) < 0.1 * c_shift["stat"],
      f"null_mean={c_same['null_mean']:+.4f} vs shifted stat {c_shift['stat']:.4f}")
check("null band has positive spread", c_same["null_q95"] > c_same["null_mean"])

# ──────────────────────────── D. calibration ─────────────────────────────
print("\nD. rank histogram")
n, M = 400, 39
truth = rng.normal(size=n)
cal = truth[:, None] + rng.normal(size=(n, M))          # correct spread
narrow = truth[:, None] + rng.normal(size=(n, M)) * 0.2  # too confident
obs = truth + rng.normal(size=n)
h_cal = sm.rank_histogram(obs, cal, seed=0)
h_nar = sm.rank_histogram(obs, narrow, seed=0)
check("calibrated ensemble gives a uniform histogram", h_cal["uniform_p"] > 0.05,
      f"p={h_cal['uniform_p']:.3f} shape={h_cal['shape']}")
check("under-dispersed ensemble is detected", h_nar["shape"] == "under-dispersed",
      f"p={h_nar['uniform_p']:.1e} shape={h_nar['shape']}")

# ───────────────────────────── E. baselines ──────────────────────────────
print("\nE. baseline samplers reproduce their fitted laws")
p = np.array([.05, .02, .30, .08, .10, .03, .06, .02, .34])
s = iid_sampler(p)
draws = s(np.zeros(200000, dtype=np.int64), 0, np.random.default_rng(0))
emp = np.bincount(draws, minlength=10)[1:10] / len(draws)
check("iid sampler reproduces the marginal", np.abs(emp - p).max() < 0.005,
      f"max|d|={np.abs(emp - p).max():.4f}")

P = rng.random((10, 9)) + 0.05
P /= P.sum(axis=1, keepdims=True)
ms = markov_sampler(P)
prev = np.full(200000, 3, dtype=np.int64)
d2 = ms(prev, 0, np.random.default_rng(0))
emp2 = np.bincount(d2, minlength=10)[1:10] / len(d2)
check("markov sampler reproduces its transition row",
      np.abs(emp2 - P[3]).max() < 0.005, f"max|d|={np.abs(emp2 - P[3]).max():.4f}")

# ───────────────────────────── F. end to end ─────────────────────────────
print("\nF. end to end: does the pipeline separate i.i.d. from real sequences?")
cfg = config5.get_config("pilot")
config5.apply_vocab_level(cfg, 7)
dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
DATA = pathlib.Path(os.environ["PRODUCTGPT_DATA"])
recs = dsm.load_json_dataset(str(DATA / cfg["data_file"]))[:240]
ds = dsm.TransformerDataset(recs, max_events=96)
items = [ds[i] for i in range(len(ds))]
batch = dsm.collate_multistream(items)
vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
env = AcquisitionEnv(vocab).fit(items, holdout_from=28)

START = 48
real_seq = real_sequences(batch, start=START)
F_real = sm.functionals(real_seq["decisions"], real_seq["alive"],
                        real_seq["obtained"], real_seq["lto"], start=START)

pm = fit_marginal(items, holdout_from=28)
PM = fit_markov(items, holdout_from=28)
iid = simulate(batch, env, vocab, iid_sampler(pm), start=START, n_rep=1, seed=5)
mkv = simulate(batch, env, vocab, markov_sampler(PM), start=START, n_rep=1, seed=5)
F_iid = sm.functionals(iid["decisions"], iid["alive"], iid["obtained"], iid["lto"],
                       start=START)
F_mkv = sm.functionals(mkv["decisions"], mkv["alive"], mkv["obtained"], mkv["lto"],
                       start=START)
check("baseline output has the real sequences' shape",
      F_iid.shape == F_real.shape == F_mkv.shape, f"{F_real.shape}")

sc = sm.Scaler.fit(F_real)
Zr, Zi, Zm = sc(F_real), sc(F_iid), sc(F_mkv)
check("standardised features are finite",
      np.isfinite(Zr).all() and np.isfinite(Zi).all() and np.isfinite(Zm).all())

c_iid = sm.compare(Zr, Zi, n_boot=60, seed=2)
check("the pipeline rejects the i.i.d. baseline", not c_iid["inside_band"],
      f"stat={c_iid['stat']:.3f} vs q95={c_iid['null_q95']:.3f}, z={c_iid['z']:.1f}")
auc_iid = sm.c2st(Zr, Zi, seed=0)
check("C2ST sees the i.i.d. baseline", auc_iid["auc"] > 0.7, f"AUC={auc_iid['auc']:.3f}")

c_mkv = sm.compare(Zr, Zm, n_boot=60, seed=2)
check("markov1 scores closer to real than i.i.d. does",
      c_mkv["stat"] < c_iid["stat"],
      f"markov {c_mkv['stat']:.3f} < iid {c_iid['stat']:.3f}")

top = np.argsort(auc_iid["importance"])[::-1][:4]
print("     functionals the classifier used against i.i.d.: "
      + ", ".join(NAMES[i] for i in top))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f_ in FAIL:
        print(f"   FAILED: {f_}")
    sys.exit(1)
