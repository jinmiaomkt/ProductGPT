"""R47 part A: model-free forgetting curves (pre-registered in EXPERIMENTS.md R47).

Unit: an occasion at which a featured product is on offer (offer slot 0 =
figure banner A, 1 = figure banner B, 2-3 = weapon banner; gacha_env's
FEATURED_SLOTS). Outcome: a pull on that slot's banner at that occasion.

Satiation gap at an owned occasion = y - baseline, where the baseline is
P(pull) among NOT-owned occasions in the same stratum (campaign x slot x spend
tercile x pity bin on that banner). Owned means the player acquired the product
before the occasion (row t carries o_(t-1), so ownership uses rows <= t).

Curves: the gap against elapsed time since the last acquisition of the product,
in occasions and in hours (cumulative IPT):
  within  the acquisition was in the same campaign as the offer
  across  it was in an earlier campaign (a rerun)
Slope of the gap on log(1 + hours), with a 1,000-resample customer bootstrap CI.

Calibration campaigns only (<= 27): the holdout stays untouched. Aggregates only.

    python scripts/r47_partA_model_free.py --out results/r47/partA.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
import model_multistream_state_space as mm  # noqa: E402
from shared.features import load_feature_tensor  # noqa: E402

SLOT_DECS = {0: (3, 4), 1: (5, 6), 2: (7, 8), 3: (7, 8)}   # featured slot -> its banner's decisions
SLOT_BANNER = {0: 1, 1: 2, 2: 3, 3: 3}                     # pity_depth banner index
HOUR_BINS = [0, 1, 6, 24, 72, 168, 504, np.inf]
OCC_BINS = [1, 2, 4, 8, 16, 32, 64, 128, np.inf]
PITY_BINS = [0, 20, 50, 75, np.inf]
# Days into the campaign, measured from the customer's first row of that
# campaign. InsertNotBuy_GenerateJSON_IPT.R lays a fixed interval grid from each
# campaign's StartingDate over the player's active campaigns and inserts a
# no-buy row at the end of each empty interval. So a customer's first row of a
# campaign lies within one interval of the campaign's start: a proxy for
# calendar days into the campaign, accurate to one interval.
DAY_BINS = [0, 2, 5, 10, 15, np.inf]


def binlab(edges, i):
    lo, hi = edges[i], edges[i + 1]
    return f"[{lo:g}, {hi:g})" if np.isfinite(hi) else f">= {lo:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-campaign", type=int, default=27)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-stratum", type=int, default=20)
    ap.add_argument("--strata-days", type=int, default=0,
                    help="ROBUSTNESS (added 28 Oct 2026, after the pre-registered run): 1 = also "
                         "stratify the baseline by days into the campaign, so late-campaign "
                         "owned occasions are compared with late-campaign non-owners")
    ap.add_argument("--out", default="results/r47/partA.json")
    a = ap.parse_args()

    cfg = config5.get_config("hpcc")
    config5.apply_vocab_level(cfg, 7)
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    mm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"], cfg["unk_prod_id"])
    raw = dsm.load_json_dataset(str(config5.data_path(cfg)))
    ds = dsm.TransformerDataset(raw, ai_rate=cfg["ai_rate"], lto_len=4, obtained_len=10,
                                prev_dec_len=1, max_events=None, shift_obtained=True)
    feat = load_feature_tensor(config5.feature_file_path(cfg), id_column=cfg.get("feature_id_column"),
                               first_prod_id=cfg["first_prod_id"], last_prod_id=cfg["last_prod_id"],
                               max_token_id=cfg["vocab_size_src"] - 1)
    rarity = feat[:, 0].float()
    lo, hi = cfg["first_prod_id"], cfg["last_prod_id"]
    print(f"[data] {len(ds):,} customers; campaigns <= {a.max_campaign}")

    # spend terciles: purchase decisions (1-8) in the calibration campaigns
    spend = []
    for i in range(len(ds)):
        it = ds[i]
        m = it["campaign"] <= a.max_campaign
        spend.append(int(((it["label"] >= 1) & (it["label"] <= 8) & m).sum()))
    cuts = np.quantile(spend, [1 / 3, 2 / 3])
    terc = np.digitize(spend, cuts)

    recs = defaultdict(list)          # column -> list
    for i in range(len(ds)):
        it = ds[i]
        lab, camp = it["label"].numpy(), it["campaign"].numpy()
        lto, obt = it["lto"].numpy(), it["obtained"].numpy()
        T = np.cumsum(it["ipt"].numpy()) if "ipt" in it else np.arange(len(lab), dtype=float)
        pity = mm.pity_depth(it["obtained"][None], it["prev_decision"][None], rarity)[0].numpy()
        rows_of = defaultdict(list)
        for r in range(obt.shape[0]):
            for p in set(obt[r].tolist()):
                if lo <= p <= hi:
                    rows_of[p].append(r)
        rows_of = {p: np.asarray(v) for p, v in rows_of.items()}
        t_start = {}
        for r in range(len(camp)):
            if camp[r] > 0 and camp[r] not in t_start:
                t_start[int(camp[r])] = T[r]
        ok = (lab >= 1) & (lab <= 9) & (camp >= 1) & (camp <= a.max_campaign)
        for t in np.flatnonzero(ok):
            for j in range(4):
                p = int(lto[t, j])
                if not (lo <= p <= hi):
                    continue
                b = SLOT_BANNER[j]
                cap = mm.HARD_PITY5[b]
                pr = float(pity[t, 2 * b]) * cap
                y = int(lab[t] in SLOT_DECS[j])
                rr = rows_of.get(p)
                owned, occ, hrs, within = 0, np.nan, np.nan, -1
                if rr is not None:
                    k = np.searchsorted(rr, t, side="right") - 1
                    if k >= 0:
                        r = int(rr[k])
                        owned = 1
                        occ = t - r + 1
                        hrs = float(T[t] - T[r - 1]) if r >= 1 else np.nan
                        within = int(camp[r - 1] == camp[t]) if r >= 1 else -1
                recs["cust"].append(i)
                recs["slot"].append(j)
                recs["camp"].append(int(camp[t]))
                recs["terc"].append(int(terc[i]))
                recs["pbin"].append(int(np.digitize(pr, PITY_BINS[1:-1])))
                recs["dbin"].append(int(np.digitize((T[t] - t_start[int(camp[t])]) / 24.0,
                                                    DAY_BINS[1:-1])))
                recs["y"].append(y)
                recs["owned"].append(owned)
                recs["occ"].append(occ)
                recs["hrs"].append(hrs)
                recs["within"].append(within)
    R = {k: np.asarray(v) for k, v in recs.items()}
    n = len(R["y"])
    print(f"[data] {n:,} featured-offer occasions; {int(R['owned'].sum()):,} with the product owned")

    # baseline per stratum from not-owned occasions
    key = (R["camp"] * 1000 + R["slot"] * 100 + R["terc"] * 10 + R["pbin"])
    if a.strata_days:
        key = key * 10 + R["dbin"]
        print("[robustness] baseline strata also include days into the campaign "
              f"(bins {DAY_BINS[:-1]} days)")
        print("[robustness] occasions per day bin: " +
              ", ".join(f"{binlab(DAY_BINS, k_)}: {int((R['dbin'] == k_).sum()):,}"
                        for k_ in range(len(DAY_BINS) - 1)))
    base_sum, base_n = defaultdict(float), defaultdict(int)
    for k_, y_, o_ in zip(key, R["y"], R["owned"]):
        if not o_:
            base_sum[k_] += y_
            base_n[k_] += 1
    base = np.array([base_sum[k_] / base_n[k_] if base_n[k_] >= a.min_stratum else np.nan
                     for k_ in key])
    resid = R["y"] - base
    usable = (R["owned"] == 1) & np.isfinite(base) & np.isfinite(R["hrs"])
    print(f"[gap] {int(usable.sum()):,} owned occasions with a usable baseline")

    out = {"strata_days": bool(a.strata_days),
           "n_offer_occasions": int(n), "n_owned": int(R["owned"].sum()),
           "n_owned_usable": int(usable.sum()), "max_campaign": a.max_campaign, "curves": {}}
    rng = np.random.default_rng(0)
    for name, wflag in (("within", 1), ("across", 0)):
        m = usable & (R["within"] == wflag)
        cur = {"n": int(m.sum()), "mean_gap": float(resid[m].mean()) if m.any() else float("nan"),
               "hours": [], "occasions": []}
        for col, edges in (("hrs", HOUR_BINS), ("occ", OCC_BINS)):
            bi = np.digitize(R[col][m], edges[1:-1])
            rows = []
            for k_ in range(len(edges) - 1):
                sel = bi == k_
                rows.append({"bin": binlab(edges, k_), "n": int(sel.sum()),
                             "gap": float(resid[m][sel].mean()) if sel.any() else float("nan"),
                             "p_owned": float(R["y"][m][sel].mean()) if sel.any() else float("nan"),
                             "baseline": float(base[m][sel].mean()) if sel.any() else float("nan")})
            cur["hours" if col == "hrs" else "occasions"] = rows
        # slope of the gap on log(1 + hours), customer bootstrap
        x, yv, cu = np.log1p(R["hrs"][m]), resid[m], R["cust"][m]
        custs = np.unique(cu)
        agg = np.zeros((len(custs), 5))
        idx = {c: k_ for k_, c in enumerate(custs)}
        for xi, yi, ci in zip(x, yv, cu):
            row = agg[idx[ci]]
            row += (1, xi, yi, xi * yi, xi * xi)

        def slope(rows):
            s = rows.sum(0)
            nn, sx, sy, sxy, sxx = s
            den = nn * sxx - sx * sx
            return (nn * sxy - sx * sy) / den if den > 0 else np.nan

        est = slope(agg)
        boots = [slope(agg[rng.integers(0, len(custs), len(custs))]) for _ in range(a.boot)]
        ci = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
        shrinks = bool((ci[0] > 0 or ci[1] < 0) and np.sign(est) == -np.sign(cur["mean_gap"]))
        cur.update({"slope_log1p_hours": float(est), "slope_ci95": ci, "customers": int(len(custs)),
                    "gap_shrinks": shrinks})
        # by spend tercile
        cur["mean_gap_by_tercile"] = {str(k_): float(resid[m & (R["terc"] == k_)].mean())
                                      for k_ in range(3) if (m & (R["terc"] == k_)).any()}
        out["curves"][name] = cur
        print(f"\n== {name}: {cur['n']:,} owned occasions, {len(custs):,} customers, "
              f"mean gap {cur['mean_gap']:+.4f}")
        print(f"   slope on log(1+hours) {est:+.5f}, 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}] -> "
              f"{'gap SHRINKS with time' if shrinks else 'no detectable shrinkage'}")
        print(f"   {'hours since acquired':<22}{'n':>9}{'gap':>9}{'P(pull|own)':>13}{'baseline':>10}")
        for r_ in cur["hours"]:
            print(f"   {r_['bin']:<22}{r_['n']:>9,}{r_['gap']:>+9.4f}{r_['p_owned']:>13.4f}"
                  f"{r_['baseline']:>10.4f}")
        print(f"   {'occasions since':<22}{'n':>9}{'gap':>9}")
        for r_ in cur["occasions"]:
            print(f"   {r_['bin']:<22}{r_['n']:>9,}{r_['gap']:>+9.4f}")
        print(f"   mean gap by spend tercile: {cur['mean_gap_by_tercile']}")

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
