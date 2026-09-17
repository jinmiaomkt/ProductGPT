"""
Two pre-build checks for the additive attention stock (proposed R26).

CHECK 1 -- is there anything for a copy-tier term to learn?
    For the 39 limited-time 5-star products (ids 18-56, the only products the
    data identifies individually), how many copies do customers hold by the end
    of their history? Duplicates of characters upgrade them up to a cap of 7
    copies; weapons up to 5. If almost nobody holds a second copy, g(tier) has
    nothing to fit.

CHECK 2 -- does substitution show up, and does it survive stable taste?
    Unit: a customer x occasion x character banner (Fig A = decisions 3/4,
    Fig B = 5/6; LTO block order is [Fig5A, Fig5B, Weapon5A, Weapon5B] per
    GenerateJSON.R) whose featured 5-star Y the customer does NOT yet own.
    Outcome: the customer pulls on that banner at that occasion.
    Treatment: the customer already owns a DIFFERENT limited-time 5-star
    character of Y's element.

    (a) raw pull rates, owners vs non-owners of a same-element character,
        within strata of how many limited-time characters the customer owns;
    (b) a linear probability model with customer x element fixed effects and
        offered-product fixed effects. Customer x element effects absorb stable
        taste for an element ("does not care for Pyro"), so the coefficient is
        identified only by CHANGES in ownership within customer and element --
        the timing argument. Controls: log number of limited-time characters
        owned (depletion / general engagement) and log occasions since the last
        limited-time 5-star acquisition. SEs clustered by customer.

    Substitution predicts a negative coefficient in (b). Pure stable taste
    predicts (a) negative but (b) near zero.

Acquisitions at row t are recorded in row t's obtained block (the raw,
unshifted data), so "owned before t" uses rows < t only.

Aggregate statistics only -- never uids, token values or per-customer records.

USAGE
    python scripts/copies_substitution_check.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_multistream import load_json_dataset, parse_token_ids  # noqa: E402

LTO_LO, LTO_HI = 18, 56
LTO_W, OBT_W = 4, 10
RATE = LTO_W + OBT_W + 1
ELEMENTS = ["Ice", "Rock", "Water", "Fire", "Thunder", "Wind", "Grass"]


def product_table(root: str):
    df = pd.read_excel(Path(root) / "SelectedFigureWeaponEmbeddingIndex.xlsx")
    idcol = [c for c in df.columns if str(c).startswith("NewProductIndex6")][0]
    fig, elem = {}, {}
    for _, r in df.iterrows():
        pid = int(r[idcol])
        fig[pid] = int(r["type_figure"]) == 1
        hot = [e for e in ELEMENTS if f"Ethnicity{e}" in df.columns and r[f"Ethnicity{e}"] == 1]
        elem[pid] = hot[0] if len(hot) == 1 else None
    return fig, elem


def demean(df, cols, groups, iters=30):
    """Alternating projections: residualise cols on several sets of fixed effects."""
    out = df[cols].astype(float).copy()
    for _ in range(iters):
        for g in groups:
            out = out - out.groupby(df[g]).transform("mean")
    return out


def main() -> None:
    root = os.environ.get("PRODUCTGPT_DATA")
    if not root:
        sys.exit("PRODUCTGPT_DATA is not set")
    fig, elem = product_table(root)
    lto_ids = [p for p in range(LTO_LO, LTO_HI + 1) if p in fig]
    cap = {p: 7 if fig[p] else 5 for p in lto_ids}

    recs = load_json_dataset(str(Path(root) / "clean_list_int_wide4_simple6_IPT.json"))

    # ---------------------------------------------------------------- check 1
    holdings = Counter()          # copies-held bucket -> number of customer x product holdings
    dup_acq = over_cap_acq = all_acq = 0
    cust_with_dup = cust_with_any = 0
    by_type = {True: Counter(), False: Counter()}

    rows = []                     # check 2 panel
    for ci, rec in enumerate(recs):
        ai = parse_token_ids(rec.get("AggregateInput"))
        dec = parse_token_ids(rec.get("Decision", []))
        n = min(len(ai) // RATE, len(dec))
        count = Counter()
        last_acq = None
        for t in range(n):
            base = t * RATE
            lto = ai[base:base + LTO_W]
            # check 2 observations use ownership BEFORE row t
            owned_figs = [p for p in count if fig.get(p)]
            n_owned = len(owned_figs)
            elem_owned = Counter(elem[p] for p in owned_figs)
            since = (t - last_acq) if last_acq is not None else None
            for slot, pulls in ((0, (3, 4)), (1, (5, 6))):
                y = lto[slot]
                if not (LTO_LO <= y <= LTO_HI) or not fig.get(y) or elem.get(y) is None:
                    continue
                if count[y] > 0:
                    continue
                rows.append((ci, elem[y], y, int(dec[t] in pulls),
                             int(elem_owned[elem[y]] > 0), n_owned,
                             -1 if since is None else since))
            # then update ownership with row t's acquisitions
            for p in ai[base + LTO_W:base + LTO_W + OBT_W]:
                if LTO_LO <= p <= LTO_HI and p in cap:
                    all_acq += 1
                    count[p] += 1
                    dup_acq += count[p] >= 2
                    over_cap_acq += count[p] > cap[p]
                    last_acq = t
        held = [(p, c) for p, c in count.items() if c > 0]
        if held:
            cust_with_any += 1
            cust_with_dup += any(c >= 2 for _, c in held)
        for p, c in held:
            b = c if c <= cap[p] else "over cap"
            holdings[b] += 1
            by_type[fig[p]][b] += 1

    print("=" * 78)
    print("CHECK 1  copies of limited-time 5-stars held at end of history")
    print("=" * 78)
    print(f"customers holding any limited-time 5-star: {cust_with_any:,}")
    print(f"  of whom hold >= 2 copies of at least one: {cust_with_dup / max(cust_with_any, 1):.1%}")
    print(f"acquisitions of limited-time 5-stars: {all_acq:,}")
    print(f"  that were duplicates (copy >= 2):   {dup_acq / max(all_acq, 1):.1%}")
    print(f"  beyond the cap (7 char / 5 weapon): {over_cap_acq / max(all_acq, 1):.2%}")
    for label, is_fig, c in (("characters (cap 7)", True, 7), ("weapons (cap 5)", False, 5)):
        tot = sum(by_type[is_fig].values())
        if not tot:
            continue
        parts = "  ".join(f"{k}:{by_type[is_fig][k] / tot:.1%}"
                          for k in list(range(1, c + 1)) + ["over cap"] if by_type[is_fig][k])
        print(f"holdings, {label}, n={tot:,}:  copies  {parts}")

    # ---------------------------------------------------------------- check 2
    df = pd.DataFrame(rows, columns=["cust", "elem", "y", "pull", "same", "n_owned", "since"])
    print()
    print("=" * 78)
    print("CHECK 2  substitution: pull on a character banner whose 5-star is not yet owned")
    print("=" * 78)
    print(f"observations (customer x occasion x banner): {len(df):,}   customers: {df.cust.nunique():,}")
    print(f"base pull rate: {df.pull.mean():.4f}   share owning a same-element character: {df.same.mean():.3f}")

    print("\n(a) raw pull rates by number of limited-time characters owned")
    df["stratum"] = pd.cut(df.n_owned, [-1, 0, 1, 2, 4, 1000], labels=["0", "1", "2", "3-4", "5+"])
    tab = df.groupby(["stratum", "same"], observed=True).pull.agg(["mean", "size"]).unstack("same")
    for s in tab.index:
        m0 = tab.loc[s, ("mean", 0)] if ("mean", 0) in tab.columns else np.nan
        m1 = tab.loc[s, ("mean", 1)] if ("mean", 1) in tab.columns else np.nan
        n1 = tab.loc[s, ("size", 1)] if ("size", 1) in tab.columns else 0
        print(f"  owns {s:>3}:  no same-element {m0:.4f}   same-element {m1:.4f}   "
              f"(n same = {0 if pd.isna(n1) else int(n1):,})")

    print("\n(b) linear probability, customer x element FE + offered-product FE, SE clustered by customer")
    df["ce"] = df.cust.astype(str) + "_" + df.elem
    df["log_owned"] = np.log1p(df.n_owned)
    df["has_since"] = (df.since >= 0).astype(int)
    df["log_since"] = np.where(df.since >= 0, np.log1p(df.since.clip(lower=0)), 0.0)
    # singleton customer x element cells carry no within variation
    df = df[df.groupby("ce").pull.transform("size") > 1].copy()
    for spec, xs in (("same only", ["same"]),
                     ("+ controls", ["same", "log_owned", "has_since", "log_since"])):
        d = demean(df, ["pull"] + xs, ["ce", "y"])
        X, yv = d[xs].to_numpy(), d["pull"].to_numpy()
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ yv
        e = yv - X @ beta
        score = pd.DataFrame(X * e[:, None]).groupby(df.cust.to_numpy()).sum().to_numpy()
        V = XtX_inv @ (score.T @ score) @ XtX_inv
        se = np.sqrt(np.diag(V))
        within = int((df.groupby("ce").same.transform("nunique") > 1).sum())
        print(f"  [{spec}]  obs {len(df):,}; obs in cells where 'same' changes: {within:,}")
        for name, b, s in zip(xs, beta, se):
            print(f"      {name:<10} {b:+.5f}  (se {s:.5f}, t {b / s:+.2f})")
    print(f"\n  for scale: base pull rate {df.pull.mean():.4f}")


if __name__ == "__main__":
    main()
