"""
Draw the ESTIMATED substitution structure: the attribute kernel's coefficients,
and the product map they imply.

What this is, and how it differs from the map we had to withdraw. The stage-5
embedding map turned out to be the attribute table reflected back (the id branch
alone sits at chance). This is a different object:

    kappa_theta(j,p) proportional to exp( theta_self*1[j=p]
                                          + sum_a w_a * 1[attribute a matches] )

The PRODUCT POSITIONS come from attributes we already know. What is ESTIMATED is
the METRIC -- which attributes actually carry satiation, and how strongly. That
is a small number of identified parameters (the rank argument: a low-dimensional
kernel fits inside the span the calendar provides, a free P-by-P kernel does
not), so the picture is licensed where the free-kernel picture was not.

Call it a satiation-weighted attribute map, never a perception map.

    python3 scripts/kernel_map.py --coef results/r30/kernel_coef.json \
        --vocab <ProductVocab7.xlsx> --out results/paper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def implied_kernel(theta_self: float, w: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Row-stochastic kappa from the attribute matches, as the model builds it."""
    P = feats.shape[0]
    same = (feats[:, None, :] == feats[None, :, :]).astype(float)       # (P,P,F)
    logits = same @ w + theta_self * np.eye(P)
    logits -= logits.max(axis=1, keepdims=True)
    k = np.exp(logits)
    return k / k.sum(axis=1, keepdims=True)


def mds(d: np.ndarray, dim: int = 2) -> np.ndarray:
    """Classical multidimensional scaling on a distance matrix."""
    n = len(d)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(-vals)
    vals, vecs = vals[order][:dim], vecs[:, order][:, :dim]
    return vecs * np.sqrt(np.maximum(vals, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coef", required=True, help="json with theta_self and attr_w")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coef = json.loads(Path(a.coef).read_text())
    names = coef["columns"]
    w = np.asarray(coef["attr_w"], dtype=float)
    theta_self = float(coef["attr_self"])
    tab = pd.read_excel(a.vocab).sort_values("NewProductIndex7")
    tab = tab[(tab["NewProductIndex7"] >= coef["first"]) & (tab["NewProductIndex7"] <= coef["last"])]
    feats = tab[names].fillna(0).to_numpy(dtype=float)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. which attributes carry satiation -----------------------------
    order = np.argsort(-np.abs(w))[:12]
    seeds = np.asarray(coef.get("attr_w_seeds", [w]), dtype=float)
    err = seeds.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(w)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(order))
    ax.barh(y, w[order][::-1], xerr=err[order][::-1], capsize=2.5, ecolor="#5F5E5A",
            color=["#378ADD" if v >= 0 else "#D85A30" for v in w[order][::-1]])
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order][::-1], fontsize=8)
    ax.axvline(0, color="#888780", lw=0.8)
    ax.set_xlabel("estimated coefficient on 'shares this attribute'", fontsize=9)
    ax.set_title(f"What carries satiation across products\n"
                 f"(own-product weight {theta_self:.2f}; larger = more substitutable)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "kernel_coefficients.png", dpi=200)
    plt.close(fig)

    # ---- 2. the map the estimated metric implies -------------------------
    k = implied_kernel(theta_self, w, feats)
    sym = 0.5 * (k + k.T)
    np.fill_diagonal(sym, sym.max())
    d = np.sqrt(np.maximum(sym.max() - sym, 0))
    xy = mds(d)
    rarity = tab["Rarity"].fillna(0).to_numpy()
    is_fig = tab["type_figure"].fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for r, marker in ((5, "o"), (4, "s"), (3, "^")):
        for f, fill in ((1, True), (0, False)):
            m = (rarity == r) & (is_fig == f)
            if not m.any():
                continue
            ax.scatter(xy[m, 0], xy[m, 1], marker=marker, s=46,
                       facecolors="#378ADD" if fill else "none",
                       edgecolors="#185FA5", linewidths=1.1,
                       label=f"{int(r)}-star {'character' if f else 'weapon'}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Products positioned by the ESTIMATED substitution metric\n"
                 "(positions from known attributes; the metric is what is estimated)",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "kernel_map.png", dpi=200)
    plt.close(fig)

    off = k[~np.eye(len(k), dtype=bool)]
    print(f"own-product weight theta_self = {theta_self:.3f}")
    print(f"largest attribute coefficients:")
    for i in order[:6]:
        print(f"   {names[i]:<28} {w[i]:+.4f}")
    print(f"kappa: diagonal mean {np.diag(k).mean():.3f}, "
          f"off-diagonal mean {off.mean():.5f}, max off-diagonal {off.max():.4f}")
    print(f"wrote {out / 'kernel_coefficients.png'} and {out / 'kernel_map.png'}")


if __name__ == "__main__":
    main()
