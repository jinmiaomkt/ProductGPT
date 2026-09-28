"""
How much of the product map did the model LEARN, and how much did we feed it?

`SpecialPlusFeatureLookup` builds a product's vector as

    id_embed(p)  +  gamma * feat_proj(attributes(p))

so products sharing attributes are pushed together BY CONSTRUCTION. The
stage-5 map (rarity purity 0.91 against 0.41 chance) therefore cannot be read
as evidence that the model discovered rarity: the feature branch alone would
produce clustering by rarity even at initialisation.

This splits the two branches and computes the same neighbour purity on each:

    combined   what stage 5 reported
    id only    what the model learned from behaviour ALONE
    feature    what our input table implies, learned projection applied

If id-only purity is near chance, the map is our attribute table reflected
back. Parameters only -- no data, no forward pass.

    python3 scripts/embedding_decompose.py --tags b17_hyb_v7 b17_tf_v7 b17_gru_v7
"""
from __future__ import annotations

import argparse
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

GROUPS = {
    "rarity": ["Rarity"],
    "weapon type": ["WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow",
                    "WeaponTypeMagic", "WeaponTypePolearm"],
    "element": ["EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire",
                "EthnicityThunder", "EthnicityWind"],
    "figure vs weapon": ["type_figure"],
}


def labels_for(tab: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if len(cols) == 1:
        return tab[cols[0]].fillna(-1).astype(float).to_numpy()
    block = tab[cols].fillna(0).to_numpy()
    lab = block.argmax(1).astype(float)
    lab[block.sum(1) == 0] = -1
    return lab


def purity(emb: np.ndarray, lab: np.ndarray, k: int = 5):
    ok = lab >= 0
    e = emb[ok]
    n = np.linalg.norm(e, axis=1, keepdims=True)
    e = e / np.where(n > 1e-9, n, 1.0)
    l = lab[ok]
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    _, cnt = np.unique(l, return_counts=True)
    chance = float(((cnt / cnt.sum()) ** 2).sum())
    return float(np.mean(l[nn] == l[:, None])), chance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--vocab", default=os.path.expanduser(
        "~/ProductGPT/data/perproduct/ProductVocab7.xlsx"))
    a = ap.parse_args()

    tab = pd.read_excel(a.vocab).sort_values("NewProductIndex7")
    for stem in a.tags:
        rows = defaultdict(lambda: defaultdict(list))
        gammas = []
        for ck in sorted(Path(a.runs_root).glob(f"*_b4_{stem}_s[0-9]/**/best.pt")):
            state = torch.load(ck, map_location="cpu", weights_only=False)
            cfg, sd = state["cfg"], state["model_state_dict"]
            first, last = cfg["first_prod_id"], cfg["last_prod_id"]
            ids = np.arange(first, last + 1)
            t = tab[(tab["NewProductIndex7"] >= first) & (tab["NewProductIndex7"] <= last)]
            key_id = [k for k in sd if k.endswith("product_embed.id_embed.weight")]
            key_pr = [k for k in sd if k.endswith("product_embed.feat_proj.weight")]
            key_g = [k for k in sd if k.endswith("product_embed.gamma")]
            if not key_id or not key_pr:
                print(f"  {stem}: no product embedding in this checkpoint")
                break
            id_emb = sd[key_id[0]][ids].float().numpy()
            feats = torch.as_tensor(
                t[[c for c in tab.columns if c in sum(GROUPS.values(), [])
                   or c not in ("ProductID", "Name", "NewProductIndex7", "v2_code",
                                "NewProductIndex6")]].fillna(0).to_numpy(dtype=float),
                dtype=torch.float32)
            W = sd[key_pr[0]].float()
            if feats.shape[1] != W.shape[1]:
                feats = feats[:, :W.shape[1]] if feats.shape[1] > W.shape[1] else \
                    torch.cat([feats, torch.zeros(len(feats), W.shape[1] - feats.shape[1])], 1)
            feat_emb = (feats @ W.T).numpy()
            gamma = float(sd[key_g[0]]) if key_g else 1.0
            gammas.append(gamma)
            combined = id_emb + gamma * feat_emb
            for name, cols in GROUPS.items():
                if not all(c in t.columns for c in cols):
                    continue
                lab = labels_for(t, cols)
                for branch, e in (("combined", combined), ("id only", id_emb),
                                  ("feature only", feat_emb)):
                    p, ch = purity(e, lab)
                    rows[name][branch].append(p)
                    rows[name]["chance"] = ch
        if not rows:
            continue
        print(f"\n{stem}   (gamma = {st.mean(gammas):.2f}, "
              f"{len(gammas)} seeds)")
        print(f"  {'attribute':<18} {'chance':>8} {'combined':>10} {'id only':>10} "
              f"{'feature only':>13}")
        for name, d in rows.items():
            print(f"  {name:<18} {d['chance']:>8.3f} "
                  f"{st.mean(d['combined']):>10.3f} {st.mean(d['id only']):>10.3f} "
                  f"{st.mean(d['feature only']):>13.3f}")


if __name__ == "__main__":
    main()
