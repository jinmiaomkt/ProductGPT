"""
R30 stage 5: the product-embedding map, from the frozen level-7 models.

WHAT THIS IS, AND IS NOT. R34 showed our offer rotation does not identify a
substitution kernel, so this map is NOT a perception map, a preference map or a
substitution matrix. It is a description of what the trained model GROUPS
TOGETHER when predicting the next decision. That is worth showing -- if the
model has learned that 3-star weapons behave alike, it should be visible -- but
it licenses no claim about consumer preference.

Because an embedding space is only defined up to rotation, the map itself is
shown for one seed while the STATISTICS are rotation-invariant and computed per
seed, so their spread across seeds is reported:

    neighbour purity   for each product, the share of its k nearest neighbours
                       (cosine) sharing its rarity / weapon type / element
    separation         mean within-group cosine minus mean between-group cosine

A purity near the group's population share means no structure; higher means the
model groups by that attribute.

    python scripts/stage5_embeddings.py --tags b17_hyb_v7 b17_tf_v7 \\
        --out-dir results/r30/stage5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config5
import dataset_multistream
import model_multistream_state_space
from eval_per_occasion import build_model
from shared.features import load_feature_tensor
from train5_multistream import build_loaders, pick_device, set_seed

GROUPS = {
    "rarity": ["Rarity"],
    "weapon type": ["WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow",
                    "WeaponTypeMagic", "WeaponTypePolearm"],
    "element": ["EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire",
                "EthnicityThunder", "EthnicityWind"],
    "figure vs weapon": ["type_figure"],
    "limited (LTO)": ["LTO"],
}


def labels_for(tab: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """One label per product: the single column's value, or the argmax of a one-hot block."""
    if len(cols) == 1:
        return tab[cols[0]].fillna(-1).astype(float).to_numpy()
    block = tab[cols].fillna(0).to_numpy()
    lab = block.argmax(1).astype(float)
    lab[block.sum(1) == 0] = -1
    return lab


def purity(emb: np.ndarray, lab: np.ndarray, k: int = 5) -> tuple[float, float]:
    ok = lab >= 0
    e = emb[ok] / np.linalg.norm(emb[ok], axis=1, keepdims=True).clip(1e-9)
    l = lab[ok]
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :k]
    share = float(np.mean(l[nn] == l[:, None]))
    same = l[:, None] == l[None, :]
    off = ~np.eye(len(l), dtype=bool)
    sep = float(sim[same & off].mean() - sim[~same & off].mean())
    # chance level: probability two random products share a label
    _, cnt = np.unique(l, return_counts=True)
    chance = float(((cnt / cnt.sum()) ** 2).sum())
    return share, sep, chance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--runs-root", default="/storage/home/jinmiao/ProductGPT/runs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    device = pick_device()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for stem in a.tags:
        stats, maps = {}, None
        for ck in sorted(Path(a.runs_root).glob(f"*_b4_{stem}_s[0-9]/**/best.pt")):
            state = torch.load(ck, map_location=device, weights_only=False)
            cfg = state["cfg"]
            if int(cfg.get("vocab_level", 6)) != 7:
                print(f"[emb] {ck.parent.parent.parent.name}: not level 7, skipped")
                continue
            dataset_multistream.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
            model_multistream_state_space.set_product_range(
                cfg["first_prod_id"], cfg["last_prod_id"], cfg["unk_prod_id"])
            set_seed(cfg["seed"])
            _, _, _, num_users = build_loaders(cfg)
            feat = load_feature_tensor(config5.feature_file_path(cfg),
                                       id_column=cfg.get("feature_id_column"),
                                       first_prod_id=cfg["first_prod_id"],
                                       last_prod_id=cfg["last_prod_id"],
                                       max_token_id=cfg["vocab_size_src"] - 1)
            model = build_model(cfg, feat, num_users, device)
            model.load_state_dict(state["model_state_dict"], strict=False)
            model.eval()
            ids = torch.arange(cfg["first_prod_id"], cfg["last_prod_id"] + 1, device=device)
            with torch.no_grad():
                emb = model.product_embed(ids).float().cpu().numpy()      # (P, d_model)

            tab = pd.read_excel(config5.feature_file_path(cfg)).sort_values("NewProductIndex7")
            tab = tab[(tab["NewProductIndex7"] >= cfg["first_prod_id"])
                      & (tab["NewProductIndex7"] <= cfg["last_prod_id"])]
            if len(tab) != len(emb):
                raise SystemExit(f"table/embedding mismatch: {len(tab)} vs {len(emb)}")
            for name, cols in GROUPS.items():
                if not all(c in tab.columns for c in cols):
                    continue
                share, sep, chance = purity(emb, labels_for(tab, cols), a.k)
                stats.setdefault(name, []).append({"purity": share, "separation": sep,
                                                   "chance": chance})
            if maps is None:                       # keep the first seed for the picture
                c = emb - emb.mean(0, keepdims=True)
                u, s, vt = np.linalg.svd(c, full_matrices=False)
                xy = (u[:, :2] * s[:2])
                maps = {"xy": xy.tolist(),
                        "product_index7": tab["NewProductIndex7"].astype(int).tolist(),
                        "rarity": tab["Rarity"].fillna(-1).tolist(),
                        "type_figure": tab["type_figure"].fillna(-1).tolist(),
                        "explained": (s[:2] ** 2 / (s ** 2).sum()).tolist(),
                        "seed": int(cfg["seed"])}
        if not stats:
            print(f"[emb] no level-7 runs for {stem}")
            continue
        res = {"stem": stem, "k": a.k, "map": maps,
               "stats": {n: {"purity_mean": float(np.mean([x["purity"] for x in v])),
                             "purity_sd": float(np.std([x["purity"] for x in v], ddof=1))
                             if len(v) > 1 else 0.0,
                             "separation_mean": float(np.mean([x["separation"] for x in v])),
                             "chance": v[0]["chance"], "n_seeds": len(v)}
                         for n, v in stats.items()}}
        (out / f"embmap_{stem}.json").write_text(json.dumps(res), encoding="utf-8")
        print(f"\n[emb] {stem}  ({res['stats'][list(res['stats'])[0]]['n_seeds']} seeds)")
        print(f"  {'attribute':<18} {'purity':>16} {'chance':>8} {'lift':>7} {'separation':>11}")
        for n, v in res["stats"].items():
            print(f"  {n:<18} {v['purity_mean']:>8.3f}+/-{v['purity_sd']:<6.3f} "
                  f"{v['chance']:>8.3f} {v['purity_mean'] / v['chance']:>7.2f} "
                  f"{v['separation_mean']:>11.3f}")


if __name__ == "__main__":
    main()
