"""R41: what happens AFTER a five-star hit, in the REAL holdout data?

Jin's hypothesis (13 Oct): winning a limited-time five-star figure triggers
positive emotion and a different, heavier purchase sequence -- the root of
compulsive buying -- so a free-running model that samples the lottery is
exploring legitimate branches that the single realised path never shows.

Row t carries the outcome of the decision at t-1 (obtained block) and the
decision at t (label). Table 1: among rows whose previous decision was a
pull, grouped by what the pull delivered. Table 2 (the quasi-experiment):
previous decision = a TEN-pull on a figure banner, stratified by the pity
depth BEFORE that pull, so only the lottery outcome differs between groups.
Aggregates only; nothing row-level is printed.

    python scripts/r41_post_hit.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "shared"):
    sys.path.insert(0, str(p))
import config5                                               # noqa: E402
import dataset_multistream as dsm                            # noqa: E402
from shared.features import load_feature_tensor               # noqa: E402
from train5_multistream import build_loaders                  # noqa: E402
import model_multistream_state_space as mm                  # noqa: E402
from model_multistream_state_space import pity_depth, HARD_PITY5  # noqa: E402

REV = np.array([1, 10, 1, 10, 1, 10, 1, 10, 0], dtype=float)
WINDOWS = (5, 20)
KEYS = ["n", "notbuy", "reg", "fig", "wep", "rev"] + \
       [f"{k}_{w}" for w in WINDOWS for k in ("live", "rev", "nb")]


def accumulate(a, m, y, c_live, c_rev, c_nb, S):
    bi, ti = np.nonzero(m)
    a["n"] += len(ti)
    yy = y[bi, ti]
    a["notbuy"] += (yy == 9).sum(); a["reg"] += np.isin(yy, (1, 2)).sum()
    a["fig"] += np.isin(yy, (3, 4, 5, 6)).sum(); a["wep"] += np.isin(yy, (7, 8)).sum()
    a["rev"] += REV[yy - 1].sum()
    for w in WINDOWS:
        e = np.minimum(ti + w, S)
        a[f"live_{w}"] += (c_live[bi, e] - c_live[bi, ti]).sum()
        a[f"rev_{w}"] += (c_rev[bi, e] - c_rev[bi, ti]).sum()
        a[f"nb_{w}"] += (c_nb[bi, e] - c_nb[bi, ti]).sum()


def show(title, acc):
    print(f"\n{title}")
    print(f"{'group':44s} {'n':>7s} {'P(NotBuy)':>9s} {'P(reg)':>7s} {'P(fig)':>7s} {'P(wep)':>7s} "
          f"{'E[rev]':>7s} | " + "  ".join(f"rev/occ nxt{w:<2d} NotBuy nxt{w:<2d}" for w in WINDOWS))
    for g, a in acc.items():
        n = max(a["n"], 1)
        row = (f"{g:44s} {int(a['n']):7d} {a['notbuy']/n:9.3f} {a['reg']/n:7.3f} {a['fig']/n:7.3f} "
               f"{a['wep']/n:7.3f} {a['rev']/n:7.2f} | ")
        row += "  ".join(f"{a[f'rev_{w}']/max(a[f'live_{w}'],1):12.2f} {a[f'nb_{w}']/max(a[f'live_{w}'],1):12.3f}"
                         for w in WINDOWS)
        print(row)


def main() -> None:
    cfg = config5.get_config("hpcc")
    config5.apply_vocab_level(cfg, 7)
    cfg["num_workers"] = 0
    cfg["max_events"] = 2048          # full history: the pity counter needs it
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    mm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"],
                         cfg.get("unk_prod_id", cfg["last_prod_id"] + 1))
    _, _, tests, _ = build_loaders(cfg)
    cell = tests["outsample_users_holdout_period"].dataset
    feat = load_feature_tensor(config5.feature_file_path(cfg),
                               id_column=cfg.get("feature_id_column"),
                               first_prod_id=cfg["first_prod_id"],
                               last_prod_id=cfg["last_prod_id"],
                               max_token_id=cfg["vocab_size_src"] - 1)
    rarity_t = feat[:, 0].float()
    rarity = rarity_t.numpy()
    lo, hi = cfg["first_prod_id"], cfg["last_prod_id"]

    t1 = ["featured FIGURE 5-star won", "featured WEAPON 5-star won", "other 5-star won", "pulled, no 5-star"]
    t2 = [f"{o}, pity before pull {p}" for p in ("< 30", "30-69", ">= 70")
          for o in ("won the featured figure", "5-star but not featured (50/50 lost)", "no 5-star")]
    acc1 = {g: {k: 0.0 for k in KEYS} for g in t1}
    acc2 = {g: {k: 0.0 for k in KEYS} for g in t2}

    for i in range(0, len(cell), 16):
        b = dsm.collate_multistream([cell[j] for j in range(i, min(i + 16, len(cell)))])
        y = b["label"].numpy(); lto = b["lto"].numpy(); obt = b["obtained"].numpy()
        B, S = y.shape
        prev = b["prev_decision"].numpy().reshape(B, S)
        live = y > 0
        valid = (obt >= lo) & (obt <= hi)
        rar = rarity[np.clip(obt, 0, rarity.shape[0] - 1)]
        is5 = ((rar >= 4.5) & valid).any(-1)
        feat_fig = np.zeros((B, S), bool); feat_wep = np.zeros((B, S), bool)
        feat_fig[:, 1:] = (((obt[:, 1:, :] == lto[:, :-1, 0:1]) | (obt[:, 1:, :] == lto[:, :-1, 1:2]))
                           & valid[:, 1:, :]).any(-1) & (lto[:, :-1, 0:2] > 0).any(-1)
        feat_wep[:, 1:] = (((obt[:, 1:, :] == lto[:, :-1, 2:3]) | (obt[:, 1:, :] == lto[:, :-1, 3:4]))
                           & valid[:, 1:, :]).any(-1) & (lto[:, :-1, 2:4] > 0).any(-1)
        pulled_prev = (prev >= 1) & (prev <= 8)
        base = live & pulled_prev; base[:, 0] = False
        rev = np.where(live, REV[np.clip(y, 1, 9) - 1], 0.0)
        nb = (y == 9).astype(float)
        c_live = np.concatenate([np.zeros((B, 1)), np.cumsum(live, 1)], 1)
        c_rev = np.concatenate([np.zeros((B, 1)), np.cumsum(rev, 1)], 1)
        c_nb = np.concatenate([np.zeros((B, 1)), np.cumsum(nb, 1)], 1)
        g1 = {t1[0]: base & feat_fig & is5, t1[1]: base & feat_wep & is5 & ~feat_fig,
              t1[2]: base & is5 & ~feat_fig & ~feat_wep, t1[3]: base & ~is5}
        for g, m in g1.items():
            accumulate(acc1[g], m, y, c_live, c_rev, c_nb, S)

        # ---- table 2: ten-pull on a figure banner at t-1, by pity BEFORE that pull
        pity = pity_depth(b["obtained"], b["prev_decision"].reshape(B, S), rarity_t).numpy()
        pity_before = np.zeros((B, S))
        figA = prev == 4; figB = prev == 6
        pity_before[:, 1:] = np.where(figA[:, 1:], pity[:, :-1, 2] * HARD_PITY5[1],
                             np.where(figB[:, 1:], pity[:, :-1, 4] * HARD_PITY5[2], 0.0))
        ten = base & (figA | figB)
        outcome = {"won the featured figure": feat_fig & is5,
                   "5-star but not featured (50/50 lost)": is5 & ~feat_fig,
                   "no 5-star": ~is5}
        for pname, pm in (("< 30", pity_before < 30), ("30-69", (pity_before >= 30) & (pity_before < 70)),
                          (">= 70", pity_before >= 70)):
            for oname, om in outcome.items():
                accumulate(acc2[f"{oname}, pity before pull {pname}"], ten & pm & om,
                           y, c_live, c_rev, c_nb, S)

    show("TABLE 1 -- real holdout (campaigns 28-30): what follows a pull, by what it delivered", acc1)
    show("TABLE 2 -- quasi-experiment: previous decision = TEN-pull on a figure banner, same pity stratum", acc2)


if __name__ == "__main__":
    main()
