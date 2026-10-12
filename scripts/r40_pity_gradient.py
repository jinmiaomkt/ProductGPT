"""R40: has the model internalised the PITY GRADIENT, or just "a hit happened"?

R38 showed an explicit pity-depth stream adds nothing to one-step fit although
pity depth is informative on its own. That is a redundancy result: the model
already extracts whatever one-step-relevant information the counter carries.
What it does not show is WHICH information -- the full goal-gradient (purchase
hazard rising toward the hard-pity cap, resetting after a hit) or merely the
simpler cue that a five-star just arrived.

This probe scores a checkpoint WITHOUT the pity stream on the holdout cell,
computes pity depth externally from the acquisition stream, bins rows by pulls
since the last five-star on each banner, and compares the model's predicted
probability of buying on that banner with the observed rate, bin by bin.

    predicted tracks observed across bins, including the rise in the soft-pity
    region and the drop at zero  ->  the gradient is internalised: "learned".
    predicted is flat across bins while observed rises  ->  only the reset cue.

Inference only, on existing checkpoints.

    python scripts/r40_pity_gradient.py --configs b29_hl2 --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared"):
    sys.path.insert(0, str(p))

import dataset_multistream as dsm  # noqa: E402
from model_multistream_state_space import HARD_PITY5, pity_depth  # noqa: E402
from rollout import load_checkpoint  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
BANNERS = {1: ("figA", (2, 3), 0), 2: ("figB", (4, 5), 1), 3: ("wep", (6, 7), (2, 3))}
# bins on raw pulls since the last 5-star; the soft-pity region is 74-89 (63-79 weapon)
EDGES = [0, 1, 11, 31, 51, 61, 71, 81, 91, 10_000]
LABELS = ["0", "1-10", "11-30", "31-50", "51-60", "61-70", "71-80", "81-90", "91+"]


def ckpt(tag, seed, runs):
    hits = sorted(runs.glob(f"*_{tag}_s{seed}/gen5_multistream/hpcc/best.pt"))
    if not hits:
        raise FileNotFoundError(f"{tag} s{seed}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["b29_hl2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--out", default="results/r40/pity_gradient.json")
    ap.add_argument("--legacy-camp-bias", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    runs = Path(a.runs_dir)

    state = torch.load(str(ckpt(a.configs[0], a.seeds[0], runs)),
                       map_location="cpu", weights_only=False)
    cfg = dict(state["cfg"])
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    from train5_multistream import build_loaders
    _, _, tests, _ = build_loaders(cfg)
    cell = tests["outsample_users_holdout_period"].dataset
    idx = list(range(len(cell)))
    chunks = [idx[i:i + a.chunk] for i in range(0, len(idx), a.chunk)]
    out = {"edges": EDGES, "labels": LABELS, "results": {}}

    for tag in a.configs:
        for sd in a.seeds:
            model, mcfg = load_checkpoint(ckpt(tag, sd, runs), device=a.device,
                                          legacy_camp_bias=a.legacy_camp_bias)
            rarity = model.product_embed.feat_tbl[:, 0].float().cpu()
            acc = defaultdict(lambda: np.zeros(3))      # (banner, bin) -> [n, obs, pred]
            with torch.no_grad():
                for ch in chunks:
                    b = dsm.collate_multistream([cell[i] for i in ch])
                    uid = torch.zeros(b["label"].shape[0], dtype=torch.long)
                    kw = {}
                    if b.get("ipt") is not None:
                        kw["ipt"] = b["ipt"].to(a.device)
                    if getattr(model, "camp_bias", None) is not None and "campaign" in b:
                        kw["campaign"] = b["campaign"].to(a.device)
                    for k in ("inv_init_count", "inv_init_last"):
                        if k in b:
                            kw[k] = b[k].to(a.device)
                    o = model(b["lto"].to(a.device), b["obtained"].to(a.device),
                              b["prev_decision"].to(a.device), uid.to(a.device), **kw)
                    lg = (o[0] if isinstance(o, (tuple, list)) else o)[..., 1:10]
                    prob = torch.softmax(lg.float(), dim=-1).cpu().numpy()      # (B,S,9)
                    y = b["label"].numpy()
                    live = y != 0
                    pity = pity_depth(b["obtained"], b["prev_decision"], rarity).numpy()
                    lto = b["lto"].numpy()
                    for bid, (name, cls, slots) in BANNERS.items():
                        slots = slots if isinstance(slots, tuple) else (slots,)
                        offered = np.zeros(y.shape, dtype=bool)
                        for s in slots:
                            offered |= lto[:, :, s] > 0
                        m = live & offered
                        if not m.any():
                            continue
                        raw = pity[..., 2 * bid][m] * HARD_PITY5[bid]
                        obs = np.isin(y[m] - 1, cls).astype(float)
                        pred = prob[m][:, list(cls)].sum(axis=1)
                        bins = np.digitize(raw, EDGES[1:], right=False)
                        for k in np.unique(bins):
                            sel = bins == k
                            acc[(name, int(k))] += [sel.sum(), obs[sel].sum(), pred[sel].sum()]
            rec = {}
            for (name, k), (n, obs, pred) in sorted(acc.items()):
                if n >= 200:
                    rec[f"{name}|{LABELS[k]}"] = {"n": int(n), "observed": obs / n,
                                                  "predicted": pred / n}
            out["results"][f"{tag}_s{sd}"] = rec
            print(f"\n{tag}_s{sd}")
            for name in ("figA", "figB", "wep"):
                rows = [(LABELS[k], v) for key, v in rec.items() for k in range(len(LABELS))
                        if key == f"{name}|{LABELS[k]}"]
                if not rows:
                    continue
                o = np.array([v["observed"] for _, v in rows]); p_ = np.array([v["predicted"] for _, v in rows])
                corr = np.corrcoef(o, p_)[0, 1] if len(o) > 2 else float("nan")
                print(f"  {name}: corr(observed, predicted) across {len(rows)} bins = {corr:.3f}")
                print("    " + "  ".join(f"{l}:{v['observed']:.3f}/{v['predicted']:.3f}" for l, v in rows))
            del model

    dest = Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
