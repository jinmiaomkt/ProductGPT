"""R39: does the per-product identity embedding help on RETURNING products and
hurt on NEW ones?

The pooled comparison -- attributes-only 0.8757 against both 0.8756 -- averages
two regimes that should behave oppositely, and a co-author's objection is what
exposed this:

  campaign 28       100% of its 197,110 decision rows have a product on offer
                    that was NEVER offered during calibration. Its id_p row was
                    never updated, so the full model is adding an UNTRAINED
                    vector to the offer representation -- noise the
                    attributes-only model does not carry.
  campaigns 29-30   0% new products. Every index was trained, so a per-product
                    fixed effect can do what a fixed effect is for.

If identity helps on 29-30 and hurts on 28, the pooled 0.0001 is two effects
cancelling, and the claim "the identity branch is idle" is wrong.

Also reported: per-class NLL inside campaign 28, to see whether any
attributes-only advantage is concentrated in the decision classes belonging to
the banner that hosts the new product. That is a WITHIN-campaign cut, so it does
not rely on comparing 28 against 29-30, which differ in more than novelty.

    python scripts/r39_embedding_split.py --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
from rollout import load_checkpoint  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
N_CLASSES = 9
BANNER_OF_SLOT = {0: "figA", 1: "figB", 2: "wep", 3: "wep"}
CLASSES_OF_BANNER = {"reg": (1, 2), "figA": (3, 4), "figB": (5, 6), "wep": (7, 8)}
LABELS = ["Buy1Reg", "Buy10Reg", "Buy1FigA", "Buy10FigA", "Buy1FigB",
          "Buy10FigB", "Buy1Wep", "Buy10Wep", "NotBuy"]


def ckpt(tag, seed, runs):
    hits = sorted(runs.glob(f"*_{tag}_s{seed}/gen5_multistream/hpcc/best.pt"))
    if not hits:
        raise FileNotFoundError(f"{tag} s{seed}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["b32_attronly", "b29_hl2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--out", default="results/r39/embedding_split.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    runs = Path(a.runs_dir)

    state = torch.load(str(ckpt(a.configs[0], a.seeds[0], runs)),
                       map_location="cpu", weights_only=False)
    cfg = dict(state["cfg"])
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    LO, HI = cfg["first_prod_id"], cfg["last_prod_id"]

    from train5_multistream import build_loaders
    tr_dl, _, tests, _ = build_loaders(cfg)
    cell = tests["outsample_users_holdout_period"].dataset
    base = tr_dl.dataset

    # Which product indices were OFFERED during calibration? Taken from the
    # TRAINING customers, because that is what determined which id_p rows were
    # ever updated.
    off_cal = set()
    for i in range(len(base)):
        it = base[i]
        lto = np.asarray(it["lto"]); lab = np.asarray(it["label"])
        camp = np.asarray(it["campaign"])
        S = int((lab != 0).sum())
        for t in range(S):
            if int(camp[t]) < 28:
                for p in lto[t]:
                    if LO <= int(p) <= HI:
                        off_cal.add(int(p))
    print(f"[novelty] {len(off_cal)} product indices trained")

    idx = list(range(len(cell)))
    chunks = [idx[i:i + a.chunk] for i in range(0, len(idx), a.chunk)]
    out = {"n_trained_indices": len(off_cal), "results": {}}

    for tag in a.configs:
        for sd in a.seeds:
            try:
                cp = ckpt(tag, sd, runs)
            except FileNotFoundError as e:
                print(f"  skip {tag} s{sd}: {e}")
                continue
            model, _ = load_checkpoint(cp, device=a.device, legacy_camp_bias=True)
            # sum of NLL and count, per group
            acc = defaultdict(lambda: [0.0, 0])
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
                    lg = (o[0] if isinstance(o, (tuple, list)) else o)[..., 1:1 + N_CLASSES]
                    y = b["label"].to(a.device)
                    m = y != 0
                    if not m.any():
                        continue
                    nll = F.cross_entropy(lg[m], y[m] - 1, reduction="none").cpu().numpy()
                    camp = b["campaign"].cpu().numpy()[m.cpu().numpy()]
                    ycls = (y[m] - 1).cpu().numpy()
                    # per-row: does the offer carry an untrained index, and which
                    # banner hosts it?
                    lto_np = b["lto"].cpu().numpy()
                    mm = m.cpu().numpy()
                    novel_ban = np.full(mm.sum(), "", dtype=object)
                    k = 0
                    for bi in range(mm.shape[0]):
                        for t in range(mm.shape[1]):
                            if not mm[bi, t]:
                                continue
                            nb = ""
                            for s in range(lto_np.shape[2]):
                                p = int(lto_np[bi, t, s])
                                if LO <= p <= HI and p not in off_cal:
                                    nb = BANNER_OF_SLOT.get(s, "")
                                    break
                            novel_ban[k] = nb
                            k += 1
                    grp28 = camp >= 28
                    for name, sel in (
                        ("campaign28", grp28 & (camp == 28)),
                        ("campaign29_30", grp28 & (camp >= 29)),
                        ("all_holdout", grp28),
                    ):
                        if sel.any():
                            acc[name][0] += float(nll[sel].sum()); acc[name][1] += int(sel.sum())
                    # within campaign 28: classes of the banner hosting the new
                    # product, versus every other class
                    in28 = camp == 28
                    if in28.any():
                        host = np.array([nb for nb in novel_ban])
                        for bn in ("figA", "figB", "wep"):
                            hostrows = in28 & (host == bn)
                            if not hostrows.any():
                                continue
                            cls = CLASSES_OF_BANNER[bn]
                            is_host_cls = np.isin(ycls + 1, cls)
                            for nm, sel in ((f"c28_{bn}_hostclass", hostrows & is_host_cls),
                                            (f"c28_{bn}_otherclass", hostrows & ~is_host_cls)):
                                if sel.any():
                                    acc[nm][0] += float(nll[sel].sum()); acc[nm][1] += int(sel.sum())
            rec = {k: (v[0] / v[1], v[1]) for k, v in acc.items() if v[1] > 0}
            out["results"][f"{tag}_s{sd}"] = rec
            print(f"  {tag}_s{sd}: " + "  ".join(
                f"{k}={v[0]:.4f}" for k, v in sorted(rec.items()) if k.startswith(("campaign", "all"))))
            del model

    dest = Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
