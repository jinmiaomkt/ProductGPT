"""
R33: read the estimated STRUCTURAL parameters out of the finished runs.

The ladder's NLL differences are small by construction -- R30 stage 5 showed
the whole stock path is worth ~0.01 nats. What R33 is for is the quantities:

    half-life      occasions for a holding's effect to halve (Guadagni-Little)
    tier weights   what the 2nd and 3rd copy of a product count for
    attr kernel    which attributes carry satiation across products

Loads parameters only -- no data, no forward pass.

    python3 scripts/r33_report.py --prefixes b20_ b21_
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch


def softplus(x: float) -> float:
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.path.expanduser("~/ProductGPT/runs"))
    ap.add_argument("--prefixes", nargs="+", default=["b20_", "b21_"])
    a = ap.parse_args()

    rows = defaultdict(list)
    for prefix in a.prefixes:
        for ck in sorted(Path(a.runs_root).glob(f"*_b4_{prefix}*/**/best.pt")):
            tag = ck.parent.parent.parent.name.split("_b4_")[-1]
            group = re.sub(r"_s\d+$", "", tag)
            st_ = torch.load(ck, map_location="cpu", weights_only=False)
            sd = st_["model_state_dict"]
            rec = {"seed": st_["cfg"].get("seed")}
            for k, v in sd.items():
                if k.endswith("inventory_slots.log_half_life"):
                    rec["half_life"] = softplus(float(v))
                elif k.endswith("inventory_slots.tier_raw"):
                    steps = torch.sigmoid(v.float())
                    g = torch.cat([torch.ones(1), torch.cumprod(steps, 0)])
                    rec["tier"] = [round(float(x), 3) for x in g]
                elif k.endswith("inventory_slots.attr_self"):
                    rec["attr_self"] = float(v)
                elif k.endswith("inventory_slots.attr_w"):
                    w = v.float()
                    rec["attr_w_absmax"] = float(w.abs().max())
                    rec["attr_w_top"] = [int(i) for i in torch.argsort(-w.abs())[:5]]
            fin = ck.parent / "final.json"
            if fin.exists():
                try:
                    cells = json.loads(fin.read_text()).get("test_cells", {})
                    cell = cells.get("outsample_users_holdout_period", {})
                    rec["holdout_nll"] = cell.get("nll")
                except json.JSONDecodeError:
                    pass
            rows[group].append(rec)

    if not rows:
        raise SystemExit("no runs found")
    print(f"{'configuration':<26} {'n':>2} {'half-life':>18} {'tier weights':>26} "
          f"{'holdout NLL':>14}")
    for group in sorted(rows):
        v = rows[group]
        hl = [r["half_life"] for r in v if "half_life" in r]
        hol = [r["holdout_nll"] for r in v if r.get("holdout_nll") is not None]
        hl_s = (f"{st.mean(hl):.1f} +/-{st.stdev(hl):.1f}" if len(hl) > 1
                else (f"{hl[0]:.1f}" if hl else "-"))
        tiers = [r["tier"] for r in v if "tier" in r]
        t_s = "-"
        if tiers:
            m = [st.mean(t[i] for t in tiers) for i in range(len(tiers[0]))]
            t_s = "[" + ", ".join(f"{x:.2f}" for x in m) + "]"
        h_s = (f"{st.mean(hol):.4f}+/-{st.stdev(hol):.4f}" if len(hol) > 1 else "-")
        print(f"{group:<26} {len(v):>2} {hl_s:>18} {t_s:>26} {h_s:>14}")
        aw = [r for r in v if "attr_self" in r]
        if aw:
            print(f"{'':<26}    attribute kernel: self "
                  f"{st.mean(r['attr_self'] for r in aw):.2f}, largest |coefficient| "
                  f"{st.mean(r['attr_w_absmax'] for r in aw):.3f}")


if __name__ == "__main__":
    main()
