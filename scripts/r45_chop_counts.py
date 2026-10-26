"""R45 data preparation: what each chop setting removes, before any model runs.

For every (mode, depth) on the grid, counts over the real level-7 data
(aggregates only):
  - training customers (in-sample, campaigns < 27) and their training labels,
    and how many labels / rows the chop removes, plus how many customers are
    left with no training label at all;
  - the out-of-sample x holdout cell: how much history the chop removes before
    the scored events, which it never touches (the cut stops at campaign 27).

    python scripts/r45_chop_counts.py            # on HPCC (PRODUCTGPT_DATA set)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream  # noqa: E402
import model_multistream_state_space as mm  # noqa: E402
import train5_multistream as t5  # noqa: E402
from censor6 import chop_index  # noqa: E402

GRID = [("calendar", k) for k in (1, 2, 4, 8)] + [("count", d) for d in (64, 128, 256, 512)]


def main() -> None:
    cfg = config5.get_config("hpcc")
    cfg.update({"vocab_level": 7, "max_events": None, "split_mode": "both", "val_mode": "late",
                "val_from": 27, "holdout_flag": "feature", "num_workers": 0, "batch_size": 4})
    config5.apply_vocab_level(cfg, 7)
    dataset_multistream.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    mm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"], cfg["unk_prod_id"])
    tr, _, tests, _ = t5.build_loaders(cfg)
    cells = {"train (in-sample, campaigns < 27)": tr.dataset,
             "out-of-sample x holdout": tests["outsample_users_holdout_period"].dataset}
    vf = int(cfg["val_from"])
    for name, ds in cells.items():
        items = [ds[i] for i in range(len(ds))]
        lab = [((it["label"] >= 1) & (it["label"] <= 9)) for it in items]
        n_lab = sum(int(m.sum()) for m in lab)
        n_rows = sum(int(it["label"].numel()) for it in items)
        n_cust = sum(1 for m in lab if bool(m.any()))
        print(f"\n== {name}: {n_cust:,} customers with labels, {n_lab:,} labels, {n_rows:,} rows")
        print(f"   {'chop':<14}{'rows removed':>14}{'share':>8}{'labels removed':>16}{'share':>8}"
              f"{'cust. left w/o labels':>23}")
        for mode, depth in GRID:
            rows_rm = labs_rm = lost = 0
            for it, m in zip(items, lab):
                s = chop_index(it["campaign"], mode, depth, vf)
                rows_rm += s
                labs_rm += int(m[:s].sum())
                if bool(m.any()) and not bool(m[s:].any()):
                    lost += 1
            print(f"   {mode + ' ' + str(depth):<14}{rows_rm:>14,}{rows_rm / n_rows:>8.1%}"
                  f"{labs_rm:>16,}{labs_rm / max(n_lab, 1):>8.1%}{lost:>23,}")


if __name__ == "__main__":
    main()
