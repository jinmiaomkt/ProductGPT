"""Gate G2: every R46 state variable against the label, before any fit is read.

On the train rows (in-sample customers, campaigns < 27), each feature column
is cut into at most 20 quantile bins, and two numbers are computed:

    MI(feature; y)                      in nats
    max over bins of n >= 1,000 of      max_k P(y = k | bin)

A column is FLAGGED when MI > 0.30 nats (R20's leaky clock carried 0.40) or
a bin of at least 1,000 rows is more than 99% one decision. A flag is not a
verdict: the column is then explained (it is history the model may use) or
dropped. The decayed stocks depend on a parameter, so they are tested at
alpha = 0.5 and 0.9. Prints aggregates only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from parametric import data as D, fit as FT, model as M   # noqa: E402

MI_MAX, PURITY, MIN_BIN = 0.30, 0.99, 1_000


def mi_and_purity(x: np.ndarray, y: np.ndarray):
    u = np.unique(x)
    if len(u) <= 20:
        b = np.searchsorted(u, x)
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 21)[1:-1]))
        b = np.searchsorted(edges, x, side="right")
    nb = int(b.max()) + 1
    joint = np.zeros((nb, 9))
    np.add.at(joint, (b, y - 1), 1.0)
    n = joint.sum()
    pb, py = joint.sum(1) / n, joint.sum(0) / n
    pj = joint / n
    nz = pj > 0
    mi = float((pj[nz] * np.log(pj[nz] / (pb[:, None] * py[None, :])[nz])).sum())
    big = joint.sum(1) >= MIN_BIN
    purity = float((joint[big].max(1) / joint[big].sum(1)).max()) if big.any() else float("nan")
    return mi, purity, nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tab, F = FT.load_features(Path(a.table), Path(a.feats))
    roles = D.roles(tab)
    m = roles["train"]
    y = tab["label"][m].astype(np.int64)
    cols = {}
    for k, v in F.items():
        if k == "alive":
            continue
        arr = v[m]
        arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr[:, None]
        for j in range(arr.shape[1]):
            cols[f"{k}[{j}]"] = arr[:, j].astype(np.float64)
    # decayed stocks at two decays
    ind = torch.as_tensor(F["ind"], dtype=torch.float64)
    for alpha in (0.5, 0.9):
        L = M.decay(ind, torch.tensor(alpha, dtype=torch.float64)).numpy()[m]
        for j in range(L.shape[1]):
            cols[f"stock{alpha}[{j}]"] = L[:, j]
    rows, flagged = [], []
    for name, x in cols.items():
        mi, pur, nb = mi_and_purity(x, y)
        flag = mi > MI_MAX or (pur == pur and pur > PURITY)
        rows.append({"feature": name, "mi": mi, "max_bin_purity": pur, "bins": nb, "flag": flag})
        if flag:
            flagged.append(name)
    rows.sort(key=lambda r: -r["mi"])
    print(f"G2 on {int(m.sum()):,} train rows; label entropy "
          f"{-(np.bincount(y - 1, minlength=9) / len(y) * np.log(np.bincount(y - 1, minlength=9) / len(y) + 1e-300)).sum():.4f} nats")
    print(f"{'feature':<22}{'MI':>8}{'purity':>9}{'bins':>6}")
    for r in rows[:40]:
        print(f"{r['feature']:<22}{r['mi']:>8.4f}{r['max_bin_purity']:>9.4f}{r['bins']:>6}"
              f"{'  FLAG' if r['flag'] else ''}")
    print(f"G2 flagged: {flagged if flagged else 'none'}")
    Path(a.out).write_text(json.dumps({"rows": rows, "flagged": flagged}, indent=1))


if __name__ == "__main__":
    main()
