"""R48 step 4: pair the zero-shot LLM with the plain GRU, occasion by occasion.

Inputs: the R48 shards (gated by scripts/r48_summarise.py) and one file per GRU
seed from scripts/eval_per_occasion.py run with
--cells outsample_users_calib_period,outsample_users_holdout_period.

Gates, per GRU seed:
    - the file holds exactly the LLM's occasions in those cells: the same
      customers, positions, labels and cells, each once
    - its per-cell mean NLL matches the run's final.json to 5e-4, R31's rule
      (these files are float32 on CPU; the runs were evaluated in bf16 on a GPU)
The two files hash customer ids differently (eval_per_occasion: signed first 8
bytes of sha1; parametric.data: unsigned, shifted right by one), so the GRU's
hashes are converted before matching.

The GRU's per-occasion NLL is the mean over seeds, as in R31. The statistic is
d = NLL(LLM) - NLL(GRU), positive = the GRU is better, with a cluster bootstrap
over customers. One set of draws is shared by every subgroup, so contrasts
between subgroups are paired draw by draw.

Reported (aggregates only):
    pre-registered: the headline cell: LLM, GRU, d
    exploratory: by position in the 1,024-occasion window, per cell: LLM, GRU,
    the train-share constant, d, and the share of the GRU's gain over the
    constant that the LLM achieves, (constant - LLM) / (constant - GRU); the
    change in d from the first to the last position bin; d per decision in the
    headline cell; d against each seed separately

USAGE (HPCC)
    python scripts/r48_pair_gru.py --table <r46>/occasions_v6_S1024.npz \
        --llm-dir <runs>/r48 --llm-tag qwen3_8b_out --n-shards 2 \
        --gru-dir <runs>/r48/gru_b17 --runs-root <runs> \
        --gru-tags b17_gru_v6_s1 b17_gru_v6_s2 b17_gru_v6_s3 b17_gru_v6_s4 b17_gru_v6_s5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_zeroshot.score import CELL_NAMES, HEADLINE  # noqa: E402
from parametric.data import load, roles  # noqa: E402

CELLS = (3, 4)                    # out-of-sample x calibration, x holdout
POS_BINS = (0, 64, 256, 512, 768, 1024)


def to_parametric_hash(h: np.ndarray) -> np.ndarray:
    """eval_per_occasion's signed sha1 hash -> parametric.data.uid_hash."""
    return (h.astype(np.int64).view(np.uint64) >> np.uint64(1)).astype(np.int64)


def sort_rows(uid, pos):
    return np.lexsort((pos, uid))


class Bootstrap:
    """Cluster bootstrap over customers with one set of draws for every statistic."""

    def __init__(self, cid: np.ndarray, draws: int, seed: int):
        self.cid, self.C = cid, int(cid.max()) + 1
        idx = np.random.default_rng(seed).integers(0, self.C, (draws, self.C))
        self.W = np.stack([np.bincount(r, minlength=self.C) for r in idx]).astype(np.float64)

    def reps(self, m, x):
        S = np.bincount(self.cid[m], x[m], minlength=self.C)
        N = np.bincount(self.cid[m], minlength=self.C).astype(np.float64)
        n = self.W @ N                      # a draw can miss a small subgroup entirely
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(S.sum() / N.sum()), np.where(n > 0, (self.W @ S) / n, np.nan)


def summ(point, reps):
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    out = {"est": point, "ci95": [float(lo), float(hi)]}
    if np.isnan(reps).any():
        out["draws_without_rows"] = int(np.isnan(reps).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="R48: pair the zero-shot LLM with the GRU")
    ap.add_argument("--table", required=True, help="R46 occasions_v6_S1024.npz")
    ap.add_argument("--llm-dir", required=True)
    ap.add_argument("--llm-tag", required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--gru-dir", required=True)
    ap.add_argument("--gru-tags", nargs="+", required=True)
    ap.add_argument("--runs-root", required=True, help="holds gen5_hpcc_S1024_b4_<tag>/")
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = Path(a.llm_dir)
    parts = [load(run / f"{a.llm_tag}_shard{k}.npz") for k in range(a.n_shards)]
    L = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    keep = np.isin(L["cell"], CELLS)
    L = {k: v[keep] for k, v in L.items()}
    o = sort_rows(L["uid_hash"], L["pos"])
    L = {k: v[o] for k, v in L.items()}

    gate, gru = {}, []
    for tag in a.gru_tags:
        G = load(Path(a.gru_dir) / f"{tag}.npz")
        G["uid_hash"] = to_parametric_hash(G["uid_hash"])
        o = sort_rows(G["uid_hash"], G["pos"])
        G = {k: v[o] for k, v in G.items()}
        same = len(G["nll"]) == len(L["nll"])
        g = {"rows": int(len(G["nll"])), "rows_llm": int(len(L["nll"]))}
        if same:
            g["same_occasions"] = bool((G["uid_hash"] == L["uid_hash"]).all()
                                       and (G["pos"].astype(np.int64) == L["pos"]).all())
            g["same_labels"] = bool((G["label"] == L["label"]).all())
            g["same_cells"] = bool((G["cell"] == L["cell"]).all())
        final = json.loads((Path(a.runs_root) / f"gen5_hpcc_S1024_b4_{tag}" / "gen5_multistream"
                            / "hpcc" / "final.json").read_text())
        g["nll_vs_final"] = {}
        for c in CELLS:
            name = CELL_NAMES[c]
            m = G["cell"] == c
            mine = float(G["nll"][m].astype(np.float64).mean())
            rec = float(final["test_cells"][name]["nll"])
            g["nll_vs_final"][name] = {"file": mine, "final_json": rec, "diff": mine - rec}
        g["pass"] = bool(same and g["same_occasions"] and g["same_labels"] and g["same_cells"]
                         and all(abs(v["diff"]) <= a.tol for v in g["nll_vs_final"].values()))
        gate[tag] = g
        gru.append(G["nll"].astype(np.float64) if same else None)
    gate_pass = all(g["pass"] for g in gate.values())

    out = {"llm_tag": a.llm_tag, "gru_tags": a.gru_tags, "tol": a.tol, "gate": gate,
           "gate_pass": gate_pass}
    if not all(x is not None for x in gru):
        (run / f"{a.llm_tag}_vs_gru.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        raise SystemExit("GATE FAILED: a GRU file does not hold the LLM's occasions")

    tab = load(Path(a.table))
    shares = np.bincount(tab["label"][roles(tab)["train"]].astype(np.int64), minlength=10)[1:]
    shares = shares / shares.sum()
    lab = L["label"].astype(np.int64)
    pos = L["pos"].astype(np.int64)
    llm = L["nll"].astype(np.float64)
    grus = np.stack(gru)
    gm = grus.mean(0)
    const = -np.log(shares[lab - 1])
    d = llm - gm

    _, cid = np.unique(L["uid_hash"], return_inverse=True)
    B = Bootstrap(cid, a.draws, a.seed)
    out["customers"] = int(B.C)
    out["draws"] = a.draws

    h = L["cell"] == HEADLINE
    out["headline"] = {k: summ(*B.reps(h, x)) for k, x in
                       (("llm", llm), ("gru_seed_mean", gm), ("constant", const), ("d_llm_minus_gru", d))}
    out["headline"]["n"] = int(h.sum())
    out["headline"]["d_per_seed"] = {t: float((llm[h] - grus[i][h]).mean())
                                     for i, t in enumerate(a.gru_tags)}
    out["headline"]["d_per_class"] = {}
    for c in range(1, 10):
        m = h & (lab == c)
        if m.any():
            out["headline"]["d_per_class"][str(c)] = dict(summ(*B.reps(m, d)), n=int(m.sum()))

    expl = {}
    for c in CELLS:
        rows, first, last = [], None, None
        for lo, hi in zip(POS_BINS[:-1], POS_BINS[1:]):
            m = (L["cell"] == c) & (pos >= lo) & (pos < hi)
            if not m.any():
                continue
            (pl, rl), (pg, rg), (pc, rc), (pd, rd) = (B.reps(m, x) for x in (llm, gm, const, d))
            share = (pc - pl) / (pc - pg)
            rows.append({"pos": [lo, hi], "n": int(m.sum()),
                         "llm": summ(pl, rl), "gru": summ(pg, rg), "constant": summ(pc, rc),
                         "d_llm_minus_gru": summ(pd, rd),
                         "share_of_gru_gain": summ(share, (rc - rl) / (rc - rg))})
            first = (pd, rd) if first is None else first
            last = (pd, rd)
        expl[CELL_NAMES[c]] = {"bins": rows,
                               "d_last_minus_first_bin": summ(last[0] - first[0], last[1] - first[1])}
    out["exploratory_by_position"] = expl

    (run / f"{a.llm_tag}_vs_gru.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if not gate_pass:
        raise SystemExit("GATE FAILED")


if __name__ == "__main__":
    main()
