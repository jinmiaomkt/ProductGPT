"""R42: does the free-running model get the LUCKY BRANCH right?

R41 measured, in the real holdout, what follows a five-star win: same ten-pull
on a figure banner, same pity depth before it, winning the featured figure
makes the next twenty occasions heavier and moves the next decision toward
the weapon and regular banners. Jin's reading of the free-running arm is that
a model which re-rolls the lottery is exploring exactly such branches, so the
generated population may legitimately differ from the single realised path.

This probe rolls the model out free-running on the R36 cell (same 300
customers, same seeds, same context) and computes the SAME post-hit tables on
the generated sequences as R41 computed on the real ones -- and on the real
sequences of the same customers, same rows, for a like-for-like comparison.
It then splits every generated replicate by whether the simulated lottery
delivered a featured figure early, and reports population shares for lucky
and unlucky replicates separately.

    lucky and unlucky replicates both drift (NotBuy ~0.18)  -> exposure bias;
    the drift is concentrated in lucky replicates and their post-win behaviour
    matches R41                                              -> the branch is
    modelled correctly and carries the gap.

Aggregates only. Inference only.

    python scripts/r42_generated_post_hit.py --configs b36_triplet b36_duplet \
        --seeds 1 2 3 --max-events 384 --limit-users 300 --n-rep 8 --chunk 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
from baselines import real_sequences  # noqa: E402
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from rollout import first_row_of_campaign, load_checkpoint, rollout  # noqa: E402
from model_multistream_state_space import HARD_PITY5, pity_depth  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
HOLDOUT_FROM = 28
REV = np.array([1, 10, 1, 10, 1, 10, 1, 10, 0], dtype=float)
WINDOWS = (5, 20)
STRATA = (("< 30", 0, 30), ("30-69", 30, 70), (">= 70", 70, 10**9))
T1 = ["featured figure 5-star", "featured weapon 5-star", "other 5-star", "no 5-star"]
OUTCOMES = ["won featured figure", "50/50 lost", "no 5-star"]
KEYS = ["n", "notbuy", "reg", "fig", "wep", "rev"] + \
       [f"{k}_{w}" for w in WINDOWS for k in ("live", "rev", "nb")]


def ckpt_path(tag: str, seed: int, runs: Path) -> Path:
    # gen 5 writes <run>/gen5_multistream/hpcc/, gen 6 <run>/gen6_streaming/hpcc/
    hits = sorted(runs.glob(f"*_{tag}_s{seed}/*/hpcc/best.pt"))
    if not hits:
        raise FileNotFoundError(f"no checkpoint for {tag} seed {seed} under {runs}")
    return hits[0]


def collate_chunk(ds, idx):
    b = dsm.collate_multistream([ds[i] for i in idx])
    if "user_idx" not in b:
        b["user_idx"] = torch.zeros(b["label"].shape[0], dtype=torch.long)
    return b


class Acc:
    """Accumulates the post-hit tables and the replicate split across chunks."""

    def __init__(self):
        self.t1 = {g: {k: 0.0 for k in KEYS} for g in T1}
        self.t2 = {f"{o} | pity {s[0]}": {k: 0.0 for k in KEYS} for s in STRATA for o in OUTCOMES}
        self.pop = {"live": 0.0, "nb": 0.0, "rev": 0.0, "ten_fig": 0.0, "ten_fig_won": 0.0}
        self.split = {k: {"seq": 0, "live": 0.0, "nb": 0.0, "rev": 0.0}
                      for k in ("early featured win", "no early featured win")}

    @staticmethod
    def _add(a, m, y, c_live, c_rev, c_nb, S):
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

    def add(self, dec, prev, obt, lto, rarity_t, lo, hi, early_k=30):
        """dec: (N,S) decisions, 0 outside the scored span; prev: (N,S) decision at
        t-1 as the row-t input; obt: (N,S,10) row t = outcome of the decision at
        t-1; lto: (N,S,4)."""
        N, S = dec.shape
        live = dec > 0
        valid = (obt >= lo) & (obt <= hi)
        rarity = rarity_t.numpy()
        rar = rarity[np.clip(obt, 0, rarity.shape[0] - 1)]
        is5 = ((rar >= 4.5) & valid).any(-1)
        feat_fig = np.zeros((N, S), bool); feat_wep = np.zeros((N, S), bool)
        feat_fig[:, 1:] = (((obt[:, 1:, :] == lto[:, :-1, 0:1]) | (obt[:, 1:, :] == lto[:, :-1, 1:2]))
                           & valid[:, 1:, :]).any(-1) & (lto[:, :-1, 0:2] > 0).any(-1)
        feat_wep[:, 1:] = (((obt[:, 1:, :] == lto[:, :-1, 2:3]) | (obt[:, 1:, :] == lto[:, :-1, 3:4]))
                           & valid[:, 1:, :]).any(-1) & (lto[:, :-1, 2:4] > 0).any(-1)
        base = np.zeros((N, S), bool)
        base[:, 1:] = live[:, 1:] & live[:, :-1]          # previous decision is in the span
        pulled_prev = (prev >= 1) & (prev <= 8)
        base &= pulled_prev
        rev = np.where(live, REV[np.clip(dec, 1, 9) - 1], 0.0)
        nb = (dec == 9).astype(float)
        c_live = np.concatenate([np.zeros((N, 1)), np.cumsum(live, 1)], 1)
        c_rev = np.concatenate([np.zeros((N, 1)), np.cumsum(rev, 1)], 1)
        c_nb = np.concatenate([np.zeros((N, 1)), np.cumsum(nb, 1)], 1)

        g1 = {T1[0]: base & feat_fig & is5, T1[1]: base & feat_wep & is5 & ~feat_fig,
              T1[2]: base & is5 & ~feat_fig & ~feat_wep, T1[3]: base & ~is5}
        for g, m in g1.items():
            self._add(self.t1[g], m, dec, c_live, c_rev, c_nb, S)

        pity = pity_depth(torch.as_tensor(obt), torch.as_tensor(prev), rarity_t).numpy()
        figA = prev == 4; figB = prev == 6
        before = np.zeros((N, S))
        before[:, 1:] = np.where(figA[:, 1:], pity[:, :-1, 2] * HARD_PITY5[1],
                        np.where(figB[:, 1:], pity[:, :-1, 4] * HARD_PITY5[2], 0.0))
        ten = base & (figA | figB)
        oc = {OUTCOMES[0]: feat_fig & is5, OUTCOMES[1]: is5 & ~feat_fig, OUTCOMES[2]: ~is5}
        for sname, a_, b_ in STRATA:
            sm_ = (before >= a_) & (before < b_)
            for oname, om in oc.items():
                self._add(self.t2[f"{oname} | pity {sname}"], ten & sm_ & om,
                          dec, c_live, c_rev, c_nb, S)

        self.pop["live"] += live.sum(); self.pop["nb"] += nb.sum(); self.pop["rev"] += rev.sum()
        self.pop["ten_fig"] += ten.sum(); self.pop["ten_fig_won"] += (ten & feat_fig & is5).sum()

        # replicate split: a featured-figure win within the first early_k live rows
        order = np.cumsum(live, 1)                       # live-row rank, 1-based on live rows
        early = (feat_fig & is5 & live & (order <= early_k)).any(1)
        for flag, name in ((early, "early featured win"), (~early, "no early featured win")):
            sel = flag & live.any(1)
            s = self.split[name]
            s["seq"] += int(sel.sum()); s["live"] += live[sel].sum()
            s["nb"] += nb[sel].sum(); s["rev"] += rev[sel].sum()

    def report(self):
        def row(a):
            n = max(a["n"], 1)
            out = {"n": int(a["n"]), "P_notbuy": a["notbuy"] / n, "P_reg": a["reg"] / n,
                   "P_fig": a["fig"] / n, "P_wep": a["wep"] / n, "E_rev": a["rev"] / n}
            for w in WINDOWS:
                out[f"rev_per_occ_next{w}"] = a[f"rev_{w}"] / max(a[f"live_{w}"], 1)
                out[f"notbuy_next{w}"] = a[f"nb_{w}"] / max(a[f"live_{w}"], 1)
            return out
        pop = {"notbuy_share": self.pop["nb"] / max(self.pop["live"], 1),
               "rev_per_occ": self.pop["rev"] / max(self.pop["live"], 1),
               "ten_pulls_fig": int(self.pop["ten_fig"]),
               "featured_win_rate_per_ten_pull": self.pop["ten_fig_won"] / max(self.pop["ten_fig"], 1)}
        split = {k: {"sequences": v["seq"], "notbuy_share": v["nb"] / max(v["live"], 1),
                     "rev_per_occ": v["rev"] / max(v["live"], 1)} for k, v in self.split.items()}
        return {"table1": {g: row(a) for g, a in self.t1.items()},
                "table2": {g: row(a) for g, a in self.t2.items()},
                "population": pop, "replicate_split": split}


def show(name, rep):
    print(f"\n=== {name}")
    p = rep["population"]
    print(f"  population: NotBuy {p['notbuy_share']:.3f}  rev/occ {p['rev_per_occ']:.2f}  "
          f"figure ten-pulls {p['ten_pulls_fig']}  featured win rate per ten-pull {p['featured_win_rate_per_ten_pull']:.3f}")
    for k, v in rep["replicate_split"].items():
        print(f"  {k:24s}: {v['sequences']:5d} sequences  NotBuy {v['notbuy_share']:.3f}  rev/occ {v['rev_per_occ']:.2f}")
    print(f"  {'table 1: previous pull delivered':40s} {'n':>7s} {'P(NB)':>6s} {'P(reg)':>6s} {'P(fig)':>6s} {'P(wep)':>6s} "
          f"{'E[rev]':>6s} {'rev/occ 5':>9s} {'NB 5':>6s} {'rev/occ 20':>10s} {'NB 20':>6s}")
    for tbl in ("table1", "table2"):
        if tbl == "table2":
            print(f"  {'table 2: ten-pull on a figure banner':40s}")
        for g, r in rep[tbl].items():
            print(f"  {g:40s} {r['n']:7d} {r['P_notbuy']:6.3f} {r['P_reg']:6.3f} {r['P_fig']:6.3f} {r['P_wep']:6.3f} "
                  f"{r['E_rev']:6.2f} {r['rev_per_occ_next5']:9.2f} {r['notbuy_next5']:6.3f} "
                  f"{r['rev_per_occ_next20']:10.2f} {r['notbuy_next20']:6.3f}")


def prev_from(batch_prev: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Row-t input 'previous decision': the real one, overwritten by the generated
    decision at t-1 wherever one exists -- the rollout buffer's convention."""
    prev = batch_prev.copy()
    m = dec[:, :-1] > 0
    prev[:, 1:][m] = dec[:, :-1][m]
    return prev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["b36_triplet", "b36_duplet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--limit-users", type=int, default=300)
    ap.add_argument("--n-rep", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--out", default="results/r42/generated_post_hit.json")
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--legacy-camp-bias", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    runs = Path(a.runs_dir)

    first = ckpt_path(a.configs[0], a.seeds[0], runs)
    state = torch.load(str(first), map_location="cpu", weights_only=False)
    cfg = dict(state["cfg"])
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    if a.max_events:
        cfg["max_events"] = int(a.max_events)
    lo, hi = cfg["first_prod_id"], cfg["last_prod_id"]

    from train5_multistream import build_loaders
    tr_dl, _, tests, _ = build_loaders(cfg)
    cell = tests["outsample_users_holdout_period"].dataset
    train_ds = tr_dl.dataset
    rng = np.random.default_rng(0)
    pick = rng.permutation(len(cell))[:a.limit_users]
    pick.sort()
    nfit = min(len(train_ds), 600)
    fit_items = [train_ds[i] for i in range(nfit)]
    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    chunks = [pick[i:i + a.chunk] for i in range(0, len(pick), a.chunk)]
    print(f"[cell] {len(pick)} customers, {len(chunks)} chunks, context {cfg['max_events']}")

    out = {"cell": "outsample_users_holdout_period", "n_users": int(len(pick)),
           "n_rep": a.n_rep, "early_k": 30, "results": {}}
    rarity_t = None

    for tag in a.configs:
        for sd in a.seeds:
            model, mcfg = load_checkpoint(ckpt_path(tag, sd, runs), device=a.device,
                                          legacy_camp_bias=a.legacy_camp_bias)
            if rarity_t is None:
                rarity_t = model.product_embed.feat_tbl[:, 0].float().cpu()
                # ---- the real side, once, on the same customers and rows
                acc_r = Acc()
                for ch in chunks:
                    b = collate_chunk(cell, ch)
                    st = first_row_of_campaign(b["campaign"], HOLDOUT_FROM)
                    r = real_sequences(b, start=int(st.min()))
                    acc_r.add(r["decisions"], b["prev_decision"].numpy().reshape(r["decisions"].shape),
                              r["obtained"], r["lto"], rarity_t, lo, hi)
                out["results"]["real"] = acc_r.report()
                show("REAL (same 300 customers, holdout rows)", out["results"]["real"])
            t0 = time.time()
            acc = Acc()
            for ch in chunks:
                b = collate_chunk(cell, ch)
                st = first_row_of_campaign(b["campaign"], HOLDOUT_FROM)
                r = rollout(model, b, env, vocab, start=int(st.min()),
                            n_rep=a.n_rep, seed=1000 + sd, device=a.device)
                prev_base = np.repeat(b["prev_decision"].numpy().reshape(b["label"].shape),
                                      a.n_rep, axis=0)
                acc.add(r["decisions"], prev_from(prev_base, r["decisions"]),
                        r["obtained"], r["lto"], rarity_t, lo, hi)
            out["results"][f"{tag}_s{sd}"] = acc.report()
            show(f"{tag}_s{sd} free-running ({time.time() - t0:.0f}s)", out["results"][f"{tag}_s{sd}"])
            del model

    dest = Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
