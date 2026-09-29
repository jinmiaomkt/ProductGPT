"""R36c: do models TIED on one-step accuracy separate as generators?

    python scripts/r36c_generate.py --probe                  # time one chunk
    python scripts/r36c_generate.py --configs b29_hl2 b18_hyb_stock_nostock \
        --seeds 1 2 3 --limit-users 300 --n-rep 8

Stage 4 found no family wins: eight cells inside 0.0194 nats against a seed sd
of 0.0075, and deleting the stock path entirely cost 0.0137 -- a statistical
tie.  So the decision-level criterion cannot tell a model that has a satiation
mechanism from one that does not.  Satiation is a SEQUENCE-level mechanism: it
governs whether a customer who has just acquired five copies stops pulling.

H1 predicts these tied models separate here, and that the no-stock model is the
one that separates, on the satiation signature.

The cell scored is exactly the one the headline NLL comes from --
`outsample_users_holdout_period` out of build_loaders, so the ranking by one-step
NLL and the ranking by sequence discrepancy are computed on the same customers
and the same campaigns.  The environment is calibrated on TRAIN customers'
calibration rows only, so it never touches the cell being scored.
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
import seq_metrics as sm  # noqa: E402
from baselines import (fit_marginal, fit_markov, iid_sampler,  # noqa: E402
                       markov_sampler, real_sequences, simulate)
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from rollout import first_row_of_campaign, load_checkpoint, rollout  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
HOLDOUT_FROM = 28


def ckpt_path(tag: str, seed: int, runs: Path) -> Path:
    hits = sorted(runs.glob(f"*_{tag}_s{seed}/gen5_multistream/hpcc/best.pt"))
    if not hits:
        raise FileNotFoundError(f"no checkpoint for {tag} seed {seed} under {runs}")
    return hits[0]


def collate_chunk(ds, idx):
    b = dsm.collate_multistream([ds[i] for i in idx])
    if "user_idx" not in b:
        b["user_idx"] = torch.zeros(b["label"].shape[0], dtype=torch.long)
    return b


def score(res, start_vec):
    """Functionals over each sequence's own holdout span."""
    return sm.functionals(res["decisions"], res["alive"], res["obtained"],
                          res["lto"], start=int(start_vec.min()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["b29_hl2", "b18_hyb_stock_nostock",
                             "b18_hyb_stock_tokens", "b18_hyb_stock_counts0"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--limit-users", type=int, default=300)
    ap.add_argument("--n-rep", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--out", default="results/r36/r36c.json")
    ap.add_argument("--probe", action="store_true",
                    help="time one chunk of one config and stop")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    runs = Path(a.runs_dir)

    # ---------------------------------------------------------------- data
    first = ckpt_path(a.configs[0], a.seeds[0], runs)
    print(f"[cfg] reading data configuration from {first.parent.parent.parent.name}")
    state = torch.load(str(first), map_location="cpu", weights_only=False)
    cfg = dict(state["cfg"])
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])

    from train5_multistream import build_loaders
    tr_dl, _, tests, num_users = build_loaders(cfg)
    cell = tests["outsample_users_holdout_period"].dataset
    train_ds = tr_dl.dataset
    print(f"[cell] outsample_users_holdout_period: {len(cell)} customers")

    rng = np.random.default_rng(0)
    pick = rng.permutation(len(cell))[:a.limit_users]
    pick.sort()
    print(f"[cell] scoring {len(pick)} of them")

    # Environment and baselines are fitted on TRAIN customers' CALIBRATION rows.
    nfit = min(len(train_ds), 600)
    fit_items = [train_ds[i] for i in range(nfit)]
    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    p_marg = fit_marginal(fit_items, holdout_from=HOLDOUT_FROM)
    P_mkv = fit_markov(fit_items, holdout_from=HOLDOUT_FROM)
    print(f"[env] calibrated on {nfit} training customers")

    chunks = [pick[i:i + a.chunk] for i in range(0, len(pick), a.chunk)]

    # ------------------------------------------------------------- real set
    F_real, starts = [], []
    for ch in chunks:
        b = collate_chunk(cell, ch)
        st = first_row_of_campaign(b["campaign"], HOLDOUT_FROM)
        starts.append(st)
        F_real.append(score(real_sequences(b, start=int(st.min())), st))
    F_real = np.vstack(F_real)
    print(f"[real] {F_real.shape[0]} sequences, {F_real.shape[1]} functionals")

    scaler = sm.Scaler.fit(F_real)
    Zr = scaler(F_real)
    out = {"cell": "outsample_users_holdout_period", "n_users": int(len(pick)),
           "n_rep": a.n_rep, "functionals": sm.functional_names(), "results": {}}

    def evaluate(name, F):
        Z = scaler(F)
        cmp_ = sm.compare(Zr, Z, n_boot=120, seed=3)
        c = sm.c2st(Zr, Z, seed=0)
        rec = {"energy": cmp_["stat"], "null_q95": cmp_["null_q95"],
               "z": cmp_["z"], "inside_band": cmp_["inside_band"],
               "p_value": cmp_["p_value"], "auc": c["auc"], "auc_ci": c["ci95"],
               "top_functionals": [sm.functional_names()[i]
                                   for i in np.argsort(c["importance"])[::-1][:5]]}
        out["results"][name] = rec
        print(f"  {name:<28} energy {rec['energy']:+.3f} (q95 {rec['null_q95']:.3f}, "
              f"z {rec['z']:>7.1f})  AUC {rec['auc']:.3f}  "
              f"{'INSIDE' if rec['inside_band'] else 'outside'}")
        return rec

    # ------------------------------------------------------------ baselines
    print("\n[baselines]")
    for nm, samp in (("iid", iid_sampler(p_marg)), ("markov1", markov_sampler(P_mkv))):
        F = []
        for ch, st in zip(chunks, starts):
            b = collate_chunk(cell, ch)
            r = simulate(b, env, vocab, samp, start=int(st.min()),
                         n_rep=a.n_rep, seed=17)
            F.append(score(r, st))
        evaluate(nm, np.vstack(F))

    # --------------------------------------------------------------- models
    print("\n[models]")
    for tag in a.configs:
        for sd in a.seeds:
            try:
                cp = ckpt_path(tag, sd, runs)
            except FileNotFoundError as e:
                print(f"  skip {tag} s{sd}: {e}")
                continue
            model, mcfg = load_checkpoint(cp, device=a.device)
            t0 = time.time()
            F = []
            for ci, (ch, st) in enumerate(zip(chunks, starts)):
                b = collate_chunk(cell, ch)
                r = rollout(model, b, env, vocab, start=int(st.min()),
                            n_rep=a.n_rep, seed=1000 + sd, device=a.device)
                F.append(score(r, st))
                if a.probe:
                    dt = time.time() - t0
                    print(f"  [probe] chunk of {len(ch)} users x {a.n_rep} reps "
                          f"in {dt:.1f}s -> {dt*len(chunks)/60:.1f} min per "
                          f"config-seed, {dt*len(chunks)*len(a.configs)*len(a.seeds)/3600:.1f} h total")
                    return
            evaluate(f"{tag}_s{sd}", np.vstack(F))
            print(f"      ({time.time() - t0:.0f}s)")
            del model

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
