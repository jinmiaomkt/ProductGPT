"""Is the model really a worse generator than i.i.d., or is the rollout wrong?

R36c's first result put the trained model's sequence discrepancy ABOVE an
i.i.d. sampler's.  That is either a strong finding or a bug, and the teacher-
forced arm is the control that separates them:

    free-running   model conditions on its OWN sampled decisions and the
                   environment's acquisitions
    teacher-forced model samples at each row but the streams stay REAL

If teacher-forced is close to the real data and free-running is far, the gap is
the feedback loop -- compounding error, a genuine result.  If teacher-forced is
ALSO far, the fault is in the per-step distribution or in the scoring, and
nothing about rollouts has been demonstrated.

    python scripts/r36c_diagnose.py --ckpt checkpoints/r36c/b29_hl2_s1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
import seq_metrics as sm  # noqa: E402
from baselines import (fit_marginal, iid_sampler, real_sequences,  # noqa: E402
                       simulate)
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from rollout import first_row_of_campaign, load_checkpoint, rollout  # noqa: E402

HOLDOUT_FROM = 28


def shares(dec, alive, start):
    m = alive.copy()
    m[:, :start] = False
    d = dec[m]
    d = d[d > 0]
    return np.bincount(d, minlength=10)[1:10] / max(len(d), 1), len(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/r36c/b29_hl2_s1.pt")
    ap.add_argument("--users", type=int, default=24)
    ap.add_argument("--n-rep", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(state["cfg"])
    cfg["max_events"] = a.max_events
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    from train5_multistream import build_loaders
    tr_dl, _, tests, _ = build_loaders(cfg)
    cell = tests["outsample_users_holdout_period"].dataset
    train_ds = tr_dl.dataset

    fit_items = [train_ds[i] for i in range(min(len(train_ds), 400))]
    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    p_marg = fit_marginal(fit_items, holdout_from=HOLDOUT_FROM)

    rng = np.random.default_rng(0)
    pick = np.sort(rng.permutation(len(cell))[:a.users])
    chunks = [pick[i:i + a.chunk] for i in range(0, len(pick), a.chunk)]
    model, _ = load_checkpoint(a.ckpt, device=a.device)

    def collate(ch):
        b = dsm.collate_multistream([cell[i] for i in ch])
        b["user_idx"] = torch.zeros(b["label"].shape[0], dtype=torch.long)
        return b

    acc = {k: [] for k in ("real", "free", "tf", "iid")}
    sh = {k: [] for k in acc}
    for ch in chunks:
        b = collate(ch)
        st = int(first_row_of_campaign(b["campaign"], HOLDOUT_FROM).min())
        runs = {
            "real": real_sequences(b, start=st),
            "free": rollout(model, b, env, vocab, start=st, n_rep=a.n_rep,
                            seed=7, device=a.device),
            "tf": rollout(model, b, env, vocab, start=st, n_rep=a.n_rep, seed=7,
                          device=a.device, teacher_forced=True),
            "iid": simulate(b, env, vocab, iid_sampler(p_marg), start=st,
                            n_rep=a.n_rep, seed=7),
        }
        for k, r in runs.items():
            acc[k].append(sm.functionals(r["decisions"], r["alive"], r["obtained"],
                                         r["lto"], start=st))
            sh[k].append(shares(r["decisions"], r["alive"], st))

    F = {k: np.vstack(v) for k, v in acc.items()}
    scaler = sm.Scaler.fit(F["real"])
    Z = {k: scaler(v) for k, v in F.items()}

    print("\n=== decision shares over the generated span ===")
    lab = ["Buy1Reg", "Buy10Reg", "Buy1FigA", "Buy10FigA", "Buy1FigB",
           "Buy10FigB", "Buy1Wep", "Buy10Wep", "NotBuy"]
    print(f"{'':>10}" + "".join(f"{x:>10}" for x in lab))
    for k in ("real", "free", "tf", "iid"):
        v = np.vstack([s[0] for s in sh[k]]).mean(axis=0)
        print(f"{k:>10}" + "".join(f"{x:>10.3f}" for x in v))

    print("\n=== sequence discrepancy vs real ===")
    for k in ("free", "tf", "iid"):
        c = sm.compare(Z["real"], Z[k], n_boot=60, seed=1)
        print(f"  {k:<6} energy {c['stat']:+.3f}  null q95 {c['null_q95']:.3f}  "
              f"z {c['z']:>7.1f}  {'INSIDE' if c['inside_band'] else 'outside'}")

    print("\nIf tf is close to real and free is far, the gap is the feedback loop.")
    print("If tf is also far, the fault is the per-step distribution or the scoring.")


if __name__ == "__main__":
    main()
