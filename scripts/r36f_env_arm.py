"""Is the generative failure the MODEL, or my model of the lottery?

R36c's control separated the feedback loop from the per-step distribution:
teacher forcing (streams real, decisions sampled) scores inside the null band,
free running (decisions AND acquisitions re-drawn) scores far outside. But that
leaves two things confounded in the failing arm, which a co-author rightly
pressed on: loot boxes are innately random, so some of the discrepancy might be
my ACQUISITION ENVIRONMENT re-rolling the lottery rather than the model's
decisions drifting.

The missing cell:

                      acquisitions real      acquisitions simulated
    decisions real    the data itself        ARM C  <- this script
    decisions sampled teacher-forced 0.307   free-running 2.352

Arm C replays each customer's TRUE decisions and lets the environment re-draw
what they received. No model is involved, so it isolates the lottery.

    inside the band  -> the environment is faithful, and the failure is the
                        model's decision feedback, as claimed.
    outside the band -> the environment's randomness is itself creating the
                        discrepancy, and the free-running number is not
                        attributable to the model alone.

    python scripts/r36f_env_arm.py --users 300 --n-rep 8
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
from baselines import fit_marginal, iid_sampler, real_sequences, simulate  # noqa: E402
from gacha_env import AcquisitionEnv, PityState, ProductTable  # noqa: E402
from rollout import _replicate, _warm_pity, first_row_of_campaign  # noqa: E402

HOLDOUT_FROM = 28


def arm_c(batch, env, vocab, *, start: int, n_rep: int, seed: int):
    """Replay the TRUE decisions; let the environment re-draw the acquisitions."""
    buf = _replicate(batch, n_rep)
    B, S = buf["label"].shape
    rng = np.random.default_rng(seed)
    alive = (buf["label"] != 0).cpu().numpy()
    truth = buf["label"].cpu().numpy()
    obtained = buf["obtained"].cpu().numpy().copy()
    offers = buf["lto"].cpu().numpy()

    states = [PityState() for _ in range(B)]
    _warm_pity(states, buf, vocab, env, start)

    gen = np.zeros((B, S), dtype=np.int64)
    for t in range(start, S):
        if not alive[:, t].any():
            break
        y = np.where(alive[:, t], truth[:, t], 0)
        gen[:, t] = y
        if t + 1 >= S:
            break
        act = alive[:, t] & (y > 0)
        for b in np.flatnonzero(act):
            obtained[b, t + 1] = env.step(int(y[b]), offers[b, t], states[b], rng)
    return {"decisions": gen, "alive": alive, "start": start,
            "obtained": obtained, "lto": offers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--n-rep", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--max-events", type=int, default=384)
    a = ap.parse_args()

    cfg = config5.get_config("hpcc")
    config5.apply_vocab_level(cfg, 7)
    cfg["max_events"] = a.max_events
    cfg["split_mode"] = "both"
    cfg["val_mode"] = "late"
    cfg["val_from"] = 27
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])

    from train5_multistream import build_loaders
    tr_dl, _, tests, _ = build_loaders(cfg)
    cell = tests["outsample_users_holdout_period"].dataset
    train_ds = tr_dl.dataset

    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
    fit_items = [train_ds[i] for i in range(min(len(train_ds), 600))]
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    p_marg = fit_marginal(fit_items, holdout_from=HOLDOUT_FROM)
    print(f"[env] calibrated on {len(fit_items)} training customers")

    rng = np.random.default_rng(0)
    pick = np.sort(rng.permutation(len(cell))[:a.users])
    chunks = [pick[i:i + a.chunk] for i in range(0, len(pick), a.chunk)]

    def collate(ch):
        b = dsm.collate_multistream([cell[i] for i in ch])
        b["user_idx"] = torch.zeros(b["label"].shape[0], dtype=torch.long)
        return b

    F = {"real": [], "armC": [], "iid": []}
    for ch in chunks:
        b = collate(ch)
        st = int(first_row_of_campaign(b["campaign"], HOLDOUT_FROM).min())
        runs = {
            "real": real_sequences(b, start=st),
            "armC": arm_c(b, env, vocab, start=st, n_rep=a.n_rep, seed=11),
            "iid": simulate(b, env, vocab, iid_sampler(p_marg), start=st,
                            n_rep=a.n_rep, seed=11),
        }
        for k, r in runs.items():
            F[k].append(sm.functionals(r["decisions"], r["alive"], r["obtained"],
                                       r["lto"], start=st))
    F = {k: np.vstack(v) for k, v in F.items()}
    sc = sm.Scaler.fit(F["real"])
    Z = {k: sc(v) for k, v in F.items()}

    print(f"\n[cell] {F['real'].shape[0]} real sequences, "
          f"{F['armC'].shape[0]} arm-C replicates\n")
    print(f"{'arm':<34}{'energy':>9}{'null q95':>10}{'z':>8}  verdict")
    for k, label in (("armC", "ARM C: real decisions, simulated acq."),
                     ("iid", "i.i.d. decisions (reference)")):
        c = sm.compare(Z["real"], Z[k], n_boot=200, seed=5)
        print(f"{label:<34}{c['stat']:>+9.3f}{c['null_q95']:>10.3f}{c['z']:>8.1f}  "
              f"{'INSIDE' if c['inside_band'] else 'outside'}")
    r = sm.c2st(Z["real"], Z["armC"], seed=0)
    print(f"\narm C classifier AUC {r['auc']:.3f} "
          f"(95% CI {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}); 0.5 = indistinguishable")
    print("top functionals:", ", ".join(
        sm.functional_names()[i] for i in np.argsort(r["importance"])[::-1][:5]))


if __name__ == "__main__":
    main()
