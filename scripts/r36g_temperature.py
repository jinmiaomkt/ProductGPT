"""Does the SAMPLING RULE explain the generative failure?

The gen-4 inference pipeline (evaluation/infer_new_campaign_*.py) sampled
decisions the same way we do -- multinomial over the softmax of the nine
decision logits -- but it added a learned probability CALIBRATOR (temperature,
Platt or vector scaling, fitted per checkpoint) before sampling. Our rollout
pins temperature at 1.0 and applies no calibrator.

Temperature scaling is the dominant calibrator in that family and is a pure
reweighting of the same distribution, so sweeping it is a direct test of whether
the sampling rule is implicated. The direction matters: the model over-produces
purchases and under-produces NotBuy, and NotBuy is the majority class, so
SHARPENING (T < 1) should push mass back toward it.

Either answer is informative.
  some T lands inside the band -> the sampling rule carries real weight, and the
                                  old pipeline's calibration was doing work.
  no T does                    -> the failure is structural, and T = 1 was not
                                  the problem. Note that a model which only
                                  looks right when sharpened is not reproducing
                                  the data distribution anyway: over-sharpening
                                  collapses variance, which the 27 functionals
                                  will catch elsewhere.

    python scripts/r36g_temperature.py --users 120 --n-rep 4
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
from baselines import real_sequences  # noqa: E402
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from rollout import first_row_of_campaign, load_checkpoint, rollout  # noqa: E402

HOLDOUT_FROM = 28
SHARE_LABELS = ["Buy1Reg", "Buy10Reg", "Buy1FigA", "Buy10FigA",
                "Buy1FigB", "Buy10FigB", "Buy1Wep", "Buy10Wep", "NotBuy"]


def shares(res, start):
    m = res["alive"].copy()
    m[:, :start] = False
    d = res["decisions"][m]
    d = d[d > 0]
    return np.bincount(d, minlength=10)[1:10] / max(len(d), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/r36c/b29_hl2_s1.pt")
    ap.add_argument("--users", type=int, default=120)
    ap.add_argument("--n-rep", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--max-events", type=int, default=384)
    ap.add_argument("--temps", nargs="+", type=float,
                    default=[0.5, 0.7, 0.85, 1.0, 1.3])
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

    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
    env = AcquisitionEnv(vocab).fit(
        [train_ds[i] for i in range(min(len(train_ds), 400))],
        holdout_from=HOLDOUT_FROM)
    model, _ = load_checkpoint(a.ckpt, device=a.device, legacy_camp_bias=True)

    rng = np.random.default_rng(0)
    pick = np.sort(rng.permutation(len(cell))[:a.users])
    chunks = [pick[i:i + a.chunk] for i in range(0, len(pick), a.chunk)]

    def collate(ch):
        b = dsm.collate_multistream([cell[i] for i in ch])
        b["user_idx"] = torch.zeros(b["label"].shape[0], dtype=torch.long)
        return b

    F_real, sh_real, starts = [], [], []
    for ch in chunks:
        b = collate(ch)
        st = int(first_row_of_campaign(b["campaign"], HOLDOUT_FROM).min())
        starts.append(st)
        r = real_sequences(b, start=st)
        F_real.append(sm.functionals(r["decisions"], r["alive"], r["obtained"],
                                     r["lto"], start=st))
        sh_real.append(shares(r, st))
    F_real = np.vstack(F_real)
    sc = sm.Scaler.fit(F_real)
    Zr = sc(F_real)
    print(f"\n[cell] {F_real.shape[0]} real sequences")
    print(f"real NotBuy share {np.vstack(sh_real).mean(0)[8]:.3f}\n")

    print(f"{'T':>6}{'energy':>9}{'null q95':>10}{'z':>8}{'NotBuy':>9}{'Buy10Wep':>10}  verdict")
    for T in a.temps:
        F, SH = [], []
        for ch, st in zip(chunks, starts):
            b = collate(ch)
            r = rollout(model, b, env, vocab, start=st, n_rep=a.n_rep, seed=23,
                        temperature=T, device=a.device)
            F.append(sm.functionals(r["decisions"], r["alive"], r["obtained"],
                                    r["lto"], start=st))
            SH.append(shares(r, st))
        c = sm.compare(Zr, sc(np.vstack(F)), n_boot=120, seed=5)
        s_ = np.vstack(SH).mean(0)
        print(f"{T:>6.2f}{c['stat']:>+9.3f}{c['null_q95']:>10.3f}{c['z']:>8.1f}"
              f"{s_[8]:>9.3f}{s_[7]:>10.3f}  "
              f"{'INSIDE' if c['inside_band'] else 'outside'}")


if __name__ == "__main__":
    main()
