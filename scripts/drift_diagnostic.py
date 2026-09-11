"""
Turn --track-holdout runs into verdicts on two questions from EXPERIMENTS.md R13.

  STEP 1 -- under-training or drift?
      Holdout NLL peaks at epoch 3-4 and then degrades. Two stories fit:
        (a) an early model is under-trained, sits near the class frequencies,
            and those happen to suit the holdout period better;
        (b) later epochs learn calibration-period structure that does not
            transfer, i.e. genuine temporal drift.
      Three measurements separate them, all read from history.json:
        * gain over the constant predictor: marginal entropy - NLL. If the
          holdout-best epoch barely beats "predict the class frequencies",
          story (a) is plausible. If it beats it clearly, the early model has
          learned real conditional structure that transfers.
        * predicted vs empirical class distribution, early vs late, in each
          period (the check agreed in conversation).
        * prior-matched NLL: rescale predictions so their mean matches the
          period's class frequencies. The share of the late-epoch degradation
          this removes is the share due to a shift in class FREQUENCIES; the
          rest is a shift in P(y | history).

  STEP 2 -- does late-calibration validation track the holdout?
      For each run: the epoch validation would pick, the epoch the holdout
      actually peaks, and the holdout NLL given up by trusting validation.
      Late validation "works" if that cost is small and the two epochs close.

The holdout numbers come from --track-holdout and are diagnostic only. Nothing
here selects a model; it measures whether a selection rule can be trusted.

USAGE
    python3 scripts/drift_diagnostic.py RUN_DIR [RUN_DIR ...]
    python3 scripts/drift_diagnostic.py ~/ProductGPT/runs/*diag2*

RUN_DIR may be a run directory or the directory holding history.json.
Pure standard library, so it runs under the cluster's system python3.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

LABELS = ["Buy1_Reg", "Buy10_Reg", "Buy1_FigA", "Buy10_FigA", "Buy1_FigB",
          "Buy10_FigB", "Buy1_Wep", "Buy10_Wep", "NotBuy"]
CELL = "ho_outsample_users_holdout_period"


def find_history(path: str) -> Optional[str]:
    if os.path.isfile(path) and path.endswith("history.json"):
        return path
    hits = glob.glob(os.path.join(path, "**", "history.json"), recursive=True)
    return hits[0] if hits else None


def tv(a: List[float], b: List[float]) -> float:
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def at(h: List[Dict[str, Any]], ep: int) -> Dict[str, Any]:
    return next(e for e in h if e["epoch"] == ep)


def analyse(hist_path: str) -> Optional[Dict[str, Any]]:
    h = json.load(open(hist_path))
    tracked = [e for e in h if f"{CELL}_nll" in e]
    if len(tracked) < 3:
        print(f"  skipped: fewer than 3 epochs carry holdout tracking")
        return None

    cfg_path = os.path.join(os.path.dirname(hist_path), "final.json")
    cfg = json.load(open(cfg_path)).get("cfg", {}) if os.path.exists(cfg_path) else {}
    val_mode = cfg.get("val_mode", "customers")

    ep_val = min(h, key=lambda e: e["nll"])["epoch"]
    ep_ho = min(tracked, key=lambda e: e[f"{CELL}_nll"])["epoch"]
    ep_last = tracked[-1]["epoch"]
    k = CELL

    # ---- per-epoch table ------------------------------------------------
    print(f"  val_mode={val_mode}"
          + (f" (validation = campaign >= {cfg.get('val_from')})" if val_mode == "late" else
             " (validation = held-out customers x calibration period)"))
    print(f"\n  {'ep':>3} | {'val NLL':>8} | {'HO NLL':>8} {'HO prior-m':>11} "
          f"{'HO H(y)':>8} {'gain':>7} {'TV':>6}")
    for e in tracked:
        mark = ("  <- val picks" if e["epoch"] == ep_val else "") + \
               ("  <- HOLDOUT best" if e["epoch"] == ep_ho else "")
        gain = e[f"{k}_marginal_entropy"] - e[f"{k}_nll"]
        print(f"  {e['epoch']:>3} | {e['nll']:>8.4f} | {e[f'{k}_nll']:>8.4f} "
              f"{e[f'{k}_nll_prior_matched']:>11.4f} {e[f'{k}_marginal_entropy']:>8.4f} "
              f"{gain:>+7.4f} {e[f'{k}_tv_pred_true']:>6.3f}{mark}")
    print("  gain = H(y) - NLL: nats beaten over the best CONSTANT predictor.")
    print("  TV   = total variation between mean predicted and true class distribution.")

    b, L = at(h, ep_ho), at(h, ep_last)

    # ---- STEP 1 ---------------------------------------------------------
    print("\n  STEP 1 -- under-training or drift?")
    g_best = b[f"{k}_marginal_entropy"] - b[f"{k}_nll"]
    g_last = L[f"{k}_marginal_entropy"] - L[f"{k}_nll"]
    print(f"    gain over constant predictor: epoch {ep_ho} {g_best:+.4f} nats, "
          f"epoch {ep_last} {g_last:+.4f} nats")

    print(f"\n    class distributions (holdout period), epoch {ep_ho} vs {ep_last}:")
    print(f"    {'class':<11}{'true':>8}{'pred@' + str(ep_ho):>10}{'pred@' + str(ep_last):>10}")
    for i, name in enumerate(LABELS):
        print(f"    {name:<11}{L[f'{k}_true_dist'][i]:>8.3f}"
              f"{b[f'{k}_pred_dist'][i]:>10.3f}{L[f'{k}_pred_dist'][i]:>10.3f}")
    print(f"    TV(pred, true): epoch {ep_ho} {b[f'{k}_tv_pred_true']:.3f}  "
          f"epoch {ep_last} {L[f'{k}_tv_pred_true']:.3f}")

    if "true_dist" in b:
        shift = tv(b["true_dist"], L[f"{k}_true_dist"])
        print(f"    TV(validation-period true, holdout true) = {shift:.3f}  "
              "<- how far the class mix itself moved between periods")
        print(f"    TV(pred, true) on VALIDATION: epoch {ep_ho} "
              f"{b.get('tv_pred_true', float('nan')):.3f}  epoch {ep_last} "
              f"{L.get('tv_pred_true', float('nan')):.3f}")

    d_raw = L[f"{k}_nll"] - b[f"{k}_nll"]
    d_pm = L[f"{k}_nll_prior_matched"] - b[f"{k}_nll_prior_matched"]
    if ep_ho == ep_last or d_raw <= 1e-9:
        share = float("nan")
        print(f"\n    holdout NLL was still improving at the last epoch ({ep_last}),")
        print("    so there is no degradation to decompose. Train longer (PATIENCE=99).")
    else:
        share = (d_raw - d_pm) / d_raw
        print(f"\n    holdout NLL degradation epoch {ep_ho} -> {ep_last}: {d_raw:+.4f} nats")
        print(f"    after oracle prior-matching:              {d_pm:+.4f} nats")
        print(f"    share explained by class-frequency shift: {share:.0%}")

    # ---- STEP 2 ---------------------------------------------------------
    v = at(h, ep_val)
    cost = v[f"{k}_nll"] - b[f"{k}_nll"]
    print("\n  STEP 2 -- does validation track the holdout?")
    print(f"    validation picks epoch {ep_val}; holdout peaks at epoch {ep_ho} "
          f"(gap {ep_val - ep_ho:+d} epochs)")
    print(f"    holdout NLL at validation's pick: {v[f'{k}_nll']:.4f} vs best "
          f"{b[f'{k}_nll']:.4f} -> cost {cost:+.4f} nats")
    return {"val_mode": val_mode, "ep_val": ep_val, "ep_ho": ep_ho, "cost": cost,
            "gain_best": g_best, "share_prior": share, "d_raw": d_raw}


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    summary = []
    for p in paths:
        hp = find_history(os.path.expanduser(p))
        if not hp:
            continue
        # history.json lives at <run>/gen5_multistream/<profile>/history.json
        tag = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(hp)))))
        print("=" * 78)
        print(tag)
        print("=" * 78)
        r = analyse(hp)
        if r:
            summary.append((tag, r))
        print()

    if len(summary) > 1:
        print("=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"  {'run':<38}{'val':>10}{'val ep':>8}{'HO ep':>7}{'cost':>9}{'prior%':>8}")
        for tag, r in summary:
            print(f"  {tag[:37]:<38}{r['val_mode']:>10}{r['ep_val']:>8}{r['ep_ho']:>7}"
                  f"{r['cost']:>+9.4f}"
                  + (f"{r['share_prior']:>8.0%}" if r["share_prior"] == r["share_prior"]
                     else f"{'n/a':>8}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
