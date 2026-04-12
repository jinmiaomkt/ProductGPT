#!/usr/bin/env python3
"""
select_best_mixture_flash.py

Reads all completed Ray Tune trials for MixtureFlash and prints a ranked table.
"""
import json
from pathlib import Path

def main():
    base = Path("./ray_results/MixtureFlash_RayTune")
    if not base.exists():
        print(f"[ERROR] {base} not found")
        return

    trials = []
    for trial_dir in sorted(base.iterdir()):
        if not trial_dir.is_dir():
            continue
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue

        best_nll = float("inf")
        best_metrics = {}
        best_epoch = -1
        last_epoch = -1
        cfg = {}

        for line in result_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            ep = r.get("epoch", -1)
            last_epoch = max(last_epoch, ep)
            nll = r.get("val_nll", float("inf"))

            if nll < best_nll:
                best_nll = nll
                best_epoch = ep
                best_metrics = r

            # Grab config from first line
            if not cfg:
                for k in ("dm_heads", "N", "dropout", "lr", "tau", "gamma",
                           "warmup_steps", "data_frac", "batch_size",
                           "dff_mult", "label_smoothing", "num_mixture_heads",
                           "weight_decay"):
                    if k in r.get("config", {}):
                        cfg[k] = r["config"][k]

        if best_nll == float("inf"):
            continue

        # Try to read config from params.json
        params_file = trial_dir / "params.json"
        if params_file.exists() and not cfg:
            cfg = json.loads(params_file.read_text())

        trials.append({
            "trial_dir": str(trial_dir),
            "trial_name": trial_dir.name,
            "val_nll": best_nll,
            "epoch_at_best": best_epoch,
            "val_hit": best_metrics.get("val_hit", 0),
            "val_f1_macro": best_metrics.get("val_f1_macro", 0),
            "last_epoch": last_epoch,
            **{f"cfg_{k}": v for k, v in cfg.items()},
        })

    if not trials:
        print("[INFO] No completed trials found.")
        return

    trials.sort(key=lambda t: t["val_nll"])

    print(f"\n{'='*100}")
    print(f"{'Rank':>4} {'val_nll':>10} {'ep':>4} {'hit':>8} {'f1':>8} "
          f"{'dm_heads':>12} {'N':>3} {'MH':>3} {'drop':>6} {'lr':>10} {'bs':>3}")
    print(f"{'='*100}")

    for i, t in enumerate(trials[:20]):
        print(f"{i+1:>4} {t['val_nll']:>10.4f} {t['epoch_at_best']:>4} "
              f"{t.get('val_hit', 0):>8.4f} {t.get('val_f1_macro', 0):>8.4f} "
              f"{str(t.get('cfg_dm_heads', '?')):>12} {t.get('cfg_N', '?'):>3} "
              f"{t.get('cfg_num_mixture_heads', '?'):>3} "
              f"{t.get('cfg_dropout', 0):>6.3f} "
              f"{t.get('cfg_lr', 0):>10.6f} "
              f"{t.get('cfg_batch_size', '?'):>3}")

    best = trials[0]
    print(f"\n{'='*100}")
    print(f"BEST TRIAL: {best['trial_name']}")
    print(f"  val_nll:  {best['val_nll']:.6f}")
    print(f"  val_hit:  {best.get('val_hit', 'N/A')}")
    print(f"  val_f1:   {best.get('val_f1_macro', 'N/A')}")
    print(f"  epoch:    {best['epoch_at_best']}")
    for k, v in best.items():
        if k.startswith("cfg_"):
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()