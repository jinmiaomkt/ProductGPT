"""R45 part B: a model trained on complete histories, scoring censored ones.

For each gen-6 run, scores the out-of-sample x holdout cell with every
customer's history chopped (calendar, first k active campaigns) at each depth,
with the computed state restarting at zero (Z) or supplied from the chopped
rows (T). The scored events never change (the cut stops at campaign 27).

Also reports, for Z, how wrong the restarted state is against the truth: mean
absolute error of log(1 + inventory count) over product slots, and of the pity
depth, averaged over the scored rows.

Depth 0 is the self-check: it must reproduce the run's final.json NLL.

    python scripts/r45_eval_chopped.py --tags <frozen R44 gen-6 tag> --seeds 1 2 3 4 5
    python scripts/r45_eval_chopped.py --run-dirs <run dir> --depths 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "simulation",
          REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import model_multistream_state_space as mm  # noqa: E402
import train5_multistream as t5  # noqa: E402
import train6_streaming as t6  # noqa: E402  (also routes the chop fields through t5.evaluate)
from censor6 import ChopView, chop_index  # noqa: E402
from rollout import load_checkpoint  # noqa: E402
from shared.features import load_feature_tensor  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
CELL = "outsample_users_holdout_period"


def runs(a):
    if a.run_dirs:
        for d in a.run_dirs:
            hits = sorted(Path(d).glob("*/hpcc/best.pt"))
            if not hits:
                raise SystemExit(f"no */hpcc/best.pt under {d}")
            yield Path(d).name, None, hits[0]
        return
    for tag in a.tags:
        for s in a.seeds:
            hits = sorted(Path(a.runs_dir).glob(f"*_{tag}_s{s}/*/hpcc/best.pt"))
            if hits:
                yield tag, s, hits[0]
            else:
                print(f"[skip] no checkpoint for {tag} seed {s}")


@torch.no_grad()
def state_error(cell, k, val_from, rarity):
    """Z's restarted state against the truth, over the rows each customer keeps
    and is scored on: MAE of log1p(count) per slot, and of the pity depth."""
    inv_err = pity_err = 0.0
    n = 0
    for i in range(len(cell)):
        it = cell[i]
        m = (it["label"] >= 1) & (it["label"] <= 9)
        if not bool(m.any()):
            continue
        s = chop_index(it["campaign"], "calendar", k, val_from)
        if s == 0:
            n += int(m.sum())
            continue
        o, p = it["obtained"][None], it["prev_decision"][None]
        c_full, _ = mm.additive_inventory(o)
        c_zero, _ = mm.additive_inventory(o[:, s:])
        p_full = mm.pity_depth(o, p, rarity)
        p_zero = mm.pity_depth(o[:, s:], p[:, s:], rarity)
        keep = m[s:]
        inv = (torch.log1p(c_full[0, s:]) - torch.log1p(c_zero[0])).abs().mean(-1)
        pit = (p_full[0, s:] - p_zero[0]).abs().mean(-1)
        inv_err += float(inv[keep].sum())
        pity_err += float(pit[keep].sum())
        n += int(keep.sum())
    return inv_err / max(n, 1), pity_err / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--run-dirs", nargs="*", default=[])
    ap.add_argument("--depths", nargs="*", type=int, default=[0, 1, 2, 4, 8])
    ap.add_argument("--states", nargs="*", default=["zero", "true"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.003)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/r45/partB.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)

    out, bad = [], []
    cache = {}
    for tag, seed, ckpt in runs(a):
        model, cfg = load_checkpoint(ckpt, device=str(device))
        if cfg.get("gen") != 6:
            raise SystemExit(f"{ckpt} is not a gen-6 run")
        cfg = dict(cfg)
        adtype = t5.amp_dtype(device) if cfg.get("amp") else None
        final = json.loads((ckpt.parent / "final.json").read_text())
        key = (cfg.get("data_file"), cfg.get("max_users"), cfg.get("seed") if cfg.get("max_users") else None)
        if key not in cache:
            _, _, tests, _ = t5.build_loaders(dict(cfg))
            feat = load_feature_tensor(config5.feature_file_path(cfg),
                                       id_column=cfg.get("feature_id_column"),
                                       first_prod_id=cfg["first_prod_id"],
                                       last_prod_id=cfg["last_prod_id"],
                                       max_token_id=cfg["vocab_size_src"] - 1)
            cache[key] = (tests[CELL].dataset, feat[:, 0].float())
        cell, rarity = cache[key]
        vf = int(cfg["val_from"])
        model.chunk = int(cfg.get("eval_chunk", 1024))
        for k in a.depths:
            for state in (["zero"] if k == 0 else a.states):
                ds = ChopView(cell, "calendar" if k else "none", k, state, vf)
                dl, _, _ = t6.make_loader(ds, a.batch, False, 0, 0)
                m = t5.evaluate(model, dl, device, adtype)
                rec = {"tag": tag, "seed": seed, "depth": k, "state": state if k else "none",
                       "nll": m["nll"], "events": int(m["n"]), "hit": m["hit"],
                       "f1_macro": m["f1_macro"]}
                if k and state == "zero":
                    rec["inv_log_mae"], rec["pity_mae"] = state_error(cell, k, vf, rarity)
                if k == 0:
                    ref = final["test_cells"][CELL]
                    ok = abs(m["nll"] - ref["nll"]) < a.tol and int(m["n"]) == int(ref["n"])
                    rec["check_ok"] = bool(ok)
                    if not ok:
                        bad.append(f"{tag} s{seed}")
                out.append(rec)
                extra = (f"  inv_log_mae {rec['inv_log_mae']:.3f} pity_mae {rec['pity_mae']:.3f}"
                         if "inv_log_mae" in rec else "")
                chk = f"  (final {final['test_cells'][CELL]['nll']:.4f}, " \
                      f"{'ok' if rec.get('check_ok') else 'MISMATCH'})" if k == 0 else ""
                print(f"{tag} s{seed}  k={k} {rec['state']:<5} nll {m['nll']:.4f} over "
                      f"{int(m['n']):,} events{chk}{extra}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    if bad:
        print("SELF-CHECK FAILED (depth 0 does not reproduce final.json): " + ", ".join(bad))
        sys.exit(1)
    print("depth-0 self-check passed for every run")


if __name__ == "__main__":
    main()
