"""R45 part C: score every arm on the main and the censored cells.

Arm F (R44's frozen gen-6 runs) never saw the censored file, so its final.json
has no censored cells; this scores them, with the same partition the inclusion
runs used. The inclusion arms are scored the same way, so every number in the
reading comes from one script.

Self-check: every cell already in a run's final.json must be reproduced (NLL
within --tol, the same event count). For F that is the three main cells; for
the inclusion arms it is all five.

    python scripts/r45c_eval.py --tags b39_g6w384 r45c_I0 r45c_I --seeds 1 2 3 4 5
    python scripts/r45c_eval.py --run-dirs <run dir> --out results/r45/partC_test.json
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

import train5_multistream as t5  # noqa: E402
import train6_streaming as t6  # noqa: E402  (also routes the censoring flag through t5.evaluate)
from censor6 import CensorMark  # noqa: E402
from rollout import load_checkpoint  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
EXTRA = "perproduct/censored_list_int_wide4_simple7_IPT.json"
CELLS = ("outsample_users_holdout_period", "insample_users_holdout_period",
         "censored_outsample_holdout_period", "censored_insample_holdout_period")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--run-dirs", nargs="*", default=[])
    ap.add_argument("--extra", default=EXTRA)
    ap.add_argument("--tol", type=float, default=0.003)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="results/r45/partC_eval.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)

    out, bad, cache = [], [], {}
    for tag, seed, ckpt in runs(a):
        model, cfg = load_checkpoint(ckpt, device=str(device))
        if cfg.get("gen") != 6:
            raise SystemExit(f"{ckpt} is not a gen-6 run")
        cfg = dict(cfg)
        own = cfg.get("extra_data_file")
        if own and own != a.extra:
            raise SystemExit(f"{ckpt} trained with {own}, not {a.extra}")
        trained = bool(own) and bool(cfg.get("extra_train", True))
        cfg.update({"extra_data_file": a.extra, "extra_train": trained})
        adtype = t5.amp_dtype(device) if cfg.get("amp") else None
        final = json.loads((ckpt.parent / "final.json").read_text())
        key = (cfg.get("data_file"), cfg.get("max_users"), trained)
        if key not in cache:
            _, _, tests, _ = t5.build_loaders(cfg)
            uids = cfg.pop("_extra_uids")
            cache[key] = {c: CensorMark(tests[c].dataset, uids) for c in CELLS}
        model.chunk = int(cfg.get("eval_chunk", 1024))
        rec = {"tag": tag, "seed": seed, "run": ckpt.parent.parent.parent.name,
               "censor_flag": bool(cfg.get("censor_flag")), "extra_trained": trained}
        for c in CELLS:
            dl, _, _ = t6.make_loader(cache[key][c], a.batch, False, 0, 0)
            m = t5.evaluate(model, dl, device, adtype)
            rec[c] = {"nll": m["nll"], "n": int(m["n"]), "hit": m["hit"], "f1_macro": m["f1_macro"]}
            ref = final.get("test_cells", {}).get(c)
            if ref is not None:
                ok = abs(m["nll"] - ref["nll"]) < a.tol and int(m["n"]) == int(ref["n"])
                rec[c]["check_ok"] = bool(ok)
                if not ok:
                    bad.append(f"{tag} s{seed} {c}")
            chk = (f"  (final {ref['nll']:.4f}, {'ok' if rec[c]['check_ok'] else 'MISMATCH'})"
                   if ref is not None else "")
            print(f"{tag} s{seed}  {c:<34} nll {m['nll']:.4f} over {int(m['n']):,}{chk}", flush=True)
        out.append(rec)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    if bad:
        print("SELF-CHECK FAILED (does not reproduce final.json): " + ", ".join(bad))
        sys.exit(1)
    print("self-check passed: every cell in every final.json reproduced")


if __name__ == "__main__":
    main()
