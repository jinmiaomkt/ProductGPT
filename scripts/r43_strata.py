"""R43 secondary analysis: headline-cell NLL by history length.

Splits the out-of-sample x holdout cell by each customer's FULL history length
in the data file: more than 1,024 occasions (the customers gen 5's window
truncates) against at most 1,024. Pre-registered in EXPERIMENTS.md R43 as
reported with no decision attached, and written and tested before any
batch-39 result was read.

    python scripts/r43_strata.py --tags b39_g5rev2 b39_g6full b39_g6w384 b39_g6w128 \
        --seeds 1 2 3 4 5 --out results/r43/strata.json
    python scripts/r43_strata.py --run-dirs <run dir> [<run dir> ...]     # explicit runs

Self-check per run: the event-weighted NLL over both strata must equal the
run's own final.json NLL on the cell, over the same number of events. The
scoring mirrors train5_multistream.evaluate (same forward call, same autocast),
so a failure means the strata are not the headline numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "simulation",
          REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
import train5_multistream as t5  # noqa: E402
from rollout import load_checkpoint  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")


def find_runs(a):
    if a.run_dirs:
        out = []
        for d in a.run_dirs:
            hits = sorted(Path(d).glob("*/hpcc/best.pt"))
            if not hits:
                raise SystemExit(f"no */hpcc/best.pt under {d}")
            out.append((Path(d).name, None, hits[0]))
        return out
    out = []
    for tag in a.tags:
        for s in a.seeds:
            hits = sorted(Path(a.runs_dir).glob(f"*_{tag}_s{s}/*/hpcc/best.pt"))
            if hits:
                out.append((tag, s, hits[0]))
            else:
                print(f"[skip] no checkpoint for {tag} seed {s}")
    return out


_raw_len = {}


def full_lengths(cfg):
    """uid -> number of occasions in the data file (the loader's n_full)."""
    path = str(config5.data_path(cfg))
    if path not in _raw_len:
        ai_rate = int(cfg["ai_rate"])
        out = {}
        for rec in dsm.load_json_dataset(path):
            ai = dsm.parse_token_ids(rec["AggregateInput"])
            dec = dsm.parse_token_ids(rec.get("Decision", []))
            n = len(ai) // ai_rate
            if dec:
                n = min(n, len(dec))
            out[dsm.TransformerDataset._uid(rec)] = n
        _raw_len[path] = out
    return _raw_len[path]


_cells = {}


def cell_for(cfg, name):
    key = (cfg.get("data_file"), cfg.get("max_events"), cfg.get("max_users"),
           cfg.get("seed") if cfg.get("max_users") else None, name)
    if key not in _cells:
        _, _, tests, _ = t5.build_loaders(dict(cfg))
        _cells[key] = tests[name].dataset
    return _cells[key]


@torch.no_grad()
def score(model, cfg, cell, device, batch):
    """Per-uid (sum of NLL, events) on the cell, scored as t5.evaluate scores it."""
    model.eval()
    if getattr(model, "streaming", False):
        model.chunk = int(cfg.get("eval_chunk", 1024))
    adtype = t5.amp_dtype(device) if cfg.get("amp") else None
    t5.set_unknown_user_to_mean(model, cfg.get("_trained_user_indices", []))
    lens = [(int(cell[i]["lto"].size(0)), i) for i in range(len(cell))]
    order = [i for _, i in sorted(lens)]
    acc = {}
    for k in range(0, len(order), batch):
        items = [cell[i] for i in order[k:k + batch]]
        if not any(bool((it["label"] != 0).any()) for it in items):
            continue
        b = dsm.collate_multistream(items)
        lto = b["lto"].to(device)
        obt = b["obtained"].to(device)
        prev = b["prev_decision"].to(device)
        uid = b["user_id"].to(device)
        tgt = b["label"].to(device)
        ipt = b["ipt"].to(device) if "ipt" in b else None
        inv = t5.inventory_kwargs(model, b, device)
        ctx = (torch.autocast("cuda", dtype=adtype) if adtype is not None
               else torch.autocast("cpu", enabled=False))
        with ctx:
            logits = model(lto, obt, prev, uid, ipt, **inv)
        prob = F.softmax(logits.float()[..., 1:10], dim=-1)
        m = (tgt >= 1) & (tgt <= 9)
        rows, cols = torch.nonzero(m, as_tuple=True)
        p = prob[rows, cols, tgt[rows, cols] - 1].clamp_min(1e-12)
        nll = (-torch.log(p)).cpu().numpy()
        r = rows.cpu().numpy()
        for j, u in enumerate(b["uid"]):
            sel = nll[r == j]
            if sel.size:
                s, n = acc.get(u, (0.0, 0))
                acc[u] = (s + float(sel.sum()), n + int(sel.size))
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--run-dirs", nargs="*", default=[])
    ap.add_argument("--cell", default="outsample_users_holdout_period")
    ap.add_argument("--cut", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.003)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="results/r43/strata.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)

    out, failed = [], []
    for tag, seed, ckpt in find_runs(a):
        model, cfg = load_checkpoint(ckpt, device=str(device))
        cfg = dict(cfg)
        final = json.loads((ckpt.parent / "final.json").read_text())
        cell = cell_for(cfg, a.cell)
        # build_loaders filled cfg["_trained_user_indices"] on its own copy;
        # with the user embedding off (every R43 arm) it is not needed.
        lengths = full_lengths(cfg)
        acc = score(model, cfg, cell, device, a.batch)
        rec = {"tag": tag, "seed": seed, "ckpt": str(ckpt), "gen": cfg.get("gen", 5)}
        tot_s = tot_n = 0
        for name, keep in (("long", lambda n: n > a.cut), ("short", lambda n: n <= a.cut)):
            us = [u for u in acc if keep(lengths[u])]
            s = sum(acc[u][0] for u in us)
            n = sum(acc[u][1] for u in us)
            rec[name] = {"customers": len(us), "events": n, "nll": s / n if n else float("nan")}
            tot_s += s
            tot_n += n
        rec["all"] = {"customers": len(acc), "events": tot_n, "nll": tot_s / max(tot_n, 1)}
        ref = final["test_cells"][a.cell]
        rec["final_nll"], rec["final_events"] = float(ref["nll"]), int(ref["n"])
        ok = abs(rec["all"]["nll"] - rec["final_nll"]) < a.tol and tot_n == rec["final_events"]
        rec["check_ok"] = bool(ok)
        if not ok:
            failed.append(f"{tag} s{seed}")
        print(f"{tag:<34} s{seed}  all {rec['all']['nll']:.4f} (final {rec['final_nll']:.4f}, "
              f"{'ok' if ok else 'MISMATCH'})  >{a.cut}: {rec['long']['nll']:.4f} "
              f"[{rec['long']['customers']} cust, {rec['long']['events']:,} ev]  "
              f"<={a.cut}: {rec['short']['nll']:.4f} [{rec['short']['customers']} cust, "
              f"{rec['short']['events']:,} ev]", flush=True)
        out.append(rec)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    if failed:
        print("SELF-CHECK FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("self-check passed for every run")


if __name__ == "__main__":
    main()
