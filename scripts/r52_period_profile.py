"""R52: where the out-of-sample x calibration cell's extra nats come from.

On every family the out-of-sample x calibration cell (campaigns < 28, players
never trained on) scores about 0.09-0.10 nats WORSE than the same players'
holdout cell (campaigns >= 28), although the model trained on that period.
This script scores the frozen checkpoints once more, keeps one row per scored
occasion, and lets the read-out split the cell by campaign, by tenure (occasions
since the player's first record), by class, and by player group, so the gap can
be attributed to (a) class mix, (b) cold start, (c) calendar time. Diagnostic
only: no model is selected or changed, and the rows are the same events the
run's final.json already scored (self-checked below).

    python scripts/r52_period_profile.py --tags b39_g6w384 --seeds 1 2 3 4 5     # score
    python scripts/r52_period_profile.py --read --tags b39_g6w384 g5_c06         # aggregate

Scoring mirrors train5_multistream.evaluate (same forward call, same autocast,
same chunk), on the base dataset with NO role mask, so every occasion of every
player is scored in one pass and the three test cells are recovered as filters:
    in-sample  x holdout      : group 0, holdout flag 1
    out-sample x calibration  : group 1, holdout flag 0
    out-sample x holdout      : group 1, holdout flag 1
    in-sample  x calibration  : group 0, holdout flag 0   (training FIT, not a forecast)
Self-check per run: each test cell's event count must equal final.json's n and
its mean NLL must be within --tol of final.json's nll; otherwise the run is
flagged and the script exits 1.

Per-event rows go to --out-dir/<tag>_s<seed>.npz: uid (blake2b hash, never
printed), grp, hf, camp, pos (0-based occasion index in the player's FULL
history), label, nll. Nothing about an individual player is printed.
"""
from __future__ import annotations

import argparse
import hashlib
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
CELLS = {
    "insample_users_calib_period_FIT": (0, 0),
    "insample_users_holdout_period": (0, 1),
    "outsample_users_calib_period": (1, 0),
    "outsample_users_holdout_period": (1, 1),
}
CHECKED = ("insample_users_holdout_period", "outsample_users_calib_period",
           "outsample_users_holdout_period")
LABELS = t5.DECISION_LABELS
TENURE_EDGES = [0, 10, 50, 200, 500, 1000, 10**9]
TENURE_NAMES = ["1-10", "11-50", "51-200", "201-500", "501-1000", ">1000"]


def uid_hash(u: str) -> int:
    return int.from_bytes(hashlib.blake2b(u.encode(), digest_size=8).digest(), "little") >> 1


def find_runs(a):
    if a.run_dirs:
        out = []
        for d in a.run_dirs:
            hits = sorted(Path(d).glob("*/hpcc/best.pt"))
            if not hits:
                raise SystemExit(f"no */hpcc/best.pt under {d}")
            out.append((Path(d).name, 0, hits[0]))
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


_splits = {}


def split_for(cfg):
    """(base dataset with the unknown-user rows already mapped, in-sample idx, out idx, trained idx)."""
    key = (cfg.get("data_file"), cfg.get("max_events"), cfg.get("max_users"),
           cfg.get("seed") if cfg.get("max_users") else None, cfg.get("split_seed"),
           cfg.get("val_mode"), cfg.get("val_from"))
    if key not in _splits:
        c = dict(cfg)
        _, _, tests, _ = t5.build_loaders(c)
        ins = tests["insample_users_holdout_period"].dataset
        oos = tests["outsample_users_calib_period"].dataset
        base = oos.base.base
        assert ins.base.base is base
        assert not set(ins.indices) & set(oos.indices)
        _splits[key] = (base, list(ins.indices), list(oos.indices),
                        list(c.get("_trained_user_indices", [])))
    return _splits[key]


def pad_long(ts, S, fill=0):
    out = torch.full((len(ts), S), fill, dtype=torch.long)
    for i, t in enumerate(ts):
        out[i, :t.numel()] = t
    return out


@torch.no_grad()
def score_group(model, cfg, base, indices, grp, device, batch, lengths):
    """One row per scored occasion of every player in `indices`."""
    model.eval()
    if getattr(model, "streaming", False):
        model.chunk = int(cfg.get("eval_chunk", 1024))
    adtype = t5.amp_dtype(device) if cfg.get("amp") else None
    order = [j for _, j in sorted((int(base._label[j].numel()), j) for j in indices)]
    cols = {k: [] for k in ("uid", "hf", "camp", "pos", "label", "nll")}
    for k in range(0, len(order), batch):
        items = [base.__getitem__(j) for j in order[k:k + batch]]
        if not any(bool(((it["label"] >= 1) & (it["label"] <= 9)).any()) for it in items):
            continue
        b = dsm.collate_multistream(items)
        S = int(b["label"].size(1))
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
        rows, cc = torch.nonzero(m, as_tuple=True)
        p = prob[rows, cc, tgt[rows, cc] - 1].clamp_min(1e-12)
        nll = (-torch.log(p)).cpu().numpy().astype(np.float32)
        r = rows.cpu().numpy()
        c = cc.cpu().numpy()
        hf = pad_long([it["holdout_feature"] for it in items], S).numpy()
        camp = pad_long([it["campaign"] for it in items], S).numpy()
        hashes = np.array([uid_hash(it["uid"]) for it in items], dtype=np.int64)
        if lengths is None:
            offs = np.zeros(len(items), dtype=np.int64)
        else:
            offs = np.array([lengths[it["uid"]] - int(it["label"].numel()) for it in items],
                            dtype=np.int64)
        cols["uid"].append(hashes[r])
        cols["hf"].append(hf[r, c].astype(np.int8))
        cols["camp"].append(camp[r, c].astype(np.int16))
        cols["pos"].append((offs[r] + c).astype(np.int32))
        cols["label"].append(tgt[rows, cc].cpu().numpy().astype(np.int8))
        cols["nll"].append(nll)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["grp"] = np.full(out["nll"].shape, grp, dtype=np.int8)
    return out


def score_main(a) -> None:
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    dest = Path(a.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for tag, seed, ckpt in find_runs(a):
        model, cfg = load_checkpoint(ckpt, device=str(device))
        cfg = dict(cfg)
        final = json.loads((ckpt.parent / "final.json").read_text())
        base, ins, oos, trained = split_for(cfg)
        t5.set_unknown_user_to_mean(model, trained)
        lengths = full_lengths(cfg) if cfg.get("max_events") else None
        parts = [score_group(model, cfg, base, ins, 0, device, a.batch, lengths),
                 score_group(model, cfg, base, oos, 1, device, a.batch, lengths)]
        rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        rec = {"tag": tag, "seed": seed, "ckpt": str(ckpt), "gen": cfg.get("gen", 5),
               "rows": int(rows["nll"].size), "cells": {}}
        ok = True
        for name, (g, h) in CELLS.items():
            sel = (rows["grp"] == g) & (rows["hf"] == h)
            n, mean = int(sel.sum()), float(rows["nll"][sel].mean()) if sel.any() else float("nan")
            cell = {"events": n, "nll": mean}
            if name in CHECKED:
                ref = final["test_cells"][name]
                cell["final_events"], cell["final_nll"] = int(ref["n"]), float(ref["nll"])
                cell["ok"] = bool(n == int(ref["n"]) and abs(mean - float(ref["nll"])) < a.tol)
                ok &= cell["ok"]
            rec["cells"][name] = cell
        rec["check_ok"] = bool(ok)
        if not ok:
            failed.append(f"{tag} s{seed}")
        print(f"{tag:<28} s{seed}  rows {rec['rows']:,}  " + "  ".join(
            f"{k.split('_')[0][:3]}x{k.split('_')[2][:3]} {v['nll']:.4f}"
            + (f"({'ok' if v['ok'] else 'MISMATCH ' + format(v['final_nll'], '.4f')})"
               if "ok" in v else "(fit)")
            for k, v in rec["cells"].items()), flush=True)
        np.savez_compressed(dest / f"{tag}_s{seed}.npz", **rows)
        (dest / f"{tag}_s{seed}.check.json").write_text(json.dumps(rec, indent=2))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if failed:
        print("SELF-CHECK FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("self-check passed for every run")


# ------------------------------------------------------------------ read-out
def entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    f = np.bincount(labels, minlength=10)[1:10].astype(np.float64)
    f = f[f > 0] / f.sum()
    return float(-(f * np.log(f)).sum())


def mean_nll(d, sel) -> float:
    return float(d["nll"][sel].mean()) if sel.any() else float("nan")


def tenure_bucket(pos: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(TENURE_EDGES[1:]), pos + 1, side="left")


def read_run(path: Path) -> dict:
    d = dict(np.load(path))
    out = {"events": int(d["nll"].size)}
    grp, hf, camp, pos, lab, nll = (d[k] for k in ("grp", "hf", "camp", "pos", "label", "nll"))
    o_cal, o_hol = (grp == 1) & (hf == 0), (grp == 1) & (hf == 1)
    i_cal, i_hol = (grp == 0) & (hf == 0), (grp == 0) & (hf == 1)
    out["cells"] = {n: {"events": int(s.sum()), "nll": mean_nll(d, s), "entropy": entropy(lab[s])}
                    for n, s in (("in_cal_fit", i_cal), ("in_hol", i_hol), ("out_cal", o_cal),
                                 ("out_hol", o_hol))}
    gap = out["cells"]["out_cal"]["nll"] - out["cells"]["out_hol"]["nll"]
    out["gap"] = gap

    # (B) by campaign, out-of-sample and in-sample players
    camps = sorted(int(x) for x in np.unique(camp))
    byc = {}
    for g, name in ((1, "out"), (0, "in")):
        byc[name] = {}
        for cp in camps:
            s = (grp == g) & (camp == cp)
            if not s.any():
                continue
            byc[name][str(cp)] = {"events": int(s.sum()),
                                  "players": int(np.unique(d["uid"][s]).size),
                                  "nll": mean_nll(d, s), "entropy": entropy(lab[s])}
    out["by_campaign"] = byc

    # (C) by tenure bucket, out-of-sample players, each period
    tb = tenure_bucket(pos)
    byt = {}
    for per, s0 in (("cal", o_cal), ("hol", o_hol)):
        byt[per] = {}
        for i, nm in enumerate(TENURE_NAMES):
            s = s0 & (tb == i)
            byt[per][nm] = {"events": int(s.sum()), "nll": mean_nll(d, s), "entropy": entropy(lab[s])}
    out["by_tenure_out"] = byt

    # (D) by campaign for mature out-of-sample players (tenure > 200 occasions)
    mature = pos >= 200
    out["by_campaign_out_mature"] = {
        str(cp): {"events": int(((grp == 1) & (camp == cp) & mature).sum()),
                  "nll": mean_nll(d, (grp == 1) & (camp == cp) & mature)}
        for cp in camps if ((grp == 1) & (camp == cp) & mature).any()}

    # (E) Kitagawa decomposition of the out-of-sample gap by class
    w_cal = np.bincount(lab[o_cal], minlength=10)[1:10] / o_cal.sum()
    w_hol = np.bincount(lab[o_hol], minlength=10)[1:10] / o_hol.sum()
    per_class = {}
    comp = within = 0.0
    for k in range(1, 10):
        nc, nh = mean_nll(d, o_cal & (lab == k)), mean_nll(d, o_hol & (lab == k))
        per_class[LABELS[k]] = {"w_cal": float(w_cal[k - 1]), "w_hol": float(w_hol[k - 1]),
                                "nll_cal": nc, "nll_hol": nh}
        comp += (w_cal[k - 1] - w_hol[k - 1]) * nh
        within += w_cal[k - 1] * (nc - nh)
    out["kitagawa_class"] = {"gap": gap, "composition": float(comp), "within": float(within),
                             "composition_share": float(comp / gap) if gap else float("nan"),
                             "per_class": per_class}

    # (F) cold-start counterfactuals: drop each player's first k occasions
    cf = {}
    for k in (10, 50, 100, 200, 500):
        s = o_cal & (pos >= k)
        v = mean_nll(d, s)
        cf[str(k)] = {"events": int(s.sum()), "nll": v,
                      "gap_closed_share": float((gap - (v - out["cells"]["out_hol"]["nll"])) / gap)}
    out["drop_first_k_out_cal"] = cf

    # (G) campaign windows, out-of-sample players
    win = {}
    for name, lo, hi in (("c1-10", 1, 10), ("c11-20", 11, 20), ("c20-27", 20, 27),
                         ("c25-27", 25, 27), ("c27", 27, 27), ("c28", 28, 28), ("c29", 29, 29),
                         ("c30", 30, 30)):
        s = (grp == 1) & (camp >= lo) & (camp <= hi)
        win[name] = {"events": int(s.sum()), "nll": mean_nll(d, s), "entropy": entropy(lab[s])}
    out["campaign_windows_out"] = win

    # (H) tenure in campaigns (campaign minus the player's first campaign), out-of-sample
    first = {}
    for u, cp in zip(d["uid"], camp):
        if u not in first or cp < first[u]:
            first[u] = int(cp)
    tc = camp.astype(np.int64) - np.array([first[u] for u in d["uid"]], dtype=np.int64)
    byk = {}
    for name, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8+", 8, 10**6)):
        for per, s0 in (("cal", o_cal), ("hol", o_hol)):
            s = s0 & (tc >= lo) & (tc <= hi)
            byk[f"{name}/{per}"] = {"events": int(s.sum()), "nll": mean_nll(d, s)}
    out["by_campaign_tenure_out"] = byk
    out["first_campaign_hist_out"] = {str(k): int(v) for k, v in sorted(
        zip(*np.unique([first[u] for u in np.unique(d["uid"][grp == 1])], return_counts=True)))}
    return out


def agg(vals):
    v = np.array([x for x in vals if x == x], dtype=np.float64)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size))


def read_main(a) -> None:
    src = Path(a.out_dir)
    for tag in a.tags:
        files = sorted(src.glob(f"{tag}_s*.npz"))
        if not files:
            print(f"[skip] no rows for {tag}")
            continue
        runs = [read_run(f) for f in files]
        print(f"\n==== {tag}: {len(runs)} seeds, {runs[0]['events']:,} rows each ====")
        print("cells (mean over seeds; sd):")
        for n in ("in_cal_fit", "in_hol", "out_cal", "out_hol"):
            m, s, _ = agg(r["cells"][n]["nll"] for r in runs)
            print(f"  {n:<11} {m:.4f} ({s:.4f})  events {runs[0]['cells'][n]['events']:,}  "
                  f"entropy {runs[0]['cells'][n]['entropy']:.4f}")
        gm, gs, _ = agg(r["gap"] for r in runs)
        print(f"  gap out_cal - out_hol = {gm:+.4f} ({gs:.4f})")

        print("\n(B) by campaign: out-of-sample players | in-sample players (<28 is FIT)")
        print(f"  {'camp':>4} {'events':>9} {'players':>7} {'nll':>7} {'sd':>6} {'entropy':>7} | "
              f"{'events':>9} {'nll':>7}")
        for cp in sorted(runs[0]["by_campaign"]["out"], key=int):
            o = runs[0]["by_campaign"]["out"][cp]
            m, s, _ = agg(r["by_campaign"]["out"][cp]["nll"] for r in runs)
            i = runs[0]["by_campaign"]["in"].get(cp)
            im = agg(r["by_campaign"]["in"][cp]["nll"] for r in runs)[0] if i else float("nan")
            print(f"  {cp:>4} {o['events']:>9,} {o['players']:>7,} {m:>7.4f} {s:>6.4f} "
                  f"{o['entropy']:>7.4f} | {(i or {}).get('events', 0):>9,} {im:>7.4f}")

        print("\n(C) by tenure (occasions since the player's first record), out-of-sample")
        print(f"  {'tenure':>9} {'cal events':>11} {'cal nll':>8} {'entropy':>7} | "
              f"{'hol events':>11} {'hol nll':>8}")
        for nm in TENURE_NAMES:
            c = runs[0]["by_tenure_out"]["cal"][nm]
            h = runs[0]["by_tenure_out"]["hol"][nm]
            cm = agg(r["by_tenure_out"]["cal"][nm]["nll"] for r in runs)[0]
            hm = agg(r["by_tenure_out"]["hol"][nm]["nll"] for r in runs)[0]
            print(f"  {nm:>9} {c['events']:>11,} {cm:>8.4f} {c['entropy']:>7.4f} | "
                  f"{h['events']:>11,} {hm:>8.4f}")

        print("\n(D) by campaign, out-of-sample players with tenure > 200 occasions")
        line = []
        for cp in sorted(runs[0]["by_campaign_out_mature"], key=int):
            m = agg(r["by_campaign_out_mature"][cp]["nll"] for r in runs)[0]
            line.append(f"c{cp}:{m:.3f}[{runs[0]['by_campaign_out_mature'][cp]['events'] // 1000}k]")
        print("  " + "  ".join(line))

        print("\n(E) Kitagawa decomposition of the out-of-sample gap by class")
        cm, cs, _ = agg(r["kitagawa_class"]["composition"] for r in runs)
        wm, ws, _ = agg(r["kitagawa_class"]["within"] for r in runs)
        sh = agg(r["kitagawa_class"]["composition_share"] for r in runs)[0]
        print(f"  gap {gm:+.4f} = composition {cm:+.4f} ({cs:.4f}) + within-class {wm:+.4f} ({ws:.4f});"
              f" composition share {sh:.2f}")
        print(f"  {'class':<11} {'w_cal':>6} {'w_hol':>6} {'nll_cal':>8} {'nll_hol':>8} {'diff':>7}")
        for k in range(1, 10):
            pc = [r["kitagawa_class"]["per_class"][LABELS[k]] for r in runs]
            nc, nh = agg(p["nll_cal"] for p in pc)[0], agg(p["nll_hol"] for p in pc)[0]
            print(f"  {LABELS[k]:<11} {pc[0]['w_cal']:>6.3f} {pc[0]['w_hol']:>6.3f} {nc:>8.4f} "
                  f"{nh:>8.4f} {nc - nh:>+7.4f}")

        print("\n(F) drop each out-of-sample player's first k occasions from the calibration cell")
        for k in ("10", "50", "100", "200", "500"):
            m = agg(r["drop_first_k_out_cal"][k]["nll"] for r in runs)[0]
            sh = agg(r["drop_first_k_out_cal"][k]["gap_closed_share"] for r in runs)[0]
            print(f"  k={k:>3}: nll {m:.4f}  events {runs[0]['drop_first_k_out_cal'][k]['events']:,}  "
                  f"gap closed {sh:+.2f}")

        print("\n(G) campaign windows, out-of-sample players")
        for nm, v in runs[0]["campaign_windows_out"].items():
            m, s, _ = agg(r["campaign_windows_out"][nm]["nll"] for r in runs)
            print(f"  {nm:<7} nll {m:.4f} ({s:.4f})  events {v['events']:>9,}  entropy {v['entropy']:.4f}")

        print("\n(H) tenure in campaigns (campaign - player's first campaign), out-of-sample")
        for key, v in runs[0]["by_campaign_tenure_out"].items():
            m = agg(r["by_campaign_tenure_out"][key]["nll"] for r in runs)[0]
            print(f"  {key:<8} nll {m:.4f}  events {v['events']:>9,}")
        print("  first-campaign histogram (out-of-sample players): "
              + ", ".join(f"c{k}:{v}" for k, v in runs[0]["first_campaign_hist_out"].items()))

        summary = {"tag": tag, "seeds": len(runs), "runs": runs}
        (src / f"profile_{tag}.json").write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {src / f'profile_{tag}.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--run-dirs", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.003)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default="results/r52")
    ap.add_argument("--read", action="store_true", help="aggregate existing rows instead of scoring")
    a = ap.parse_args()
    if a.read:
        read_main(a)
    else:
        score_main(a)


if __name__ == "__main__":
    main()
