"""R43 step 0: does the rollout engine generate from a gen-6 checkpoint on REAL data?

Runs on the smoke checkpoint (job 42862) before anything in batch 39 is
submitted. CPU only (the GPU is hidden below); reads PRODUCTGPT_DATA.

    python scripts/r43_rollout_probe.py \
        --run-dir /storage/home/jinmiao/ProductGPT/runs/gen6_hpcc_W384_C512_b4_g6smoke_s1

Gates -- all four must pass:
  G1  load       rollout.load_checkpoint rebuilds the gen-6 model from best.pt.
  G2  same fn    teacher-forced NLL on the out-of-sample x holdout cell, computed
                 here in fp32, equals the trainer's final.json (bf16) within 0.01
                 nats, over the same number of scored events.
  G3  same draws on real customers and the fitted environment, the streaming
                 rollout and gen 5's prefix-recompute loop give identical
                 decisions and acquisitions from the same seed, free running and
                 teacher forced, over the first 60 generated occasions of the 8
                 customers whose holdout starts earliest (the reference loop's
                 cost grows with the square of history length). The synthetic
                 version is test T9.
  G4  runs       a larger streaming rollout completes: every live holdout row gets
                 a valid decision and nothing is generated anywhere else. Reports
                 the time per generated occasion, which prices step 2.

Writes results/r43/probe.json and exits 1 if any gate fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU by design

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "simulation",
          REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from rollout import first_row_of_campaign, load_checkpoint, rollout  # noqa: E402

HOLDOUT_FROM = 28
CELL = "outsample_users_holdout_period"
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}", flush=True)
    if not ok:
        FAIL.append(name)
    return ok


def collate(ds, idx):
    b = dsm.collate_multistream([ds[i] for i in idx])
    b["user_idx"] = torch.zeros(b["label"].shape[0], dtype=torch.long)
    return b


def live_customers(cell, n: int):
    """The first n customers in the cell with at least one holdout row."""
    out = []
    for i in range(len(cell)):
        if bool((cell[i]["label"] != 0).any()):
            out.append(i)
            if len(out) == n:
                break
    return out


def earliest_holdout(cell, n: int):
    """The n customers whose holdout starts earliest (shortest real prefix)."""
    rows = []
    for i in range(len(cell)):
        live = torch.nonzero(cell[i]["label"] != 0).view(-1)
        if live.numel():
            rows.append((int(live[0]), i))
    return [i for _, i in sorted(rows)[:n]]


def cut(b, n: int):
    """Keep rows [0, n) of every per-row tensor in a collated batch."""
    out = dict(b)
    for k, v in b.items():
        if torch.is_tensor(v) and v.dim() >= 2 and k not in ("inv_init_count", "inv_init_last"):
            out[k] = v[:, :n]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--equiv-users", type=int, default=8)
    ap.add_argument("--equiv-rep", type=int, default=2)
    ap.add_argument("--equiv-steps", type=int, default=60,
                    help="generated occasions compared in G3. The reference loop re-reads "
                         "the whole prefix at every step, so its cost grows with the square "
                         "of history length")
    ap.add_argument("--big-users", type=int, default=32)
    ap.add_argument("--big-rep", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/r43/probe.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    hits = sorted(Path(a.run_dir).glob("*/hpcc/best.pt"))
    if not hits:
        raise SystemExit(f"no */hpcc/best.pt under {a.run_dir}")
    ckpt = hits[0]
    final = json.loads((ckpt.parent / "final.json").read_text())
    rec = {"ckpt": str(ckpt), "gates": {}}

    print("G1 load", flush=True)
    model, cfg = load_checkpoint(ckpt, device="cpu")
    ok = bool(getattr(model, "streaming", False)) and cfg.get("gen") == 6
    check("gen-6 model rebuilt from best.pt", ok,
          f"window={getattr(model, 'window', None)} chunk={getattr(model, 'chunk', None)}")
    rec["gates"]["G1"] = ok
    if not ok:
        raise SystemExit(1)

    from train5_multistream import build_loaders
    tr_dl, _, tests, _ = build_loaders(dict(cfg))
    cell = tests[CELL].dataset
    train_ds = tr_dl.dataset

    print("G2 the same function as the trainer", flush=True)
    t0 = time.time()
    tot, n = 0.0, 0
    camp_on = getattr(model, "camp_bias", None) is not None
    with torch.no_grad():
        idx = list(range(len(cell)))
        for k in range(0, len(idx), 16):
            b = collate(cell, idx[k:k + 16])
            y = b["label"]
            m = (y >= 1) & (y <= 9)
            if not bool(m.any()):
                continue
            lg = model(b["lto"], b["obtained"], b["prev_decision"], b["user_idx"],
                       campaign=b["campaign"] if camp_on else None).float()
            tot += float(F.cross_entropy(lg[..., 1:10][m], y[m] - 1, reduction="sum"))
            n += int(m.sum())
    nll = tot / max(n, 1)
    ref = float(final["test_cells"][CELL]["nll"])
    ref_n = int(final["test_cells"][CELL]["n"])
    ok = abs(nll - ref) < 0.01 and n == ref_n
    check("teacher-forced NLL equals final.json (fp32 here, bf16 there)", ok,
          f"{nll:.4f} vs {ref:.4f} over {n:,} vs {ref_n:,} events ({time.time() - t0:.0f}s)")
    rec["gates"]["G2"] = ok
    rec["tf_nll"], rec["final_nll"], rec["events"] = nll, ref, n

    print("G3 the same draws on real data", flush=True)
    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
    fit_items = [train_ds[i] for i in range(min(len(train_ds), 600))]
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    # The reference loop re-reads the whole prefix at every step, so its cost
    # grows with the square of history length: the first version of this probe
    # (job 42863) needed about 13 h on a CPU node. Compare the first
    # `equiv_steps` generated occasions of the customers whose holdout starts
    # earliest -- same gate, bounded cost.
    b = collate(cell, earliest_holdout(cell, a.equiv_users))
    start = int(first_row_of_campaign(b["campaign"], HOLDOUT_FROM).min())
    b = cut(b, start + a.equiv_steps)
    S3 = int(b["label"].shape[1])
    passes = a.equiv_users * a.equiv_rep * sum(t + 1 for t in range(start, S3))
    print(f"  {a.equiv_users} customers x {a.equiv_rep} replicates, rows {start}..{S3 - 1}; "
          f"the reference loop processes ~{passes:,} rows per mode", flush=True)
    rec["G3"] = {"start": start, "rows": S3}
    g3 = True
    for tf in (False, True):
        name = "teacher forced" if tf else "free running"
        model.streaming = False                   # gen 5's loop: whole prefix each step
        t0 = time.time()
        ref_r = rollout(model, b, env, vocab, start=start, n_rep=a.equiv_rep, seed=7,
                        device="cpu", teacher_forced=tf)
        t_ref = time.time() - t0
        model.streaming = True
        t0 = time.time()
        got = rollout(model, b, env, vocab, start=start, n_rep=a.equiv_rep, seed=7,
                      device="cpu", teacher_forced=tf)
        t_got = time.time() - t0
        same_d = np.array_equal(ref_r["decisions"], got["decisions"])
        same_o = np.array_equal(ref_r["obtained"], got["obtained"])
        n_gen = int((got["decisions"] > 0).sum())
        agree = float((ref_r["decisions"] == got["decisions"]).mean())
        ok = same_d and same_o and n_gen > 0
        g3 &= ok
        check(f"{name}: identical decisions and acquisitions", ok,
              f"{n_gen:,} generated occasions, agreement {agree:.6f}; prefix recompute "
              f"{t_ref:.0f}s, streaming {t_got:.0f}s")
        rec["G3"][name] = {"identical": bool(ok), "generated": n_gen, "agreement": agree,
                           "secs_recompute": t_ref, "secs_streaming": t_got}
    rec["gates"]["G3"] = g3

    print("G4 a larger streaming rollout", flush=True)
    b = collate(cell, live_customers(cell, a.big_users))
    start = int(first_row_of_campaign(b["campaign"], HOLDOUT_FROM).min())
    t0 = time.time()
    r = rollout(model, b, env, vocab, start=start, n_rep=a.big_rep, seed=11, device="cpu")
    dt = time.time() - t0
    dec, alive = r["decisions"], r["alive"]
    valid = bool(((dec[alive] >= 1) & (dec[alive] <= 9)).all())
    silent = bool((dec[~alive] == 0).all())
    steps = dec.shape[1] - start
    ok = valid and silent and int(alive.sum()) > 0
    check("every live row gets a valid decision, nothing else is generated", ok,
          f"{int(alive.sum()):,} live rows over {dec.shape[0]} sequences; {steps} steps in "
          f"{dt:.0f}s = {1000 * dt / max(steps, 1):.0f} ms per step for the whole batch")
    shares = np.bincount(dec[alive], minlength=10)[1:10] / max(int(alive.sum()), 1)
    real = b["label"].repeat_interleave(a.big_rep, dim=0).numpy()
    real_sh = np.bincount(real[alive], minlength=10)[1:10] / max(int(alive.sum()), 1)
    print("  decision shares, generated vs real (informative only -- a 2-epoch smoke model):")
    print("   " + " ".join(f"{x:.3f}" for x in shares))
    print("   " + " ".join(f"{x:.3f}" for x in real_sh))
    rec["gates"]["G4"] = ok
    rec["G4"] = {"sequences": int(dec.shape[0]), "live_rows": int(alive.sum()), "steps": steps,
                 "secs": dt, "shares_generated": shares.tolist(), "shares_real": real_sh.tolist()}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"\nwrote {out}")
    if FAIL:
        print(f"{len(FAIL)} gate(s) FAILED: " + "; ".join(FAIL))
        sys.exit(1)
    print("ALL GATES PASSED")


if __name__ == "__main__":
    main()
