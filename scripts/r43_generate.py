"""R43 step 2: generation, gen 6 against the frozen gen 5 (EXPERIMENTS.md R43 step 2).

    python scripts/r43_generate.py --family g5            # G5 c06 (free, teacher forced),
                                                          #   arm A (free), the yardsticks
    python scripts/r43_generate.py --family g6            # G6 (free, teacher forced)
    python scripts/r43_generate.py --family g6 --probe    # time the g6 rollouts (GPU)
    python scripts/r43_generate.py --family g5 --smoke    # every arm on 12 customers

R36i's protocol, unchanged: the outsample_users_holdout_period cell, 300 customers
drawn with default_rng(0), 8 replicates, chunks of 6, the environment and
baselines fitted on 600 training customers' calibration rows, a split-half band
(120 splits, seed 3), rollout seed 1000 + training seed.

ONE FRAME FOR BOTH FAMILIES. Every job builds R36i's frame: the dataset at a
384-occasion window, from G5 c06 s1's configuration. The real set, the scaler,
the band, the environment and the customer draw all come from it, so the two
jobs compute them identically. Gen 5 generates in that frame (its trained 1,024
does not fit; R36c method note). Gen 6 generates over each customer's FULL
history, from a second build, and its output is cropped back to the frame
before scoring: the satiation functionals count holdings from the first row they
are given, so scoring gen 6 on its own longer rows would score a different real
set. The job refuses unless the cropped real sequences give the frame's real
functionals bit for bit.

Arm A (b39_g5rev2, untuned gen 5) is an extra arm with no decision attached.
The decision is read by scripts/r43_step2_read.py from the two jobs' outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "simulation",
          REPO / "shared", REPO / "scripts"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import seq_metrics as sm  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
from baselines import (fit_marginal, fit_markov, iid_sampler,  # noqa: E402
                       markov_sampler, real_sequences, simulate)
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from r36c_generate import ckpt_path, collate_chunk, score  # noqa: E402
from r36f_env_arm import arm_c  # noqa: E402
from r36i_rerun import (DATA_KEYS as R36I_DATA_KEYS, integrity, evaluate,  # noqa: E402
                        pooled_counts, summarise)
from rollout import first_row_of_campaign, load_checkpoint, rollout  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
CELL = "outsample_users_holdout_period"
HOLDOUT_FROM = 28
FIX_COMMIT = "7aca2a3"
PREFIX_Q95 = 0.07812581333995453      # R36c/e/i, n_boot 120, seed 3
MAX_EVENTS = 384
NULL_SEED = 3

# Training seed -> tag. Gen-5 c06's seed 1 is the stage-2 run, seeds 2-5 batch 42.
G5C06 = {1: "b40_g5_c06", 2: "b42_g5_c06", 3: "b42_g5_c06", 4: "b42_g5_c06", 5: "b42_g5_c06"}
ARM_A = {s: "b39_g5rev2" for s in range(1, 6)}
G6 = {s: "b39_g6w384" for s in range(1, 6)}
REF = ("b40_g5_c06", 1)                # the frame's configuration
ARMS = {                               # family -> [(arm, seed->tag, teacher_forced)]
    "g5": [("fr_g5c06", G5C06, False), ("tf_g5c06", G5C06, True), ("fr_A", ARM_A, False)],
    "g6": [("fr_g6", G6, False), ("tf_g6", G6, True)],
}
GEN = {"b40_g5_c06": 5, "b42_g5_c06": 5, "b39_g5rev2": 5, "b39_g6w384": 6}
# The families differ, by design, in gen and in max_events; nothing else that
# decides customers, rows or inputs may differ.
DATA_KEYS = tuple(k for k in R36I_DATA_KEYS if k != "gen")


def git(*args: str):
    import subprocess
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def item_len(ds, i: int) -> int:
    return int(ds[i]["lto"].size(0))


def align(cell5, cell6, i5: int, i6: int) -> int:
    """Offset of the frame's rows inside the full history, after checking that
    they are the same rows. Gen 5 keeps the LAST rows (tail truncation); its
    first row's `obtained` is padding (shift_obtained has no predecessor)."""
    a, b = cell5[i5], cell6[i6]
    L5, L6 = int(a["lto"].size(0)), int(b["lto"].size(0))
    off = L6 - L5
    bad = []
    if off < 0:
        bad.append(f"frame longer than the full history ({L5} > {L6})")
    else:
        for k in ("label", "lto", "campaign"):
            if k in a and not torch.equal(a[k], b[k][off:off + L5]):
                bad.append(k)
        if not torch.equal(a["obtained"][1:], b["obtained"][off + 1:off + L5]):
            bad.append("obtained")
    if bad:
        raise SystemExit(f"customer {a['uid']}: the frame's rows are not the full "
                         f"history's last {L5} rows ({', '.join(bad)}) -- refusing")
    return off


def to_frame(r: dict, b5: dict, offs: np.ndarray, lens: np.ndarray, n_rep: int):
    """Crop a full-history result to the frame of the gen-5 batch b5. Returns
    the cropped result and how many live or generated rows fell outside it."""
    lab5 = b5["label"].cpu().numpy()
    B5, S5 = lab5.shape
    dec = np.zeros((B5 * n_rep, S5), dtype=np.int64)
    obt = np.repeat(b5["obtained"].cpu().numpy(), n_rep, axis=0)
    lto = np.repeat(b5["lto"].cpu().numpy(), n_rep, axis=0)
    alive = np.repeat(lab5 != 0, n_rep, axis=0)
    outside = 0
    for j in range(B5 * n_rep):
        c = j // n_rep
        o, L = int(offs[c]), int(lens[c])
        dec[j, :L] = r["decisions"][j, o:o + L]
        obt[j, 1:L] = r["obtained"][j, o + 1:o + L]
        if not np.array_equal(r["alive"][j, o:o + L], alive[j, :L]):
            outside += 1_000_000            # cannot happen if align() passed; flag loudly
        outside += int((r["decisions"][j, :o] > 0).sum() + (r["decisions"][j, o + L:] > 0).sum())
        outside += int(r["alive"][j, :o].sum() + r["alive"][j, o + L:].sum())
    return {"decisions": dec, "alive": alive, "obtained": obt, "lto": lto}, outside


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("g5", "g6"), required=True)
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="every arm of the family on 12 customers x 2 replicates, one seed. "
                         "A wiring check; its numbers mean nothing.")
    ap.add_argument("--probe", action="store_true",
                    help="g6 only: time the free-running rollout of G6 s1 on the first and "
                         "the longest chunk, and price the job")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    runs = Path(a.runs_dir)
    fam = a.family
    if a.probe and fam != "g6":
        raise SystemExit("--probe prices the g6 job only")
    out_path = Path(a.out or f"results/r43/{'smoke_' if a.smoke else ''}step2_{fam}.json")

    if a.smoke:
        n_users, n_rep, chunk, n_boot = 12, 2, 6, 20
        train_seeds, draws_c, draws_b = (1,), (11,), (17,)
    else:
        n_users, n_rep, chunk, n_boot = 300, 8, 6, 120
        train_seeds, draws_c, draws_b = (1, 2, 3, 4, 5), (11, 12, 13), (17, 18, 19)
        if a.device != "cuda":
            raise SystemExit("the full run and the probe need a GPU; use --smoke for a wiring check")
    if chunk * n_rep * MAX_EVENTS > 65535:
        raise SystemExit(f"chunk {chunk} x {n_rep} x {MAX_EVENTS} exceeds CUDA's 65,535 "
                         "attention batches (R36i, job 42868)")

    # ------------------------------------------------------------- the code
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", FIX_COMMIT, "HEAD").returncode != 0:
        raise SystemExit(f"HEAD {head} does not contain the rollout fix {FIX_COMMIT}")
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "simulation", "scripts",
                "gen5_multistream", "gen6_streaming", "shared").stdout.strip()
    print(f"[code] HEAD {head}, contains {FIX_COMMIT}; "
          f"{'UNCOMMITTED CHANGES:' + chr(10) + dirty if dirty else 'clean'}")
    if dirty and not a.smoke:
        raise SystemExit("uncommitted changes to the code this job runs")

    # ------------------------------------------------ all fifteen checkpoints
    # Every job checks all of them, so a bad checkpoint fails either job early.
    seeds_all = (1, 2, 3, 4, 5)
    paths = {}
    for m in (G5C06, ARM_A, G6):
        for s in seeds_all:
            paths[(m[s], s)] = ckpt_path(m[s], s, runs)
    ref_cfg = torch.load(str(paths[REF]), map_location="cpu", weights_only=False)["cfg"]
    print("[ckpt]")
    for (t, s), p in paths.items():
        c = torch.load(str(p), map_location="cpu", weights_only=False)["cfg"]
        ss = float(c.get("sched_sampling", 0.0) or 0.0)
        print(f"  {t}_s{s}: {p.parent.parent.parent.name}  gen {c.get('gen', 5)}  rev "
              f"{c.get('model_rev', 1)}  camp_fe {c.get('camp_fe')}  sched_sampling {ss}  "
              f"max_events {c.get('max_events')}  window {c.get('window')}  lr {c.get('lr')}")
        diff = [k for k in DATA_KEYS if c.get(k) != ref_cfg.get(k)]
        if diff:
            raise SystemExit(f"{t}_s{s} disagrees with {REF[0]}_s{REF[1]} on {diff}")
        if c.get("model_rev", 1) != 2:
            raise SystemExit(f"{t}_s{s} is model_rev {c.get('model_rev', 1)}, not 2")
        if not c.get("camp_fe"):
            raise SystemExit(f"{t}_s{s} has no campaign control")
        if ss != 0.0:
            raise SystemExit(f"{t}_s{s} carries scheduled sampling {ss}")
        if int(c.get("gen", 5)) != GEN[t]:
            raise SystemExit(f"{t}_s{s} is gen {c.get('gen', 5)}, expected {GEN[t]}")
        if GEN[t] == 6 and (c.get("window") != 384 or c.get("max_events") is not None):
            raise SystemExit(f"{t}_s{s}: window {c.get('window')}, max_events "
                             f"{c.get('max_events')}; the frozen gen 6 is window 384, full history")

    # ---------------------------------------------------------- the frame
    from train5_multistream import build_loaders
    cfg5 = dict(ref_cfg)
    cfg5["max_events"] = MAX_EVENTS
    dsm.set_product_range(cfg5["first_prod_id"], cfg5["last_prod_id"])
    print(f"[frame] {REF[0]}_s{REF[1]}'s configuration, context {ref_cfg['max_events']} -> "
          f"{MAX_EVENTS}; pre-window inventory carried by inv_init")
    tr_dl, _, tests, _ = build_loaders(cfg5)
    cell5 = tests[CELL].dataset
    train_ds = tr_dl.dataset
    pick = np.sort(np.random.default_rng(0).permutation(len(cell5))[:n_users])
    uids = [str(cell5[i]["uid"]) for i in pick]
    uid_sha = hashlib.sha256("\n".join(uids).encode()).hexdigest()[:16]
    print(f"[cell] {CELL}: {len(cell5)} customers, scoring {len(pick)} "
          f"(customer-list sha256 {uid_sha})")

    fit_items = [train_ds[i] for i in range(min(len(train_ds), 600))]
    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg5))
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    p_marg = fit_marginal(fit_items, holdout_from=HOLDOUT_FROM)
    P_mkv = fit_markov(fit_items, holdout_from=HOLDOUT_FROM)
    print(f"[env] calibrated on {len(fit_items)} training customers (frame build)")

    chunks = [pick[i:i + chunk] for i in range(0, len(pick), chunk)]
    batches = [collate_chunk(cell5, ch) for ch in chunks]
    owns = [first_row_of_campaign(b["campaign"], HOLDOUT_FROM) for b in batches]
    lens = [np.array([item_len(cell5, i) for i in ch]) for ch in chunks]

    F_real = np.vstack([score(real_sequences(b, start=int(st.min())), st)
                        for b, st in zip(batches, owns)])
    scaler = sm.Scaler.fit(F_real)
    Zr = scaler(F_real)
    null = sm.null_band(Zr, n_boot=n_boot, seed=NULL_SEED)
    q95 = float(np.quantile(null, 0.95))
    real_counts = sum(pooled_counts(b["label"].cpu().numpy(), b["label"].cpu().numpy() != 0)
                      for b in batches)
    same_as_prefix = (not a.smoke) and abs(q95 - PREFIX_Q95) < 1e-9
    print(f"[real] {F_real.shape[0]} sequences, {F_real.shape[1]} functionals, "
          f"{int(real_counts.sum()):,} holdout rows")
    print(f"[band] split-half null, {n_boot} splits: mean {null.mean():+.4f}  "
          f"sd {null.std(ddof=1):.4f}  q95 {q95:.7f}")
    if not a.smoke:
        print(f"[band] R36i q95 {PREFIX_Q95:.7f}: "
              + ("REPRODUCED exactly, R36i's customers and functionals" if same_as_prefix else
                 f"NOT REPRODUCED (got {q95:.10f}): the families still share this band, but "
                 "R36i's numbers cannot be quoted alongside"))

    # ------------------------------------------- gen 6: the full-history build
    batches6, owns6, offs = None, None, None
    if fam == "g6":
        cfg6 = dict(torch.load(str(paths[("b39_g6w384", 1)]), map_location="cpu",
                               weights_only=False)["cfg"])
        _, _, tests6, _ = build_loaders(cfg6)
        cell6 = tests6[CELL].dataset
        where6 = {str(cell6[i]["uid"]): i for i in range(len(cell6))}
        missing = [u for u in uids if u not in where6]
        if missing:
            raise SystemExit(f"{len(missing)} of the frame's customers are not in the "
                             f"full-history cell -- refusing")
        chunks6 = [np.array([where6[str(cell5[i]["uid"])] for i in ch]) for ch in chunks]
        offs = [np.array([align(cell5, cell6, i5, i6) for i5, i6 in zip(c5, c6)])
                for c5, c6 in zip(chunks, chunks6)]
        batches6 = [collate_chunk(cell6, ch) for ch in chunks6]
        owns6 = [first_row_of_campaign(b["campaign"], HOLDOUT_FROM) for b in batches6]
        F6 = []
        for b6, b5, o, L, st in zip(batches6, batches, offs, lens, owns):
            fr, outside = to_frame(real_sequences(b6, start=0), b5, o, L, 1)
            if outside:
                raise SystemExit(f"real rows outside the frame: {outside} -- refusing")
            F6.append(score(fr, st))
        F6 = np.vstack(F6)
        same = np.array_equal(F6, F_real, equal_nan=True)
        long_ = int(sum((o > 0).sum() for o in offs))
        print(f"[full] gen-6 build: {len(cell6)} customers in the cell; {long_} of the "
              f"{len(pick)} have rows before the frame (up to {max(int(o.max()) for o in offs)})")
        print(f"[full] cropped real sequences give the frame's real functionals: "
              f"{'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            raise SystemExit("the full-history build does not reproduce the frame -- refusing")

    # ------------------------------------------------------------- the probe
    if a.probe:
        model, _ = load_checkpoint(paths[("b39_g6w384", 1)], device=a.device,
                                   legacy_camp_bias=False)
        steps = [int(b["label"].shape[1]) for b in batches6]
        longest = int(np.argmax(steps))
        rates = []
        for ci in sorted({0, longest}):
            t0 = time.time()
            rollout(model, batches6[ci], env, vocab, start=int(owns6[ci].min()),
                    n_rep=n_rep, seed=1001, device=a.device)
            if a.device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            rates.append(dt / steps[ci])
            print(f"[probe] chunk {ci}: {steps[ci]} rows in {dt:.1f}s "
                  f"({1000 * dt / steps[ci]:.1f} ms per row, {len(chunks[ci])} customers x {n_rep})")
        per_rollout = max(rates) * sum(steps)
        n_roll = sum(len(train_seeds) for _ in ARMS["g6"])
        print(f"[probe] {sum(steps):,} rows over {len(batches6)} chunks: about "
              f"{per_rollout / 60:.0f} min per rollout at the slower rate, "
              f"{n_roll * per_rollout / 3600:.1f} h for the {n_roll} g6 rollouts")
        if a.device == "cuda":
            print(f"[probe] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        return

    out = {"family": fam, "cell": CELL, "head": head, "smoke": a.smoke,
           "n_users": int(len(pick)), "n_rep": n_rep, "chunk": chunk, "max_events": MAX_EVENTS,
           "customer_sha256": uid_sha, "functionals": sm.functional_names(),
           "null_band": {"n_boot": n_boot, "seed": NULL_SEED, "q95": q95,
                         "mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                         "r36i_q95": PREFIX_Q95, "reproduces_r36i": same_as_prefix},
           "real": {"shares": (real_counts / real_counts.sum()).tolist(),
                    "holdout_rows": int(real_counts.sum())},
           "checkpoints": {f"{t}_s{s}": str(p) for (t, s), p in paths.items()},
           "arms": {}, "problems": []}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {"real": F_real}
    npz = out_path.with_name(out_path.stem + "_functionals.npz")

    def save() -> None:
        out_path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        np.savez_compressed(npz, functionals=np.array(sm.functional_names()), n_rep=n_rep,
                            customer_sha256=uid_sha, **raw)

    def run_arm(name: str, label: int, gen, teacher_forced: bool = False) -> None:
        """gen(k) -> (result in the frame, integrity dict, rows outside the frame)."""
        t0 = time.time()
        F, cnt = [], np.zeros(9, dtype=np.int64)
        integ = {"rows_alive": 0, "rows_mismatched": 0, "real_rows_altered": 0,
                 "rows_outside_frame": 0}
        for k, st in enumerate(owns):
            r, ig, outside = gen(k)
            F.append(score(r, st))
            cnt += pooled_counts(r["decisions"], r["alive"])
            for key, v in ig.items():
                integ[key] += v
            integ["rows_outside_frame"] += outside
        raw[f"{name}__{label}"] = np.vstack(F)
        rec = evaluate(Zr, scaler(raw[f"{name}__{label}"]), n_boot)
        rec.update(shares=(cnt / max(cnt.sum(), 1)).tolist(), integrity=integ,
                   seconds=round(time.time() - t0))
        if rec["null_q95"] != q95:
            out["problems"].append(f"{name} {label}: band {rec['null_q95']} is not the shared {q95}")
        if integ["rows_mismatched"] or integ["real_rows_altered"] or integ["rows_outside_frame"]:
            out["problems"].append(f"{name} {label}: integrity {integ}")
        if integ["rows_alive"] != n_rep * int(real_counts.sum()):
            out["problems"].append(f"{name} {label}: {integ['rows_alive']} live rows, "
                                   f"expected {n_rep * int(real_counts.sum())}")
        out["arms"].setdefault(name, {"runs": {}})["runs"][str(label)] = rec
        lo, hi = rec["auc_ci"]
        print(f"  {name:<10} {label:>4}  energy {rec['energy']:+.3f}  z {rec['z']:>6.1f}  "
              f"{'INSIDE ' if rec['inside_band'] else 'outside'}  AUC {rec['auc']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  NotBuy {rec['shares'][8]:.3f}  "
              f"top {', '.join(rec['top_functionals'][:3])}  "
              f"mismatch {integ['rows_mismatched']} altered {integ['real_rows_altered']} "
              f"outside {integ['rows_outside_frame']}  ({rec['seconds']}s)", flush=True)
        save()

    def in_frame(r, k, tf):
        return r, integrity(r, batches[k], owns[k], n_rep, tf), 0

    # -------------------------------------------- the yardsticks (g5 job, CPU)
    if fam == "g5":
        print("\n[yardsticks]")
        for d in draws_c:
            run_arm("armC", d, lambda k, d=d: in_frame(arm_c(
                batches[k], env, vocab, start=int(owns[k].min()), n_rep=n_rep, seed=d), k, False))
        for nm, samp in (("iid", iid_sampler(p_marg)), ("markov1", markov_sampler(P_mkv))):
            for d in draws_b:
                run_arm(nm, d, lambda k, d=d, samp=samp: in_frame(simulate(
                    batches[k], env, vocab, samp, start=int(owns[k].min()),
                    n_rep=n_rep, seed=d), k, False))

    # ------------------------------------------------------------ model arms
    print("\n[model arms]")
    for name, tags, tf in ARMS[fam]:
        for sd in train_seeds:
            model, _ = load_checkpoint(paths[(tags[sd], sd)], device=a.device,
                                       legacy_camp_bias=False)
            if fam == "g5":
                gen = (lambda k, sd=sd, tf=tf, model=model: in_frame(rollout(
                    model, batches[k], env, vocab, start=int(owns[k].min()), n_rep=n_rep,
                    seed=1000 + sd, device=a.device, teacher_forced=tf), k, tf))
            else:
                def gen(k, sd=sd, tf=tf, model=model):
                    r = rollout(model, batches6[k], env, vocab, start=int(owns6[k].min()),
                                n_rep=n_rep, seed=1000 + sd, device=a.device, teacher_forced=tf)
                    ig = integrity(r, batches6[k], owns6[k], n_rep, tf)
                    fr, outside = to_frame(r, batches[k], offs[k], lens[k], n_rep)
                    return fr, ig, outside
            run_arm(name, sd, gen, teacher_forced=tf)
            del model
            if a.device == "cuda":
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- report
    for v in out["arms"].values():
        v["summary"] = summarise(v["runs"])
    save()
    lab = ["Buy1Reg", "Buy10Reg", "Buy1FigA", "Buy10FigA", "Buy1FigB", "Buy10FigB",
           "Buy1Wep", "Buy10Wep", "NotBuy"]
    print(f"\n=== {fam}: {len(pick)} customers x {n_rep} replicates, band q95 {q95:.4f} ===")
    print(f"{'arm':<10}{'n':>3}{'energy':>9}{'sd':>8}{'min':>8}{'max':>8}{'z':>8}"
          f"{'in':>4}{'AUC':>7}  top functionals")
    for nm, v in out["arms"].items():
        s = v["summary"]
        print(f"{nm:<10}{s['n']:>3}{s['energy_mean']:>+9.3f}{s['energy_sd']:>8.3f}"
              f"{s['energy_min']:>+8.3f}{s['energy_max']:>+8.3f}{s['z_mean']:>8.1f}"
              f"{s['inside_band']:>4}{s['auc_mean']:>7.3f}  {', '.join(s['top_functionals'][:3])}")
    print(f"\n{'':>10}" + "".join(f"{x:>10}" for x in lab))
    print(f"{'real':>10}" + "".join(f"{x:>10.3f}" for x in out["real"]["shares"]))
    for nm, v in out["arms"].items():
        print(f"{nm:>10}" + "".join(f"{x:>10.3f}" for x in v["summary"]["shares_mean"]))
    if a.smoke:
        print("\n(smoke run: wiring only; the numbers mean nothing)")
    if out["problems"]:
        print("\nPROBLEMS:")
        for p in out["problems"]:
            print("  " + p)
    print(f"\nwrote {out_path} and {npz} ({len(raw) - 1} arm runs + real)")
    if out["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
