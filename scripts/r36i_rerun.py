"""R36i: the R36 two-by-two and yardstick, re-run on the fixed rollout engine.

    python scripts/r36i_rerun.py              # the pre-registered run (GPU)
    python scripts/r36i_rerun.py --smoke      # every arm on 12 customers, one seed

Every number behind results/figs/generative_failure (panels A and B) ran on the
engine with the four defects fixed in 7aca2a3. The two-by-two also combined runs
that shared neither customers, checkpoint nor band: teacher forcing was b29_hl2
on 60 customers (job 41859, band 0.373), free running the b33 control on 300
(41897), arm C a separate local run. This scores every arm in ONE process: one
dataset build, one draw of 300 customers, one scaler, one split-half null band.

    arm            what is re-drawn                         seeds
    armC           acquisitions (decisions stay real)       draws 11, 12, 13
    iid, markov1   decisions and acquisitions               draws 17, 18, 19
    tf_b33         decisions; both streams stay real        training 1-3
    fr_b33         decisions and acquisitions, fed back     training 1-3
    fr_b35_ss015   as fr_b33, scheduled sampling p = 0.15   training 1-3
    fr_b35_ss035   as fr_b33, scheduled sampling p = 0.35   training 1-3

The two model cells of the two-by-two use the same checkpoints (b33_campfe_hl2)
and the same rollout seed (1000 + training seed). The cell, the customer draw,
the scaler and the band settings are R36c/e's, so the real-vs-real q95 must
reproduce 0.0781258 exactly. If it does not, the customers or the functionals
differ from the pre-fix runs: the new numbers stand on their own, but the
comparison with the pre-fix ones is void, and the job says so.

The verdicts are computed here, by the rules pre-registered in EXPERIMENTS.md
(R36i), so reading the log involves no judgement about thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared",
          REPO / "scripts"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import seq_metrics as sm  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
from baselines import (fit_marginal, fit_markov, iid_sampler,  # noqa: E402
                       markov_sampler, real_sequences, simulate)
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from r36c_generate import ckpt_path, collate_chunk, score  # noqa: E402
from r36f_env_arm import arm_c  # noqa: E402
from rollout import first_row_of_campaign, load_checkpoint, rollout  # noqa: E402

RUNS = Path("/storage/home/jinmiao/ProductGPT/runs")
CELL = "outsample_users_holdout_period"
HOLDOUT_FROM = 28
FIX_COMMIT = "7aca2a3"
PREFIX_Q95 = 0.07812581333995453      # R36c (41840) and R36e (41897), n_boot 120, seed 3
MAX_EVENTS = 384
NULL_SEED = 3

B33 = "b33_campfe_hl2"
MODEL_ARMS = [                         # (arm, tag, teacher_forced)
    ("tf_b33", B33, True),
    ("fr_b33", B33, False),
    ("fr_b35_ss015", "b35_ss015", False),
    ("fr_b35_ss035", "b35_ss035", False),
]
SS_P = {B33: 0.0, "b35_ss015": 0.15, "b35_ss035": 0.35}
# What decides which customers, rows and inputs the dataset holds. Every
# checkpoint must agree with the one the dataset is built from, or the arms
# would not be scored on the same thing.
DATA_KEYS = ("vocab_level", "first_prod_id", "last_prod_id", "unk_prod_id", "data_file",
             "feature_file", "split_mode", "split_seed", "user_holdout_frac",
             "holdout_flag", "val_mode", "val_from", "shift_obtained", "ai_rate",
             "lto_len", "obtained_len", "prev_dec_len", "gen")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def pooled_counts(dec: np.ndarray, alive: np.ndarray) -> np.ndarray:
    d = dec[alive & (dec > 0)]
    return np.bincount(d, minlength=10)[1:10].astype(np.int64)


def integrity(res, batch, own: np.ndarray, n_rep: int, teacher_forced: bool) -> dict:
    """What every arm must satisfy: a decision on every live holdout row and on
    no other row, and the REAL acquisitions on every row the arm had no business
    changing (rows up to each sequence's own first holdout row; all rows when
    teacher-forced)."""
    dec, alive = res["decisions"], res["alive"]
    real_obt = np.repeat(batch["obtained"].cpu().numpy(), n_rep, axis=0)
    S = real_obt.shape[1]
    if teacher_forced:
        keep = np.ones((real_obt.shape[0], S), dtype=bool)
    else:
        keep = np.arange(S)[None, :] <= np.repeat(own, n_rep)[:, None]
    return {"rows_alive": int(alive.sum()),
            "rows_mismatched": int(((dec > 0) != alive).sum()),
            "real_rows_altered": int((res["obtained"] != real_obt)[keep].sum())}


def defect3_reach(alive: np.ndarray, start: int) -> tuple:
    """Sequences and rows the pre-fix loop never generated: it stopped at the first
    row >= start where no sequence in the chunk was alive."""
    dead = np.flatnonzero(~alive[:, start:].any(axis=0))
    if dead.size == 0:
        return 0, 0
    lost = alive[:, start + int(dead[0]):].sum(axis=1)
    return int((lost > 0).sum()), int(lost.sum())


def evaluate(Zr: np.ndarray, Z: np.ndarray, n_boot: int) -> dict:
    c = sm.compare(Zr, Z, n_boot=n_boot, seed=NULL_SEED)
    k = sm.c2st(Zr, Z, seed=0)
    imp = np.asarray(k["importance"], dtype=float)
    names = sm.functional_names()
    return {"energy": c["stat"], "null_q95": c["null_q95"], "z": c["z"],
            "p_value": c["p_value"], "inside_band": c["inside_band"],
            "auc": k["auc"], "auc_ci": list(k["ci95"]), "importance": imp.tolist(),
            "top_functionals": [names[i] for i in np.argsort(imp)[::-1][:5]]}


def summarise(runs: dict) -> dict:
    r = list(runs.values())
    e = np.array([x["energy"] for x in r])
    a = np.array([x["auc"] for x in r])
    z = np.array([x["z"] for x in r])
    imp = np.mean([x["importance"] for x in r], axis=0)
    sd = (lambda v: float(v.std(ddof=1))) if len(r) > 1 else (lambda v: float("nan"))
    names = sm.functional_names()
    return {"n": len(r), "energy_mean": float(e.mean()), "energy_sd": sd(e),
            "energy_min": float(e.min()), "energy_max": float(e.max()),
            "z_mean": float(z.mean()), "auc_mean": float(a.mean()), "auc_sd": sd(a),
            "auc_ci_lo_min": float(min(x["auc_ci"][0] for x in r)),
            "auc_ci_hi_max": float(max(x["auc_ci"][1] for x in r)),
            "auc_ci_has_half": int(sum(x["auc_ci"][0] <= 0.5 <= x["auc_ci"][1] for x in r)),
            "inside_band": int(sum(bool(x["inside_band"]) for x in r)),
            "shares_mean": np.mean([x["shares"] for x in r], axis=0).tolist(),
            "top_functionals": [names[i] for i in np.argsort(imp)[::-1][:5]]}


def differs(x: dict, y: dict, band: float) -> dict:
    """R43's rule at three seeds: x - y counts as a difference only if it has the
    same sign on every seed pair (paired by seed number) and |mean| > the band."""
    keys = sorted(set(x) & set(y))
    d = np.array([x[k] - y[k] for k in keys])
    same_sign = bool(len(d) > 0 and (np.all(d > 0) or np.all(d < 0)))
    return {"per_seed": dict(zip(keys, d.tolist())), "mean": float(d.mean()),
            "counts": bool(same_sign and abs(d.mean()) > band)}


def verdicts(out: dict) -> dict:
    """The rules pre-registered in EXPERIMENTS.md, R36i. Nothing here is tuned."""
    q = out["null_band"]["q95"]
    A = {k: {s: r["energy"] for s, r in v["runs"].items()} for k, v in out["arms"].items()}
    v = {}
    v["V1_lottery_harmless"] = all(e <= q for e in A["armC"].values())
    v["V2_closed_loop_fails"] = all(e > q for e in A["fr_b33"].values())
    v["V3_teacher_forced_inside"] = all(e <= q for e in A["tf_b33"].values())
    gap = differs(A["fr_b33"], A["tf_b33"], q)
    ratio = {s: A["tf_b33"][s] / A["fr_b33"][s] for s in A["tf_b33"] if s in A["fr_b33"]}
    v["V4_loop_makes_most_of_gap"] = bool(gap["counts"] and gap["mean"] > 0
                                          and all(x < 0.5 for x in ratio.values()))
    mean = {k: float(np.mean(list(A[k].values()))) for k in A}
    order = ["armC", "tf_b33", "markov1", "fr_b33"]
    v["V5_ordering"] = all(mean[x] < mean[y] for x, y in zip(order, order[1:]))
    v["ordering_means"] = {k: mean[k] for k in order}
    if all(v[k] for k in ("V1_lottery_harmless", "V2_closed_loop_fails",
                          "V4_loop_makes_most_of_gap", "V5_ordering")):
        v["diagnosis"] = ("SURVIVES as stated" if v["V3_teacher_forced_inside"]
                          else "SURVIVES in weak form: teacher forcing is close, not inside")
    else:
        v["diagnosis"] = "DOES NOT SURVIVE"
    v["fr_minus_tf"] = gap
    v["tf_over_fr"] = ratio
    v["gap_removed_by_tf"] = {s: 1.0 - x for s, x in ratio.items()}
    m = {k: float(np.mean(list(A[k].values()))) for k in ("fr_b33", "fr_b35_ss015", "fr_b35_ss035")}
    v["remedy_monotone_in_p"] = bool(m["fr_b33"] > m["fr_b35_ss015"] > m["fr_b35_ss035"])
    v["remedy_ss035_vs_b33"] = differs(A["fr_b35_ss035"], A["fr_b33"], q)
    v["remedy_ss015_vs_b33"] = differs(A["fr_b35_ss015"], A["fr_b33"], q)
    v["remedy_gap_closed_ss035"] = 1.0 - m["fr_b35_ss035"] / m["fr_b33"]
    v["remedy_gap_closed_ss015"] = 1.0 - m["fr_b35_ss015"] / m["fr_b33"]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--out", default="results/r36i/r36i.json")
    ap.add_argument("--smoke", action="store_true",
                    help="every arm on 12 customers x 2 replicates, one seed each. A "
                         "wiring check; its numbers mean nothing.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    runs = Path(a.runs_dir)

    if a.smoke:
        n_users, n_rep, chunk, n_boot = 12, 2, 6, 20
        train_seeds, draws_c, draws_b = (1,), (11,), (17,)
    else:
        # chunk 6 is what R36c (41840) and R36e (41897) actually ran (/tmp/r36e.pbs),
        # not r36c_generate's default of 32; job 42868 crashed at 32.
        n_users, n_rep, chunk, n_boot = 300, 8, 6, 120
        train_seeds, draws_c, draws_b = (1, 2, 3), (11, 12, 13), (17, 18, 19)
        if a.device != "cuda":
            raise SystemExit("the full run needs a GPU (twelve rollouts of about 13 min "
                             "each on an L40S); use --smoke for a wiring check")
    # The satiation blocks attend over (sequences x prefix rows) as one batch, and
    # CUDA's efficient attention cannot launch more than 65,535 of them: beyond
    # that it fails with "invalid configuration argument" (job 42868 at 256 x 384).
    if chunk * n_rep * MAX_EVENTS > 65535:
        raise SystemExit(f"chunk {chunk} x {n_rep} replicates x {MAX_EVENTS} rows = "
                         f"{chunk * n_rep * MAX_EVENTS:,} attention batches; the limit is 65,535")

    # ------------------------------------------------------------- the code
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", FIX_COMMIT, "HEAD").returncode != 0:
        raise SystemExit(f"HEAD {head} does not contain the rollout fix {FIX_COMMIT}")
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "simulation",
                "scripts", "gen5_multistream", "shared").stdout.strip()
    print(f"[code] HEAD {head}, contains {FIX_COMMIT}; "
          f"{'UNCOMMITTED CHANGES:' + chr(10) + dirty if dirty else 'clean'}")
    if dirty and not a.smoke:
        raise SystemExit("uncommitted changes to the code this job runs")

    # ---------------------------------------------------- the nine checkpoints
    # All nine are checked even in --smoke, so a bad checkpoint fails in minutes.
    tags = [B33, "b35_ss015", "b35_ss035"]
    paths = {(t, s): ckpt_path(t, s, runs) for t in tags for s in (1, 2, 3)}
    base = None
    print("[ckpt]")
    for (t, s), p in paths.items():
        c = torch.load(str(p), map_location="cpu", weights_only=False)["cfg"]
        base = base or dict(c)
        ss = float(c.get("sched_sampling", 0.0) or 0.0)
        print(f"  {t}_s{s}: {p.parent.parent.parent.name}  rev {c.get('model_rev', 1)}  "
              f"camp_fe {c.get('camp_fe')}  sched_sampling {ss}  "
              f"inventory {c.get('inventory')}  decay_init {c.get('decay_init')}")
        diff = [k for k in DATA_KEYS if c.get(k) != base.get(k)]
        if diff:
            raise SystemExit(f"{t}_s{s} disagrees with {B33}_s1 on {diff}")
        if abs(ss - SS_P[t]) > 1e-9:
            raise SystemExit(f"{t}_s{s} carries sched_sampling {ss}, expected {SS_P[t]}")
        if not c.get("camp_fe"):
            raise SystemExit(f"{t}_s{s} has no campaign FE; this job keeps camp_bias as trained")

    # ------------------------------------------------------------- the data
    cfg = dict(base)
    cfg["max_events"] = MAX_EVENTS
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    print(f"[ctx] context window {base['max_events']} -> {MAX_EVENTS}; pre-window "
          "inventory carried by inv_init (fixed engine), attention context is not")
    from train5_multistream import build_loaders
    tr_dl, _, tests, _ = build_loaders(cfg)
    cell = tests[CELL].dataset
    train_ds = tr_dl.dataset
    pick = np.sort(np.random.default_rng(0).permutation(len(cell))[:n_users])
    uid_sha = hashlib.sha256("\n".join(str(cell[i]["uid"]) for i in pick)
                             .encode()).hexdigest()[:16]
    print(f"[cell] {CELL}: {len(cell)} customers, scoring {len(pick)} "
          f"(customer-list sha256 {uid_sha})")

    # Environment and baselines on TRAIN customers' calibration rows, as R36c.
    fit_items = [train_ds[i] for i in range(min(len(train_ds), 600))]
    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    p_marg = fit_marginal(fit_items, holdout_from=HOLDOUT_FROM)
    P_mkv = fit_markov(fit_items, holdout_from=HOLDOUT_FROM)
    print(f"[env] calibrated on {len(fit_items)} training customers")

    chunks = [pick[i:i + chunk] for i in range(0, len(pick), chunk)]
    batches = [collate_chunk(cell, ch) for ch in chunks]
    owns = [first_row_of_campaign(b["campaign"], HOLDOUT_FROM) for b in batches]

    # ------------------------------------------------- real data and the band
    F_real = np.vstack([score(real_sequences(b, start=int(st.min())), st)
                        for b, st in zip(batches, owns)])
    scaler = sm.Scaler.fit(F_real)
    Zr = scaler(F_real)
    null = sm.null_band(Zr, n_boot=n_boot, seed=NULL_SEED)
    q95 = float(np.quantile(null, 0.95))
    real_counts = sum(pooled_counts(b["label"].cpu().numpy(), b["label"].cpu().numpy() != 0)
                      for b in batches)
    reach = [defect3_reach(b["label"].cpu().numpy() != 0, int(st.min()))
             for b, st in zip(batches, owns)]
    same_as_prefix = (not a.smoke) and abs(q95 - PREFIX_Q95) < 1e-9
    print(f"[real] {F_real.shape[0]} sequences, {F_real.shape[1]} functionals, "
          f"{int(real_counts.sum()):,} holdout rows")
    print(f"[band] split-half null, {n_boot} splits: mean {null.mean():+.4f}  "
          f"sd {null.std(ddof=1):.4f}  q95 {q95:.7f}")
    if not a.smoke:
        print(f"[band] pre-fix q95 {PREFIX_Q95:.7f}: "
              + ("REPRODUCED exactly, same customers and functionals as R36c/e"
                 if same_as_prefix else
                 f"NOT REPRODUCED (got {q95:.10f}) -- the comparison with pre-fix "
                 "numbers is VOID"))
    print(f"[defect 3] on this chunking the pre-fix loop dropped "
          f"{sum(x[0] for x in reach)} of {len(pick)} customers' rows "
          f"({sum(x[1] for x in reach):,} rows)")

    out = {"cell": CELL, "head": head, "smoke": a.smoke, "n_users": int(len(pick)),
           "n_rep": n_rep, "chunk": chunk, "max_events": MAX_EVENTS,
           "customer_sha256": uid_sha, "functionals": sm.functional_names(),
           "null_band": {"n_boot": n_boot, "seed": NULL_SEED, "q95": q95,
                         "mean": float(null.mean()), "sd": float(null.std(ddof=1)),
                         "prefix_q95": PREFIX_Q95, "reproduces_prefix": same_as_prefix},
           "real": {"shares": (real_counts / real_counts.sum()).tolist(),
                    "holdout_rows": int(real_counts.sum())},
           "defect3_prefix_reach": {"customers": sum(x[0] for x in reach),
                                    "rows": sum(x[1] for x in reach)},
           "checkpoints": {f"{t}_s{s}": str(p) for (t, s), p in paths.items()},
           "arms": {}, "problems": []}
    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The raw (unstandardised) per-sequence functionals of every arm run, so the
    # 27 per-statistic tables can be computed afterwards on CPU. In an arm's
    # matrix, row j is a replicate of real row j // n_rep.
    raw = {"real": F_real}
    npz = dest.with_name(dest.stem + "_functionals.npz")

    def save() -> None:
        dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        np.savez_compressed(npz, functionals=np.array(sm.functional_names()),
                            n_rep=n_rep, customer_sha256=uid_sha, **raw)

    def run_arm(name: str, label: int, gen, teacher_forced: bool = False) -> None:
        t0 = time.time()
        F, cnt = [], np.zeros(9, dtype=np.int64)
        integ = {"rows_alive": 0, "rows_mismatched": 0, "real_rows_altered": 0}
        for b, st in zip(batches, owns):
            r = gen(b, int(st.min()))
            F.append(score(r, st))
            cnt += pooled_counts(r["decisions"], r["alive"])
            for k, v in integrity(r, b, st, n_rep, teacher_forced).items():
                integ[k] += v
        raw[f"{name}__{label}"] = np.vstack(F)
        rec = evaluate(Zr, scaler(raw[f"{name}__{label}"]), n_boot)
        rec.update(shares=(cnt / max(cnt.sum(), 1)).tolist(), integrity=integ,
                   seconds=round(time.time() - t0))
        if rec["null_q95"] != q95:
            out["problems"].append(f"{name} {label}: band {rec['null_q95']} is not the shared {q95}")
        if integ["rows_mismatched"] or integ["real_rows_altered"]:
            out["problems"].append(f"{name} {label}: integrity {integ}")
        if integ["rows_alive"] != n_rep * int(real_counts.sum()):
            out["problems"].append(f"{name} {label}: {integ['rows_alive']} live rows, "
                                   f"expected {n_rep * int(real_counts.sum())}")
        out["arms"].setdefault(name, {"runs": {}})["runs"][str(label)] = rec
        lo, hi = rec["auc_ci"]
        print(f"  {name:<14} {label:>4}  energy {rec['energy']:+.3f}  z {rec['z']:>6.1f}  "
              f"{'INSIDE ' if rec['inside_band'] else 'outside'}  AUC {rec['auc']:.3f} "
              f"[{lo:.3f}, {hi:.3f}]  NotBuy {rec['shares'][8]:.3f}  "
              f"top {', '.join(rec['top_functionals'][:3])}  "
              f"mismatch {integ['rows_mismatched']} altered {integ['real_rows_altered']}  "
              f"({rec['seconds']}s)", flush=True)
        save()

    # --------------------------------------------- arms without a model (CPU)
    print("\n[arms without a model]")
    for d in draws_c:
        run_arm("armC", d, lambda b, s, d=d: arm_c(b, env, vocab, start=s,
                                                   n_rep=n_rep, seed=d))
    for nm, samp in (("iid", iid_sampler(p_marg)), ("markov1", markov_sampler(P_mkv))):
        for d in draws_b:
            run_arm(nm, d, lambda b, s, d=d, samp=samp: simulate(
                b, env, vocab, samp, start=s, n_rep=n_rep, seed=d))

    # ------------------------------------------------------ model arms (GPU)
    print("\n[model arms]")
    for name, tag, tf in MODEL_ARMS:
        for sd in train_seeds:
            # b33/b35 were trained after the 29 Sep collate fix: their campaign
            # effects are estimates, so legacy_camp_bias stays False.
            model, _ = load_checkpoint(paths[(tag, sd)], device=a.device,
                                       legacy_camp_bias=False)
            run_arm(name, sd, lambda b, s, sd=sd, tf=tf, model=model: rollout(
                model, b, env, vocab, start=s, n_rep=n_rep, seed=1000 + sd,
                device=a.device, teacher_forced=tf), teacher_forced=tf)
            del model
            if a.device == "cuda":
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------- report
    for v in out["arms"].values():
        v["summary"] = summarise(v["runs"])
    out["verdicts"] = verdicts(out)
    save()

    lab = ["Buy1Reg", "Buy10Reg", "Buy1FigA", "Buy10FigA", "Buy1FigB", "Buy10FigB",
           "Buy1Wep", "Buy10Wep", "NotBuy"]
    print(f"\n=== summary: {len(pick)} customers x {n_rep} replicates, "
          f"band q95 {q95:.4f} ===")
    print(f"{'arm':<14}{'n':>3}{'energy':>9}{'sd':>8}{'min':>8}{'max':>8}{'z':>8}"
          f"{'in':>4}{'AUC':>7}{'sd':>7}  top functionals")
    print(f"{'real (band)':<14}{'':>3}{null.mean():>+9.3f}{null.std(ddof=1):>8.3f}"
          f"{'':>8}{q95:>8.3f}")
    for nm, v in out["arms"].items():
        s = v["summary"]
        print(f"{nm:<14}{s['n']:>3}{s['energy_mean']:>+9.3f}{s['energy_sd']:>8.3f}"
              f"{s['energy_min']:>+8.3f}{s['energy_max']:>+8.3f}{s['z_mean']:>8.1f}"
              f"{s['inside_band']:>4}{s['auc_mean']:>7.3f}{s['auc_sd']:>7.3f}  "
              f"{', '.join(s['top_functionals'][:3])}")
    print("\n=== decision shares over the holdout rows (pooled) ===")
    print(f"{'':>14}" + "".join(f"{x:>10}" for x in lab))
    print(f"{'real':>14}" + "".join(f"{x:>10.3f}" for x in out["real"]["shares"]))
    for nm, v in out["arms"].items():
        print(f"{nm:>14}" + "".join(f"{x:>10.3f}" for x in v["summary"]["shares_mean"]))

    print("\n=== pre-registered verdicts (EXPERIMENTS.md R36i) ===")
    for k, val in out["verdicts"].items():
        print(f"  {k}: {json.dumps(val, default=float)}")
    if a.smoke:
        print("\n(smoke run: the verdicts exercise the code only; the numbers mean nothing)")
    if out["problems"]:
        print("\nPROBLEMS:")
        for p in out["problems"]:
            print("  " + p)
    print(f"\nwrote {dest} and {npz} ({len(raw) - 1} arm runs + real)")
    if out["problems"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
