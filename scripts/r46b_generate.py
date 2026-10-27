"""R46 part B: the classical ladder as generators, on the R36i yardstick.

    python scripts/r46b_generate.py --reference              # band, i.i.d., Markov-1 (gate B3)
    python scripts/r46b_generate.py --rung R6 --mode fr      # one rung, one mode, draws 17-19
    python scripts/r46b_generate.py --rung R3 --mode tf --smoke

Pre-registered in EXPERIMENTS.md, R46 part B (27 Oct 2026). The cell, the 300
customers, the chunking, the environment, the scaler, the band and the C2ST
are R36i's (scripts/r36i_rerun.py), rebuilt the same way in this process, so
every number here sits on R36i's yardstick. R36i's own runs are the
comparators: scripts/r46b_compare.py reads them from its files, nothing of
R36i is re-run except i.i.d. and Markov-1, as gate B3.

Gates, each logged before any rollout is scored:
  B1  each customer's level-7 grid aligns with its level-6 window
  B2  the online state is T1's: every block, and the probabilities, on the
      real streams; then, in the rollouts, the probabilities each sequence
      draws its first holdout decision from (every holdout decision when
      teacher-forced)
  B3  the real functionals equal R36i's saved ones; the band is 0.0781258;
      (reference) i.i.d. and Markov-1 reproduce R36i's energies
  B4  per run: decisions on exactly the live rows, real acquisitions untouched
CPU only. Nothing here reads a uid except to hash it; nothing prints one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared", REPO / "scripts"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
import seq_metrics as sm  # noqa: E402
from baselines import (fit_marginal, fit_markov, iid_sampler,  # noqa: E402
                       markov_sampler, real_sequences, simulate)
from gacha_env import AcquisitionEnv, ProductTable  # noqa: E402
from r36c_generate import ckpt_path, collate_chunk, score  # noqa: E402
from r36i_rerun import (B33, CELL, FIX_COMMIT, HOLDOUT_FROM, MAX_EVENTS,  # noqa: E402
                        NULL_SEED, PREFIX_Q95, RUNS, evaluate, integrity, pooled_counts)
from rollout import first_row_of_campaign  # noqa: E402

from parametric import data as D, model as M, policy as PL, state as ST  # noqa: E402

R46 = Path("/storage/home/jinmiao/ProductGPT/runs/r46")
R36I = Path("/storage/home/jinmiao/ProductGPT/work_r36i/results/r36i")
# the part-A estimates; R6 is the fit part A selected (K = 6, start 1)
TAGS = {"M0": "M0", "M1": "M1", "R0": "R0", "R1": "R1", "R1b": "R1b", "R2": "R2",
        "R3": "R3", "R4": "R4", "R6": "R6_K6_s1"}
DRAWS = (17, 18, 19)
TOL = 1e-5


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────── R36i's setup, rebuilt ───────────────────────────
def r36i_setup(runs: Path, smoke: bool) -> dict:
    """scripts/r36i_rerun.py main(), from the data build to the band."""
    if smoke:
        n_users, n_rep, chunk, n_boot = 12, 2, 6, 20
    else:
        n_users, n_rep, chunk, n_boot = 300, 8, 6, 120
    c = torch.load(str(ckpt_path(B33, 1, runs)), map_location="cpu", weights_only=False)["cfg"]
    cfg = dict(c)
    cfg["max_events"] = MAX_EVENTS
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    from train5_multistream import build_loaders
    tr_dl, _, tests, _ = build_loaders(cfg)
    cell = tests[CELL].dataset
    train_ds = tr_dl.dataset
    pick = np.sort(np.random.default_rng(0).permutation(len(cell))[:n_users])
    uid_sha = hashlib.sha256("\n".join(str(cell[i]["uid"]) for i in pick).encode()).hexdigest()[:16]
    log(f"[cell] {CELL}: {len(cell)} customers, scoring {len(pick)} (customer-list sha256 {uid_sha})")
    fit_items = [train_ds[i] for i in range(min(len(train_ds), 600))]
    vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))
    env = AcquisitionEnv(vocab).fit(fit_items, holdout_from=HOLDOUT_FROM)
    chunks = [pick[i:i + chunk] for i in range(0, len(pick), chunk)]
    batches = [collate_chunk(cell, ch) for ch in chunks]
    owns = [first_row_of_campaign(b["campaign"], HOLDOUT_FROM) for b in batches]
    F_real = np.vstack([score(real_sequences(b, start=int(st.min())), st)
                        for b, st in zip(batches, owns)])
    scaler = sm.Scaler.fit(F_real)
    Zr = scaler(F_real)
    null = sm.null_band(Zr, n_boot=n_boot, seed=NULL_SEED)
    q95 = float(np.quantile(null, 0.95))
    real_counts = sum(pooled_counts(b["label"].cpu().numpy(), b["label"].cpu().numpy() != 0)
                      for b in batches)
    log(f"[real] {F_real.shape[0]} sequences, {int(real_counts.sum()):,} holdout rows; "
        f"band q95 {q95:.7f}")
    return dict(cfg=cfg, cell=cell, pick=pick, chunks=chunks, batches=batches, owns=owns,
                env=env, vocab=vocab, fit_items=fit_items, F_real=F_real, scaler=scaler, Zr=Zr,
                q95=q95, null=null, real_counts=real_counts, n_rep=n_rep, n_boot=n_boot,
                uid_sha=uid_sha)


# ───────────────────────────────── gates ─────────────────────────────────
def b1_alignment(S: dict, tab: dict, map76: np.ndarray) -> tuple:
    """Per chunk: the customers' table rows and the offset of grid row 0."""
    pos = {int(h): i for i, h in enumerate(tab["uid_hash"])}
    rows_by_chunk, off_by_chunk = [], []
    bad = {"missing": 0, "in_sample": 0, "negative_offset": 0, "prev": 0, "campaign": 0,
           "lto": 0, "obtained": 0, "label": 0, "holdout_rows": 0, "row0_not_empty": 0}
    # Amendment B-1: TransformerDataset applies the R1 shift AFTER truncation, so
    # row 0 of a truncated window carries padding where a longer window holds
    # the real o_(start-1). For a customer whose 384-row window is truncated
    # (offset > 0), grid row 0's obtained block must be empty at level 7 and is
    # not compared; every other row must match exactly.
    info = {"truncated_windows": 0, "row0_level6_nonempty": 0}
    for ch, b in zip(S["chunks"], S["batches"]):
        rows, offs = [], []
        for j, i in enumerate(ch):
            item = S["cell"][int(i)]
            h = D.uid_hash(str(item["uid"]))
            if h not in pos:
                bad["missing"] += 1
                rows.append(0)
                offs.append(0)
                continue
            r = pos[h]
            bad["in_sample"] += int(tab["is_out"][r] != 1)
            L7 = int(item["label"].numel())
            off = int(tab["length"][r]) - L7
            bad["negative_offset"] += int(off < 0)
            rows.append(r)
            offs.append(max(off, 0))
            w = slice(max(off, 0), max(off, 0) + L7)
            lab7 = b["label"][j, :L7].numpy().astype(np.int64)
            bad["prev"] += int((b["prev_decision"][j, :L7].numpy() != tab["prev"][r, w]).sum())
            bad["campaign"] += int((b["campaign"][j, :L7].numpy() != tab["campaign"][r, w]).sum())
            bad["lto"] += int((map76[b["lto"][j, :L7].numpy()] != tab["lto"][r, w]).sum())
            o7 = map76[b["obtained"][j, :L7].numpy()]
            o6 = tab["obtained"][r, w]
            if off > 0:
                info["truncated_windows"] += 1
                bad["row0_not_empty"] += int((o7[0] != 0).any())
                info["row0_level6_nonempty"] += int((o6[0] != 0).any())
                o7, o6 = o7[1:], o6[1:]
            bad["obtained"] += int((o7 != o6).sum())
            live = lab7 != 0
            bad["label"] += int((lab7[live] != tab["label"][r, w][live]).sum())
            bad["holdout_rows"] += int((live != (tab["holdout"][r, w] == 1)).sum())
        rows_by_chunk.append(np.array(rows))
        off_by_chunk.append(np.array(offs))
    return rows_by_chunk, off_by_chunk, bad, info


def b2_replay(spec, theta, ctx, tab, rows, feats: Path, occ: Path) -> dict:
    """The online state on the real streams against T1's features and probabilities."""
    seq = PL.sequences(tab, rows)
    prob, blocks = PL.replay(spec, theta, ctx, seq)
    L = seq["label"].shape[1]
    alive = np.arange(L)[None, :] < seq["length"][:, None]
    out = {"blocks": {}}
    with np.load(feats) as z:
        for k in PL.BLOCKS:
            ref = z[k][rows, :L]
            out["blocks"][k] = float(np.abs(blocks[k][alive].astype(np.float64)
                                            - ref[alive].astype(np.float64)).max())
    with np.load(occ) as z:
        uh, pos, pr = z["uid_hash"], z["pos"].astype(np.int64), z["prob"].astype(np.float64)
    where = {int(h): i for i, h in enumerate(tab["uid_hash"][rows])}
    sel = np.array([int(h) in where for h in uh])
    bi = np.array([where[int(h)] for h in uh[sel]], dtype=np.int64)
    out["prob_rows"] = int(sel.sum())
    out["prob"] = float(np.abs(prob[bi, pos[sel]] - pr[sel]).max()) if sel.any() else float("nan")
    out["t1_prob"] = (uh[sel], pos[sel], pr[sel])
    return out


# ───────────────────────────────── a run ─────────────────────────────────
def run_arm(S: dict, gen, *, teacher_forced: bool) -> tuple:
    F, cnt, probs = [], np.zeros(9, dtype=np.int64), []
    integ = {"rows_alive": 0, "rows_mismatched": 0, "real_rows_altered": 0}
    for ci, (b, st) in enumerate(zip(S["batches"], S["owns"])):
        r = gen(ci, b, int(st.min()))
        F.append(score(r, st))
        cnt += pooled_counts(r["decisions"], r["alive"])
        for k, v in integrity(r, b, st, S["n_rep"], teacher_forced).items():
            integ[k] += v
        probs.append(r.get("prob"))
    return np.vstack(F), cnt, integ, probs


def record(S: dict, F: np.ndarray, cnt, integ, t0: float, problems: list, label: str) -> dict:
    rec = evaluate(S["Zr"], S["scaler"](F), S["n_boot"])
    rec.update(shares=(cnt / max(cnt.sum(), 1)).tolist(), integrity=integ,
               seconds=round(time.time() - t0))
    expected = S["n_rep"] * int(S["real_counts"].sum())
    if rec["null_q95"] != S["q95"]:
        problems.append(f"{label}: band {rec['null_q95']} is not the shared {S['q95']}")
    if integ["rows_mismatched"] or integ["real_rows_altered"]:
        problems.append(f"{label}: integrity {integ}")
    if integ["rows_alive"] != expected:
        problems.append(f"{label}: {integ['rows_alive']} live rows, expected {expected}")
    lo, hi = rec["auc_ci"]
    log(f"  {label:<12} energy {rec['energy']:+.3f}  z {rec['z']:>6.1f}  "
        f"{'INSIDE ' if rec['inside_band'] else 'outside'}  AUC {rec['auc']:.3f} [{lo:.3f}, {hi:.3f}]  "
        f"NotBuy {rec['shares'][8]:.3f}  top {', '.join(rec['top_functionals'][:3])}  "
        f"mismatch {integ['rows_mismatched']} altered {integ['real_rows_altered']}  ({rec['seconds']}s)")
    return rec


def first_row_check(S: dict, probs: list, t1: tuple, rows_by_chunk, off_by_chunk, tab,
                    teacher_forced: bool) -> dict:
    """B2 in the rollout: the probabilities at each sequence's first holdout row
    (every holdout row when teacher-forced) against T1's."""
    uh, pos, pr = t1
    key = {(int(h), int(p)): v for h, p, v in zip(uh, pos, pr)}
    worst, n, missing = 0.0, 0, 0
    nr = S["n_rep"]
    for ci, (b, st) in enumerate(zip(S["batches"], S["owns"])):
        P = probs[ci]
        alive = (b["label"] != 0).numpy()
        for j in range(len(S["chunks"][ci])):
            h = int(tab["uid_hash"][rows_by_chunk[ci][j]])
            ts = np.flatnonzero(alive[j])
            if not teacher_forced:
                ts = ts[:1]
            for t in ts:
                k = (h, int(off_by_chunk[ci][j] + t))
                if k not in key:
                    missing += 1
                    continue
                for rr in range(nr):
                    worst = max(worst, float(np.abs(P[j * nr + rr, t] - key[k]).max()))
                    n += 1
    return {"max_abs": worst, "rows_compared": n, "rows_missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=sorted(TAGS))
    ap.add_argument("--mode", choices=("fr", "tf"))
    ap.add_argument("--reference", action="store_true",
                    help="the band, i.i.d. and Markov-1 at draws 17-19: gate B3")
    ap.add_argument("--smoke", action="store_true",
                    help="12 customers x 2 replicates, one draw; a wiring check, its numbers mean nothing")
    ap.add_argument("--runs-dir", default=str(RUNS))
    ap.add_argument("--r46", default=str(R46))
    ap.add_argument("--r36i", default=str(R36I))
    ap.add_argument("--out-dir", default=str(R46 / "partB"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NCPUS", "4")))
    a = ap.parse_args()
    if not a.reference and not (a.rung and a.mode):
        raise SystemExit("give --reference, or --rung and --mode")
    torch.set_num_threads(a.threads)
    t_start = time.time()
    r46, r36i = Path(a.r46), Path(a.r36i)
    draws = DRAWS[:1] if a.smoke else DRAWS
    arm = "ref" if a.reference else f"{a.mode}_{a.rung}"
    out_dir = Path(a.out_dir) / ("smoke" if a.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    if git("merge-base", "--is-ancestor", FIX_COMMIT, "HEAD").returncode != 0:
        raise SystemExit(f"HEAD {head} does not contain the rollout fix {FIX_COMMIT}")
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "parametric", "simulation",
                "scripts", "gen5_multistream", "shared").stdout.strip()
    log(f"R46 part B, arm {arm}{' (SMOKE)' if a.smoke else ''}: HEAD {head}, contains {FIX_COMMIT}; "
        f"{'UNCOMMITTED CHANGES:' + chr(10) + dirty if dirty else 'clean'}")
    if dirty and not a.smoke:
        raise SystemExit("uncommitted changes to the code this job runs")

    S = r36i_setup(Path(a.runs_dir), a.smoke)
    problems, gates = [], {}
    q_ok = abs(S["q95"] - PREFIX_Q95) < 1e-9
    gates["B3_band"] = {"q95": S["q95"], "prefix_q95": PREFIX_Q95, "reproduced": q_ok}
    if not a.smoke:
        with np.load(r36i / "r36i_functionals.npz") as z:
            # NaN marks a functional undefined for a customer (1,725 of 8,100 cells)
            same_real = bool(np.array_equal(z["real"], S["F_real"], equal_nan=True))
        gates["B3_real_functionals_equal_r36i"] = same_real
        log(f"[B3] band q95 {S['q95']:.7f} vs R36i {PREFIX_Q95:.7f}: "
            f"{'REPRODUCED' if q_ok else 'NOT REPRODUCED'}; real functionals "
            f"{'identical to' if same_real else 'DIFFER from'} R36i's")
        if not (q_ok and same_real):
            problems.append("B3: the yardstick does not reproduce R36i's")
    out = {"arm": arm, "head": head, "smoke": a.smoke, "n_users": int(len(S["pick"])),
           "n_rep": S["n_rep"], "customer_sha256": S["uid_sha"], "q95": S["q95"],
           "functionals": sm.functional_names(), "gates": gates, "runs": {}, "problems": problems}
    raw = {"real": S["F_real"]}

    def save() -> None:
        (out_dir / f"{arm}.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
        np.savez_compressed(out_dir / f"{arm}_functionals.npz", functionals=np.array(sm.functional_names()),
                            n_rep=S["n_rep"], **raw)

    if a.reference:
        p_marg = fit_marginal(S["fit_items"], holdout_from=HOLDOUT_FROM)
        P_mkv = fit_markov(S["fit_items"], holdout_from=HOLDOUT_FROM)
        ref = json.loads((r36i / "r36i.json").read_text()) if not a.smoke else None
        out["runs"] = {"iid": {}, "markov1": {}}
        for nm, samp in (("iid", iid_sampler(p_marg)), ("markov1", markov_sampler(P_mkv))):
            for d in draws:
                t0 = time.time()
                F, cnt, integ, _ = run_arm(S, lambda ci, b, s, d=d, samp=samp: simulate(
                    b, S["env"], S["vocab"], samp, start=s, n_rep=S["n_rep"], seed=d),
                    teacher_forced=False)
                raw[f"{nm}__{d}"] = F
                rec = record(S, F, cnt, integ, t0, problems, f"{nm} {d}")
                if ref is not None:
                    e0 = ref["arms"][nm]["runs"][str(d)]["energy"]
                    rec["r36i_energy"] = e0
                    if abs(rec["energy"] - e0) > 1e-6:
                        problems.append(f"B3: {nm} {d} energy {rec['energy']:.9f} vs R36i {e0:.9f}")
                out["runs"][nm][str(d)] = rec
                save()
        gates["B3_reference_reproduces"] = not any(p.startswith("B3: iid") or p.startswith("B3: markov")
                                                   for p in problems)
    else:
        tag = TAGS[a.rung]
        res = json.loads((r46 / tag / "result.json").read_text())
        spec = M.Spec(res["rung"], K=res.get("K", 1))
        theta = torch.tensor(res["theta"], dtype=torch.float64)
        out.update(tag=tag, rung=a.rung, mode=a.mode, K=spec.K, n_params=spec.size)
        log(f"[policy] {tag}: {spec.size} parameters, K = {spec.K}, "
            f"{'teacher-forced' if a.mode == 'tf' else 'free-running'}")
        tab = D.load(r46 / "occasions_v6_S1024.npz")
        prod = ST.product_table()
        ctx = PL.Context(tab, prod)
        map76 = PL.level7_to_6(config5.feature_file_path(S["cfg"]))

        rows_by_chunk, off_by_chunk, bad, info = b1_alignment(S, tab, map76)
        gates["B1_alignment"] = dict(bad, **info)
        ok1 = not any(bad.values())
        log(f"[B1] {sum(len(c) for c in S['chunks'])} customers; mismatches {bad}; "
            f"{info['truncated_windows']} truncated 384-row windows, whose row 0 is empty at "
            f"level 7 (level 6 holds an outcome there for {info['row0_level6_nonempty']}): "
            f"{'PASS' if ok1 else 'FAIL'}")
        if not ok1:
            problems.append(f"B1: {bad}")

        rows_all = np.concatenate(rows_by_chunk)
        t0 = time.time()
        b2 = b2_replay(spec, theta, ctx, tab, rows_all, r46 / "features_v6_S1024.npz",
                       r46 / tag / "occ_outsample_users_holdout_period.npz")
        t1 = b2.pop("t1_prob")
        worst = max(b2["blocks"].values())
        ok2 = worst <= TOL and b2["prob"] <= TOL and b2["prob_rows"] == int(S["real_counts"].sum())
        gates["B2_replay"] = b2
        log(f"[B2] replay on {len(rows_all)} customers ({time.time() - t0:.0f}s): worst block "
            f"{max(b2['blocks'], key=b2['blocks'].get)} {worst:.1e}; probabilities {b2['prob']:.1e} "
            f"on {b2['prob_rows']:,} T1 rows: {'PASS' if ok2 else 'FAIL'}")
        if not ok2:
            problems.append(f"B2 replay: blocks {worst:.2e}, probs {b2['prob']:.2e} on {b2['prob_rows']} rows")

        tf = a.mode == "tf"
        for d in draws:
            t0 = time.time()

            def gen(ci, b, s, d=d):
                rec_p: dict = {}
                r = PL.policy_rollout(spec, theta, ctx, PL.sequences(tab, rows_by_chunk[ci]),
                                      off_by_chunk[ci], b, S["env"], S["vocab"], map76,
                                      start=s, n_rep=S["n_rep"], seed=d, teacher_forced=tf,
                                      record=rec_p)
                r["prob"] = rec_p["prob"]
                return r

            F, cnt, integ, probs = run_arm(S, gen, teacher_forced=tf)
            raw[f"{arm}__{d}"] = F
            rec = record(S, F, cnt, integ, t0, problems, f"{arm} {d}")
            chk = first_row_check(S, probs, t1, rows_by_chunk, off_by_chunk, tab, tf)
            rec["B2_rollout"] = chk
            ok = chk["max_abs"] <= TOL and chk["rows_missing"] == 0
            log(f"[B2] draw {d}: {'every holdout row' if tf else 'first holdout rows'} against T1, "
                f"{chk['rows_compared']:,} compared, max {chk['max_abs']:.1e}, missing "
                f"{chk['rows_missing']}: {'PASS' if ok else 'FAIL'}")
            if not ok:
                problems.append(f"B2 rollout draw {d}: {chk}")
            out["runs"][str(d)] = rec
            save()

    out["seconds"] = round(time.time() - t_start)
    save()
    log(f"wrote {out_dir / (arm + '.json')} and its functionals ({len(raw) - 1} runs + real) "
        f"in {out['seconds']}s")
    if a.smoke:
        log("(smoke run: the numbers exercise the code only)")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
