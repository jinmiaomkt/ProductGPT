"""R46 part C: identification. One replicate per call.

    python scripts/r46c_identify.py --job pboot --dgp R6 --rep 3     # C1
    python scripts/r46c_identify.py --job cboot --rep 7              # C2
    python scripts/r46c_identify.py --job keane --null b --fit R4 --rep 0   # C3
    ... --smoke                                                      # 150 customers, 2 rounds

Pre-registered in EXPERIMENTS.md, R46 part C (27 Oct 2026).
  C1  pboot: simulate R6 or R4 at its part-A estimate on the real design,
      refit the same rung from the truth, sandwich SEs.
  C2  cboot: resample the in-sample customers with replacement, refit R6 from
      its estimate on the real rows.
  C3  keane: simulate the heterogeneity-only null H6 (a: as fitted; b: its
      continuous version) on the real design, fit R4 (part A's default start)
      or R6 (H6 embedded).
The H6 fits themselves are `python -m parametric.fit --rung H6 ...`
(scripts/r46c_cpu.pbs, STAGE=H6).

Gates, each logged and stored: C-G1 design, C-G2 simulation, C-G3 optimum,
C-G4 bootstrap weights, C-G5 pairing (the read-out compares checksums).
Training rows only: no holdout row is read. CPU only. Counts, shares and
checksums only; no uid is printed.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parametric import data as D, fit as FT, identify as IC, model as M, policy as PL, state as ST  # noqa: E402

F64 = torch.float64
R46 = Path(os.environ.get("R46", "/storage/home/jinmiao/ProductGPT/runs/r46"))
# the part-A estimates: R6 from stage 3 (same theta as the T1 fit, with SEs)
TRUTH = {"R6": ("se/R6_K6_s1/result.json", 6), "R4": ("R4/result.json", 1)}
SEEDS = {"pboot": {"R6": 46000, "R4": 46100}, "cboot": 46200, "keane": {"a": 46300, "b": 46400}}
CONT_SEED, K_BIG, K_H6 = 46499, 200, 6
N_RECORD = 100


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def commit() -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def load_result(rel: str) -> dict:
    return json.loads((R46 / rel).read_text())


def best_h6(out_root: Path) -> dict:
    """H6's estimate: of the five starts, the best training NLL (R6's rule)."""
    paths = sorted(glob.glob(str(out_root / "h6" / f"H6_K{K_H6}_s*" / "result.json")))
    if len(paths) != 5:
        raise SystemExit(f"expected 5 H6 starts, found {len(paths)}")
    res = [json.loads(Path(p).read_text()) for p in paths]
    best = min(res, key=lambda r: r["train_nll"])
    log(f"H6: best of {len(res)} starts by training NLL is {best['tag']} ({best['train_nll']:.6f}); "
        f"spread {max(r['train_nll'] for r in res) - best['train_nll']:.6f}")
    return best


# ───────────────────────────── simulate on the real design ─────────────────────────────
def simulated_world(tab, roles, prod, ctx, spec, theta, seed, max_users):
    users, T, seq = IC.design(tab, roles, max_users=max_users)
    log(f"design: {len(users)} customers, {int(T.sum()):,} rows through the last training row")
    env = IC.level6_env(tab, prod, users)
    rec = {"users": np.unique(np.linspace(0, len(users) - 1, min(N_RECORD, len(users))).astype(np.int64))}
    t0 = time.time()
    sim = IC.simulate(spec, theta, ctx, seq, T, env, seed=seed, record=rec)
    log(f"simulated {spec.rung} (K={spec.K}) seed {seed} in {time.time() - t0:.0f}s")
    tab_s = IC.synthetic_table(tab, users, T, seq, sim)
    roles_s = D.roles(tab_s)
    g1 = IC.check_design(tab, roles, users, T, tab_s, roles_s)
    log(f"[C-G1] design: {g1['customers']} customers, {g1['train_rows']:,} training rows; "
        f"equal {g1['equal']}  {'PASS' if g1['ok'] else 'FAIL'}")
    t0 = time.time()
    F_s = ST.build(tab_s, prod, calendar=ctx.calendar)
    log(f"state.build on the simulated table in {time.time() - t0:.0f}s")
    g2 = IC.check_simulation(spec, theta, F_s, tab_s, rec, sim["cls"])
    log(f"[C-G2] simulation: {g2['users']} customers, block max {g2['block_max']:.1e}, "
        f"prob max {g2['prob_max']:.1e}{' BAD ' + str(g2['bad_blocks']) if g2['bad_blocks'] else ''}  "
        f"{'PASS' if g2['ok'] else 'FAIL'}")
    S = tab_s["label"].shape[1]
    real_world = IC.world_summary({k: tab[k][users, :S] for k in ("label", "prev", "obtained", "lto")},
                                  roles["train"][users, :S], prod)
    sim_world = IC.world_summary(tab_s, roles_s["train"], prod)
    log(f"decision shares on training rows: real {real_world['decision_shares']}")
    log(f"                                  sim  {sim_world['decision_shares']}")
    cs = IC.checksum(tab_s["label"], tab_s["obtained"])
    meta = dict(design=g1, simulation=g2, world_real=real_world, world_sim=sim_world,
                sim_checksum=cs, classes_drawn=np.bincount(sim["cls"], minlength=spec.K).tolist()
                if spec.K <= 10 else int(len(np.unique(sim["cls"]))))
    if not g1["ok"] or not g2["ok"]:
        raise SystemExit("a part C gate failed: nothing is fitted")
    return tab_s, F_s, roles_s, meta


def refit(spec, theta0, tab_s, F_s, roles_s, *, ridge, max_rounds, se, truth=None, weights=None,
          users=None):
    users = np.arange(tab_s["label"].shape[0]) if users is None else users
    chunks = FT.make_chunks(tab_s, F_s, roles_s, users, weights=weights)
    rows = roles_s["train"].sum(1)
    n = float((rows[users] * (1 if weights is None else weights[users])).sum())
    l2 = ridge / n
    out = {"n_rows": n}
    if truth is not None:
        out["obj_truth"] = FT.objective(spec, truth, chunks, n, l2, backward=False)
    out["obj_start"] = FT.objective(spec, theta0, chunks, n, l2, backward=False)
    t0 = time.time()
    theta, hist = FT.fit(spec, theta0, chunks, n, l2=l2, max_rounds=max_rounds)
    out["fit_seconds"] = time.time() - t0
    out["obj_hat"] = FT.objective(spec, theta, chunks, n, l2, backward=False)
    out["train_nll"] = FT.objective(spec, theta, chunks, n, 0.0, backward=False)
    out["fit_history"] = hist
    log(f"fitted {spec.rung} K={spec.K} in {out['fit_seconds']:.0f}s: penalised objective start "
        f"{out['obj_start']:.8f} -> {out['obj_hat']:.8f}"
        + (f" (truth {out['obj_truth']:.8f})" if truth is not None else ""))
    if truth is not None:
        out["G3_ok"] = bool(out["obj_hat"] <= out["obj_truth"] + 1e-9)
        log(f"[C-G3] optimum: estimate {'no worse' if out['G3_ok'] else 'WORSE'} than the truth "
            f"(diff {out['obj_hat'] - out['obj_truth']:+.2e})  {'PASS' if out['G3_ok'] else 'FAIL'}")
    V = None
    if se:
        t0 = time.time()
        V, diag = FT.sandwich(spec, theta, chunks)
        out["se"] = torch.sqrt(torch.clamp(torch.diagonal(V), min=0)).tolist()
        out["se_diag"] = {k: v for k, v in diag.items()}
        log(f"sandwich in {time.time() - t0:.0f}s: negative definite {diag['negative_definite']}, "
            f"not identified {len(diag['not_identified'])}, ridge {len(diag['ridge'])}")
    out["quantities"] = IC.quantities_with_se(spec, theta, V)
    return theta, out


# ───────────────────────────── the three jobs ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=("pboot", "cboot", "keane"))
    ap.add_argument("--dgp", choices=("R6", "R4"))
    ap.add_argument("--null", choices=("a", "b"))
    ap.add_argument("--fit", choices=("R4", "R6"))
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="150 customers, 2 L-BFGS rounds")
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--out-dir", default=str(R46 / "partC"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NCPUS", "8")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t_start = time.time()
    out_root = Path(a.out_dir)
    max_users = 150 if a.smoke else None
    max_rounds = 2 if a.smoke else 15
    if a.job == "pboot":
        assert a.dgp, "--dgp"
        tag, seed = f"C1_{a.dgp}_r{a.rep:02d}", SEEDS["pboot"][a.dgp] + a.rep
    elif a.job == "cboot":
        tag, seed = f"C2_R6_r{a.rep:02d}", SEEDS["cboot"] + a.rep
    else:
        assert a.null and a.fit, "--null, --fit"
        tag, seed = f"C3{a.null}_{a.fit}_r{a.rep:02d}", SEEDS["keane"][a.null] + a.rep
    if a.smoke:
        tag = "smoke_" + tag
    out = out_root / a.job / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    log(f"R46 part C {tag}  seed {seed}  threads {a.threads}  commit {commit()}  torch {torch.__version__}")

    tab = D.load(R46 / "occasions_v6_S1024.npz")
    roles = D.roles(tab)
    prod = ST.product_table()
    res = dict(tag=tag, job=a.job, seed=seed, rep=a.rep, commit=commit(), smoke=a.smoke, ridge=a.ridge)

    if a.job == "pboot":
        rel, K = TRUTH[a.dgp]
        src = load_result(rel)
        spec = M.Spec(a.dgp, K=K)
        truth = torch.tensor(src["theta"], dtype=F64)
        ctx = PL.Context(tab, prod)
        tab_s, F_s, roles_s, meta = simulated_world(tab, roles, prod, ctx, spec, truth, seed, max_users)
        theta, fit = refit(spec, truth, tab_s, F_s, roles_s, ridge=a.ridge, max_rounds=max_rounds,
                           se=True, truth=truth)
        res.update(meta, fit=fit, rung=a.dgp, K=spec.K, labels=FT.param_labels(spec),
                   theta_true=truth.tolist(), theta_hat=theta.tolist(),
                   quantities_true={k: float(v) for k, v in IC.quantities(spec, truth).items()})

    elif a.job == "cboot":
        rel, K = TRUTH["R6"]
        src = load_result(rel)
        spec = M.Spec("R6", K=K)
        est = torch.tensor(src["theta"], dtype=F64)
        tab_f, F = FT.load_features(R46 / "occasions_v6_S1024.npz", R46 / "features_v6_S1024.npz")
        inu = np.flatnonzero((tab_f["is_out"] == 0) & roles["train"].any(1))
        if max_users:
            inu = inu[:max_users]
        w = IC.bootstrap_weights(len(inu), seed)
        weights = np.zeros(tab_f["label"].shape[0], dtype=np.int64)
        weights[inu] = w
        g4 = dict(customers=int(len(inu)), weight_sum=int(w.sum()), distinct=int((w > 0).sum()),
                  max_weight=int(w.max()), ok=int(w.sum()) == len(inu))
        log(f"[C-G4] weights: sum {g4['weight_sum']} over {g4['customers']} customers, "
            f"{g4['distinct']} distinct, max {g4['max_weight']}  {'PASS' if g4['ok'] else 'FAIL'}")
        if not g4["ok"]:
            raise SystemExit("C-G4 failed")
        theta, fit = refit(spec, est, tab_f, F, roles, ridge=a.ridge, max_rounds=max_rounds, se=False,
                           weights=weights, users=inu[w > 0])
        log(f"weighted training rows {fit['n_rows']:,.0f}")
        res.update(bootstrap=g4, fit=fit, rung="R6", K=K, labels=FT.param_labels(spec),
                   theta_est=est.tolist(), theta_hat=theta.tolist())

    else:
        h6 = best_h6(out_root)
        sh = M.Spec("H6", K=K_H6)
        th_h6 = torch.tensor(h6["theta"], dtype=F64)
        if a.null == "a":
            spec_dgp, th_dgp, cont = sh, th_h6, None
        else:
            spec_dgp, th_dgp, cont = IC.continuous_h6(th_h6, K_H6, K_big=K_BIG, seed=CONT_SEED)
            log(f"null (b): {K_BIG} classes; class-constant mean target {cont['target_mean']} "
                f"drawn {cont['draw_mean']}; covariance rank {cont['rank']}")
        ctx = PL.Context(tab, prod)
        tab_s, F_s, roles_s, meta = simulated_world(tab, roles, prod, ctx, spec_dgp, th_dgp, seed, max_users)
        r6 = load_result(TRUTH["R6"][0])
        s6 = M.Spec("R6", K=6)
        a_dec_real = float(s6.unpack(torch.tensor(r6["theta"], dtype=F64))["a_dec"][0])
        if a.fit == "R4":
            spec = M.Spec("R4")
            lab = tab_s["label"][roles_s["train"]].astype(np.int64)
            theta0 = spec.init(marginal=np.bincount(lab - 1, minlength=9) / len(lab))
            truth = None
        else:
            spec = M.Spec("R6", K=K_H6)
            theta0 = IC.embed_h6(th_h6, K_H6, a_dec=a_dec_real)
            truth = theta0 if a.null == "a" else None
        theta, fit = refit(spec, theta0, tab_s, F_s, roles_s, ridge=a.ridge, max_rounds=max_rounds,
                           se=False, truth=truth)
        res.update(meta, fit=fit, rung=a.fit, K=spec.K, null=a.null, h6_tag=h6["tag"],
                   h6_train_nll=h6["train_nll"], continuous=cont, labels=FT.param_labels(spec),
                   theta_start=theta0.tolist(), theta_hat=theta.tolist())

    res["seconds"] = time.time() - t_start
    out.write_text(json.dumps(res, indent=1))
    q = res["fit"]["quantities"]
    show = [k for k in IC.SD_QUANTITIES if k in q]
    log("quantities: " + ", ".join(f"{k} {q[k]['value']:+.3f}" for k in show))
    log(f"done {tag} in {res['seconds']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
