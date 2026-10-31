"""R46 part D: banner loyalty. One fit per call.

    python scripts/r46d_banner.py --job real --arm R6B --start 0         # D1 on the real rows
    python scripts/r46d_banner.py --job real --arm K12 --start 2         # D2 on the real rows
    python scripts/r46d_banner.py --job null1 --rep 3                    # table 3: simulate, save, fit R6
    python scripts/r46d_banner.py --job null2 --rep 3 --arm K9 --start 1
    ... --smoke                                                          # 150 customers, 2 rounds

Pre-registered in EXPERIMENTS.md, R46 part D (approved by Jin 31 Oct 2026).
  null1  null (b) exactly as part C (H6's best start, 200 classes from seed
         46499), simulated on the real design with seed 46500 + rep and saved;
         R6 (K = 6) fitted from part C's start (H6 embedded, the real a_dec).
  real, null2  every start is built from the table's K = 6 R6 estimate (part
         A's on the real rows, null1's on a null table):
         R6B     start 0: a_ban = a_dec; start 1: a_ban at a 90-occasion half-life
         K9, K12 split the 3 / 6 heaviest classes; start s moves the copies'
                 class constants by eps (0, 0.1, 0.5)[s] * z, seed 46900 + s
Gates: C-G1 and C-G2 (null1, part C's code), C-G5 (null2 reads null1's table:
checksums of the table and of the features rebuilt from it), and D-G2's
inputs: each fit stores the objective it is compared with (R6's at the table's
estimate for R6B, the nested eps = 0 start for K > 6). The read-out is
scripts/r46d_compare.py. Training rows only; the real fits score campaign 27
only, so no holdout row is read. CPU only. Counts and checksums only; no uid
is printed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parametric import data as D, fit as FT, identify as IC, model as M, policy as PL, state as ST  # noqa: E402
import r46c_identify as RC  # noqa: E402  part C's simulated world, H6 choice and part-A estimate

F64 = torch.float64
R46 = RC.R46
SEED_TABLE, SEED_SPLIT = 46500, 46900
EPS = (0.0, 0.1, 0.5)
HL_START_B = 90.0
ARMS = {"R6B": (6, 2), "K9": (9, 3), "K12": (12, 3)}       # arm -> (K, starts)
log, commit = RC.log, RC.commit


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def features_checksum(F: dict) -> str:
    return IC.checksum(*[F[k] for k in sorted(F)])


def start_for(arm: str, s: int, theta6: torch.Tensor):
    """(spec, start, (reference spec, reference theta), info) for one start."""
    spec6 = M.Spec("R6", K=6)
    K, n_starts = ARMS[arm]
    if not 0 <= s < n_starts:
        raise SystemExit(f"{arm} has starts 0-{n_starts - 1}")
    if arm == "R6B":
        spec = M.Spec("R6B", K=6)
        P6 = spec6.unpack(theta6)
        a = float(P6["a_dec"][0]) if s == 0 else logit(0.5 ** (1.0 / HL_START_B))
        theta0 = IC.pack(spec, {**P6, "a_ban": torch.tensor([a], dtype=F64)})
        info = dict(a_ban_start=a, hl_ban_start=math.log(0.5) / math.log(1.0 / (1.0 + math.exp(-a))))
        return spec, theta0, (spec6, theta6), info
    spec, theta0, info = IC.split_classes(spec6, theta6, K, eps=EPS[s], seed=SEED_SPLIT + s)
    _, nested, _ = IC.split_classes(spec6, theta6, K, eps=0.0, seed=SEED_SPLIT)
    return spec, theta0, (spec, nested), info


def fit_one(spec, theta0, ref, tab, F, roles, users, *, ridge, max_rounds, se):
    """Part C's refit, plus the reference objective D-G2 compares with."""
    chunks = FT.make_chunks(tab, F, roles, users)
    n = float(roles["train"][users].sum())
    l2 = ridge / n
    out = {"n_rows": n, "n_users": int(len(users))}
    out["obj_ref"] = FT.objective(ref[0], ref[1], chunks, n, l2, backward=False)
    out["obj_start"] = FT.objective(spec, theta0, chunks, n, l2, backward=False)
    t0 = time.time()
    theta, hist = FT.fit(spec, theta0, chunks, n, l2=l2, max_rounds=max_rounds)
    out["fit_seconds"] = time.time() - t0
    out["obj_hat"] = FT.objective(spec, theta, chunks, n, l2, backward=False)
    out["train_nll"] = FT.objective(spec, theta, chunks, n, 0.0, backward=False)
    out["fit_history"] = hist
    log(f"fitted {spec.rung} K={spec.K} in {out['fit_seconds']:.0f}s: penalised objective start "
        f"{out['obj_start']:.8f} -> {out['obj_hat']:.8f} (reference {out['obj_ref']:.8f}, "
        f"diff {out['obj_hat'] - out['obj_ref']:+.2e})")
    V = None
    if se:
        t0 = time.time()
        V, diag = FT.sandwich(spec, theta, chunks)
        out["se"] = torch.sqrt(torch.clamp(torch.diagonal(V), min=0)).tolist()
        out["se_diag"] = diag
        log(f"sandwich in {time.time() - t0:.0f}s: negative definite {diag['negative_definite']}, "
            f"not identified {len(diag['not_identified'])}, ridge {len(diag['ridge'])}")
    out["quantities"] = IC.quantities_with_se(spec, theta, V)
    out["derived"] = FT.transformed(spec, theta, V)
    return theta, out


def score_val(spec, theta, tab, F, roles, max_users=None) -> dict:
    """Campaign 27 (the `val` role) only, on every customer part A scored there."""
    eval_u = np.flatnonzero(roles["val"].any(1))
    if max_users:
        eval_u = eval_u[:max_users]
    ch = FT.make_chunks(tab, F, roles, eval_u)
    d = FT.score(spec, theta, ch, roles, tab, eval_roles=("val",))["val"]
    return FT.metrics(d["label"].astype(np.int64), d["prob"])


def show(q: dict) -> str:
    keys = [k for k in ("loy", "inc", "tenl", "repeat_med", "hl_dec", "hl_ban", "hl_out") if k in q]
    return ", ".join(f"{k} {q[k]['value']:+.3f}" for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=("real", "null1", "null2"))
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="150 customers, 2 L-BFGS rounds")
    ap.add_argument("--ridge", type=float, default=0.01)
    ap.add_argument("--out-dir", default=str(R46 / "partD"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("NCPUS", "8")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t_start = time.time()
    root = Path(a.out_dir)
    pre = "smoke_" if a.smoke else ""
    max_users = 150 if a.smoke else None
    max_rounds = 2 if a.smoke else 15
    if a.job != "null1" and not a.arm:
        raise SystemExit("--arm is required for real and null2")
    if a.job == "real":
        tag, out = f"{pre}D_real_{a.arm}_s{a.start}", root / "real"
    elif a.job == "null1":
        tag, out = f"{pre}D_null_base_r{a.rep:02d}", root / "null" / "stage1"
    else:
        tag, out = f"{pre}D_null_{a.arm}_s{a.start}_r{a.rep:02d}", root / "null" / "fits"
    out.mkdir(parents=True, exist_ok=True)
    table_path = root / "null" / "tables" / f"{pre}table_r{a.rep:02d}.npz"
    stage1_path = root / "null" / "stage1" / f"{pre}D_null_base_r{a.rep:02d}.json"
    log(f"R46 part D {tag}  threads {a.threads}  commit {commit()}  torch {torch.__version__}")

    res = dict(tag=tag, job=a.job, arm=a.arm, start=a.start, rep=a.rep, commit=commit(),
               smoke=a.smoke, ridge=a.ridge)
    prod = ST.product_table()

    if a.job == "real":
        tab, F = FT.load_features(R46 / "occasions_v6_S1024.npz", R46 / "features_v6_S1024.npz")
        roles = D.roles(tab)
        users = np.flatnonzero((tab["is_out"] == 0) & roles["train"].any(1))
        if max_users:
            users = users[:max_users]
        theta6 = torch.tensor(RC.load_result(RC.TRUTH["R6"][0])["theta"], dtype=F64)
        spec, theta0, ref, info = start_for(a.arm, a.start, theta6)
        log(f"{a.arm} start {a.start} from part A's R6 (K = 6): {info}")
        theta, fit = fit_one(spec, theta0, ref, tab, F, roles, users, ridge=a.ridge,
                             max_rounds=max_rounds, se=a.arm == "R6B")
        fit["val"] = score_val(spec, theta, tab, F, roles, max_users)
        log(f"campaign 27: n {fit['val']['n']:,}  NLL {fit['val']['nll']:.5f}")
        if a.arm == "R6B" and a.start == 0:
            fit["val_R6_K6"] = score_val(M.Spec("R6", K=6), theta6, tab, F, roles, max_users)
            log(f"campaign 27, part A's R6 (K = 6) on the same users: NLL {fit['val_R6_K6']['nll']:.5f}")
        res.update(fit=fit, start_info=info, rung=spec.rung, K=spec.K, labels=FT.param_labels(spec),
                   theta_start=theta0.tolist(), theta_hat=theta.tolist(),
                   quantities_ref={k: float(v) for k, v in IC.quantities(M.Spec("R6", K=6), theta6).items()})

    else:
        tab = D.load(R46 / "occasions_v6_S1024.npz")
        roles = D.roles(tab)
        ctx = PL.Context(tab, prod)
        if a.job == "null1":
            h6 = RC.best_h6(R46 / "partC")
            th_h6 = torch.tensor(h6["theta"], dtype=F64)
            spec_dgp, th_dgp, cont = IC.continuous_h6(th_h6, RC.K_H6, K_big=RC.K_BIG, seed=RC.CONT_SEED)
            log(f"null (b): {RC.K_BIG} classes (seed {RC.CONT_SEED}); class-constant mean target "
                f"{cont['target_mean']} drawn {cont['draw_mean']}; covariance rank {cont['rank']}")
            seed = SEED_TABLE + a.rep
            tab_s, F_s, roles_s, meta = RC.simulated_world(tab, roles, prod, ctx, spec_dgp, th_dgp,
                                                           seed, max_users)
            table_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(table_path, **tab_s)
            fcs = features_checksum(F_s)
            log(f"saved the table ({meta['sim_checksum']}; features {fcs}) -> {table_path}")
            r6 = RC.load_result(RC.TRUTH["R6"][0])
            a_dec_real = float(M.Spec("R6", K=6).unpack(torch.tensor(r6["theta"], dtype=F64))["a_dec"][0])
            spec = M.Spec("R6", K=RC.K_H6)
            theta0 = IC.embed_h6(th_h6, RC.K_H6, a_dec=a_dec_real)
            theta, fit = fit_one(spec, theta0, (spec, theta0), tab_s, F_s, roles_s,
                                 np.arange(tab_s["label"].shape[0]), ridge=a.ridge,
                                 max_rounds=max_rounds, se=False)
            res.update(meta, seed=seed, table=str(table_path), feat_checksum=fcs, fit=fit, rung="R6",
                       K=spec.K, h6_tag=h6["tag"], h6_train_nll=h6["train_nll"], continuous=cont,
                       labels=FT.param_labels(spec), theta_start=theta0.tolist(), theta_hat=theta.tolist())
        else:
            s1 = json.loads(stage1_path.read_text())
            tab_s = D.load(table_path)
            cs = IC.checksum(tab_s["label"], tab_s["obtained"])
            t0 = time.time()
            F_s = ST.build(tab_s, prod, calendar=ctx.calendar)
            fcs = features_checksum(F_s)
            g5 = dict(table=cs == s1["sim_checksum"], features=fcs == s1["feat_checksum"],
                      stage1=s1["tag"])
            g5["ok"] = g5["table"] and g5["features"]
            log(f"[C-G5] table {cs} vs stage 1 {s1['sim_checksum']}, features {fcs} vs {s1['feat_checksum']} "
                f"(rebuilt in {time.time() - t0:.0f}s)  {'PASS' if g5['ok'] else 'FAIL'}")
            if not g5["ok"]:
                raise SystemExit("C-G5 failed: this fit would not read stage 1's table")
            roles_s = D.roles(tab_s)
            theta6 = torch.tensor(s1["theta_hat"], dtype=F64)
            spec, theta0, ref, info = start_for(a.arm, a.start, theta6)
            log(f"{a.arm} start {a.start} from stage 1's R6 (K = 6): {info}")
            theta, fit = fit_one(spec, theta0, ref, tab_s, F_s, roles_s, np.arange(tab_s["label"].shape[0]),
                                 ridge=a.ridge, max_rounds=max_rounds, se=False)
            res.update(pairing=g5, seed=s1["seed"], fit=fit, start_info=info, rung=spec.rung, K=spec.K,
                       labels=FT.param_labels(spec), theta_start=theta0.tolist(), theta_hat=theta.tolist())

    res["seconds"] = time.time() - t_start
    (out / f"{tag}.json").write_text(json.dumps(res, indent=1))
    log("quantities: " + show(res["fit"]["quantities"]))
    log(f"done {tag} in {res['seconds']:.0f}s -> {out / (tag + '.json')}")


if __name__ == "__main__":
    main()
