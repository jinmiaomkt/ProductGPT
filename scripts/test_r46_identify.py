"""R46 part C unit tests: the simulator on a design, the Keane nulls, the weights.

CPU, synthetic panels from parametric/simulate.py, a few minutes:

    python scripts/test_r46_identify.py

  I1  the part-A rungs' parameter layouts are unchanged (H6 was added beside them)
  I2  H6 embedded in R6 gives H6's probabilities and class weights exactly
  I3  the Tracker's class-specific probabilities equal model.log_probs on
      state.build features, class by class (R6 and H6)
  I4  integer bootstrap weights equal duplicated customers: objective and gradient
  I5  simulate() on a design: gates C-G1 and C-G2 pass, decisions sit on the
      live rows only, and a second run with the same seed is identical (C-G5)
  I6  quantities(): the repeat premium and the class-weighted loyalties equal
      the numpy formulas part A's read-out used
  I7  continuous_h6(): 200 classes reproduce the class-weighted mean and
      spread of the six class constants within Monte Carlo error
  I8  a small parametric bootstrap for R4: the refit from the truth is no
      worse than the truth (C-G3)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from parametric import data as D, fit as FT, identify as IC, model as M, policy as PL  # noqa: E402
from parametric import simulate as SIM, state as ST  # noqa: E402

RESULTS = []
F64 = torch.float64
SIZES = {"M0": 8, "M1": 80, "R0": 30, "R1": 102, "R1b": 103, "R2": 107, "R3": 115, "R4": 137}
R4_NAMES = ["lag", "fig", "rr_a", "rr_b", "unoff_b", "wep", "phase", "bawa", "runs", "a_dec", "a_out",
            "loy", "inc", "tenl", "out", "cop_f", "cop_w", "elem", "owned", "pity0", "pity1", "pity2",
            "pity3", "guar", "wloss"]
R6_NAMES = R4_NAMES + ["d_asc", "d_inc", "d_loy", "d_nb", "cls"]


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def panel(prod, N=60, S=48, seed=5):
    """A synthetic table from R6 (K = 3), uneven lengths, non-empty inventory."""
    spec = M.Spec("R6", K=3)
    tab, _ = SIM.simulate_panel(spec, SIM.true_theta(spec, seed=seed), prod, N=N, S=S, seed=seed)
    rng = np.random.default_rng(seed)
    tab["length"] = rng.integers(S // 3, S + 1, N).astype(np.int32)
    tab["length"][0] = S
    for k in ("label", "prev", "campaign", "ipt"):
        tab[k][np.arange(S)[None, :] >= tab["length"][:, None]] = 0
    inv = rng.integers(0, 3, (N, ST.NP)) * (rng.random((N, ST.NP)) < 0.15)
    inv[: N // 3] = 0
    tab["inv_init"] = inv.astype(np.float32)
    return tab


def h6_theta(K: int, seed: int) -> torch.Tensor:
    spec = M.Spec("H6", K=K)
    th = SIM.true_theta(spec, seed=seed)
    off = spec.offsets()
    a, b = off["cls"]
    th[a:b] = torch.linspace(-0.5, 0.5, b - a, dtype=F64)
    return th


def i1_layouts():
    bad = [r for r, n in SIZES.items() if M.Spec(r).size != n]
    s4, s6 = M.Spec("R4"), M.Spec("R6", K=6)
    ok = not bad and s4.names == R4_NAMES and s6.names == R6_NAMES and s6.size == 197
    h = M.Spec("H6", K=6)
    check("I1 part-A layouts unchanged", ok,
          f"sizes {'ok' if not bad else bad}; R4 {s4.size}, R6 K=6 {s6.size}; H6 K=6 {h.size} params {h.names}")


def i2_embed(tab, prod):
    X = SIM.to_X(ST.build(tab, prod))
    th = h6_theta(3, 7)
    lp_h, pi_h = M.log_probs(M.Spec("H6", K=3), th, X)
    th6 = IC.embed_h6(th, 3, a_dec=0.4)
    lp_6, pi_6 = M.log_probs(M.Spec("R6", K=3), th6, X)
    e1 = float((lp_h - lp_6).abs().max())
    e2 = float((pi_h - pi_6).abs().max())
    check("I2 H6 embedded in R6", e1 < 1e-12 and e2 < 1e-15, f"probs {e1:.1e}, class weights {e2:.1e}")


def i3_classes(tab, prod):
    F = ST.build(tab, prod)
    X = SIM.to_X(F)
    alive = F["alive"]
    ctx = PL.Context(tab, prod)
    seq = PL.sequences(tab, np.arange(tab["label"].shape[0]))
    for spec, th in ((M.Spec("R6", K=3), SIM.true_theta(M.Spec("R6", K=3), seed=9)),
                     (M.Spec("H6", K=3), h6_theta(3, 9))):
        lp, _ = M.log_probs(spec, th, X)
        ref = lp.exp().numpy()
        tr = PL.Tracker(spec, th, ctx, seq)
        worst = 0.0
        for u in range(tr.L):
            act = u < tr.length
            for c in range(spec.K):
                P = tr.propose(np.full(tr.B, u), cls=np.full(tr.B, c))
                worst = max(worst, float(np.abs(P[act] - ref[c, act, u]).max()) if act.any() else 0.0)
            tr.commit(tr.label[:, u], act)
        check(f"I3 class-specific Tracker probabilities {spec.rung}", worst < 1e-6, f"max err {worst:.1e}")


def i4_weights(tab, prod):
    m = 5
    dup = {k: np.concatenate([v, v[:m]], 0) for k, v in tab.items()}
    F1, F2 = ST.build(tab, prod), ST.build(dup, prod)
    r1, r2 = D.roles(tab), D.roles(dup)
    N = tab["label"].shape[0]
    w = np.ones(N, np.int64)
    w[:m] = 2
    spec = M.Spec("R6", K=3)
    th = SIM.true_theta(spec, seed=13)
    c1 = FT.make_chunks(tab, F1, r1, np.arange(N), weights=w)
    c2 = FT.make_chunks(dup, F2, r2, np.arange(N + m))
    n = float(r2["train"].sum())
    out = []
    for ch in (c1, c2):
        t = th.clone().requires_grad_(True)
        f = FT.objective(spec, t, ch, n, 1e-4, backward=True)
        out.append((f, t.grad.clone()))
    ef = abs(out[0][0] - out[1][0])
    eg = float((out[0][1] - out[1][1]).abs().max())
    check("I4 bootstrap weights = duplicated customers", ef < 1e-12 and eg < 1e-12,
          f"objective {ef:.1e}, gradient {eg:.1e}")


def run_sim(tab, prod, spec, th, seed):
    roles = D.roles(tab)
    users, T, seq = IC.design(tab, roles)
    env = IC.level6_env(tab, prod, users)
    ctx = PL.Context(tab, prod)
    rec = {"users": np.arange(len(users))}
    sim = IC.simulate(spec, th, ctx, seq, T, env, seed=seed, record=rec)
    tab_s = IC.synthetic_table(tab, users, T, seq, sim)
    return roles, users, T, ctx, rec, sim, tab_s


def i5_simulate(tab, prod):
    spec = M.Spec("R6", K=3)
    th = SIM.true_theta(spec, seed=21)
    roles, users, T, ctx, rec, sim, tab_s = run_sim(tab, prod, spec, th, 101)
    roles_s = D.roles(tab_s)
    g1 = IC.check_design(tab, roles, users, T, tab_s, roles_s)
    F_s = ST.build(tab_s, prod, calendar=ctx.calendar)
    g2 = IC.check_simulation(spec, th, F_s, tab_s, rec, sim["cls"])
    S = tab_s["label"].shape[1]
    alive = np.arange(S)[None, :] < T[:, None]
    live_ok = bool(((tab_s["label"] >= 1) == alive).all())
    again = run_sim(tab, prod, spec, th, 101)[-1]
    same = IC.checksum(tab_s["label"], tab_s["obtained"]) == IC.checksum(again["label"], again["obtained"])
    bought = int(((tab_s["obtained"] >= ST.FIRST) & (tab_s["obtained"] <= ST.LAST)).any(-1).sum())
    check("I5 simulate on a design", g1["ok"] and g2["ok"] and live_ok and same,
          f"C-G1 {g1['equal']}; C-G2 blocks {g2['block_max']:.1e} probs {g2['prob_max']:.1e}; "
          f"live rows only {live_ok}; same seed same table {same}; rows with acquisitions {bought}")


def i6_quantities():
    spec = M.Spec("R6", K=3)
    th = SIM.true_theta(spec, seed=17)
    P = spec.unpack(th)
    q = IC.quantities(spec, th)
    w = torch.softmax(torch.cat([torch.zeros(1, dtype=F64), P["cls"]]), 0).numpy()
    zk = lambda v: np.concatenate([[0.0], v.numpy()])
    loy = float(P["loy"][0]) + float((w * zk(P["d_loy"])).sum())
    inc = float(P["inc"][0]) + float((w * zk(P["d_inc"])).sum())
    lag = P["lag"].numpy()
    rep = np.array([lag[j + 1, j] - np.delete(lag[1:, j], j).mean() for j in range(8)])
    e = max(abs(float(q["loy"]) - loy), abs(float(q["inc"]) - inc),
            abs(float(q["repeat_med"]) - float(np.median(rep))),
            max(abs(float(q[f"repeat[{j}]"]) - rep[j]) for j in range(8)))
    hl = float(q["hl_dec"])
    a = float(torch.sigmoid(P["a_dec"][0]))
    e = max(e, abs(hl - np.log(0.5) / np.log(a)))
    s4 = M.Spec("R4")
    q4 = IC.quantities(s4, SIM.true_theta(s4, seed=17))
    keys_ok = set(IC.SD_QUANTITIES) <= set(q4) and set(IC.HG_QUANTITIES) <= set(q4)
    check("I6 quantities match part A's formulas", e < 1e-12 and keys_ok, f"max err {e:.1e}; keys {keys_ok}")


def i7_continuous():
    th = h6_theta(6, 23)
    sb, tb, info = IC.continuous_h6(th, 6, K_big=200, seed=46499)
    m, dm = np.array(info["target_mean"]), np.array(info["draw_mean"])
    sd, dsd = np.array(info["target_sd"]), np.array(info["draw_sd"])
    ok_m = bool((np.abs(dm - m) <= 4 * sd / np.sqrt(200) + 1e-3).all())
    ok_s = bool((np.abs(dsd - sd) <= 0.25 * sd + 1e-3).all())
    ok_shape = tb.numel() == sb.size and sb.K == 200
    check("I7 continuous null matches the class moments", ok_m and ok_s and ok_shape,
          f"rank {info['rank']}; mean err max {np.abs(dm - m).max():.3f}; sd ratio "
          f"{np.round(dsd / np.maximum(sd, 1e-9), 2).tolist()}")


def i8_refit(tab, prod):
    spec = M.Spec("R4")
    th = SIM.true_theta(spec, seed=31)
    roles, users, T, ctx, rec, sim, tab_s = run_sim(tab, prod, spec, th, 202)
    roles_s = D.roles(tab_s)
    F_s = ST.build(tab_s, prod, calendar=ctx.calendar)
    ch = FT.make_chunks(tab_s, F_s, roles_s, np.arange(len(users)))
    n = float(roles_s["train"].sum())
    l2 = 0.01 / n
    f0 = FT.objective(spec, th, ch, n, l2, backward=False)
    est, _ = FT.fit(spec, th, ch, n, l2=l2, max_rounds=3, newton_iters=2)
    f1 = FT.objective(spec, est, ch, n, l2, backward=False)
    check("I8 refit from the truth is no worse (C-G3)", f1 <= f0 + 1e-9, f"truth {f0:.6f}, estimate {f1:.6f}")


def main() -> None:
    torch.set_num_threads(4)
    prod = ST.product_table()
    tab = panel(prod)
    i1_layouts()
    i2_embed(tab, prod)
    i3_classes(tab, prod)
    i4_weights(tab, prod)
    i5_simulate(tab, prod)
    i6_quantities()
    i7_continuous()
    i8_refit(tab, prod)
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
