"""R46 part D unit tests (gate D-G1): R6B, class splitting, hl_ban.

CPU, synthetic panels from parametric/simulate.py, a few minutes:

    python scripts/test_r46_banner.py

  B1  R6B with a_ban = a_dec gives R6's probabilities and class weights exactly
  B2  a_ban moves the banner-stock terms and nothing else: with loyalty off it
      changes no utility; with loyalty on the change is loyalty times the
      difference of the two banner stocks; a_dec still moves buy and ten-pull
  B3  an eps = 0 split gives the K = 6 likelihood and predictive exactly at
      K = 9 and K = 12 (R6 and R6B); the heaviest classes are split, weights
      halve, and eps > 0 moves only the copies' class constants
  B4  hl_ban, hl_dec and the class-weighted loyalty read back from a known
      parameter vector
  B5  every existing layout is unchanged (R6B is appended), R6B = R6 + a_ban,
      and K = 9 / 12 carry 233 / 269 parameters
  B6  R6B has no rollout: the Tracker and a carried stock both refuse it
  B7  part D's starts (scripts/r46d_banner.py): R6B start 0 is R6 with
      a_ban = a_dec, start 1 sits at a 90-occasion half-life, and the nested
      split's penalised objective is the K = 6 one plus the ridge on the
      duplicated class terms (D-G2's allowance)
  B8  a short R6B fit from R6's estimate ends no worse than R6 (D-G2)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parametric import data as D, fit as FT, identify as IC, model as M, policy as PL  # noqa: E402
from parametric import simulate as SIM, state as ST  # noqa: E402
import test_r46_identify as TI  # noqa: E402  part C's synthetic panel and layouts

RESULTS = []
F64 = torch.float64


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def r6_theta(K: int, seed: int) -> torch.Tensor:
    """R6 with distinct classes and uneven weights (true_theta makes them equal)."""
    spec = M.Spec("R6", K=K)
    th = SIM.true_theta(spec, seed=seed)
    off = spec.offsets()
    g = torch.Generator().manual_seed(seed + 1)
    for n, sc in (("d_asc", 0.8), ("d_inc", 0.3), ("d_loy", 0.3), ("d_nb", 0.2), ("cls", 0.7)):
        a, b = off[n]
        th[a:b] = sc * torch.randn(b - a, generator=g, dtype=F64)
    th[off["loy"][0]] = 1.5
    return th


def to_r6b(th6: torch.Tensor, K: int, a_ban: float | None = None) -> torch.Tensor:
    P = M.Spec("R6", K=K).unpack(th6)
    a = float(P["a_dec"][0]) if a_ban is None else a_ban
    return IC.pack(M.Spec("R6B", K=K), {**P, "a_ban": torch.tensor([a], dtype=F64)})


def b1_nests(X):
    K = 3
    th6 = r6_theta(K, 3)
    lp6, pi6 = M.log_probs(M.Spec("R6", K=K), th6, X)
    lpb, pib = M.log_probs(M.Spec("R6B", K=K), to_r6b(th6, K), X)
    e1, e2 = float((lp6 - lpb).abs().max()), float((pi6 - pib).abs().max())
    check("B1 R6B at a_ban = a_dec is R6", e1 < 1e-12 and e2 == 0.0, f"probs {e1:.1e}, class weights {e2:.1e}")


def b2_moves(X):
    K = 3
    sb = M.Spec("R6B", K=K)
    th = to_r6b(r6_theta(K, 5), K)
    off = sb.offsets()
    U = lambda t: M.utilities(sb, sb.unpack(t), X)
    a0 = float(th[off["a_ban"][0]])
    a1 = math.log(0.95 / 0.05)
    moved = lambda t: torch.cat([t[:off["a_ban"][0]], torch.tensor([a1], dtype=F64)])
    # loyalty off: a_ban has no path into any utility
    off_l = th.clone()
    off_l[off["loy"][0]] = 0.0
    lo, hi = off["d_loy"]
    off_l[lo:hi] = 0.0
    e_off = float((U(off_l) - U(moved(off_l))).abs().max())
    # loyalty on: the change is (loy + d_loy_k) times the change in the banner stock
    dU = U(moved(th)) - U(th)
    ind = X["ind"][..., :4]
    dS = (M.decay(ind, torch.sigmoid(torch.tensor(a1, dtype=F64)))
          - M.decay(ind, torch.sigmoid(torch.tensor(a0, dtype=F64)))) @ M.BAN
    P = sb.unpack(th)
    loy_k = P["loy"][0] + torch.cat([torch.zeros(1, dtype=F64), P["d_loy"]])
    want = loy_k[:, None, None, None] * dS[None]
    e_on = float((dU - want).abs().max())
    size_on = float(dU.abs().max())
    # NotBuy and nothing else on the buy-or-ten-pull path: column 8 never moves
    e_nb = float(dU[..., 8].abs().max())
    # a_dec still drives the buy and ten-pull stocks in R6B
    th_d = off_l.clone()
    th_d[off["a_dec"][0]] = a1
    size_dec = float((U(th_d) - U(off_l)).abs().max())
    ok = e_off == 0.0 and e_on < 1e-12 and size_on > 1e-3 and e_nb == 0.0 and size_dec > 1e-3
    check("B2 a_ban moves only the banner stock", ok,
          f"loyalty off {e_off:.1e}; on: change {size_on:.3f}, vs formula {e_on:.1e}; NotBuy {e_nb:.1e}; "
          f"a_dec still moves buy/ten-pull by {size_dec:.3f}")


def b3_split(X, y, valid, mask):
    ok_all, worst = True, 0.0
    detail = []
    for rung in ("R6", "R6B"):
        th6 = r6_theta(6, 11)
        if rung == "R6B":
            th6 = to_r6b(th6, 6, a_ban=math.log(0.8 / 0.2))
        s6 = M.Spec(rung, K=6)
        ll6 = M.customer_ll(s6, th6, X, y, mask)
        pr6 = M.predictive(s6, th6, X, y, valid)
        for K in (9, 12):
            sk, thk, info = IC.split_classes(s6, th6, K, eps=0.0, seed=46900)
            e = max(float((M.customer_ll(sk, thk, X, y, mask) - ll6).abs().max()),
                    float((M.predictive(sk, thk, X, y, valid) - pr6).abs().max()))
            worst = max(worst, e)
            ok_all &= e < 1e-10 and thk.numel() == sk.size and sk.K == K
    # the bookkeeping: heaviest split, weights halve, eps moves only the copies' constants
    s6 = M.Spec("R6", K=6)
    th6 = r6_theta(6, 11)
    w6 = torch.softmax(torch.cat([torch.zeros(1, dtype=F64), s6.unpack(th6)["cls"]]), 0).numpy()
    heavy = np.argsort(-w6, kind="stable")[:3]
    s9, t0, i0 = IC.split_classes(s6, th6, 9, eps=0.0, seed=46901)
    _, t1, _ = IC.split_classes(s6, th6, 9, eps=0.1, seed=46901)
    P0, P1 = s9.unpack(t0), s9.unpack(t1)
    w9 = torch.softmax(torch.cat([torch.zeros(1, dtype=F64), P0["cls"]]), 0).numpy()
    want_w = w6.copy()
    want_w[heavy] /= 2
    want_w = np.concatenate([want_w, w6[heavy] / 2])
    ok_w = i0["split"] == heavy.tolist() and np.abs(w9 - want_w).max() < 1e-12
    dd = (P1["d_asc"] - P0["d_asc"]).abs().sum(1)              # rows = classes 1..8
    ok_eps = float(dd[:5].max()) == 0.0 and bool((dd[5:] > 0).all())
    z = np.random.default_rng(46901).standard_normal((3, 8))
    ok_eps &= float((P1["d_asc"][5:] - P0["d_asc"][5:] - 0.1 * torch.as_tensor(z)).abs().max()) < 1e-12
    same = all(torch.equal(P0[n], P1[n]) for n in s9.names if n != "d_asc")
    detail.append(f"max err {worst:.1e}; split {i0['split']} (heaviest {heavy.tolist()}); weights {ok_w}; "
                  f"eps moves only the copies {ok_eps}; other parameters equal {same}")
    check("B3 eps = 0 split is the K = 6 model", ok_all and ok_w and ok_eps and same, "; ".join(detail))


def b4_quantities():
    K = 4
    th6 = r6_theta(K, 19)
    sb = M.Spec("R6B", K=K)
    th = to_r6b(th6, K, a_ban=math.log(0.5 ** (1 / 7) / (1 - 0.5 ** (1 / 7))))
    off = sb.offsets()
    th[off["a_dec"][0]] = math.log(0.5 ** (1 / 2.5) / (1 - 0.5 ** (1 / 2.5)))
    P = sb.unpack(th)
    w = torch.softmax(torch.cat([torch.zeros(1, dtype=F64), P["cls"]]), 0).numpy()
    loy = float(P["loy"][0]) + float((w * np.concatenate([[0.0], P["d_loy"].numpy()])).sum())
    q = IC.quantities(sb, th)
    e = max(abs(float(q["hl_ban"]) - 7.0), abs(float(q["hl_dec"]) - 2.5), abs(float(q["loy"]) - loy))
    qs = IC.quantities_with_se(sb, th, torch.eye(sb.size, dtype=F64) * 1e-4)
    se_ok = math.isfinite(qs["hl_ban"]["se"]) and qs["hl_ban"]["se"] > 0
    d = FT.transformed(sb, th, None)
    e = max(e, abs(d["half_life_banner"]["value"] - 7.0))
    check("B4 hl_ban and class-weighted loyalty", e < 1e-9 and se_ok and "hl_ban" not in IC.quantities(M.Spec("R6", K=K), th6),
          f"max err {e:.1e}; delta-method SE finite {se_ok}")


def b5_layouts():
    bad = [r for r, n in TI.SIZES.items() if M.Spec(r).size != n]
    s6, sb = M.Spec("R6", K=6), M.Spec("R6B", K=6)
    h = M.Spec("H6", K=6)
    ok = (not bad and M.Spec("R4").names == TI.R4_NAMES and s6.names == TI.R6_NAMES and s6.size == 197
          and h.size == 101 and sb.names == TI.R6_NAMES + ["a_ban"] and sb.size == 198
          and s6.offsets() == {k: v for k, v in sb.offsets().items() if k != "a_ban"}
          and M.Spec("R6", K=9).size == 233 and M.Spec("R6", K=12).size == 269
          and M.RUNGS[:11] == ("M0", "M1", "R0", "R1", "R1b", "R2", "R3", "R4", "R5", "R6", "H6"))
    check("B5 layouts unchanged; R6B = R6 + a_ban", ok,
          f"sizes {'ok' if not bad else bad}; R6 {s6.size}, R6B {sb.size}, H6 {h.size}, "
          f"K=9 {M.Spec('R6', K=9).size}, K=12 {M.Spec('R6', K=12).size}")


def b6_no_rollout(tab, prod, X):
    sb = M.Spec("R6B", K=2)
    th = to_r6b(r6_theta(2, 23), 2)
    ctx = PL.Context(tab, prod)
    seq = PL.sequences(tab, np.arange(4))
    refused = []
    try:
        PL.Tracker(sb, th, ctx, seq)
    except NotImplementedError:
        refused.append("Tracker")
    Xc = dict(X)
    Xc["LD"] = torch.zeros(*X["prev_idx"].shape, 6, dtype=F64)
    try:
        M.utilities(sb, sb.unpack(th), Xc)
    except NotImplementedError:
        refused.append("carried stock")
    check("B6 R6B has no rollout", refused == ["Tracker", "carried stock"], f"refused: {refused}")


def b7_starts(tab, prod):
    import r46d_banner as DB
    th6 = r6_theta(6, 29)
    P6 = M.Spec("R6", K=6).unpack(th6)
    s0, t0, ref0, _ = DB.start_for("R6B", 0, th6)
    s1, t1, _, i1 = DB.start_for("R6B", 1, th6)
    ok_b = (s0.rung == "R6B" and torch.equal(t0[:-1], th6) and float(t0[-1]) == float(P6["a_dec"][0])
            and abs(i1["hl_ban_start"] - 90.0) < 1e-9 and torch.equal(t1[:-1], th6) and ref0[0].rung == "R6")
    F = ST.build(tab, prod)
    roles = D.roles(tab)
    users = np.arange(tab["label"].shape[0])
    ch = FT.make_chunks(tab, F, roles, users)
    n = float(roles["train"].sum())
    l2 = 0.01 / n
    f6 = FT.objective(M.Spec("R6", K=6), th6, ch, n, l2, backward=False)
    gaps = []
    for arm in ("K9", "K12"):
        sk, tk, (rs, rt), info = DB.start_for(arm, 0, th6)
        fk = FT.objective(sk, rt, ch, n, l2, backward=False)
        pen = 0.5 * l2 * float((rt ** 2).sum() - (th6 ** 2).sum())
        gaps.append(abs(fk - f6 - pen))
        ok_b &= torch.equal(tk, rt) and info["eps"] == 0.0
        _, t2, _, i2 = DB.start_for(arm, 2, th6)
        ok_b &= i2["eps"] == 0.5 and i2["seed"] == 46902 and not torch.equal(t2, rt)
    check("B7 part D's starts", ok_b and max(gaps) < 1e-12,
          f"R6B starts ok; nested objective - K6 - ridge on duplicates: {max(gaps):.1e}")


def b8_nesting(tab, prod):
    F = ST.build(tab, prod)
    roles = D.roles(tab)
    ch = FT.make_chunks(tab, F, roles, np.arange(tab["label"].shape[0]))
    n = float(roles["train"].sum())
    # the real design's per-row ridge (0.01 over about 1.06M training rows): on
    # this small panel 0.01 / n would make a_ban's own ridge term (1.9e-5)
    # larger than three L-BFGS rounds can recover, which says nothing about D-G2
    l2 = 0.01 / 1.06e6
    s6 = M.Spec("R6", K=2)
    th6, _ = FT.fit(s6, r6_theta(2, 31), ch, n, l2=l2, max_rounds=3, newton_iters=0)
    f6 = FT.objective(s6, th6, ch, n, l2, backward=False)
    sb = M.Spec("R6B", K=2)
    thb, _ = FT.fit(sb, to_r6b(th6, 2), ch, n, l2=l2, max_rounds=3, newton_iters=0)
    fb = FT.objective(sb, thb, ch, n, l2, backward=False)
    check("B8 R6B from R6's estimate is no worse (D-G2)", fb <= f6 + 1e-6, f"R6 {f6:.6f}, R6B {fb:.6f}")


def main() -> None:
    torch.set_num_threads(1)
    prod = ST.product_table()
    tab = TI.panel(prod)
    F = ST.build(tab, prod)
    X = SIM.to_X(F)
    roles = D.roles(tab)
    y = torch.as_tensor(tab["label"].astype(np.int64))
    valid = torch.as_tensor(roles["valid"])
    mask = torch.as_tensor(roles["train"])
    b1_nests(X)
    b2_moves(X)
    b3_split(X, y, valid, mask)
    b4_quantities()
    b5_layouts()
    b6_no_rollout(tab, prod, X)
    b7_starts(tab, prod)
    b8_nesting(tab, prod)
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
