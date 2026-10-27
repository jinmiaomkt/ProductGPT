"""R46 part B unit tests: the online policy state against the batch builder.

CPU, synthetic panels from parametric/simulate.py, a few minutes:

    python scripts/test_r46_policy.py

  P1  replay: for every rung, the Tracker on the real streams reproduces every
      state.build block on every live row (uneven lengths, non-empty
      pre-window inventory) and model.predictive's probabilities
  P2  utilities() with the stocks passed in equals utilities() computing them
  P3  a right-aligned warm-up (what a rollout does before its first holdout
      row) reaches the replay's state: same probabilities at the next row
  P4  write(): a generated row changes the next row's blocks, never the
      current one (no look-ahead)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from parametric import model as M, policy as PL, simulate as SIM, state as ST  # noqa: E402

RESULTS = []
RUNGS = ("M0", "M1", "R0", "R1", "R1b", "R2", "R3", "R4", "R6")
TOL_BLOCK, TOL_PROB = 1e-5, 1e-6


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def panel(prod, N=24, S=48, seed=5):
    """A synthetic table from R6 (K = 3), then uneven lengths and inventory."""
    spec = M.Spec("R6", K=3)
    tab, _ = SIM.simulate_panel(spec, SIM.true_theta(spec, seed=seed), prod, N=N, S=S, seed=seed)
    rng = np.random.default_rng(seed)
    tab["length"] = rng.integers(S // 3, S + 1, N).astype(np.int32)
    tab["length"][0] = S
    inv = rng.integers(0, 3, (N, ST.NP)) * (rng.random((N, ST.NP)) < 0.15)
    inv[: N // 3] = 0
    tab["inv_init"] = inv.astype(np.float32)
    return tab


def p1_replay(tab, prod):
    ctx = PL.Context(tab, prod)
    F = ST.build(tab, prod)
    alive = F["alive"]
    seq = PL.sequences(tab, np.arange(tab["label"].shape[0]))
    y = torch.as_tensor(tab["label"].astype(np.int64))
    valid = torch.as_tensor(alive & (tab["label"] >= 1) & (tab["label"] <= 9))
    for rung in RUNGS:
        spec = M.Spec(rung, K=3 if rung == "R6" else 1)
        theta = SIM.true_theta(spec, seed=11)
        ref = M.predictive(spec, theta, SIM.to_X(F), y, valid).exp().numpy()
        prob, blocks = PL.replay(spec, theta, ctx, seq)
        worst = {}
        for k in PL.BLOCKS:
            a, b = blocks[k][alive], F[k][alive]
            worst[k] = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        bad = {k: v for k, v in worst.items() if v > TOL_BLOCK}
        pe = float(np.abs(prob[alive] - ref[alive]).max())
        check(f"P1 replay {rung}", not bad and pe < TOL_PROB,
              f"blocks max {max(worst.values()):.1e}{' BAD ' + str(bad) if bad else ''}; probs {pe:.1e}")


def p2_stocks(tab, prod):
    F = SIM.to_X(ST.build(tab, prod))
    spec = M.Spec("R6", K=3)
    theta = SIM.true_theta(spec, seed=3)
    P = spec.unpack(theta)
    a_d, a_o = torch.sigmoid(P["a_dec"][0]), torch.sigmoid(P["a_out"][0])
    X2 = dict(F, LD=M.decay(F["ind"][..., :6], a_d), LO=M.decay(F["ind"][..., 6:9], a_o))
    err = float((M.utilities(spec, P, F) - M.utilities(spec, P, X2)).abs().max())
    check("P2 utilities with stocks passed in", err < 1e-12, f"max err {err:.1e}")


def p3_warm(tab, prod):
    ctx = PL.Context(tab, prod)
    seq = PL.sequences(tab, np.arange(tab["label"].shape[0]))
    rng = np.random.default_rng(2)
    for rung in ("R3", "R6"):
        spec = M.Spec(rung, K=3 if rung == "R6" else 1)
        theta = SIM.true_theta(spec, seed=4)
        prob, _ = PL.replay(spec, theta, ctx, seq, blocks=False)
        pre = np.array([rng.integers(1, int(L)) for L in tab["length"]])
        tr = PL.Tracker(spec, theta, ctx, seq)
        tr.warm(pre)
        got = tr.propose(pre)
        ref = prob[np.arange(len(pre)), pre]
        err = float(np.abs(got - ref).max())
        check(f"P3 right-aligned warm-up {rung}", err < TOL_PROB, f"max err {err:.1e}")


def p4_write(tab, prod):
    ctx = PL.Context(tab, prod)
    seq = PL.sequences(tab, np.arange(4))
    spec = M.Spec("R4")
    theta = SIM.true_theta(spec, seed=6)
    u = np.array([10, 12, 14, 16])
    tr = PL.Tracker(spec, theta, ctx, seq)
    tr.warm(u)
    before, _ = tr.propose(u, keep_blocks=True)
    # overwrite row u+1 with a ten-pull on figure A that yields its featured product
    y = np.full(4, 4)
    obt = np.zeros((4, 10), np.int64)
    obt[:, 0] = tr.lto[tr.ar, u, 0]
    tr.write(np.ones(4, bool), u + 1, y, obt)
    again, _ = tr.propose(u, keep_blocks=True)
    tr.commit(tr.label[tr.ar, u], np.ones(4, bool))
    _, Fn = tr.propose(u + 1, keep_blocks=True)
    pid = obt[:, 0] - ST.FIRST
    cnt_new, cnt_old = tr._pending[0]["cnt"][tr.ar, pid], tr.c["cnt"][tr.ar, pid]
    ok_now = np.array_equal(before, again)
    ok_next = bool((Fn["prev_idx"] == 4).all()) and bool((cnt_new == cnt_old + 1).all())
    check("P4 a written row reaches the next row only", ok_now and ok_next,
          f"current row unchanged {ok_now}; next row sees it {ok_next}")


def main() -> None:
    torch.set_num_threads(4)
    prod = ST.product_table()
    tab = panel(prod)
    p1_replay(tab, prod)
    p2_stocks(tab, prod)
    p3_warm(tab, prod)
    p4_write(tab, prod)
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
