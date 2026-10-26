"""
Tests for R47's forgetting options in gen 6 (slot_state, camp_decay). CPU only.

  F1  the exact scan: _mult_scan with a constant factor rho reproduces the
      existing decayed stock, and with per-row factors it matches a
      brute-force loop of stock_t = a_t * (stock_(t-1) + copies_t).
  F2  camp_decay = 1 is the default model exactly (same logits).
  F3  camp_decay < 1 multiplies the stock by delta exactly at campaign changes
      (brute force, kernel off so the pre-kernel stock is observable), and
      delta = 0 empties it at every boundary; padding rows never count.
  F4  with camp_decay < 1 and stock_only, one pass = chunks = one occasion at
      a time (the campaign carry is exact).
  F5  stock_only: the slots' count is the thresholded decayed stock, the
      occasions-since feature is zero, the carried count stays the true count,
      and the logits differ from slot_state=full.
  F6  no decay (the infinite half-life) under stock_only: the slots' count is
      the cumulative tier-weighted stock.

    python scripts/test_r47_forgetting.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "shared"):
    sys.path.insert(0, str(p))

import model_multistream_state_space as mm  # noqa: E402
from model6_streaming import STOCK_FLOOR, StreamingHybrid, _mult_scan  # noqa: E402

FAIL = []
torch.set_num_threads(1)


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append(name)


def feat():
    g = torch.Generator().manual_seed(0)
    f = torch.randn(68, 34, generator=g)
    f[:, 0] = 3.0
    return f


def model(**over):
    kw = dict(window=5, chunk=7, vocab_size_src=68, vocab_size_tgt=18, max_seq_len=0, ai_rate=15,
              d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0, feature_tensor=feat(),
              num_users=4, encoder="gru_attn", fuse="stack", inventory="slots",
              attn_recency_bias=True, attn_time_bias="ordinal", sat_layers=2, kernel="attr",
              decay="exp", decay_init=3.0, decay_freeze=True, tier=True, n_campaigns=32)
    kw.update(over)
    torch.manual_seed(0)
    return StreamingHybrid(**kw).eval()


def history(B=3, S=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    lto = torch.randint(13, 57, (B, S, 4), generator=g)
    prev = torch.randint(1, 10, (B, S), generator=g)
    obt = torch.zeros(B, S, 10, dtype=torch.long)
    for b in range(B):
        for t in range(1, S):
            d = int(prev[b, t])
            n = 0 if d == 9 else (10 if d % 2 == 0 else 1)
            if n:
                obt[b, t, :n] = torch.randint(13, 57, (n,), generator=g)
    camp = (3 + torch.arange(S) // 7).expand(B, S).clone()
    camp[-1, -6:] = 0                    # one customer ends early: padding rows
    lto[-1, -6:] = 0
    obt[-1, -6:] = 0
    prev[-1, -6:] = 0
    return lto, obt, prev, torch.zeros(B, dtype=torch.long), camp


def rho_of(m):
    hl = F.softplus(m.inventory_slots.half_life_raw).clamp_min(1e-2)
    return float(torch.exp(-math.log(2.0) / hl))


def brute(w, a, d0):
    B, L, P = w.shape
    d, out = d0.clone().double(), []
    for t in range(L):
        d = a[:, t, None] * (d + w[:, t].double())
        out.append(d.clone())
    return torch.stack(out, 1)


def f1():
    print("F1 the exact scan")
    g = torch.Generator().manual_seed(2)
    w = torch.rand(2, 70, 5, generator=g)
    rho = 0.8
    a = torch.full((2, 70), rho, dtype=torch.float64)
    ref = rho * (mm.decayed_counts(w, torch.tensor(rho)) + w)
    got = _mult_scan(w, a, torch.zeros(2, 5))
    check(f"constant factor = existing decayed stock ({float((ref - got).abs().max()):.1e})",
          float((ref - got).abs().max()) < 1e-5)
    a2 = torch.rand(2, 70, generator=g, dtype=torch.float64) * 0.9 + 0.05
    a2[:, 33] = 1e-12
    d0 = torch.rand(2, 5, generator=g)
    diff = float((brute(w, a2, d0) - _mult_scan(w, a2, d0).double()).abs().max())
    check(f"per-row factors = brute-force loop, incl. a near-zero factor ({diff:.1e})", diff < 1e-5)


def f2():
    print("F2 camp_decay = 1 is the default model")
    lto, obt, prev, uid, camp = history()
    a, b = model(), model(camp_decay=1.0)
    with torch.no_grad():
        d = float((a(lto, obt, prev, uid, campaign=camp) - b(lto, obt, prev, uid, campaign=camp)).abs().max())
    check(f"identical logits ({d:.1e})", d == 0.0)


def f3():
    print("F3 the campaign-boundary factor")
    lto, obt, prev, uid, camp = history()
    for delta in (0.3, 0.0):
        m = model(kernel="none", camp_decay=delta)
        with torch.no_grad():
            _, _, stock, _, _ = m.computed_scan(obt, prev, m.init_stream(3, "cpu"), campaign=camp)
            g = m.inventory_slots.tier_weights(torch.float32, "cpu")
            plain = mm._plain_rows(obt, torch.float32)
            before = plain.cumsum(1) - plain
            w = mm.copy_weights(obt, before.long(), g)
            prev_c = torch.cat([torch.zeros(3, 1, dtype=torch.long), camp[:, :-1]], 1)
            bnd = (camp != prev_c) & (camp > 0) & (prev_c > 0)
            a = torch.where(bnd, torch.tensor(max(delta, 1e-12), dtype=torch.float64),
                            torch.tensor(1.0, dtype=torch.float64)) * rho_of(m)
            ref = brute(w, a, torch.zeros(3, w.size(-1)))
        diff = float((stock.double() - ref).abs().max())
        check(f"delta={delta}: stock = brute-force recursion ({diff:.1e}); "
              f"{int(bnd.sum())} boundaries, padding never one", diff < 1e-4
              and not bool(bnd[-1, -6:].any()))
        if delta == 0.0:
            rows = torch.nonzero(bnd[0]).view(-1)
            at = float((stock[0, rows] - (rho_of(m) * 0.0)).abs().max())
            check(f"delta=0 empties the stock at boundaries except this row's copies "
                  f"(max {at:.2e} where no copies arrive)",
                  all(float(stock[0, r].sum()) < 1e-6 or float(w[0, r].sum()) > 0 for r in rows.tolist()))


def f4():
    print("F4 one pass = chunks = one occasion at a time")
    lto, obt, prev, uid, camp = history()
    m = model(camp_decay=0.3, slot_state="stock_only")
    S = lto.size(1)
    with torch.no_grad():
        ref = m.stream_forward(lto, obt, prev, uid, camp, chunk=10 ** 6)
        worst = max(float((ref - m.stream_forward(lto, obt, prev, uid, camp, chunk=c)).abs().max())
                    for c in (1, 3, 7))
        st = m.init_stream(3, "cpu")
        outs = []
        lg, st = m.advance(lto[:, :11], obt[:, :11], prev[:, :11], st, uid, camp[:, :11])
        outs.append(lg)
        for t in range(11, S):
            lg, st = m.advance(lto[:, t:t + 1], obt[:, t:t + 1], prev[:, t:t + 1], st, uid,
                               camp[:, t:t + 1])
            outs.append(lg)
        step = float((ref - torch.cat(outs, 1)).abs().max())
    check(f"chunks 1/3/7 = one pass ({worst:.1e})", worst < 1e-5)
    check(f"prefix + one occasion at a time = one pass ({step:.1e})", step < 1e-5)


def f5_f6():
    print("F5/F6 stock_only")
    lto, obt, prev, uid, camp = history()
    # kernel off, so the returned (post-kernel) stock IS the pre-kernel stock
    full, so = model(kernel="none"), model(kernel="none", slot_state="stock_only")
    so.load_state_dict(full.state_dict())
    with torch.no_grad():
        cF, sF, kF, _, carF = full.computed_scan(obt, prev, full.init_stream(3, "cpu"), campaign=camp)
        cS, sS, kS, _, carS = so.computed_scan(obt, prev, so.init_stream(3, "cpu"), campaign=camp)
        want = torch.where(kS > STOCK_FLOOR, kS, torch.zeros_like(kS))
        d = float((cS - want).abs().max())
        check(f"slot count = thresholded decayed stock ({d:.1e})", d < 1e-5)
        check("occasions-since is zero", float(sS.abs().max()) == 0.0)
        check("the carried count stays the true count", torch.equal(carS["count"], carF["count"]))
        check("the stock itself is unchanged", float((kS - kF).abs().max()) == 0.0)
        lf = full(lto, obt, prev, uid, campaign=camp)
        ls = so(lto, obt, prev, uid, campaign=camp)
    check(f"logits differ from slot_state=full ({float((lf - ls).abs().max()):.1e})",
          float((lf - ls).abs().max()) > 1e-4)
    inf = model(decay="none", kernel="none", slot_state="stock_only")
    with torch.no_grad():
        c, _, _, _, _ = inf.computed_scan(obt, prev, inf.init_stream(3, "cpu"), campaign=camp)
        g = inf.inventory_slots.tier_weights(torch.float32, "cpu")
        plain = mm._plain_rows(obt, torch.float32)
        w = mm.copy_weights(obt, (plain.cumsum(1) - plain).long(), g)
        cum = w.cumsum(1)
        want = torch.where(cum > STOCK_FLOOR, cum, torch.zeros_like(cum))
    check(f"no decay: slot count = cumulative tier-weighted stock "
          f"({float((c - want).abs().max()):.1e})", float((c - want).abs().max()) < 1e-5)


def main() -> None:
    f1()
    f2()
    f3()
    f4()
    f5_f6()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all R47 forgetting checks passed")


if __name__ == "__main__":
    main()
