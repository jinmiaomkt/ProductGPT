"""
Regression tests for three gen-5 bugs found in the October 2026 code review.

1. Duplicate weights shifted by one tier. copy_weights was handed the
   inventory count INCLUDING the row being weighted, so a first copy got g[1],
   a second copy g[2], and g[0] = 1 was never used.
2. The R33 stock lagged twice. Rows already carry o_(t-1) (the R1 shift), yet
   the tier / decay / kernel path also dropped the current row, so the stock at
   row t missed the acquisitions of occasion t-1. The plain count path and
   simulation/kernel_sim.py both include them.
3. --no-product-features did nothing. It set `use_features` on the gen-5
   embedding, whose forward never read it.

Synthetic inputs only; no data needed.

    python scripts/test_stock_fixes.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model_multistream_state_space as mm  # noqa: E402
from model_multistream_state_space import build_transformer  # noqa: E402

FAILED = []
S, L, P = 5, 10, 44
FIRST, LAST, VOCAB = 13, 56, 60
A, B2 = 13, 14            # two products; index 0 and 1 in the stock


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def make(**kw):
    feat = torch.zeros(VOCAB, 34)
    feat[FIRST:LAST + 1] = (torch.rand(LAST - FIRST + 1, 34,
                                       generator=torch.Generator().manual_seed(1)) > 0.5).float()
    torch.manual_seed(0)
    return build_transformer(
        vocab_size_src=VOCAB, vocab_size_tgt=VOCAB, max_seq_len=S, d_model=32,
        n_layers=1, n_heads=4, d_ff=64, dropout=0.0, feature_tensor=feat, ai_rate=15,
        num_users=4, lto_len=4, obtained_len=10, prev_dec_len=1,
        use_user_embedding=False, encoder="gru", **kw).eval()


def history() -> torch.Tensor:
    """Rows carry o_(t-1). Product A: one copy at occasion 0 (row 1), a second
    at occasion 2 (row 3). Product B: two copies in ONE ten-pull at occasion 1
    (row 2)."""
    obt = torch.zeros(1, S, L, dtype=torch.long)
    obt[0, 1, 0] = A
    obt[0, 3, 0] = A
    obt[0, 2, 0] = B2
    obt[0, 2, 1] = B2
    return obt


def stock(model, obt):
    count, _ = mm.additive_inventory(obt)
    with torch.no_grad():
        return model.inventory_slots.stock_counts(obt, count), count


def fmt(v):
    return "[" + ", ".join(f"{float(x):.3g}" for x in v) + "]"


def main() -> None:
    obt = history()

    print("1. duplicate weights: the k-th copy gets g[k-1]   (g = [1, 0.5, 0.25])")
    m = make(inventory="slots", tier=True)
    g = m.inventory_slots.tier_weights(torch.float32, torch.device("cpu"))
    check("tier weights start at their documented init", torch.allclose(g, torch.tensor([1.0, 0.5, 0.25])),
          fmt(g))
    st, _ = stock(m, obt)
    check("two copies on separate occasions weigh 1 + 0.5", math.isclose(float(st[0, -1, 0]), 1.5, abs_tol=1e-6),
          f"final stock of A {float(st[0, -1, 0]):.3g}")
    check("two copies from one ten-pull weigh 1 + 0.5", math.isclose(float(st[0, -1, 1]), 1.5, abs_tol=1e-6),
          f"final stock of B {float(st[0, -1, 1]):.3g}")

    print("\n2. timing: the stock at row t includes occasion t-1, like the plain count")
    check("tier path: A's first copy counts from row 1", math.isclose(float(st[0, 1, 0]), 1.0, abs_tol=1e-6),
          f"A by row {fmt(st[0, :, 0])}, expected [0, 1, 1, 1.5, 1.5]")
    m = make(inventory="slots", decay="exp", decay_init=1e6, decay_freeze=True)
    st, count = stock(m, obt)
    check("decay path with an infinite half-life equals the plain count",
          torch.allclose(st, count, atol=1e-3),
          f"A by row {fmt(st[0, :, 0])} vs count {fmt(count[0, :, 0])}")
    m = make(inventory="slots", decay="exp", decay_init=3.0, decay_freeze=True)
    st, _ = stock(m, obt)
    rho = float(torch.exp(-math.log(2.0) / F.softplus(m.inventory_slots.half_life_raw)))
    want = [0.0, rho, rho ** 2, rho ** 3 + rho, rho ** 4 + rho ** 2]
    check("decay path: a copy from occasion tau weighs rho^(t - tau), as in kernel_sim",
          torch.allclose(st[0, :, 0], torch.tensor(want), atol=1e-5),
          f"A by row {fmt(st[0, :, 0])} vs {fmt(want)}")

    print("\n3. --no-product-features removes the attribute branch")
    ids = torch.arange(FIRST, LAST + 1).view(1, -1)
    m = make(use_product_features=False)
    emb = m.product_embed
    with torch.no_grad():
        off = emb(ids)
        id_only = emb.id_embed(ids)
    check("product vectors equal the identity embedding alone", torch.allclose(off, id_only),
          f"max|d| {(off - id_only).abs().max():.2e}")
    m = make()
    with torch.no_grad():
        on = m.product_embed(ids)
        id_only = m.product_embed.id_embed(ids)
    check("default still adds the attribute branch", not torch.allclose(on, id_only),
          f"max|d| {(on - id_only).abs().max():.2e}")
    from model_recurrent_baseline import build_recurrent_baseline
    gru = build_recurrent_baseline("gru", vocab_size_src=VOCAB, vocab_size_tgt=VOCAB, d_model=32,
                                   n_layers=1, d_ff=64, dropout=0.0,
                                   feature_tensor=make().product_embed.feat_tbl,
                                   use_product_features=False).eval()
    with torch.no_grad():
        d = (gru.product_embed(ids) - gru.product_embed.id_embed(ids)).abs().max()
    check("the GRU baseline honours the flag too", float(d) == 0.0, f"max|d| {float(d):.2e}")

    print("\n4. model_rev=1 rebuilds pre-fix checkpoints exactly as they were trained")
    m = make(inventory="slots", tier=True, model_rev=1)
    st, _ = stock(m, obt)
    check("rev 1 tier path keeps the old numbers", torch.allclose(st[0, :, 0], torch.tensor([0, 0, .5, .5, .75])),
          f"A by row {fmt(st[0, :, 0])}")
    m = make(inventory="slots", decay="exp", decay_init=1e6, decay_freeze=True, model_rev=1)
    st, count = stock(m, obt)
    check("rev 1 decay path keeps the old lag",
          torch.allclose(st[0, :, 0], torch.tensor([0., 0., 1., 1., 2.]), atol=1e-3),
          f"A by row {fmt(st[0, :, 0])}")
    m = make(use_product_features=False, model_rev=1)
    with torch.no_grad():
        d = (m.product_embed(ids) - m.product_embed.id_embed(ids)).abs().max()
    check("rev 1 keeps the attribute branch despite the flag (as b26 trained)", float(d) > 0,
          f"max|d| {float(d):.2e}")
    for kw in ({}, dict(inventory="slots"), dict(inventory="slots", sat_layers=2)):
        torch.manual_seed(3)
        lto = torch.randint(FIRST, LAST + 1, (2, S, 4))
        o = torch.randint(FIRST, LAST + 1, (2, S, L)) * (torch.rand(2, S, L) < 0.4)
        prev = torch.randint(1, 10, (2, S))
        with torch.no_grad():
            a = make(**kw, model_rev=1)(lto, o, prev, torch.zeros(2, dtype=torch.long))
            b = make(**kw)(lto, o, prev, torch.zeros(2, dtype=torch.long))
        check(f"without R33 options rev 1 == rev 2 {kw or '(tokens)'}", torch.equal(a, b),
              f"max|d| {(a - b).abs().max():.2e}")

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
