"""
Smoke tests for R32's hybrid designs and R33's kernel inventory.

Checks that each new mode builds, runs, produces finite logits of the right
shape, and -- the part that matters -- stays CAUSAL: row t's logits must not
move when a later row's inputs change. The block-recurrent design is the one
at risk, because it reshapes the sequence into windows.

    python scripts/test_r32_r33.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_multistream_state_space import build_transformer  # noqa: E402

FAILED = []
B, S, P = 2, 96, 44
FIRST, LAST, VOCAB = 13, 56, 60


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def make(**kw):
    feat = torch.randn(VOCAB, 34)
    feat[FIRST:LAST + 1] = (torch.rand(LAST - FIRST + 1, 34) > 0.5).float()
    torch.manual_seed(0)
    return build_transformer(
        vocab_size_src=VOCAB, vocab_size_tgt=VOCAB, max_seq_len=S, d_model=32,
        n_layers=2, n_heads=4, d_ff=64, dropout=0.0, feature_tensor=feat, ai_rate=15,
        num_users=8, lto_len=4, obtained_len=10, prev_dec_len=1,
        use_user_embedding=False, **kw)


def inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    lto = torch.randint(FIRST, LAST + 1, (B, S, 4), generator=g)
    obt = torch.randint(FIRST, LAST + 1, (B, S, 10), generator=g)
    obt[torch.rand(B, S, 10, generator=g) < 0.5] = 0
    prev = torch.randint(1, 10, (B, S), generator=g)
    uid = torch.zeros(B, dtype=torch.long)
    ipt = torch.rand(B, S, generator=g) * 24
    return lto, obt, prev, uid, ipt


def run(model, xs):
    lto, obt, prev, uid, ipt = xs
    kw = {}
    if getattr(model, "inventory_slots", None) is not None:
        kw = {"inv_init_count": torch.zeros(B, P), "inv_init_last": torch.zeros(B, P)}
    with torch.no_grad():
        return model(lto, obt, prev, uid, ipt, **kw)


def causal(model, name):
    """Row t's logits must ignore rows after t."""
    xs = inputs(0)
    base = run(model, xs)
    t = S // 2
    lto, obt, prev, uid, ipt = [x.clone() for x in xs]
    g = torch.Generator().manual_seed(99)
    lto[:, t + 1:] = torch.randint(FIRST, LAST + 1, lto[:, t + 1:].shape, generator=g)
    obt[:, t + 1:] = torch.randint(FIRST, LAST + 1, obt[:, t + 1:].shape, generator=g)
    prev[:, t + 1:] = torch.randint(1, 10, prev[:, t + 1:].shape, generator=g)
    alt = run(model, (lto, obt, prev, uid, ipt))
    d_before = (base[:, :t + 1] - alt[:, :t + 1]).abs().max().item()
    d_after = (base[:, t + 1:] - alt[:, t + 1:]).abs().max().item()
    check(f"{name}: causal", d_before < 1e-5, f"max|d| rows<=t {d_before:.2e} "
          f"(rows>t {d_after:.2e})")


STOCK = dict(inventory="slots", sat_layers=2)


def main() -> None:
    print("R32 hybrid designs\n")
    for fuse in ("gate", "stack", "seq_ra", "seq_ar", "block"):
        kw = dict(encoder="gru_attn", fuse=fuse, attn_recency_bias=True,
                  attn_time_bias="ordinal", **STOCK)
        if fuse == "block":
            kw["block_len"] = 32
        m = make(**kw).eval()
        out = run(m, inputs(0))
        check(f"fuse={fuse}: shape and finite", tuple(out.shape) == (B, S, VOCAB)
              and bool(torch.isfinite(out).all()), f"{tuple(out.shape)}")
        causal(m, f"fuse={fuse}")

    print("\nR33 kernel inventory")
    for kernel, decay, tier in (("none", "none", False), ("none", "exp", False),
                                ("none", "none", True), ("attr", "exp", True),
                                ("learned", "exp", True)):
        name = f"kernel={kernel},decay={decay},tier={int(tier)}"
        m = make(encoder="gru", kernel=kernel, decay=decay, tier=tier, **STOCK).eval()
        out = run(m, inputs(0))
        check(f"{name}: shape and finite", tuple(out.shape) == (B, S, VOCAB)
              and bool(torch.isfinite(out).all()))
        causal(m, name)

    print("\nR33 numerics")
    m = make(encoder="gru", kernel="none", decay="exp", tier=False, **STOCK).eval()
    inv = m.inventory_slots
    # a long half-life must reproduce the plain cumulative count
    from model_multistream_state_space import _plain_rows, decayed_counts
    obt = inputs(3)[1]
    rows = _plain_rows(obt, torch.float32)
    plain = torch.cumsum(rows, dim=1) - rows
    slow = decayed_counts(rows, torch.tensor(1.0))
    check("decay rho=1 equals the cumulative count",
          torch.allclose(slow, plain, atol=1e-4),
          f"max|d| {(slow - plain).abs().max():.2e}")
    # brute force against the definition, on a short sequence
    rho = torch.tensor(0.5 ** (1 / 7))
    small = rows[:1, :40]
    fast = decayed_counts(small, rho, chunk=8)[0]
    brute = torch.zeros_like(fast)
    for t in range(small.size(1)):
        for tau in range(t):
            brute[t] += small[0, tau] * rho ** (t - tau)
    check("decay matches the explicit geometric sum",
          torch.allclose(fast, brute, atol=1e-4), f"max|d| {(fast - brute).abs().max():.2e}")
    g = make(encoder="gru", kernel="none", decay="none", tier=True,
             **STOCK).eval().inventory_slots.tier_weights(torch.float32, torch.device("cpu"))
    check("tier weights are weakly decreasing and start at 1",
          bool(g[0] == 1.0) and bool((g[1:] <= g[:-1] + 1e-6).all()),
          f"{[round(float(x), 3) for x in g]}")
    k = make(encoder="gru", kernel="attr", decay="none", tier=False,
             **STOCK).eval().inventory_slots.kernel_matrix(torch.float32, torch.device("cpu"))
    check("attribute kernel rows sum to 1", torch.allclose(k.sum(1), torch.ones(P), atol=1e-5))

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
