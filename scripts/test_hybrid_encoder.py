"""
Formal tests for the recurrence + attention encoder (R25, batch 10).

For every sequence encoder -- transformer, gru, and gru_attn in both fusion
modes -- check the three properties a sequence model here must have:

1. Causality: perturbing occasion k's inputs must leave every prediction at
   occasions < k bit-identical. A hybrid has two paths to get this wrong (the
   recurrent one and the attention mask), so it is checked directly.
2. Eval equals train at dropout 0 (the fast-path trap of R20).
3. Finite output, and for the gated hybrid a gate mean in [0, 1] that is
   actually recorded, since it is reported as a diagnostic.

USAGE
    python scripts/test_hybrid_encoder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_multistream_state_space import build_transformer  # noqa: E402

FAIL = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append(name)


def make(encoder: str, fuse: str = "gate", alibi: bool = True):
    torch.manual_seed(0)
    return build_transformer(
        vocab_size_src=68, vocab_size_tgt=18, max_seq_len=32, d_model=32, n_layers=2,
        n_heads=4, d_ff=64, dropout=0.0, feature_tensor=torch.randn(68, 34), ai_rate=15,
        num_users=4, lto_len=4, obtained_len=10, prev_dec_len=1, encoder=encoder,
        fuse=fuse, attn_recency_bias=alibi, attn_time_bias="ordinal" if alibi else "none")


def main() -> None:
    g = torch.Generator().manual_seed(1)
    B, S, k = 2, 16, 7
    lto = torch.randint(13, 57, (B, S, 4), generator=g)
    obt = torch.randint(13, 57, (B, S, 10), generator=g)
    prev = torch.randint(1, 10, (B, S), generator=g)
    uid = torch.zeros(B, dtype=torch.long)
    ipt = torch.rand(B, S, generator=g) * 30

    lto2 = lto.clone(); lto2[:, k] = torch.tensor([20, 21, 22, 23])

    cases = [("transformer", "gate", True), ("gru", "gate", False),
             ("gru_attn", "gate", True), ("gru_attn", "gate", False),
             ("gru_attn", "stack", True)]
    for encoder, fuse, alibi in cases:
        label = encoder + (f"/{fuse}" + ("+alibi" if alibi else "/no-recency-bias")
                           if encoder == "gru_attn" else "")
        print(f"{label}")
        m = make(encoder, fuse, alibi).eval()
        with torch.no_grad():
            a = m(lto, obt, prev, uid, ipt)
            b = m(lto2, obt, prev, uid, ipt)
            m.train()
            c = m(lto, obt, prev, uid, ipt)
            m.eval()
        before = (a[:, :k] - b[:, :k]).abs().max().item()
        after = (a[:, k:] - b[:, k:]).abs().max().item()
        check(f"occasions < k unchanged (max diff {before:.1e})", before == 0.0)
        check(f"occasions >= k respond (max diff {after:.1e})", after > 1e-5)
        check(f"eval equals train (max diff {(a - c).abs().max().item():.1e})",
              (a - c).abs().max().item() < 1e-5)
        check("finite output", bool(torch.isfinite(a).all()))
        if encoder == "gru_attn" and fuse == "gate":
            gm = getattr(m, "gate_mean", float("nan"))
            check(f"gate mean recorded in [0,1] ({gm:.3f})", 0.0 <= gm <= 1.0)
            n_attn = sum(p.numel() for n, p in m.named_parameters() if "attn_branch" in n)
            check(f"attention branch has parameters ({n_attn:,})", n_attn > 0)

    print("PASS" if not FAIL else f"FAIL ({len(FAIL)})")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
