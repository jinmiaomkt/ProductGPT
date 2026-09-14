"""
Formal test for the R21 leak fix: row t's prediction must not depend on IPT_t.

IPT_t is the gap ENDING at row t, which reveals row t's label (see
scripts/ipt_leak_check.py). With time_bias_lag_ipt=True, perturbing IPT_t may
change predictions at rows > t but must leave rows <= t bit-for-bit unchanged
(up to float noise). With the flag False the same perturbation must change
row t -- otherwise the test cannot detect the leak it guards against.

USAGE
    python scripts/test_time_bias_causality.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_multistream_state_space import build_transformer  # noqa: E402


def make(lag: bool, mode: str = "time") -> torch.nn.Module:
    torch.manual_seed(0)
    feat = torch.randn(60, 34)
    m = build_transformer(
        vocab_size_src=60, vocab_size_tgt=60, max_seq_len=32, d_model=32,
        n_layers=2, n_heads=4, d_ff=64, dropout=0.0, feature_tensor=feat,
        ai_rate=15, num_users=4, lto_len=4, obtained_len=10, prev_dec_len=1,
        use_user_embedding=False, attn_time_bias=mode, attn_recency_bias=True,
        time_bias_lag_ipt=lag)
    return m.eval()


def main() -> None:
    B, S, t = 2, 24, 10
    g = torch.Generator().manual_seed(1)
    lto = torch.randint(13, 57, (B, S, 4), generator=g)
    obt = torch.randint(13, 57, (B, S, 10), generator=g)
    prev = torch.randint(1, 10, (B, S), generator=g)
    uid = torch.zeros(B, dtype=torch.long)
    ipt = torch.rand(B, S, generator=g) * 30
    ipt2 = ipt.clone()
    ipt2[:, t] = 24.0 if float(ipt[0, t]) < 12 else 0.0     # flip "which kind of row"

    ok = True
    # Eval must score the same model training fitted. Without autocast,
    # PyTorch's inference fast path used to mangle the additive mask (eval vs
    # train differed by ~2.4); the model now forces the standard path.
    for mode in ("ordinal", "time"):
        m = make(True, mode)
        with torch.no_grad():
            m.train()
            a = m(lto, obt, prev, uid, ipt)
            m.eval()
            b = m(lto, obt, prev, uid, ipt)
        d = (a - b).abs().max().item()
        print(f"{mode:<8} max|eval - train| (dropout 0, no autocast): {d:.2e}")
        ok &= d < 1e-4

    for lag in (True, False):
        m = make(lag)
        with torch.no_grad():
            a = m(lto, obt, prev, uid, ipt)
            b = m(lto, obt, prev, uid, ipt2)
        d_upto = (a[:, : t + 1] - b[:, : t + 1]).abs().max().item()
        d_after = (a[:, t + 1:] - b[:, t + 1:]).abs().max().item()
        d_row = (a[:, t] - b[:, t]).abs().max().item()
        print(f"lag={lag!s:<5}  max|dlogit| rows<=t: {d_upto:.2e}  row t: {d_row:.2e}  rows>t: {d_after:.2e}")
        if lag:
            ok &= d_upto < 1e-5 and d_after > 1e-4
        else:
            # Leaky clock: row t moves, and so does every later row, whose
            # distance to rows before t changed too.
            ok &= d_row > 1e-4 and d_after > 1e-4
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
