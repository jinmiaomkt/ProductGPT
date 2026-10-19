"""
Formal tests for the gen-6 streaming hybrid (gen6_streaming/).

Every claim the model's docstring makes is checked here, on random small
models and random histories, in fp32 on the CPU:

  T1  computed state, one pass: equals gen 5's additive_inventory,
      InventorySlots.stock_counts and pity_depth, for every stock option.
  T2  computed state, in pieces (rows split anywhere, single rows included):
      equals the one-pass numbers.
  T3  E1: window and chunk >= sequence length reproduces the gen-5 fuse=stack
      model with the same weights (ALiBi on/off, one-step and stacked
      satiation, pity on/off, campaign effects on).
  T4  E2: every chunk length gives the same logits in eval mode.
  T5  the window: an attention layer's output at occasion i does not depend on
      occasions before i - window + 1, and does depend on those inside it.
  T6  causality: changing occasion k leaves every logit before k bit-identical.
  T7  E3: real prefix + one occasion at a time equals one pass.
  T8  gradients: with one chunk, the chunked training step gives exactly the
      autograd gradient of the full loss; with several chunks, the structural
      stock parameters still receive a finite, non-zero gradient.
  T9  rollout: the streaming generator and gen 5's prefix-recompute loop draw
      the same decisions and acquisitions from the same seed, free running and
      teacher forced.
  T10 eval equals train at dropout 0.
  T11 a truncated batch (non-zero pre-window inventory) is refused.
  T12 the trainer's pieces: the chunk-summed loss equals gen 5's FocalLoss;
      the length-bucketed sampler yields every scored customer exactly once;
      the collate drops only rows after the last scored occasion.

USAGE
    python scripts/test_gen6_streaming.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# CPU only, wherever this runs: the GPU must stay invisible to torch. On
# 19 Oct 2026 GPU work on the laptop crashed the machine three times.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "simulation",
          REPO / "shared"):
    sys.path.insert(0, str(p))

import model_multistream_state_space as mm  # noqa: E402
from model6_streaming import StreamingHybrid  # noqa: E402

FAIL = []
torch.set_num_threads(1)


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append(name)


def feat_table():
    g = torch.Generator().manual_seed(0)
    f = torch.randn(68, 34, generator=g)
    f[:, 0] = 3.0
    f[mm.FIRST_PROD_ID:mm.LAST_PROD_ID + 1, 0] = torch.tensor(
        [5.0 if i % 7 == 0 else (4.0 if i % 3 == 0 else 3.0) for i in range(44)])
    return f


FEAT = feat_table()


def kwargs(**over):
    kw = dict(vocab_size_src=68, vocab_size_tgt=18, max_seq_len=64, ai_rate=15, d_model=32,
              n_layers=2, n_heads=4, d_ff=64, dropout=0.0, feature_tensor=FEAT, num_users=4,
              encoder="gru_attn", fuse="stack", inventory="slots",
              attn_recency_bias=True, attn_time_bias="ordinal", sat_layers=2,
              kernel="attr", decay="exp", decay_init=2.0, decay_freeze=False, tier=True,
              n_campaigns=32, pity_stream=True)
    kw.update(over)
    return kw


def pair(window=0, chunk=10 ** 6, **over):
    torch.manual_seed(0)
    g5 = mm.MultiStreamStateSpaceTransformer(**kwargs(**over))
    g6 = StreamingHybrid(window=window, chunk=chunk, **kwargs(**over))
    g6.load_state_dict(g5.state_dict())
    return g5.eval(), g6.eval()


def history(B=3, S=40, seed=1):
    """Random histories with buys of 1 and 10, empty rows, and one customer
    shorter than the rest (trailing padding)."""
    g = torch.Generator().manual_seed(seed)
    lto = torch.randint(13, 57, (B, S, 4), generator=g)
    obt = torch.zeros(B, S, 10, dtype=torch.long)
    prev = torch.randint(1, 10, (B, S), generator=g)
    for b in range(B):
        for t in range(1, S):
            d = int(prev[b, t])
            n = 0 if d == 9 else (10 if d % 2 == 0 else 1)
            if n:
                obt[b, t, :n] = torch.randint(13, 57, (n,), generator=g)
    camp = torch.arange(S).div(4, rounding_mode="floor").add(1).clamp(max=30).expand(B, S).clone()
    uid = torch.zeros(B, dtype=torch.long)
    lab = torch.randint(1, 10, (B, S), generator=g)
    short = S - 9
    lto[-1, short:] = 0
    obt[-1, short:] = 0
    prev[-1, short:] = 0
    lab[-1, short:] = 0
    camp[-1, short:] = 0
    return lto, obt, prev, uid, camp, lab


def maxdiff(a, b):
    return float((a - b).abs().max())


def t1_t2():
    print("T1/T2 computed state")
    lto, obt, prev, uid, camp, lab = history()
    S = obt.size(1)
    for opts in (dict(tier=False, decay="none", kernel="none"),
                 dict(tier=True, decay="exp", kernel="attr"),
                 dict(tier=True, decay="none", kernel="none"),
                 dict(tier=False, decay="exp", kernel="none"),
                 dict(tier=False, decay="none", kernel="learned")):
        g5, g6 = pair(**opts)
        tag = ",".join(f"{k}={v}" for k, v in opts.items())
        with torch.no_grad():
            c5, s5 = mm.additive_inventory(obt)
            st5 = g5.inventory_slots.stock_counts(obt, c5)
            p5 = mm.pity_depth(obt, prev, g5.rarity_of)
            c6, s6, st6, p6, _ = g6.computed_scan(obt, prev, g6.init_stream(obt.size(0), "cpu"))
            check(f"[{tag}] one pass = gen 5 (count {maxdiff(c5, c6):.1e}, since "
                  f"{maxdiff(s5, s6):.1e}, stock {maxdiff(st5, st6):.1e}, pity {maxdiff(p5, p6):.1e})",
                  max(maxdiff(c5, c6), maxdiff(s5, s6), maxdiff(st5, st6), maxdiff(p5, p6)) < 1e-4)
            st = g6.init_stream(obt.size(0), "cpu")
            parts = []
            for a, b in ((0, 1), (1, 8), (8, 9), (9, 25), (25, S)):
                c, s, k, p, carry = g6.computed_scan(obt[:, a:b], prev[:, a:b], st)
                parts.append((c, s, k, p))
                st = {**st, **carry, "t": b}
            cat = [torch.cat([q[i] for q in parts], dim=1) for i in range(4)]
            d = max(maxdiff(cat[0], c6), maxdiff(cat[1], s6), maxdiff(cat[2], st6), maxdiff(cat[3], p6))
            check(f"[{tag}] in pieces = one pass ({d:.1e})", d < 1e-4)


def t3():
    print("T3 E1: unlimited window, one chunk = gen-5 stack")
    lto, obt, prev, uid, camp, lab = history()
    for over in (dict(), dict(attn_recency_bias=False, attn_time_bias="none"),
                 dict(sat_layers=0), dict(pity_stream=False),
                 dict(tier=False, decay="none", kernel="none")):
        g5, g6 = pair(**over)
        with torch.no_grad():
            a = g5(lto, obt, prev, uid, campaign=camp)
            b = g6(lto, obt, prev, uid, campaign=camp)
        tag = ",".join(f"{k}={v}" for k, v in over.items()) or "incumbent options"
        check(f"[{tag}] max |gen6 - gen5| = {maxdiff(a, b):.1e}", maxdiff(a, b) < 1e-5)


def t4_t6_t10_t11():
    print("T4 E2 chunk invariance / T6 causality / T10 eval=train / T11 refusal")
    lto, obt, prev, uid, camp, lab = history()
    _, g6 = pair(window=5)
    with torch.no_grad():
        ref = g6.stream_forward(lto, obt, prev, uid, camp, chunk=10 ** 6)
        worst = max(maxdiff(ref, g6.stream_forward(lto, obt, prev, uid, camp, chunk=c))
                    for c in (1, 3, 7, 16))
    check(f"window 5: chunks 1/3/7/16 = one chunk (max {worst:.1e})", worst < 1e-5)

    k = 17
    lto2, obt2, prev2 = lto.clone(), obt.clone(), prev.clone()
    lto2[:, k] = torch.tensor([20, 21, 22, 23])
    obt2[:, k, :10] = 30
    prev2[:, k] = 2
    with torch.no_grad():
        a = g6.stream_forward(lto, obt, prev, uid, camp, chunk=7)
        b = g6.stream_forward(lto2, obt2, prev2, uid, camp, chunk=7)
    check(f"occasions < {k} bit-identical ({maxdiff(a[:, :k], b[:, :k]):.1e})",
          maxdiff(a[:, :k], b[:, :k]) == 0.0)
    check(f"occasions >= {k} respond ({maxdiff(a[:, k:], b[:, k:]):.1e})",
          maxdiff(a[:, k:], b[:, k:]) > 1e-5)

    g6.train()
    with torch.no_grad():
        c = g6.stream_forward(lto, obt, prev, uid, camp, chunk=7)
    g6.eval()
    check(f"eval equals train at dropout 0 ({maxdiff(a, c):.1e})", maxdiff(a, c) < 1e-5)

    try:
        g6(lto, obt, prev, uid, campaign=camp, inv_init_count=torch.ones(3, 44))
        check("truncated batch refused", False)
    except ValueError:
        check("truncated batch refused", True)


def t5():
    print("T5 the window")
    W = 6
    _, g6 = pair(window=W)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 30, 32, generator=g)
    j = 9
    x2 = x.clone()
    # a random change: a constant shift would be erased by the layer's LayerNorm
    x2[:, j] += torch.randn(2, 32, generator=g)
    with torch.no_grad():
        st = g6.init_stream(2, "cpu")
        a, _, _ = g6._local_attn(0, x, st, 0)
        b, _, _ = g6._local_attn(0, x2, st, 0)
    inside = maxdiff(a[:, j:j + W], b[:, j:j + W])
    outside = maxdiff(a[:, j + W:], b[:, j + W:])
    before = maxdiff(a[:, :j], b[:, :j])
    check(f"rows j..j+W-1 respond ({inside:.1e})", inside > 1e-5)
    check(f"rows >= j+W unaffected ({outside:.1e})", outside == 0.0)
    check(f"rows < j unaffected ({before:.1e})", before == 0.0)


def t7():
    print("T7 E3: prefix + one occasion at a time = one pass")
    lto, obt, prev, uid, camp, lab = history()
    S = lto.size(1)
    for W in (0, 5):
        _, g6 = pair(window=W, chunk=8)
        with torch.no_grad():
            ref = g6.stream_forward(lto, obt, prev, uid, camp)
            st = g6.init_stream(lto.size(0), "cpu")
            outs = []
            lg, st = g6.advance(lto[:, :13], obt[:, :13], prev[:, :13], st, uid, camp[:, :13])
            outs.append(lg)
            for t in range(13, S):
                lg, st = g6.advance(lto[:, t:t + 1], obt[:, t:t + 1], prev[:, t:t + 1], st, uid,
                                    camp[:, t:t + 1])
                outs.append(lg)
            got = torch.cat(outs, dim=1)
        check(f"window {W or 'all'}: step-by-step = one pass ({maxdiff(ref, got):.1e})",
              maxdiff(ref, got) < 1e-5)


def t8():
    print("T8 gradients")
    import train6_streaming as t6
    lto, obt, prev, uid, camp, lab = history()
    w = torch.ones(18)
    w[1:10] = torch.linspace(0.5, 1.5, 9)
    for chunk in (10 ** 6, 7):
        _, g6 = pair(window=5, chunk=chunk)
        g6.train()
        n_valid = int((lab != 0).sum())

        def cb(c0, c1, logits):
            (t6.weighted_ce_sum(logits, lab[:, c0:c1], w) / n_valid).backward()

        g6.zero_grad()
        g6.stream_forward(lto, obt, prev, uid, camp, chunk=chunk, on_chunk=cb)
        got = {n: p.grad.clone() for n, p in g6.named_parameters() if p.grad is not None}
        struct = [n for n in got if any(s in n for s in ("tier_raw", "attr_w", "attr_self",
                                                          "half_life_raw"))]
        if chunk >= lto.size(1):
            g6.zero_grad()
            full = g6.stream_forward(lto, obt, prev, uid, camp, chunk=chunk)
            (t6.weighted_ce_sum(full, lab, w) / n_valid).backward()
            worst = max(maxdiff(got[n], p.grad) for n, p in g6.named_parameters()
                        if p.grad is not None)
            check(f"one chunk: chunked step = full autograd gradient (max {worst:.1e})", worst < 1e-5)
        else:
            ok = all(bool(torch.isfinite(got[n]).all()) and float(got[n].abs().sum()) > 0
                     for n in struct)
            check(f"chunk {chunk}: stock parameters get finite non-zero gradients "
                  f"({', '.join(struct)})", ok and len(struct) >= 3)


class ToyEnv:
    """Deterministic given the rng: what each decision yields."""

    def step(self, y, offer, state, rng):
        n = 0 if y == 9 else (10 if y % 2 == 0 else 1)
        out = np.zeros(10, dtype=np.int64)
        if n:
            out[:n] = rng.integers(13, 57, n)
        return out

    def observe(self, *a, **k):
        return None


def t9():
    print("T9 rollout: streaming = prefix recompute, same seed")
    import rollout as ro
    lto, obt, prev, uid, camp, lab = history(B=3, S=36, seed=5)
    lab[:, :20] = 0
    batch = {"lto": lto, "obtained": obt, "prev_decision": prev, "label": lab,
             "user_idx": uid, "campaign": camp}
    _, g6 = pair(window=6, chunk=8)
    for tf in (False, True):
        g6.streaming = False                        # gen 5's loop, full prefix each step
        a = ro.rollout(g6, batch, ToyEnv(), None, start=20, n_rep=2, seed=11, teacher_forced=tf)
        g6.streaming = True
        b = ro.rollout(g6, batch, ToyEnv(), None, start=20, n_rep=2, seed=11, teacher_forced=tf)
        same = (np.array_equal(a["decisions"], b["decisions"])
                and np.array_equal(a["obtained"], b["obtained"]))
        n_gen = int((a["decisions"] > 0).sum())
        check(f"{'teacher forced' if tf else 'free running'}: identical draws "
              f"({n_gen} generated occasions)", same and n_gen > 0)


def t12():
    print("T12 trainer pieces")
    import train5_multistream as t5
    import train6_streaming as t6
    g = torch.Generator().manual_seed(4)
    logits = torch.randn(3, 20, 18, generator=g)
    tgt = torch.randint(0, 10, (3, 20), generator=g)
    w = torch.ones(18)
    w[1:10] = torch.linspace(0.5, 2.0, 9)
    for gamma in (0.0, 2.0):
        ref = t5.FocalLoss(gamma, 0, w)(logits, tgt)
        n = int((tgt != 0).sum())
        got = (t6.weighted_ce_sum(logits[:, :7], tgt[:, :7], w, gamma)
               + t6.weighted_ce_sum(logits[:, 7:], tgt[:, 7:], w, gamma)) / n
        check(f"gamma={gamma}: chunk-summed loss = gen-5 FocalLoss ({abs(float(ref - got)):.1e})",
              abs(float(ref - got)) < 1e-6)

    lengths = [0, 5, 9, 1, 0, 30, 12, 7, 7, 3]
    seen = []
    smp = t6.LengthBucketSampler(lengths, 3, shuffle=True, seed=1, pool=2)
    for ep in range(2):
        smp.set_epoch(ep)
        got = sorted(i for b in smp for i in b)
        seen.append(got)
    want = sorted(i for i, n in enumerate(lengths) if n > 0)
    check("sampler yields every scored customer once per epoch, skips the rest",
          all(s == want for s in seen))
    ev = t6.LengthBucketSampler(lengths, 3, shuffle=False)
    flat = [lengths[i] for b in ev for i in b]
    check("evaluation batches are sorted by length", flat == sorted(flat))

    def item(n_rows, last_scored):
        lab = torch.zeros(n_rows, dtype=torch.long)
        lab[last_scored] = 3
        return {"lto": torch.ones(n_rows, 4, dtype=torch.long),
                "obtained": torch.ones(n_rows, 10, dtype=torch.long),
                "prev_decision": torch.ones(n_rows, dtype=torch.long), "label": lab,
                "user_id": torch.tensor(0), "uid": "u",
                "campaign": torch.arange(n_rows), "inv_init_count": torch.zeros(44),
                "inv_init_last": torch.zeros(44)}
    b = t6.collate_scored([item(20, 11), item(15, 6)])
    check("collate keeps rows up to the batch's last scored occasion "
          f"(got {b['lto'].shape[1]} of 20)", b["lto"].shape[1] == 12 and b["label"].shape[1] == 12
          and b["campaign"].shape[1] == 12 and b["inv_init_count"].shape == (2, 44))


def main() -> None:
    t1_t2()
    t3()
    t4_t6_t10_t11()
    t5()
    t7()
    t8()
    t9()
    t12()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all gen-6 checks passed")


if __name__ == "__main__":
    main()
