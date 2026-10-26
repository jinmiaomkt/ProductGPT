"""
Tests for R45's controlled left-censoring (gen6_streaming/censor6.py and the
prefix / censoring-flag paths in model6_streaming.py). CPU only.

  C1  chop points: calendar drops exactly the first k active campaigns, count
      the first d occasions, and neither cuts into campaign val_from or later.
  C2  state="zero": the remaining rows are the full history's rows [s:],
      untouched, and there is no prefix.
  C3  state="true": with the left-padded prefix, the computed state of the
      remaining rows (count, since, stock, pity) equals the full history's
      rows [s:] exactly, in a batch whose customers have different chop points.
  C4  state="labels": inputs untouched, labels before s masked, nothing flagged.
  C5  the censoring flag is inert at initialisation (equals a flag-free model
      with the same weights), never touches uncensored customers, and moves
      censored ones once its parameters are non-zero.
  C6  gradients: the structural stock parameters get finite, non-zero gradients
      through the prefix.
  C7  no chop: ChopView with mode "none" returns the item unchanged.

    python scripts/test_r45_censor.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "shared"):
    sys.path.insert(0, str(p))

import model_multistream_state_space as mm  # noqa: E402
from censor6 import ChopView, chop_index, collate_prefix  # noqa: E402
from dataset_multistream import collate_multistream  # noqa: E402
from model6_streaming import StreamingHybrid  # noqa: E402

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
    f[13:57, 0] = torch.tensor([5.0 if i % 7 == 0 else (4.0 if i % 3 == 0 else 3.0)
                                for i in range(44)])
    return f


def model(flag=False):
    torch.manual_seed(0)
    return StreamingHybrid(
        window=5, chunk=7, censor_flag=flag, vocab_size_src=68, vocab_size_tgt=18,
        max_seq_len=0, ai_rate=15, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0,
        feature_tensor=feat(), num_users=4, encoder="gru_attn", fuse="stack",
        inventory="slots", attn_recency_bias=True, attn_time_bias="ordinal", sat_layers=2,
        kernel="attr", decay="exp", decay_init=2.0, decay_freeze=False, tier=True,
        n_campaigns=32, pity_stream=True).eval()


def item(S, seed, camp_len=6):
    """One synthetic customer: buys of 1 and 10, empty rows, campaigns in runs of
    camp_len rows starting at campaign 3, labels everywhere."""
    g = torch.Generator().manual_seed(seed)
    prev = torch.randint(1, 10, (S,), generator=g)
    obt = torch.zeros(S, 10, dtype=torch.long)
    for t in range(1, S):
        d = int(prev[t])
        n = 0 if d == 9 else (10 if d % 2 == 0 else 1)
        if n:
            obt[t, :n] = torch.randint(13, 57, (n,), generator=g)
    return {"lto": torch.randint(13, 57, (S, 4), generator=g), "obtained": obt,
            "prev_decision": prev, "label": torch.randint(1, 10, (S,), generator=g),
            "campaign": 3 + torch.arange(S) // camp_len, "uid": f"u{seed}",
            "user_id": torch.tensor(0), "inv_init_count": torch.zeros(44),
            "inv_init_last": torch.full((44,), -1e6)}


def c1():
    print("C1 chop points")
    camp = torch.tensor([3, 3, 3, 5, 5, 6, 6, 6, 27, 27, 28])
    check("calendar 1 drops campaign 3", chop_index(camp, "calendar", 1, 27) == 3)
    check("calendar 2 drops campaigns 3 and 5", chop_index(camp, "calendar", 2, 27) == 5)
    check("calendar 9 stops at campaign 27", chop_index(camp, "calendar", 9, 27) == 8)
    check("count 4 drops 4 rows", chop_index(camp, "count", 4, 27) == 4)
    check("count 100 stops at campaign 27", chop_index(camp, "count", 100, 27) == 8)
    check("depth 0 and mode none drop nothing",
          chop_index(camp, "count", 0, 27) == 0 and chop_index(camp, "none", 5, 27) == 0)


def c2_c4_c7():
    print("C2/C4/C7 views")
    base = [item(40, 1), item(31, 2)]
    full = base[0]
    z = ChopView(base, "calendar", 2, "zero", 27)[0]
    s = int(z["chopped"])
    same = all(torch.equal(z[k], full[k][s:]) for k in ("lto", "obtained", "prev_decision",
                                                         "label", "campaign"))
    check(f"zero: rows are the full rows [{s}:], no prefix, flagged",
          s == 12 and same and "prefix_obtained" not in z and int(z["censored"]) == 1)
    lab = ChopView(base, "count", 9, "labels", 27)[0]
    check("labels: inputs untouched, first 9 labels masked, not flagged",
          torch.equal(lab["obtained"], full["obtained"]) and bool((lab["label"][:9] == 0).all())
          and torch.equal(lab["label"][9:], full["label"][9:]) and int(lab["censored"]) == 0)
    n = ChopView(base, "none", 0, "zero", 27)[0]
    check("mode none: item unchanged", all(torch.equal(n[k], full[k]) for k in
                                           ("lto", "obtained", "prev_decision", "label")))


def c3():
    print("C3 true state through a left-padded prefix")
    m = model()
    base = [item(40, 1), item(33, 2), item(26, 3)]
    depth = [2, 1, 3]
    views = [ChopView([b], "calendar", d, "true", 27)[0] for b, d in zip(base, depth)]
    batch = collate_prefix(views, collate_multistream(views))
    with torch.no_grad():
        st0 = m.init_stream(3, "cpu")
        Lp = batch["prefix_obtained"].size(1)
        _, _, _, _, carry = m.computed_scan(batch["prefix_obtained"], batch["prefix_prev"],
                                            {**st0, "t": -Lp})
        c, si, sk, p, _ = m.computed_scan(batch["obtained"], batch["prev_decision"],
                                          {**st0, **carry})
        worst = 0.0
        for b, (it, v) in enumerate(zip(base, views)):
            s, n = int(v["chopped"]), int(v["obtained"].size(0))
            fc, fs, fk, fp, _ = m.computed_scan(it["obtained"][None], it["prev_decision"][None],
                                                m.init_stream(1, "cpu"))
            for got, ref in ((c, fc), (si, fs), (sk, fk), (p, fp)):
                worst = max(worst, float((got[b, :n] - ref[0, s:s + n]).abs().max()))
    check(f"count, since, stock and pity equal the full history's rows [s:] "
          f"(chops {[int(v['chopped']) for v in views]}, max diff {worst:.1e})", worst < 1e-4)


def c5():
    print("C5 censoring flag")
    m0, m1 = model(False), model(True)
    sd = m0.state_dict()
    missing = m1.load_state_dict(sd, strict=False)
    batch = collate_multistream([item(30, 4), item(30, 5)])
    args = (batch["lto"], batch["obtained"], batch["prev_decision"], batch["user_id"])
    cens = torch.tensor([1, 0])
    with torch.no_grad():
        a = m0(*args)
        b = m1(*args, censored=cens)
        check(f"inert at initialisation (max diff {float((a - b).abs().max()):.1e}); only "
              f"{sorted(missing.missing_keys)} are new", float((a - b).abs().max()) < 1e-6
              and sorted(missing.missing_keys) == ["censor_h0", "censor_vec"])
        m1.censor_vec.fill_(0.5)
        m1.censor_h0.fill_(0.3)
        c = m1(*args, censored=cens)
    check(f"uncensored customer unchanged ({float((c[1] - a[1]).abs().max()):.1e})",
          float((c[1] - a[1]).abs().max()) < 1e-6)
    check(f"censored customer moves ({float((c[0] - a[0]).abs().max()):.1e})",
          float((c[0] - a[0]).abs().max()) > 1e-4)


def c6():
    print("C6 gradients through the prefix")
    m = model()
    m.train()
    views = [ChopView([item(40, 1)], "calendar", 2, "true", 27)[0],
             ChopView([item(35, 2)], "calendar", 1, "true", 27)[0]]
    b = collate_prefix(views, collate_multistream(views))
    m.zero_grad()

    def cb(c0, c1, logits):
        logits.float().pow(2).mean().backward()

    m.stream_forward(b["lto"], b["obtained"], b["prev_decision"], b["user_id"],
                     prefix_obtained=b["prefix_obtained"], prefix_prev=b["prefix_prev"],
                     on_chunk=cb)
    g = {n: p.grad for n, p in m.named_parameters()
         if any(s in n for s in ("tier_raw", "attr_w", "half_life_raw")) and p.grad is not None}
    check(f"stock parameters get finite non-zero gradients ({', '.join(g)})",
          len(g) == 3 and all(bool(torch.isfinite(v).all()) and float(v.abs().sum()) > 0
                              for v in g.values()))


def main() -> None:
    c1()
    c2_c4_c7()
    c3()
    c5()
    c6()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all R45 censoring checks passed")


if __name__ == "__main__":
    main()
