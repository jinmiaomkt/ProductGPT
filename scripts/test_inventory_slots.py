"""
Formal tests for the additive inventory slots (R23) and satiation blocks (R24).

1. Additivity: counts equal the number of acquisitions so far, never decrease,
   and do not change on rows with an empty obtained block.
2. Lag / causality: the obtained block at row t carries o_(t-1). Perturbing it
   may move predictions at rows >= t but must leave rows < t bit-identical --
   for one attention step and for stacked blocks, on both encoders.
3. Eval equals train with dropout 0 (the fast-path trap of R20).
4. Loader continuity (needs PRODUCTGPT_DATA): with a truncated window, the
   pre-window counts plus the in-window running sum equal the count over the
   customer's full history. Reports only pass/fail and how many customers were
   checked -- no data values.

USAGE
    python scripts/test_inventory_slots.py            # 1-3, synthetic
    python scripts/test_inventory_slots.py --data     # also 4, on real data
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_multistream_state_space import additive_inventory, build_transformer  # noqa: E402

FAIL = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append(name)


def make(encoder: str, sat_layers: int):
    torch.manual_seed(0)
    return build_transformer(
        vocab_size_src=68, vocab_size_tgt=18, max_seq_len=32, d_model=32, n_layers=2,
        n_heads=4, d_ff=64, dropout=0.0, feature_tensor=torch.randn(68, 34), ai_rate=15,
        num_users=4, lto_len=4, obtained_len=10, prev_dec_len=1, encoder=encoder,
        attn_recency_bias=(encoder == "transformer"), attn_time_bias="ordinal" if encoder == "transformer" else "none",
        inventory="slots", sat_layers=sat_layers)


def synthetic():
    g = torch.Generator().manual_seed(1)
    B, S = 2, 20
    lto = torch.randint(13, 57, (B, S, 4), generator=g)
    obt = torch.randint(13, 57, (B, S, 10), generator=g)
    obt[:, ::3] = 0                     # empty rows, like inserted NotBuy days
    obt[:, 0] = 0                       # the R1 shift: row 0 has no predecessor
    prev = torch.randint(1, 10, (B, S), generator=g)
    uid = torch.zeros(B, dtype=torch.long)
    init_c = torch.zeros(B, 44)
    init_c[:, 5] = 3.0
    init_l = torch.full((B, 44), -1.0e6)
    init_l[:, 5] = -7.0
    return lto, obt, prev, uid, init_c, init_l


def main() -> None:
    lto, obt, prev, uid, init_c, init_l = synthetic()
    B, S = obt.shape[:2]

    print("1. additivity")
    count, since = additive_inventory(obt, init_c, init_l)
    brute = torch.zeros(B, S, 44)
    for b in range(B):
        run = init_c[b].clone()
        for t in range(S):
            for x in obt[b, t].tolist():
                if 13 <= x <= 56:
                    run[x - 13] += 1
            brute[b, t] = run
    check("counts equal a brute-force running tally", torch.equal(count, brute))
    check("counts never decrease", bool((count[:, 1:] >= count[:, :-1]).all()))
    empty = (obt == 0).all(-1)
    same = (count[:, 1:] == count[:, :-1]).all(-1)
    check("empty rows leave counts unchanged", bool(same[empty[:, 1:]].all()))
    reacquired = bool((obt[0, 1:3] == 18).any())      # product index 5 is token 18
    check("pre-window acquisition is remembered (count 3, 7 rows ago at row 0)",
          float(count[0, 0, 5]) == 3.0 and float(since[0, 0, 5]) == 7.0)
    check("recency keeps growing until the product is acquired again",
          reacquired or float(since[0, 2, 5]) == 9.0)

    t = 9
    obt2 = obt.clone()
    obt2[:, t] = torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    for encoder in ("transformer", "gru"):
        for L in (0, 2):
            print(f"2-3. encoder={encoder} sat_layers={L}")
            m = make(encoder, L).eval()
            kw = dict(inv_init_count=init_c, inv_init_last=init_l)
            with torch.no_grad():
                a = m(lto, obt, prev, uid, None, **kw)
                b_ = m(lto, obt2, prev, uid, None, **kw)
                m.train()
                c = m(lto, obt, prev, uid, None, **kw)
                m.eval()
            before = (a[:, :t] - b_[:, :t]).abs().max().item()
            after = (a[:, t:] - b_[:, t:]).abs().max().item()
            check(f"rows < t unchanged by o_(t-1) (max diff {before:.1e})", before == 0.0)
            check(f"rows >= t respond (max diff {after:.1e})", after > 1e-5)
            check(f"eval equals train, dropout 0 (max diff {(a - c).abs().max().item():.1e})",
                  (a - c).abs().max().item() < 1e-5)
            check("no NaN", bool(torch.isfinite(a).all()))

    if "--data" in sys.argv:
        print("4. loader continuity on real data")
        import config5
        from dataset_multistream import TransformerDataset, load_json_dataset, parse_token_ids
        cfg = config5.get_config("pilot")
        raw = load_json_dataset(str(config5.data_path(cfg)))[:200]
        ds = TransformerDataset(raw, ai_rate=15, lto_len=4, obtained_len=10, prev_dec_len=1,
                                max_events=128, base_seed=0, shift_obtained=True, truncate="tail")
        checked = mismatched = 0
        for i, rec in enumerate(raw):
            ai = parse_token_ids(rec["AggregateInput"])
            n_full = min(len(ai) // 15, len(parse_token_ids(rec.get("Decision", []))) or 10**9)
            if n_full <= 128:
                continue
            item = ds[i]
            cnt, _ = additive_inventory(item["obtained"][None], item["inv_init_count"][None],
                                        item["inv_init_last"][None])
            full = torch.zeros(44)
            for r in range(n_full - 1):          # the last row's o is never an input
                for x in ai[r * 15 + 4: r * 15 + 14]:
                    if 13 <= x <= 56:
                        full[x - 13] += 1
            checked += 1
            mismatched += int(not torch.equal(cnt[0, -1], full))
        check(f"window counts equal full-history counts ({checked} truncated customers)",
              checked > 0 and mismatched == 0)

    print("PASS" if not FAIL else f"FAIL ({len(FAIL)})")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
