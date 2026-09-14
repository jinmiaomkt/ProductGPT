"""
How much does a row's OWN inter-purchase time reveal about its own label?

WHY THIS EXISTS
---------------
Batch 5 (R20) swapped the transformer's ordinal recency bias for one built on
elapsed hours: penalty = s_h * log1p(t_i - t_j), with t = cumsum(IPT). Holdout
NLL fell from 0.886 (ordinal) to 0.706 -- a 0.18-nat jump, larger than every
architectural change so far combined, and present in the calibration period
too. That pattern suggests leakage, not recency.

The suspected channel: IPT at row t is the gap ENDING at row t. Query t's bias
toward key t-1 is -s * log1p(IPT_t), so the attention pattern at t encodes
IPT_t. But rows exist because of outcomes -- a draw creates a row, a quiet day
creates an inserted NotBuy at the 24-hour mark -- so the current gap may say
which kind of row t is before the model predicts it.

This script measures that channel directly, with no model:
    H(Y_t)               marginal label entropy
    H(Y_t | bin(IPT_t))  label entropy given the row's own gap   (leaky)
    H(Y_t | bin(IPT_t-1)) given the PREVIOUS row's gap           (legitimate)
plus the hit rate of predicting the modal label within each bin. If the own-gap
information gain is large and the lagged one is small, the time bias as built
can read the label off the clock, and R20 is not evidence for recency.

Reports ONLY aggregate statistics (entropies, shares, counts per bin). Never
uids, token values, or per-user records -- see CLAUDE.md.

USAGE
    python scripts/ipt_leak_check.py
    python scripts/ipt_leak_check.py --max-users 1000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
from dataset_multistream import load_json_dataset, parse_floats, parse_token_ids  # noqa: E402

# Bin edges in hours. Fine near zero (bursts) and around 24 h (the insertion
# interval), coarse elsewhere.
EDGES = [0.0, 1e-9, 0.02, 0.1, 0.5, 1, 2, 4, 8, 12, 20, 23.5, 24.5, 30, 48,
         72, 168, 1e9]


def bin_of(h: float) -> int:
    for k in range(len(EDGES) - 1):
        if h < EDGES[k + 1]:
            return k
    return len(EDGES) - 2


def entropy(c: Counter) -> float:
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values() if v)


def cond(table: dict) -> tuple:
    n = sum(sum(c.values()) for c in table.values())
    h = sum(sum(c.values()) / n * entropy(c) for c in table.values())
    hit = sum(max(c.values()) for c in table.values()) / n
    return h, hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-file", default="clean_list_int_wide4_simple6_IPT.json")
    ap.add_argument("--max-users", type=int, default=0)
    args = ap.parse_args()

    root = os.environ.get("PRODUCTGPT_DATA")
    if not root:
        sys.exit("PRODUCTGPT_DATA is not set")
    recs = load_json_dataset(str(Path(root) / args.data_file))
    if args.max_users:
        recs = recs[: args.max_users]

    marg = Counter()
    own = defaultdict(Counter)
    lag = defaultdict(Counter)
    own_bin_n = Counter()
    for rec in recs:
        dec = parse_token_ids(rec.get("Decision"))
        ipt = parse_floats(rec.get("IPT"))
        n = min(len(dec), len(ipt))
        for t in range(n):
            y = dec[t]
            if y < 1 or y > 9:
                continue
            g = ipt[t] if ipt[t] is not None and ipt[t] >= 0 else 0.0
            gp = ipt[t - 1] if t > 0 and ipt[t - 1] is not None and ipt[t - 1] >= 0 else -1.0
            marg[y] += 1
            own[bin_of(g)][y] += 1
            lag[bin_of(gp) if gp >= 0 else -1][y] += 1
            own_bin_n[bin_of(g)] += 1

    N = sum(marg.values())
    h0 = entropy(marg)
    hit0 = max(marg.values()) / N
    h_own, hit_own = cond(own)
    h_lag, hit_lag = cond(lag)
    print(f"users={len(recs):,}  labelled rows={N:,}")
    print(f"{'conditioning':<28}{'H(Y|.) nats':>12}{'gain':>8}{'modal hit':>11}")
    print(f"{'none (marginal)':<28}{h0:>12.4f}{0:>8.4f}{hit0:>11.4f}")
    print(f"{'own gap  IPT_t   (leaky)':<28}{h_own:>12.4f}{h0 - h_own:>8.4f}{hit_own:>11.4f}")
    print(f"{'prev gap IPT_t-1 (legit)':<28}{h_lag:>12.4f}{h0 - h_lag:>8.4f}{hit_lag:>11.4f}")

    print("\nper own-gap bin: share of rows, P(NotBuy), P(10-pull)")
    for k in sorted(own):
        c = own[k]
        n = sum(c.values())
        lo, hi = EDGES[k], EDGES[k + 1]
        ten = sum(c[y] for y in (2, 4, 6, 8)) / n
        print(f"  [{lo:>7.2f}, {hi:>9.2f}) h  share={n / N:6.3f}  "
              f"P(9)={c[9] / n:6.3f}  P(10x)={ten:6.3f}")


if __name__ == "__main__":
    main()
