"""
Characterise the event stream of an IPT data file, to decide whether a
continuous-time (marked point process) formulation is feasible and what it
would cost.

WHY THIS EXISTS
---------------
The current model is a DISCRETE-OCCASION model: the R generator inserts one
synthetic NotBuy (decision 9) row at the end of every 24-hour interval with no
draw, and the model predicts a 9-way categorical outcome per row. A
CONTINUOUS-TIME model would drop those synthetic rows entirely, keep only real
draws, and predict (time-to-next-event, 8-way mark) instead; silence would be
carried by a survival term rather than by synthetic observations.

Before building that, four things need measuring:

  1. Are the synthetic rows cleanly separable? Specifically, is every
     IsInserted==1 row decision 9, and how many decision-9 rows are REAL?
     If real NotBuy rows exist in quantity, the mark space is not simply
     "classes 1-8" and the plan needs adjusting.

  2. How much shorter do sequences get? Attention memory grows as S^2, and
     S=1536 already OOMs on the HPCC GPU while the IPT cap is 2048. Dropping
     inserted rows can only shorten sequences; this reports by how much, and
     what S would cover 90/95/99/100% of users under each representation.

  3. What does the gap distribution look like? The claim that the event index
     "mixes two clocks" (bursts of draws minutes apart, then weeks of silence)
     is testable, and the gap percentiles are also the input to the
     predicted-vs-empirical CDF figure that distinguishes the two models.

  4. Is the campaign 28-30 holdout adequately powered? The proposed comparison
     scores both models on simulated trajectories through the holdout
     campaigns. If few real draws happen there, the comparison is underpowered
     regardless of which model wins.

Reports ONLY aggregate statistics -- counts, shares, percentiles, contingency
tables. Never uids, never token values, never per-user records. Per CLAUDE.md,
data contents must not be read, printed, or copied into responses or files.

USAGE
    python scripts/measure_event_stream.py
    python scripts/measure_event_stream.py --max-users 500        # quick pass
    python scripts/measure_event_stream.py --data-file clean_list_int_wide4_simple6.json
    python scripts/measure_event_stream.py --out results/stream_stats.json

Requires PRODUCTGPT_DATA. Pure Python + numpy: no torch, no GPU, runs on the
laptop and on HPCC unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import paths
from dataset_multistream import load_json_dataset, parse_floats, parse_token_ids

# Revenue per decision class, matching rev_vec in every trainer.
REV = {1: 1.0, 2: 10.0, 3: 1.0, 4: 10.0, 5: 1.0, 6: 10.0, 7: 1.0, 8: 10.0, 9: 0.0}
NOTBUY = 9
CURRENT_CAP = 2048          # max_events for the IPT files
GAP_BUCKETS = [             # hours; upper bound of each bucket
    ("exactly 0", 0.0),     # must be first: atoms at 0 break a log-gap density
    ("<1 min", 1 / 60),
    ("1-10 min", 10 / 60),
    ("10-60 min", 1.0),
    ("1-6 h", 6.0),
    ("6-24 h", 24.0),
    ("1-7 d", 168.0),
    (">7 d", float("inf")),
]
PCTS = [1, 5, 25, 50, 75, 90, 95, 99]


def resolution(a: np.ndarray) -> Dict[str, Any]:
    """
    Infer the recording resolution of a gap field.

    Matters because a large mass of exactly-zero gaps can mean two very
    different things: genuinely simultaneous events (a modelling problem), or
    a field rounded so coarsely that distinct events collapse onto the same
    value (a data-generation problem, fixable upstream). If the smallest
    non-zero gap is q and most gaps are multiples of q, the field is quantised
    at q and the zeros are rounding, not simultaneity.
    """
    nz = a[a > 0]
    if nz.size < 100:
        return {}
    q = float(nz.min())
    if q <= 0:
        return {}
    mult = nz / q
    on_grid = float(np.isclose(mult, np.round(mult), atol=1e-6).mean())
    return {
        "smallest_nonzero_gap_hours": round(q, 6),
        "smallest_nonzero_gap_seconds": round(q * 3600, 3),
        "share_of_nonzero_gaps_on_that_grid": round(on_grid, 4),
        "n_distinct_values_below_1h": int(np.unique(nz[nz < 1.0]).size),
    }


def bucket(a: np.ndarray) -> Dict[str, Any]:
    """Bucket a gap array, with an explicit exactly-zero bucket first."""
    if a.size == 0:
        return {}
    out: Dict[str, Any] = {}
    lo = -1.0
    for name, hi in GAP_BUCKETS:
        if name == "exactly 0":
            k = int((a <= 0).sum())
        elif np.isfinite(hi):
            k = int(((a > lo) & (a <= hi)).sum())
        else:
            k = int((a > lo).sum())
        out[name] = {"n": k, "share": round(k / a.size, 4)}
        lo = max(hi, 0.0)
    return out


def pct_summary(a: np.ndarray) -> Dict[str, float]:
    """Percentile summary of a 1-D array; empty-safe."""
    if a.size == 0:
        return {}
    out = {f"p{p}": float(np.percentile(a, p)) for p in PCTS}
    out.update(mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
               n=int(a.size))
    return out


def coverage(lengths: np.ndarray, fracs=(0.50, 0.90, 0.95, 0.99, 1.00)) -> Dict[str, int]:
    """Smallest S that covers each fraction of users without truncation."""
    if lengths.size == 0:
        return {}
    return {f"{int(f * 100)}%": int(np.ceil(np.percentile(lengths, f * 100)))
            for f in fracs}


def analyse(records: List[Dict[str, Any]], holdout_from: int) -> Dict[str, Any]:
    have_ins = have_ipt = have_camp = False

    # cross-tab: IsInserted x decision, as plain counts
    xtab: Dict[int, Counter] = {0: Counter(), 1: Counter()}
    all_decisions = Counter()

    len_total: List[int] = []       # rows per user, current representation
    len_real: List[int] = []        # rows per user, continuous representation
    gaps_raw: List[float] = []      # IPT as stored, on real events
    gaps_true: List[float] = []     # reconstructed real-draw -> real-draw time
    gaps_inserted: List[float] = []  # IPT as stored, on inserted rows
    n_inserted_between: List[int] = []  # inserted rows separating two real draws

    # self-excitation probe: (previous gap, next gap) pairs on real events
    prev_gap: List[float] = []
    next_gap: List[float] = []

    per_campaign_real = Counter()
    per_campaign_inserted = Counter()
    holdout_real_users = 0
    holdout_real_events = 0
    holdout_revenue = 0.0
    users_with_no_real = 0

    for rec in records:
        dec = parse_token_ids(rec.get("Decision"))
        if not dec:
            continue

        ins = parse_token_ids(rec.get("IsInserted")) if "IsInserted" in rec else []
        ipt = parse_floats(rec.get("IPT")) if "IPT" in rec else []
        camp = parse_token_ids(rec.get("CampaignID")) if "CampaignID" in rec else []
        have_ins = have_ins or bool(ins)
        have_ipt = have_ipt or bool(ipt)
        have_camp = have_camp or bool(camp)

        # Fields are written independently by the R generator and can differ in
        # length; align to the shortest so every index is defined everywhere.
        n = len(dec)
        for other in (ins, ipt, camp):
            if other:
                n = min(n, len(other))

        n_total = n_real = 0
        user_prev_real_gap: float | None = None
        user_holdout_events = 0

        # IPT is "hours since the PREVIOUS ROW", and inserted rows are rows.
        # So during a long silence the stored IPT on the next real draw is only
        # the time since the last synthetic marker, not since the last real
        # draw. Accumulate across intervening rows to recover the true
        # inter-purchase time that a point process needs.
        acc = 0.0
        acc_rows = 0
        seen_real = False

        for t in range(n):
            y = dec[t]
            if not (1 <= y <= 9):        # PAD / UNK / SOS are not decisions
                continue
            n_total += 1
            all_decisions[y] += 1

            is_ins = int(bool(ins[t])) if ins else 0
            xtab[is_ins][y] += 1

            g = float(ipt[t]) if ipt else None
            c = int(camp[t]) if camp else None

            if g is not None and g >= 0:
                acc += g
            acc_rows += 1

            if is_ins:
                if g is not None and g >= 0:
                    gaps_inserted.append(g)
                if c is not None:
                    per_campaign_inserted[c] += 1
                continue

            # ---- real event ----
            n_real += 1
            if g is not None and g >= 0:
                gaps_raw.append(g)
                if seen_real:
                    gaps_true.append(acc)
                    n_inserted_between.append(acc_rows - 1)
                    if user_prev_real_gap is not None:
                        prev_gap.append(user_prev_real_gap)
                        next_gap.append(acc)
                    user_prev_real_gap = acc
                seen_real = True
            acc = 0.0
            acc_rows = 0
            if c is not None:
                per_campaign_real[c] += 1
                if c >= holdout_from:
                    user_holdout_events += 1
                    holdout_revenue += REV.get(y, 0.0)

        if n_total:
            len_total.append(n_total)
            len_real.append(n_real)
            if n_real == 0:
                users_with_no_real += 1
            if user_holdout_events:
                holdout_real_users += 1
                holdout_real_events += user_holdout_events

    lt = np.asarray(len_total, dtype=np.int64)
    lr = np.asarray(len_real, dtype=np.int64)
    graw = np.asarray(gaps_raw, dtype=np.float64)
    gtrue = np.asarray(gaps_true, dtype=np.float64)
    gins = np.asarray(gaps_inserted, dtype=np.float64)
    nbet = np.asarray(n_inserted_between, dtype=np.float64)

    # self-excitation: median next gap after a short vs a long previous gap
    # Medians are useless here when the field is quantised and ~half the gaps
    # round to zero, so compare the PROBABILITY that the next gap is short.
    # That statistic survives the rounding.
    excite: Dict[str, Any] = {}
    if len(prev_gap) >= 100:
        pg = np.asarray(prev_gap); ng = np.asarray(next_gap)
        short_thr = 1 / 60          # 1 minute
        long_thr = 24.0             # 1 day
        a_ = pg <= short_thr
        b_ = pg >= long_thr
        if a_.any() and b_.any():
            p_a = float((ng[a_] <= short_thr).mean())
            p_b = float((ng[b_] <= short_thr).mean())
            excite = {
                "short_gap_threshold_hours": round(short_thr, 5),
                "long_gap_threshold_hours": long_thr,
                "P(next short | prev short)": round(p_a, 4),
                "P(next short | prev long)": round(p_b, 4),
                "risk_ratio": round(p_a / p_b, 2) if p_b > 0 else None,
                "n_prev_short": int(a_.sum()),
                "n_prev_long": int(b_.sum()),
                "n_pairs": int(pg.size),
            }

    inserted_rows = sum(xtab[1].values())
    real_rows = sum(xtab[0].values())
    total_rows = inserted_rows + real_rows

    return {
        "fields_present": {"IsInserted": have_ins, "IPT": have_ipt,
                           "CampaignID": have_camp},
        "users": {
            "n_records": len(records),
            "n_with_decisions": int(lt.size),
            "n_with_zero_real_events": users_with_no_real,
        },
        "rows": {
            "total": total_rows,
            "real": real_rows,
            "inserted": inserted_rows,
            "inserted_share": round(inserted_rows / total_rows, 4) if total_rows else None,
        },
        "decision_distribution_all": dict(sorted(all_decisions.items())),
        "crosstab_isinserted_x_decision": {
            "real(IsInserted=0)": dict(sorted(xtab[0].items())),
            "inserted(IsInserted=1)": dict(sorted(xtab[1].items())),
        },
        "notbuy_audit": {
            "notbuy_total": all_decisions.get(NOTBUY, 0),
            "notbuy_inserted": xtab[1].get(NOTBUY, 0),
            "notbuy_real": xtab[0].get(NOTBUY, 0),
            "inserted_rows_that_are_not_notbuy": inserted_rows - xtab[1].get(NOTBUY, 0),
        },
        "sequence_length_discrete": pct_summary(lt.astype(float)),
        "sequence_length_continuous": pct_summary(lr.astype(float)),
        "S_needed_discrete": coverage(lt),
        "S_needed_continuous": coverage(lr),
        "truncation_at_current_cap": {
            "cap": CURRENT_CAP,
            "users_truncated_discrete": int((lt > CURRENT_CAP).sum()),
            "users_truncated_continuous": int((lr > CURRENT_CAP).sum()),
            "events_lost_discrete": int(np.clip(lt - CURRENT_CAP, 0, None).sum()),
            "events_lost_continuous": int(np.clip(lr - CURRENT_CAP, 0, None).sum()),
        },
        "gap_hours_as_stored": pct_summary(graw),
        "gap_hours_true_real_to_real": pct_summary(gtrue),
        "gap_buckets_as_stored": bucket(graw),
        "gap_buckets_true": bucket(gtrue),
        "gap_hours_on_inserted_rows": pct_summary(gins),
        "inserted_rows_between_real_draws": pct_summary(nbet),
        "zero_gap_audit": {
            "true_gaps_exactly_zero": int((gtrue <= 0).sum()) if gtrue.size else 0,
            "true_gaps_total": int(gtrue.size),
            "share": round(float((gtrue <= 0).mean()), 4) if gtrue.size else None,
        },
        "resolution_audit": resolution(graw),
        "self_excitation_probe": excite,
        "holdout": {
            "from_campaign": holdout_from,
            "users_with_real_events": holdout_real_users,
            "real_events": holdout_real_events,
            "revenue": round(holdout_revenue, 2),
        },
        "real_events_per_campaign": dict(sorted(per_campaign_real.items())),
        "inserted_rows_per_campaign": dict(sorted(per_campaign_inserted.items())),
    }


def report(st: Dict[str, Any]) -> None:
    """Human-readable summary, with the decision-relevant lines called out."""
    w = sys.stdout.write
    fp = st["fields_present"]
    w("\n=== fields present ===\n")
    for k, v in fp.items():
        w(f"  {k:<12} {'yes' if v else 'NO'}\n")
    if not fp["IsInserted"]:
        w("\n  This file has no IsInserted flag, so real and synthetic rows\n"
          "  cannot be separated. Point --data-file at an _IPT file.\n")

    r = st["rows"]
    w("\n=== rows ===\n")
    w(f"  users                {st['users']['n_with_decisions']:,}\n")
    w(f"  total rows           {r['total']:,}\n")
    w(f"  real draws           {r['real']:,}\n")
    w(f"  inserted NotBuy      {r['inserted']:,}"
      f"   ({r['inserted_share']:.1%} of all rows)\n" if r["inserted_share"] is not None
      else f"  inserted NotBuy      {r['inserted']:,}\n")

    nb = st["notbuy_audit"]
    w("\n=== can the mark space be classes 1-8? ===\n")
    w(f"  decision-9 rows, inserted  {nb['notbuy_inserted']:,}\n")
    w(f"  decision-9 rows, REAL      {nb['notbuy_real']:,}\n")
    w(f"  inserted rows that are NOT decision 9  {nb['inserted_rows_that_are_not_notbuy']:,}\n")
    if nb["notbuy_real"] == 0 and nb["inserted_rows_that_are_not_notbuy"] == 0:
        w("  -> clean. Dropping inserted rows leaves an 8-way mark space.\n")
    else:
        w("  -> NOT clean. Some decision-9 rows are real, or some inserted rows\n"
          "     are not decision 9. The 8-way mark plan needs adjusting.\n")

    sd, sc = st["sequence_length_discrete"], st["sequence_length_continuous"]
    if sd and sc:
        w("\n=== sequence length per user ===\n")
        w(f"  {'':<12}{'median':>9}{'p90':>9}{'p99':>9}{'max':>9}{'mean':>9}\n")
        w(f"  {'discrete':<12}{sd['p50']:>9.0f}{sd['p90']:>9.0f}{sd['p99']:>9.0f}"
          f"{sd['max']:>9.0f}{sd['mean']:>9.1f}\n")
        w(f"  {'continuous':<12}{sc['p50']:>9.0f}{sc['p90']:>9.0f}{sc['p99']:>9.0f}"
          f"{sc['max']:>9.0f}{sc['mean']:>9.1f}\n")
        if sd["mean"] > 0:
            w(f"  shrink factor (mean)  {sd['mean'] / max(sc['mean'], 1e-9):.2f}x\n")

    w("\n=== S needed to cover users without truncation ===\n")
    w(f"  {'':<12}{'50%':>8}{'90%':>8}{'95%':>8}{'99%':>8}{'100%':>8}\n")
    for name, key in (("discrete", "S_needed_discrete"), ("continuous", "S_needed_continuous")):
        c = st[key]
        if c:
            w(f"  {name:<12}" + "".join(f"{c[k]:>8,}" for k in ('50%', '90%', '95%', '99%', '100%')) + "\n")
    t = st["truncation_at_current_cap"]
    w(f"  at the current cap of {t['cap']:,}: "
      f"{t['users_truncated_discrete']:,} users truncated (discrete), "
      f"{t['users_truncated_continuous']:,} (continuous)\n")

    gi = st["gap_hours_on_inserted_rows"]
    if gi:
        w("\n=== IPT on inserted rows (should be the fixed grid width) ===\n")
        w(f"  median {gi['p50']:.3f}   p99 {gi['p99']:.3f}   max {gi['max']:.3f}\n")

    raw, tru = st["gap_hours_as_stored"], st["gap_hours_true_real_to_real"]
    if raw and tru:
        w("\n=== gap between consecutive real draws (hours) ===\n")
        w(f"  {'':<26}{'median':>9}{'p75':>9}{'p90':>9}{'p99':>9}{'max':>10}\n")
        w(f"  {'IPT as stored':<26}{raw['p50']:>9.3f}{raw['p75']:>9.2f}"
          f"{raw['p90']:>9.2f}{raw['p99']:>9.1f}{raw['max']:>10.1f}\n")
        w(f"  {'true real-to-real':<26}{tru['p50']:>9.3f}{tru['p75']:>9.2f}"
          f"{tru['p90']:>9.2f}{tru['p99']:>9.1f}{tru['max']:>10.1f}\n")
        if tru["max"] > raw["max"] * 1.5:
            w("  -> The stored IPT is time since the PREVIOUS ROW, and inserted rows\n"
              "     are rows. It is therefore censored by the grid width and is NOT\n"
              "     the inter-purchase time. A point process must use the accumulated\n"
              "     value, not the stored field.\n")

    nb2 = st["inserted_rows_between_real_draws"]
    if nb2:
        w(f"  inserted rows separating two real draws: median {nb2['p50']:.0f}, "
          f"p90 {nb2['p90']:.0f}, p99 {nb2['p99']:.0f}, max {nb2['max']:.0f}\n")

    if st["gap_buckets_true"]:
        w("\n  distribution of the TRUE gap:\n")
        for name, b in st["gap_buckets_true"].items():
            w(f"    {name:<12} {b['n']:>10,}  {b['share']:>7.2%}\n")

    z = st["zero_gap_audit"]
    if z.get("share"):
        w(f"\n  gaps of exactly zero: {z['true_gaps_exactly_zero']:,} "
          f"({z['share']:.1%} of {z['true_gaps_total']:,})\n")
        if z["share"] > 0.01:
            w("  -> A density over the log gap cannot place mass on an atom at 0.\n")

    res = st.get("resolution_audit") or {}
    if res:
        w("\n=== recording resolution of the gap field ===\n")
        w(f"  smallest non-zero gap        {res['smallest_nonzero_gap_hours']:.6f} h "
          f"({res['smallest_nonzero_gap_seconds']:.1f} s)\n")
        w(f"  non-zero gaps on that grid   {res['share_of_nonzero_gaps_on_that_grid']:.1%}\n")
        w(f"  distinct values below 1 h    {res['n_distinct_values_below_1h']:,}\n")
        if res["share_of_nonzero_gaps_on_that_grid"] > 0.98:
            w("  -> The field is QUANTISED at that step, so the zero gaps are rounding,\n"
              "     not simultaneity. Two ways to handle it, in order of preference:\n"
              "       (a) treat each recorded gap v as INTERVAL-CENSORED to\n"
              "           [v, v + q) and score log(F(v+q) - F(v)) instead of a\n"
              "           density. Standard for grouped duration data, needs no new\n"
              "           data, and handles the zeros without an atom.\n"
              "       (b) regenerate IPT at native timestamp precision. See the\n"
              "           2-decimal format in scripts/regenerate_ipt.py.\n")

    e = st["self_excitation_probe"]
    if e:
        w("\n=== self-excitation probe ===\n")
        w(f"  P(next draw within 1 min | previous gap <= 1 min)  "
          f"{e['P(next short | prev short)']:.3f}   (n={e['n_prev_short']:,})\n")
        w(f"  P(next draw within 1 min | previous gap >= 24 h)   "
          f"{e['P(next short | prev long)']:.3f}   (n={e['n_prev_long']:,})\n")
        if e.get("risk_ratio"):
            w(f"  risk ratio {e['risk_ratio']:.2f}x\n")
        w("  a ratio well above 1 is clustering: draws beget draws. A per-occasion\n"
          "  model cannot represent it; a self-exciting intensity can.\n")

    h = st["holdout"]
    w(f"\n=== holdout power (campaigns >= {h['from_campaign']}) ===\n")
    w(f"  users with >=1 real draw   {h['users_with_real_events']:,}\n")
    w(f"  real draws                 {h['real_events']:,}\n")
    w(f"  revenue (rev_vec units)    {h['revenue']:,.0f}\n")
    n_users = st["users"]["n_with_decisions"] or 1
    share = h["users_with_real_events"] / n_users
    w(f"  share of users active there   {share:.1%}\n")
    if share < 0.20:
        w("  WARNING: few users act in the holdout window. A per-user forecast\n"
          "  comparison there may be underpowered.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-file", default="clean_list_int_wide4_simple6_IPT.json",
                    help="file name inside PRODUCTGPT_DATA")
    ap.add_argument("--max-users", type=int, default=0,
                    help="0 = all users; a small number gives a fast smoke pass")
    ap.add_argument("--holdout-from", type=int, default=28,
                    help="first holdout campaign (28 = FeatureBasedHoldout)")
    ap.add_argument("--out", default="",
                    help="JSON output path (default: <output_dir>/event_stream_stats.json)")
    a = ap.parse_args()

    try:
        path = paths.data_file(a.data_file)
    except paths.ProductGPTPathError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"reading {path.name} ({path.stat().st_size / 1e6:.0f} MB) ...", flush=True)
    records = load_json_dataset(str(path))
    if a.max_users:
        records = records[:a.max_users]
    print(f"loaded {len(records):,} records", flush=True)

    stats = analyse(records, a.holdout_from)
    stats["source_file"] = path.name
    stats["max_users"] = a.max_users or None
    report(stats)

    out = Path(a.out) if a.out else paths.output_dir() / "event_stream_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
