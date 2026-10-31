"""R45 part C data: the left-censored players, under the sample's other filters.

The sample (CleanList in InsertNotBuy_GenerateJSON_IPT.R, ported in
regenerate_ipt.py) keeps the players observed from the account's start
(newbee_pull > 0), then applies four more filters. Part C keeps the four
filters and inverts the first: these are the players whose accounts began
before the data window. Only left-censoring changes, so the length caps stay.

Inputs (all R26-corrected, level 7, written by the same R run on 18 Sep 2026):
  perproduct/output_list_int_wide4_simple7_IPT.json   every player
  perproduct/clean_list_int_wide4_simple7_IPT.json    the 5,004 that gen 6 reads
  <E2>/Data/PlayerSummary_IPT.xlsx                    per-player counts

Self-check, before anything is written: the same filters with newbee_pull > 0
must give exactly the clean list's uids, and every one of those records must be
identical, field by field, in the two files. The first obtained block must also
carry the start-of-sequence token (132) for complete histories and the unknown
token (133) for censored ones -- the R generator's own marker.

Aggregate counts only.

    python scripts/r45c_prepare.py
    python scripts/r45c_prepare.py --out perproduct/censored_list_int_wide4_simple7_IPT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
from dataset_multistream import load_json_dataset, parse_token_ids  # noqa: E402

R, LTO_W = 15, 4
SOS7, UNK7 = 132, 133
E2 = Path.home() / "OneDrive - Singapore Management University" / "E2 Genshim Impact"


def uid_of(r) -> str:
    return str(r["uid"][0] if isinstance(r["uid"], list) else r["uid"])


def filters(ps: pd.DataFrame, a) -> pd.Series:
    """CleanList's four filters after the beginner-banner one, as in clean_lists()."""
    miss = ps["EndingCampaign"] - ps["StartingCampaign"] + 1 - ps["UniqueCampaigns"]
    return ((miss <= a.max_missing_camp) & (ps["UniqueCampaigns"] >= a.min_campaigns)
            & (ps["NumPurchases"] >= a.min_purchases) & (ps["NumPurchases"] <= a.max_purchases)
            & (ps["NumDecisions"] <= a.max_total_length))


def first_obtained(r) -> int:
    ai = parse_token_ids(r["AggregateInput"])
    return ai[LTO_W] if len(ai) >= R else -1


def campaigns(r) -> list:
    return parse_token_ids(r["CampaignID"])     # R writes a one-string list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", default="perproduct/output_list_int_wide4_simple7_IPT.json")
    ap.add_argument("--clean", default="perproduct/clean_list_int_wide4_simple7_IPT.json")
    ap.add_argument("--summary", default=str(E2 / "Data" / "PlayerSummary_IPT.xlsx"))
    ap.add_argument("--index", default=str(E2 / "Data" / "player_index.csv"))
    ap.add_argument("--out", default="perproduct/censored_list_int_wide4_simple7_IPT.json")
    ap.add_argument("--min-purchases", type=int, default=64)
    ap.add_argument("--max-purchases", type=int, default=1024)
    ap.add_argument("--max-total-length", type=int, default=2048)
    ap.add_argument("--min-campaigns", type=int, default=3)
    ap.add_argument("--max-missing-camp", type=int, default=2)
    ap.add_argument("--val-from", type=int, default=27)
    ap.add_argument("--holdout-from", type=int, default=28)
    a = ap.parse_args()
    root = Path(os.environ["PRODUCTGPT_DATA"])

    ps = pd.read_excel(a.summary)
    if "newbee_pull" not in ps.columns:       # the R26 runner's note: often already merged
        ps = ps.merge(pd.read_csv(a.index), on="uid", how="left")
    ps["uid"] = ps["uid"].astype(str)
    complete = ps["newbee_pull"] > 0           # NaN counts as not complete, as in the R
    keep = filters(ps, a)
    funnel = [("players in the summary", len(ps)),
              ("complete (newbee_pull > 0)", int(complete.sum())),
              ("complete, after the four filters", int((complete & keep).sum())),
              ("left-censored", int((~complete).sum())),
              ("left-censored, after the four filters", int((~complete & keep).sum()))]
    for name, n in funnel:
        print(f"[funnel] {name:<40} {n:>7,}")
    want_clean = set(ps.loc[complete & keep, "uid"])
    want_cens = set(ps.loc[~complete & keep, "uid"])
    purch = ps.set_index("uid")["NumPurchases"]
    total = float(purch.sum())
    print(f"[funnel] purchase share: complete kept {purch[list(want_clean)].sum() / total:.1%}, "
          f"censored kept {purch[list(want_cens)].sum() / total:.1%}")

    clean = {uid_of(r): r for r in load_json_dataset(str(root / a.clean))}
    print(f"[data] clean list: {len(clean):,} records")
    if set(clean) != want_clean:
        raise SystemExit(f"SELF-CHECK FAILED: the filters give {len(want_clean):,} complete uids, "
                         f"the clean list has {len(clean):,}; "
                         f"{len(want_clean ^ set(clean)):,} differ")
    print("[check] the filters reproduce the clean list's uid set exactly")

    allr = load_json_dataset(str(root / a.all))
    print(f"[data] all-player list: {len(allr):,} records")
    by_uid = {}
    for r in allr:
        u = uid_of(r)
        if u in by_uid:
            raise SystemExit(f"duplicate uid in {a.all}")
        by_uid[u] = r
    missing = (want_clean | want_cens) - set(by_uid)
    if missing:
        raise SystemExit(f"{len(missing):,} kept uids are absent from {a.all}")
    differ = sum(1 for u in want_clean if by_uid[u] != clean[u])
    if differ:
        raise SystemExit(f"SELF-CHECK FAILED: {differ:,} of the 5,004 records differ between files")
    print(f"[check] all {len(want_clean):,} complete records are identical in the two files")

    marks = Counter()
    for u in want_clean:
        marks[("complete", first_obtained(by_uid[u]))] += 1
    for u in want_cens:
        marks[("censored", first_obtained(by_uid[u]))] += 1
    print(f"[check] first obtained token: {dict(sorted(marks.items()))}")
    if set(marks) != {("complete", SOS7), ("censored", UNK7)}:
        raise SystemExit("SELF-CHECK FAILED: the start-of-history marker does not match the "
                         "beginner-banner filter")
    print("[check] every complete history starts with 132 and every censored one with 133")

    out = [by_uid[u] for u in sorted(want_cens)]
    tr = sum(1 for r in out if any(c < a.val_from for c in campaigns(r)))
    va = sum(1 for r in out if a.val_from in campaigns(r))
    ho = sum(1 for r in out if any(c >= a.holdout_from for c in campaigns(r)))
    rows = sum(len(campaigns(r)) for r in out)
    rows_c = sum(len(campaigns(clean[u])) for u in clean)
    print(f"[out] {len(out):,} censored players, {rows:,} rows (the 5,004 have {rows_c:,})")
    print(f"[out] with rows before campaign {a.val_from}: {tr:,}; in campaign {a.val_from}: {va:,}; "
          f"in the holdout (>= {a.holdout_from}): {ho:,}")
    dest = root / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out), encoding="utf-8")
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"[out] wrote {dest} ({dest.stat().st_size / 1e6:.0f} MB), sha256 {h}")


if __name__ == "__main__":
    main()
