"""
Python port of InsertNotBuy_GenerateJSON_IPT.R -- regenerate the IPT dataset
without needing R.

The R original is two scripts in one file. This mirrors them as subcommands:

  stage1  Drop the original ad-hoc no-draw rows, insert one synthetic NotBuy
          row at the end of every interval, carry state columns forward,
          compute HoursSinceLastDecision. Writes one .xlsx per player plus
          NotBuyInsertSummary.xlsx and PlayerSummary_IPT.xlsx.
          ONLY needed if you are changing --interval-hours. Its output already
          exists as FullDecisionSequence_IPT/.

  stage2  Read the augmented per-player files, map product indices, build the
          Decision / PreviousDecision / LTO / AggregateInput strings, apply the
          CleanList filters, and write the two JSON files.
          This is the stage that produces what the models actually read.

USAGE
  python scripts/regenerate_ipt.py stage2 --limit 50 --out-dir /tmp/try
  python scripts/regenerate_ipt.py stage2 --lag-obtained
  python scripts/regenerate_ipt.py stage1 --interval-hours 12

NOTE ON --lag-obtained
  The R pipeline writes the ObtainedProducts block for event t describing what
  the user obtained AT t. Combined with the all-zero block on inserted no-buy
  rows, that makes decision 9 exactly readable from the model's own input
  (measured: P(y=9 | block all zero) = 1.0000 over 83,664 events).
  --lag-obtained shifts the block by one event so it describes t-1, fixing the
  leak in the DATA rather than in the loader.
  If you use it, set shift_obtained=False in config5.py, or the streams get
  shifted twice.

PERFORMANCE
  There are ~16k per-player .xlsx files. Reading them with openpyxl is far
  slower than R's readxl; expect an hour or more for a full pass even in
  parallel. Use --limit while developing. On OneDrive, the files must be
  hydrated locally first ("Always keep on this device") or every read stalls
  on a network fetch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PAD10 = " ".join(["0"] * 10)
PAD10_TOKENS = " ".join(["<PAD>"] * 10)

# Decision codes, from the R: WhetherDraw-WhichPoolDraw-HowManyDraw
DECISION_MAP = {
    "1-200-1": "1", "1-200-10": "2",
    "1-301-1": "3", "1-301-10": "4",
    "1-400-1": "5", "1-400-10": "6",
    "1-302-1": "7", "1-302-10": "8",
}
SOS_DEC = "10"          # PreviousDecision sentinel at t=0
SOS_PROD_ROW = "58 " + PAD10[2:]   # "58 0 0 0 0 0 0 0 0 0"
UNK_PROD_ROW = "59 " + PAD10[2:]   # "59 0 0 0 0 0 0 0 0 0"


# ───────────────────────────── helpers ─────────────────────────────
def read_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


def build_maps(data_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    map_lv6  : NewProductIndex -> NewProductIndex6   (for ItemsJustGotIndex)
    map_lv6B : ProductIndex    -> NewProductIndex6   (for the LTO columns)
    """
    fwi = read_xlsx(data_dir / "FigureWeaponIndex6.xlsx")
    for c in ("ProductIndex", "NewProductIndex", "NewProductIndex6"):
        fwi[c] = pd.to_numeric(fwi[c], errors="coerce").astype("Int64")
    ok = fwi["NewProductIndex6"].notna()
    lv6 = {str(int(k)): str(int(v)) for k, v in
           zip(fwi.loc[ok, "NewProductIndex"], fwi.loc[ok, "NewProductIndex6"])
           if pd.notna(k)}
    lv6b = {str(int(k)): str(int(v)) for k, v in
            zip(fwi.loc[ok, "ProductIndex"], fwi.loc[ok, "NewProductIndex6"])
            if pd.notna(k)}
    return {"lv6": lv6, "lv6b": lv6b}


def map_tokens(s: Any, mapping: Dict[str, str]) -> str:
    """Map each whitespace token; tokens absent from the map keep their value."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return PAD10
    return " ".join(mapping.get(t, t) for t in str(s).split())


def interval_grid(campaign: pd.DataFrame, hours: int) -> pd.DataFrame:
    """One row per (campaign, interval), mirroring make_interval_grid()."""
    rows = []
    for cid, start, end in zip(campaign["CampaignID"],
                               pd.to_datetime(campaign["StartingDate"], utc=True),
                               pd.to_datetime(campaign["EndingDate"], utc=True)):
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        bounds = list(pd.date_range(start, end, freq=f"{hours}h"))
        if not bounds:
            continue
        if bounds[-1] < end:
            bounds.append(end)
        for i in range(len(bounds) - 1):
            rows.append({"CampaignID": cid, "IntervalIndex": i + 1,
                         "IntervalStart": bounds[i], "IntervalEnd": bounds[i + 1]})
    return pd.DataFrame(rows)


# ───────────────────────────── stage 1 ─────────────────────────────
def stage1_player(uid: str, src_dir: Path, out_dir: Path,
                  grid: pd.DataFrame) -> Optional[Dict[str, Any]]:
    f = src_dir / f"{uid}.xlsx"
    if not f.exists():
        return None
    try:
        d = read_xlsx(f)
    except Exception as e:                      # corrupt/locked file
        return {"uid": uid, "error": str(e)[:80]}
    if d.empty:
        return None

    d["gacha_time"] = pd.to_datetime(d["gacha_time"], utc=True)
    d = d.sort_values("gacha_time").reset_index(drop=True)

    n_dropped = int((d["WhetherDraw"] == 0).sum())
    d = d[d["WhetherDraw"] == 1].reset_index(drop=True)
    if d.empty:
        return None
    d["IsInserted"] = 0

    first_c, last_c = d["CampaignID"].min(), d["CampaignID"].max()
    g = grid[(grid["CampaignID"] >= first_c) & (grid["CampaignID"] <= last_c)]

    if g.empty:
        d.to_excel(out_dir / f"{uid}.xlsx", index=False)
        return {"uid": uid, "NumPurchases": len(d),
                "DroppedOriginalNoDraw": n_dropped, "InsertedRows": 0}

    nb = pd.DataFrame(index=range(len(g)), columns=d.columns, dtype=object)
    nb["gacha_time"] = list(g["IntervalEnd"] - timedelta(seconds=1))
    nb["WhetherDraw"] = 0
    nb["CampaignID"] = list(g["CampaignID"])
    nb["WhichPoolDraw"] = np.nan
    nb["HowManyDraw"] = np.nan
    nb["IsInserted"] = 1
    if "ItemsJustGot" in nb.columns:
        nb["ItemsJustGot"] = PAD10_TOKENS
    if "ItemsJustGotIndex" in nb.columns:
        nb["ItemsJustGotIndex"] = PAD10
    for c in [c for c in nb.columns if str(c).startswith("JustGot")]:
        nb[c] = False

    comb = pd.concat([d, nb], ignore_index=True)
    # R: arrange(gacha_time, desc(IsInserted == 0)) -> real rows before
    # inserted ones at an identical timestamp.
    comb["_real_first"] = (comb["IsInserted"] == 0).astype(int)
    comb = (comb.sort_values(["gacha_time", "_real_first"],
                             ascending=[True, False])
                .drop(columns="_real_first").reset_index(drop=True))

    state_cols = [c for c in comb.columns
                  if str(c).startswith(("Cum", "DrawSince", "LastFive"))]
    if "DrawInCampaign" in comb.columns:
        state_cols.append("DrawInCampaign")
    if state_cols:
        comb[state_cols] = comb[state_cols].ffill()
        for c in state_cols:
            num = pd.to_numeric(comb[c], errors="coerce")
            if num.notna().any():
                fill = comb[c].isna() & (comb["IsInserted"] == 1)
                comb.loc[fill, c] = 0

    comb["HoursSinceLastDecision"] = (
        comb["gacha_time"].diff().dt.total_seconds() / 3600.0)

    comb.to_excel(out_dir / f"{uid}.xlsx", index=False)
    return {"uid": uid,
            "NumPurchases": int((comb["IsInserted"] == 0).sum()),
            "DroppedOriginalNoDraw": n_dropped,
            "InsertedRows": int((comb["IsInserted"] == 1).sum())}


def summarise_player(uid: str, aug_dir: Path) -> Optional[Dict[str, Any]]:
    f = aug_dir / f"{uid}.xlsx"
    if not f.exists():
        return None
    try:
        d = read_xlsx(f)
    except Exception:
        return None
    if d.empty:
        return None
    pur = d[d["WhetherDraw"] == 1]
    return {
        "uid": uid,
        "StartingCampaign": d["CampaignID"].iloc[0],
        "EndingCampaign": d["CampaignID"].max(),
        "UniqueCampaigns": pur["CampaignID"].nunique(),
        "NumDecisions": len(d),
        "NumPurchases": int((d["WhetherDraw"] == 1).sum()),
        "NumInserted": int((d["WhetherDraw"] == 0).sum()),
    }


# ───────────────────────────── stage 2 ─────────────────────────────
def stage2_player(uid: str, aug_dir: Path, campaign: pd.DataFrame,
                  maps: Dict[str, Dict[str, str]], in_clean0: bool,
                  lag_obtained: bool) -> Optional[Dict[str, Any]]:
    f = aug_dir / f"{uid}.xlsx"
    if not f.exists():
        return None
    try:
        d = read_xlsx(f)
    except Exception:
        return None
    if d.empty:
        return None

    d.loc[d["WhichPoolDraw"] == 100, "WhichPoolDraw"] = 200
    d["ItemsJustGotIndex"] = d["ItemsJustGotIndex"].fillna(PAD10)
    d["ObtainedProducts_mapped"] = d["ItemsJustGotIndex"].apply(
        lambda s: map_tokens(s, maps["lv6"]))

    sentinel = SOS_PROD_ROW if in_clean0 else UNK_PROD_ROW
    if lag_obtained:
        # Event t should describe o_(t-1). Shift down one and put the
        # start-of-sequence sentinel at t=0. This is the data-side fix for the
        # leak; disable shift_obtained in config5.py if you use it.
        d["ObtainedProducts_mapped"] = (
            d["ObtainedProducts_mapped"].shift(1).fillna(sentinel))
    d.loc[d.index[0], "ObtainedProducts_mapped"] = sentinel

    dec = (d["WhetherDraw"].astype("Int64").astype(str) + "-"
           + d["WhichPoolDraw"].astype("Int64").astype(str) + "-"
           + d["HowManyDraw"].astype("Int64").astype(str))
    dec = dec.map(lambda s: DECISION_MAP.get(s, s))
    dec[d["WhetherDraw"] == 0] = "9"
    d["Decision"] = dec
    d["PreviousDecision"] = [SOS_DEC] + list(dec[:-1])

    d = d.merge(campaign, on="CampaignID", how="left").sort_values("gacha_time")

    d["LTO"] = (d["Figure5AIndex"].astype("Int64").astype(str) + " "
                + d["Figure5BIndex"].astype("Int64").astype(str) + " "
                + d["Weapon5AIndex"].astype("Int64").astype(str) + " "
                + d["Weapon5BIndex"].astype("Int64").astype(str))
    d["LTO_mapped"] = d["LTO"].apply(lambda s: map_tokens(s, maps["lv6b"]))
    d["AggregateInput"] = (d["LTO_mapped"] + " "
                           + d["ObtainedProducts_mapped"] + " "
                           + d["PreviousDecision"])

    wp = d["WhichPoolDraw"].astype("Int64").astype(str)
    hm = d["HowManyDraw"].astype("Int64").astype(str)
    wp[d["WhetherDraw"] == 0] = "0"
    hm[d["WhetherDraw"] == 0] = "0"

    if "IsInserted" not in d.columns:
        d["IsInserted"] = 0
    d["IsInserted"] = d["IsInserted"].fillna(0).astype(int)
    d["HoursSinceLastDecision"] = d["HoursSinceLastDecision"].fillna(0.0)

    j = " ".join
    return {
        "uid": str(uid),
        "CampaignID": j(d["CampaignID"].astype(str)),
        "AggregateInput": j(d["AggregateInput"]),
        "Item": j(d["ObtainedProducts_mapped"]),
        "Decision": j(d["Decision"]),
        "WhetherDraw": j(d["WhetherDraw"].astype("Int64").astype(str)),
        "WhichPoolDraw": j(wp),
        "HowManyDraw": j(hm),
        "IsInserted": j(d["IsInserted"].astype(str)),
        "IPT": j(f"{v:.2f}" for v in d["HoursSinceLastDecision"]),
        "IndexBasedHoldout": j((d["CampaignID"] >= 29).astype(int).astype(str)),
        "FeatureBasedHoldout": j((d["CampaignID"] >= 28).astype(int).astype(str)),
        "LTO": j(d["LTO_mapped"]),
    }


def clean_lists(data_dir: Path, a: argparse.Namespace) -> tuple[set, set]:
    ps = read_xlsx(data_dir / "PlayerSummary_IPT.xlsx")
    pi = pd.read_csv(data_dir / "player_index.csv")
    ps = ps.merge(pi, on="uid", how="left")
    c0 = ps[ps["newbee_pull"] > 0]
    miss = c0["EndingCampaign"] - c0["StartingCampaign"] + 1 - c0["UniqueCampaigns"]
    cl = c0[(miss <= a.max_missing_camp)
            & (c0["UniqueCampaigns"] >= a.min_campaigns)
            & (c0["NumPurchases"] >= a.min_purchases)
            & (c0["NumPurchases"] <= a.max_purchases)
            & (c0["NumDecisions"] <= a.max_total_length)]
    print(f"[clean] CleanList0 {len(c0):,} users | CleanList {len(cl):,} users")
    return set(c0["uid"].astype(str)), set(cl["uid"].astype(str))


# ───────────────────────────── cli ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["stage1", "stage2"])
    ap.add_argument("--base", default=str(Path.home() /
                    "OneDrive - Singapore Management University" / "E2 Genshim Impact"),
                    help="project root holding Data/ and the sequence folders")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N players (development)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--interval-hours", type=int, default=24,
                    help="stage1 only. The shipped IPT file used 24.")
    ap.add_argument("--lag-obtained", action="store_true",
                    help="stage2 only. Fix the label leak in the data.")
    ap.add_argument("--min-purchases", type=int, default=64)
    ap.add_argument("--max-purchases", type=int, default=1024)
    ap.add_argument("--max-total-length", type=int, default=2048)
    ap.add_argument("--min-campaigns", type=int, default=3)
    ap.add_argument("--max-missing-camp", type=int, default=2)
    a = ap.parse_args()

    base = Path(a.base)
    data_dir = base / "Data"
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    campaign = read_xlsx(data_dir / "CampaignWideIndex.xlsx")
    pi = pd.read_csv(data_dir / "player_index.csv")
    uids = [str(u) for u in pi["uid"]]
    if a.limit:
        uids = uids[: a.limit]
    print(f"[run] {a.stage}  {len(uids):,} players  workers={a.workers}")

    if a.stage == "stage1":
        src = base / "FullDecisionSequenceInt"
        out = Path(a.out_dir) if a.out_dir else base / "FullDecisionSequence_IPT"
        out.mkdir(parents=True, exist_ok=True)
        grid = interval_grid(campaign, a.interval_hours)
        print(f"[grid] {len(grid):,} intervals at {a.interval_hours}h")

        rows = []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(stage1_player, u, src, out, grid): u for u in uids}
            for n, fu in enumerate(as_completed(futs), 1):
                r = fu.result()
                if r:
                    rows.append(r)
                if n % 500 == 0:
                    print(f"  {n:,}/{len(uids):,}", flush=True)
        pd.DataFrame(rows).to_excel(data_dir / "NotBuyInsertSummary.xlsx", index=False)

        summ = []
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(summarise_player, u, out) for u in uids]
            for fu in as_completed(futs):
                r = fu.result()
                if r:
                    summ.append(r)
        ps = pd.DataFrame(summ).merge(pi, on="uid", how="left")
        ps.to_excel(data_dir / "PlayerSummary_IPT.xlsx", index=False)
        print(ps["NumDecisions"].quantile([.01, .05, .25, .5, .75, .95, .99]))
        print(f"[out] {data_dir/'PlayerSummary_IPT.xlsx'}")
        return

    # ---- stage 2 -------------------------------------------------
    aug = base / "FullDecisionSequence_IPT"
    out = Path(a.out_dir) if a.out_dir else data_dir
    out.mkdir(parents=True, exist_ok=True)
    maps = build_maps(data_dir)
    c0, cl = clean_lists(data_dir, a)
    if a.lag_obtained:
        print("[fix] --lag-obtained: ObtainedProducts will describe t-1. "
              "Set shift_obtained=False in config5.py to avoid a double shift.")

    out_list, clean_list = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(stage2_player, u, aug, campaign, maps,
                          u in c0, a.lag_obtained): u for u in uids}
        for n, fu in enumerate(as_completed(futs), 1):
            e = fu.result()
            if e:
                out_list.append(e)
                if futs[fu] in cl:
                    clean_list.append(e)
            if n % 500 == 0:
                print(f"  {n:,}/{len(uids):,}", flush=True)

    suffix = "_IPT_lagged" if a.lag_obtained else "_IPT"
    for name, payload in (("output_list_int_wide4_simple6", out_list),
                          ("clean_list_int_wide4_simple6", clean_list)):
        p = out / f"{name}{suffix}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[out] {p}  ({len(payload):,} users, {p.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
