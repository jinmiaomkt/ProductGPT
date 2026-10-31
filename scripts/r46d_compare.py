"""R46 part D read-out: the gates, I5-I7 and the predictions, by the pre-registered rules.

    python scripts/r46d_compare.py --out-dir $R46/partD
    python scripts/r46d_compare.py --out-dir $R46/partD_smoke --smoke

Reads the real fits, each null table's stage-1 R6 (K = 6) fit and its part D
fits, keeps the best start per arm and table (lowest penalised training
objective), checks every gate, and writes <out-dir>/compare.json. Nothing is
refitted. The verdicts are read only when every gate passes and every table
is complete; the smoke prints them for the code path only.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import numpy as np

R46 = Path(os.environ.get("R46", "/storage/home/jinmiao/ProductGPT/runs/r46"))
N_TABLES = 20
STARTS = {"R6B": 2, "K9": 3, "K12": 3}
TOL = 1e-6
P_HOLDS = 0.048
SD4 = ("loy", "inc", "tenl", "repeat_med")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def load(pattern: str) -> list:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]


def qv(res: dict, k: str) -> float:
    return res["fit"]["quantities"][k]["value"]


def kept(fits: list) -> dict | None:
    return min(fits, key=lambda r: r["fit"]["obj_hat"]) if fits else None


def p_one_sided(null_vals, real: float, side: str) -> float:
    v = np.asarray(null_vals, dtype=float)
    c = int((v <= real).sum()) if side == "le" else int((v >= real).sum())
    return (1 + c) / (len(v) + 1)


def p_partc(null_vals, real: float) -> float:
    """Part C's I3 rule: one-sided in the direction of the real estimate."""
    sgn = 1.0 if real >= 0 else -1.0
    return p_one_sided(sgn * np.asarray(null_vals, dtype=float), sgn * real, "ge")


def arms_of(fits: list) -> dict:
    return {arm: [r for r in fits if r["arm"] == arm] for arm in STARTS}


def dg2(by_arm: dict) -> dict:
    """D-G2 on one table: each kept fit no worse than its reference (+1e-6)."""
    out = {}
    for arm, fits in by_arm.items():
        k = kept(fits)
        if k is None:
            continue
        out[arm] = dict(ok=k["fit"]["obj_hat"] <= k["fit"]["obj_ref"] + TOL,
                        diff=k["fit"]["obj_hat"] - k["fit"]["obj_ref"], kept_start=k["start"])
    return out


def summary(v) -> dict:
    v = np.asarray(v, dtype=float)
    if not len(v):
        return {}
    return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                min=float(v.min()), max=float(v.max()))


def fmt(s: dict) -> str:
    return f"{s['mean']:.3f} ± {s['sd']:.3f} [{s['min']:.3f}, {s['max']:.3f}]" if s else "—"


# ───────────────────────────── real data ─────────────────────────────
def real_block(out: Path, pre: str) -> dict:
    fits = load(str(out / "real" / f"{pre}D_real_*.json"))
    by = arms_of(fits)
    say(f"== real data: " + ", ".join(f"{a} {len(f)}/{STARTS[a]}" for a, f in by.items()))
    if not fits:
        return {}
    ref = fits[0]["quantities_ref"]                  # part A's R6 (K = 6)
    res = dict(counts={a: len(f) for a, f in by.items()}, ref_R6=ref, dg2=dg2(by), kept={})
    for arm, f in by.items():
        k = kept(f)
        if k is None:
            continue
        q = k["fit"]["quantities"]
        res["kept"][arm] = dict(start=k["start"], obj_hat=k["fit"]["obj_hat"], val_nll=k["fit"]["val"]["nll"],
                                quantities={n: q[n] for n in q if n in SD4 + ("hl_dec", "hl_ban", "hl_out")},
                                starts={r["start"]: dict(obj_hat=r["fit"]["obj_hat"],
                                                         hl_ban=r["fit"]["quantities"].get("hl_ban", {}).get("value"),
                                                         loy=qv(r, "loy"), fit_seconds=r["fit"]["fit_seconds"])
                                        for r in f},
                                class_weights=k["fit"]["derived"].get("class_weights"))
    b = [r for r in by["R6B"] if r["start"] == 0]
    if b and "val_R6_K6" in b[0]["fit"]:
        res["val_R6_K6"] = b[0]["fit"]["val_R6_K6"]["nll"]
    # D-G3: hl_ban identified on the real data
    if by["R6B"]:
        k = kept(by["R6B"])
        qb = k["fit"]["quantities"]["hl_ban"]
        i = k["labels"].index("a_ban[0]")
        diag = k["fit"].get("se_diag", {})
        g3 = dict(hl_ban=qb["value"], se=qb["se"],
                  se_finite=bool(math.isfinite(qb["se"]) and qb["se"] > 0),
                  on_ridge=i in diag.get("ridge", []), not_identified=i in diag.get("not_identified", []))
        g3["ok"] = g3["se_finite"] and not g3["on_ridge"] and not g3["not_identified"]
        if len(by["R6B"]) == 2:
            o = {r["start"]: r for r in by["R6B"]}
            d_obj = abs(o[0]["fit"]["obj_hat"] - o[1]["fit"]["obj_hat"])
            d_hl = abs(qv(o[0], "hl_ban") - qv(o[1], "hl_ban"))
            g3.update(starts_obj_diff=d_obj, starts_hl_diff=d_hl)
            if d_obj <= TOL:
                g3["starts_agree"] = d_hl <= 2 * qb["se"]
                g3["ok"] = g3["ok"] and g3["starts_agree"]
        res["dg3"] = g3
    k6 = res.get("val_R6_K6")
    say(f"   part A's R6 (K = 6): banner loyalty {ref['loy']:.3f}, shared half-life {ref['hl_dec']:.3f}"
        + (f", campaign-27 NLL {k6:.5f}" if k6 is not None else ""))
    for arm, d in res["kept"].items():
        q = d["quantities"]
        say(f"   {arm:<4} kept start {d['start']}: objective {d['obj_hat']:.8f}, campaign-27 NLL {d['val_nll']:.5f}; "
            + ", ".join(f"{n} {q[n]['value']:.3f} ({q[n]['se']:.3f})" for n in ("loy", "hl_dec", "hl_ban") if n in q))
    say(f"   D-G2 {res['dg2']}")
    if "dg3" in res:
        say(f"   D-G3 {res['dg3']}  {'PASS' if res['dg3']['ok'] else 'FAIL'}")
    if k6 is not None and "K12" in res["kept"]:
        res["k12_beats_k6_by_0p02"] = res["kept"]["K12"]["val_nll"] < k6 - 0.02
    return res


# ───────────────────────────── null tables ─────────────────────────────
def null_block(out: Path, pre: str, smoke: bool) -> dict:
    s1 = {r["rep"]: r for r in load(str(out / "null" / "stage1" / f"{pre}D_null_base_r*.json"))}
    fits = load(str(out / "null" / "fits" / f"{pre}D_null_*_r*.json"))
    reps = sorted(s1)
    say(f"\n== null (b) tables: stage 1 {len(s1)}, part D fits {len(fits)}")
    tables, problems = {}, []
    for rep in reps:
        mine = arms_of([r for r in fits if r["rep"] == rep])
        st = s1[rep]
        t = dict(seed=st["seed"], design=st["design"]["ok"], simulation=st["simulation"]["ok"],
                 pairing=all(r["pairing"]["ok"] and r["pairing"]["stage1"] == st["tag"]
                             for f in mine.values() for r in f),
                 counts={a: len(f) for a, f in mine.items()}, dg2=dg2(mine),
                 base={n: qv(st, n) for n in SD4 + ("hl_dec",)})
        t["complete"] = all(t["counts"][a] == STARTS[a] for a in STARTS)
        for arm, f in mine.items():
            k = kept(f)
            if k is not None:
                t[arm] = {n: qv(k, n) for n in SD4 + ("hl_dec", "hl_ban") if n in k["fit"]["quantities"]}
                t[arm]["kept_start"] = k["start"]
                t[arm]["fit_seconds"] = [r["fit"]["fit_seconds"] for r in f]
        t["gates_ok"] = t["design"] and t["simulation"] and t["pairing"] and all(d["ok"] for d in t["dg2"].values())
        if not t["complete"]:
            problems.append(f"table {rep}: incomplete {t['counts']}")
        if not t["gates_ok"]:
            problems.append(f"table {rep}: gate failed (design {t['design']}, simulation {t['simulation']}, "
                            f"pairing {t['pairing']}, D-G2 {t['dg2']})")
        tables[rep] = t
    if len(reps) != N_TABLES and not smoke:
        problems.append(f"{len(reps)} of {N_TABLES} stage-1 tables")
    for p in problems:
        say(f"   {p}")
    return dict(tables=tables, problems=problems)


def col(tables: dict, arm: str, k: str) -> list:
    return [t[arm][k] for t in tables.values() if arm in t and k in t[arm]]


def old_partc() -> dict:
    """Part C's 20 null (b) R6 fits: reported beside the new ones, not tested."""
    runs = load(str(R46 / "partC" / "keane" / "C3b_R6_r*.json"))
    return dict(n=len(runs), hl_dec=summary([qv(r, "hl_dec") for r in runs]),
                loy=summary([qv(r, "loy") for r in runs]))


# ───────────────────────────── verdicts ─────────────────────────────
def verdicts(real: dict, null: dict) -> dict:
    T = null["tables"]
    ref = real["ref_R6"]
    kr = real["kept"]
    base_loy = [t["base"]["loy"] for t in T.values()]
    base_hl = [t["base"]["hl_dec"] for t in T.values()]
    v = {}
    hb_real = kr["R6B"]["quantities"]["hl_ban"]["value"]
    hb_null = col(T, "R6B", "hl_ban")
    v["I5"] = dict(real=hb_real, null=summary(hb_null), p=p_one_sided(hb_null, hb_real, "le"),
                   void=not real["dg3"]["ok"])
    v["I5"]["holds"] = (v["I5"]["p"] <= P_HOLDS) and not v["I5"]["void"]
    v["I6"] = dict(real=ref["hl_dec"], null=summary(base_hl), p=p_one_sided(base_hl, ref["hl_dec"], "le"))
    v["I6"]["holds"] = v["I6"]["p"] <= P_HOLDS
    l12_real = kr["K12"]["quantities"]["loy"]["value"]
    l12_null = col(T, "K12", "loy")
    v["I7"] = dict(real=l12_real, null=summary(l12_null), p=p_one_sided(l12_null, l12_real, "ge"))
    v["I7"]["holds"] = v["I7"]["p"] <= P_HOLDS
    v["K12_other"] = {n: dict(real=kr["K12"]["quantities"][n]["value"], null=summary(col(T, "K12", n)),
                              p=p_partc(col(T, "K12", n), kr["K12"]["quantities"][n]["value"]))
                      for n in ("inc", "tenl", "repeat_med")}
    v["K9_reported"] = {n: dict(real=kr["K9"]["quantities"][n]["value"], null=summary(col(T, "K9", n)),
                                p_descriptive=p_partc(col(T, "K9", n), kr["K9"]["quantities"][n]["value"]))
                        for n in SD4}
    v["by_K_null_loy"] = {"6": summary(base_loy), "9": summary(col(T, "K9", "loy")), "12": summary(l12_null)}
    v["by_K_real_loy"] = {"6": ref["loy"], "9": kr["K9"]["quantities"]["loy"]["value"], "12": l12_real}
    pr = {}
    pr["1 null R6 (K = 6): loyalty > real on >= 18/20, shared half-life > 50 on all, I6"] = (
        sum(x > ref["loy"] for x in base_loy) >= 18 and all(h > 50 for h in base_hl) and v["I6"]["holds"])
    pr["2 real hl_ban < 10"] = hb_real < 10
    pr["3 null hl_ban > 50 on all 20"] = all(h > 50 for h in hb_null)
    pr["4 I5"] = v["I5"]["holds"]
    pr["5 null loyalty: K = 12 mean at least a third below K = 6 mean"] = (
        np.mean(l12_null) <= (2.0 / 3.0) * np.mean(base_loy))
    pr["6 real loyalty at K = 12 within 10% of K = 6"] = abs(l12_real - ref["loy"]) <= 0.10 * abs(ref["loy"])
    pr["7 I7"] = v["I7"]["holds"]
    pr["8 inc, tenl, repeat premium pass at K = 12"] = all(d["p"] <= P_HOLDS for d in v["K12_other"].values())
    v["predictions"] = pr
    reading = {(True, True): "banner loyalty is not continuous taste of the null (b) kind (time profile and level)",
               (True, False): "state dependence on its time profile; the level test fails",
               (False, True): "not established either way; the paper does not claim it",
               (False, False): "banner loyalty cannot be told apart from continuous taste on this design"}
    v["reading"] = reading[(v["I5"]["holds"], v["I7"]["holds"])]
    if v["I5"]["void"]:
        v["reading"] += " (D-G3 failed: hl_ban not identified on the real data, I5 void; the reading rests on I7)"
    elif v["I6"]["holds"] and not v["I5"]["holds"]:
        v["reading"] += " (I6 holds, I5 fails: the short shared half-life was the buy and ten-pull stocks' memory)"
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(R46 / "partD"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    pre = "smoke_" if a.smoke else ""
    real = real_block(out, pre)
    null = null_block(out, pre, a.smoke)
    res = dict(real=real, null=null, partC_null_b=old_partc())
    pc = res["partC_null_b"]
    say(f"\n   part C's null (b) R6 fits ({pc['n']}, the lead's source; not tested): shared half-life "
        f"{fmt(pc['hl_dec'])}, banner loyalty {fmt(pc['loy'])}")
    T = null["tables"]
    if T:
        say(f"   fresh tables, R6 (K = 6): shared half-life {fmt(summary([t['base']['hl_dec'] for t in T.values()]))}, "
            f"banner loyalty {fmt(summary([t['base']['loy'] for t in T.values()]))}")
        for arm in STARTS:
            say(f"   fresh tables, {arm:<4}: banner loyalty {fmt(summary(col(T, arm, 'loy')))}"
                + (f", hl_ban {fmt(summary(col(T, arm, 'hl_ban')))}, buy/ten-pull half-life "
                   f"{fmt(summary(col(T, arm, 'hl_dec')))}" if arm == "R6B" else ""))
    complete = (real and all(real["counts"][k] == STARTS[k] for k in STARTS) and not null["problems"]
                and "dg3" in real and all(d["ok"] for d in real["dg2"].values()))
    res["complete_and_gates_pass"] = bool(complete)
    if complete or (a.smoke and real.get("kept", {}).keys() >= {"R6B", "K12"} and T):
        if a.smoke:
            # the smoke has one table and no K = 9: fill what the verdict code needs
            for t in T.values():
                t.setdefault("K9", t.get("K12", {}))
            real["kept"].setdefault("K9", real["kept"]["K12"])
            real.setdefault("dg3", {"ok": False})
        v = verdicts(real, null)
        res["verdicts"] = v
        say("\n== verdicts" + (" (SMOKE: code path only, not a test)" if a.smoke else ""))
        for n in ("I5", "I6", "I7"):
            d = v[n]
            say(f"   {n}: real {d['real']:.3f}; null {fmt(d['null'])}; p = {d['p']:.3f}  "
                f"{'HOLDS' if d['holds'] else 'FAILS'}{' (VOID)' if d.get('void') else ''}")
        for n, d in v["K12_other"].items():
            say(f"   K = 12 {n}: real {d['real']:.3f}; null {fmt(d['null'])}; p = {d['p']:.3f}")
        say(f"   banner loyalty by K, null mean: " + ", ".join(f"K={k} {s.get('mean', float('nan')):.3f}"
                                                       for k, s in v["by_K_null_loy"].items())
            + "; real: " + ", ".join(f"K={k} {x:.3f}" for k, x in v["by_K_real_loy"].items()))
        say("\n== predictions")
        for k, ok in v["predictions"].items():
            say(f"   {k}: {'holds' if ok else 'FAILS'}")
        say(f"\n   reading: {v['reading']}")
    else:
        say("\n== verdicts NOT READ: a gate failed or a fit is missing (see above)")
    if real.get("k12_beats_k6_by_0p02"):
        say("   note: K = 12 beats K = 6 on campaign 27 by more than 0.02 nats (recorded for a later registration)")
    (out / "compare.json").write_text(json.dumps(res, indent=1, default=float))
    say(f"\nwrote {out / 'compare.json'}")


if __name__ == "__main__":
    main()
