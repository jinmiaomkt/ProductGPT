"""R46 part C read-out: the verdicts I1-I4 and the predictions, by the pre-registered rules.

    python scripts/r46c_compare.py --out-dir $R46/partC
    python scripts/r46c_compare.py --out-dir $R46/partC_smoke --smoke

Reads C1 (pboot), C2 (cboot), the H6 fits and C3 (keane), checks every
replicate's gates, and writes <out-dir>/compare.json. Nothing is refitted.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from parametric import identify as IC, model as M  # noqa: E402

F64 = torch.float64
R46 = Path(os.environ.get("R46", "/storage/home/jinmiao/ProductGPT/runs/r46"))
REAL = {"R6": ("se/R6_K6_s1", 6), "R4": ("R4", 1)}
N_EXPECT = {"C1": 20, "C2": 50, "C3": 20, "H6": 5}
Z95 = 1.959964
LOG_SCALE = ("hl_dec", "hl_out")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def load(paths):
    return [json.loads(Path(p).read_text()) for p in sorted(paths)]


def real_fit(rung: str):
    rel, K = REAL[rung]
    r = json.loads((R46 / rel / "result.json").read_text())
    spec = M.Spec(rung, K=K)
    th = torch.tensor(r["theta"], dtype=F64)
    V = torch.load(R46 / rel / "cov.pt", weights_only=False)
    q = IC.quantities_with_se(spec, th, V)
    return r, spec, th, q


def stats(vals: np.ndarray, truth: float, ses: np.ndarray | None = None) -> dict:
    R = len(vals)
    bias = float(vals.mean() - truth)
    sd = float(vals.std(ddof=1)) if R > 1 else float("nan")
    out = {"truth": truth, "mean": float(vals.mean()), "bias": bias, "sd": sd,
           "bias_z": bias / (sd / math.sqrt(R)) if sd and sd > 0 else float("nan")}
    if ses is not None:
        med = float(np.median(ses))
        out.update(median_se=med, ratio=med / sd if sd and sd > 0 else float("nan"),
                   coverage=float((np.abs(vals - truth) <= Z95 * ses).mean()))
    return out


def qvals(res: dict, key: str):
    q = res["fit"]["quantities"][key]
    v, s = q["value"], q["se"]
    if key in LOG_SCALE:
        return math.log(v), s / v
    return v, s


# ───────────────────────────── C1 ─────────────────────────────
def c1(out: Path, pre: str, smoke: bool) -> dict:
    res_all = {}
    for dgp in ("R6", "R4"):
        runs = load(glob.glob(str(out / "pboot" / f"{pre}C1_{dgp}_r*.json")))
        R = len(runs)
        say(f"\n== C1 parametric bootstrap, {dgp}: {R} replicates")
        if not runs:
            continue
        gates = {"design": all(r["design"]["ok"] for r in runs),
                 "simulation": all(r["simulation"]["ok"] for r in runs),
                 "optimum": all(r["fit"]["G3_ok"] for r in runs),
                 "count": R == N_EXPECT["C1"] or smoke}
        say(f"   gates {gates}; C-G2 worst block {max(r['simulation']['block_max'] for r in runs):.1e}, "
            f"prob {max(r['simulation']['prob_max'] for r in runs):.1e}")
        real, spec, th_real, q_real = real_fit(dgp)
        labels = runs[0]["labels"]
        truth = np.array(runs[0]["theta_true"])
        TH = np.array([r["theta_hat"] for r in runs])
        SE = np.array([r["fit"]["se"] for r in runs])
        real_se = np.array([np.nan if s is None else s for s in real["se"]], dtype=float)
        ident = np.isfinite(real_se) & (real_se > 0) & np.isfinite(SE).all(0) & (SE > 0).all(0)
        rare = np.isin(labels, real.get("rare", []))
        shared = IC.shared_labels(labels)
        test = shared & ident & ~rare
        sd = TH.std(0, ddof=1) if R > 1 else np.full(len(labels), np.nan)
        bias = TH.mean(0) - truth
        with np.errstate(divide="ignore", invalid="ignore"):
            bz = bias / (sd / math.sqrt(R))
        cover = (np.abs(TH - truth) <= Z95 * SE).mean(0)
        pooled = float(cover[test].mean()) if test.any() else float("nan")
        recovered = float((np.abs(bz[test]) < 3).mean()) if test.any() else float("nan")
        say(f"   coefficients: {len(labels)}; tested (shared, identified, non-rare) {int(test.sum())}; "
            f"recovered |bias z| < 3: {recovered:.3f}; pooled 95% coverage {pooled:.3f}")
        tested = np.flatnonzero(test)
        worst = tested[np.argsort(-np.nan_to_num(np.abs(bz[tested]), nan=0.0))[:6]]
        say("   largest |bias z| among tested: " + ", ".join(f"{labels[i]} {bz[i]:+.1f}" for i in worst))
        keys = [k for k in IC.SD_QUANTITIES + IC.HG_QUANTITIES if k in runs[0]["fit"]["quantities"]]
        qs = {}
        say(f"   {'quantity':<12} {'truth':>8} {'mean':>8} {'bias z':>7} {'sd':>7} {'med SE':>7} {'ratio':>6} {'cover':>6}")
        for k in keys:
            vv = np.array([qvals(r, k) for r in runs])
            t = runs[0]["quantities_true"][k]
            t = math.log(t) if k in LOG_SCALE else t
            s = stats(vv[:, 0], t, vv[:, 1])
            s["rare"] = k in real.get("rare", [])
            qs[k] = s
            say(f"   {k + (' (log)' if k in LOG_SCALE else ''):<12} {t:>8.3f} {s['mean']:>8.3f} {s['bias_z']:>7.2f} "
                f"{s['sd']:>7.3f} {s['median_se']:>7.3f} {s['ratio']:>6.2f} {s['coverage']:>6.2f}")
        fails_i1 = [k for k, s in qs.items() if not abs(s["bias_z"]) < 3]
        fails_i2 = [k for k, s in qs.items() if not 0.67 <= s["ratio"] <= 1.5]
        I1 = not fails_i1 and recovered >= 0.90
        I2 = 0.90 <= pooled <= 0.98 and not fails_i2
        say(f"   I1 identified on the real design: {'HOLDS' if I1 else 'FAILS'}"
            f"{' (quantities with |bias z| >= 3: ' + str(fails_i1) + ')' if fails_i1 else ''}")
        say(f"   I2 sandwich usable: {'HOLDS' if I2 else 'FAILS'} (pooled coverage {pooled:.3f}; "
            f"ratio outside 0.67-1.5: {fails_i2})")
        rare_rows = {labels[i]: {"bias_z": float(bz[i]), "coverage": float(cover[i]),
                                 "ratio": float(np.median(SE[:, i]) / sd[i]) if sd[i] > 0 else float("nan")}
                     for i in np.flatnonzero(rare)}
        ws, wr = runs[0]["world_sim"], runs[0]["world_real"]
        say(f"   simulated world, decision shares: real {wr['decision_shares']}")
        say(f"                                     sim  {np.round(np.mean([r['world_sim']['decision_shares'] for r in runs], 0), 4).tolist()}")
        for b in wr["banners"]:
            say(f"   {b:<9} 5-star per item real {wr['banners'][b]['five_star_per_item']:.4f} "
                f"sim {ws['banners'][b]['five_star_per_item']:.4f}"
                + (f"; featured share real {wr['banners'][b].get('featured_share_of_5star', float('nan')):.3f} "
                   f"sim {ws['banners'][b].get('featured_share_of_5star', float('nan')):.3f}" if b != "regular" else ""))
        res_all[dgp] = dict(replicates=R, gates=gates, n_tested=int(test.sum()), recovered=recovered,
                            pooled_coverage=pooled, quantities=qs, I1=I1, I2=I2, I1_fails=fails_i1,
                            I2_fails=fails_i2, rare=rare_rows,
                            fit_seconds=[r["fit"]["fit_seconds"] for r in runs],
                            world_real=wr, world_sim_first=ws)
    return res_all


# ───────────────────────────── C2 ─────────────────────────────
def c2(out: Path, pre: str, smoke: bool) -> dict:
    runs = load(glob.glob(str(out / "cboot" / f"{pre}C2_R6_r*.json")))
    R = len(runs)
    say(f"\n== C2 customer bootstrap, R6: {R} replicates")
    if not runs:
        return {}
    gates = {"weights": all(r["bootstrap"]["ok"] for r in runs), "count": R == N_EXPECT["C2"] or smoke}
    say(f"   gates {gates}")
    real, spec, th_real, q_real = real_fit("R6")
    labels = runs[0]["labels"]
    TH = np.array([r["theta_hat"] for r in runs])
    est = np.array(real["theta"])
    rows = {}
    say(f"   {'quantity':<12} {'estimate':>9} {'boot sd':>8} {'sandwich':>8} {'ratio':>6}   95% interval            percentile")
    for k in IC.SD_QUANTITIES + IC.HG_QUANTITIES:
        if k not in q_real:
            continue
        v = np.array([r["fit"]["quantities"][k]["value"] for r in runs])
        e, se_s = q_real[k]["value"], q_real[k]["se"]
        if k in LOG_SCALE:
            la = lambda h: np.log(np.power(0.5, 1.0 / h)) - np.log1p(-np.power(0.5, 1.0 / h))
            sd_l = float(la(v).std(ddof=1)) if R > 1 else float("nan")
            lo_l, hi_l = float(la(np.array(e))) - Z95 * sd_l, float(la(np.array(e))) + Z95 * sd_l
            to_h = lambda x: math.log(0.5) / math.log(1.0 / (1.0 + math.exp(-x)))
            lo, hi = to_h(lo_l), to_h(hi_l)
            sd = float(v.std(ddof=1)) if R > 1 else float("nan")
        else:
            sd = float(v.std(ddof=1)) if R > 1 else float("nan")
            lo, hi = e - Z95 * sd, e + Z95 * sd
        pc = np.percentile(v, [2.5, 97.5]).tolist()
        ratio = sd / se_s if se_s > 0 else float("nan")
        rows[k] = dict(estimate=e, boot_sd=sd, sandwich_se=se_s, ratio=ratio, ci95=[lo, hi], percentile=pc)
        say(f"   {k:<12} {e:>9.3f} {sd:>8.3f} {se_s:>8.3f} {ratio:>6.2f}   [{lo:>7.3f}, {hi:>7.3f}]      "
            f"[{pc[0]:.3f}, {pc[1]:.3f}]")
    rare = {}
    real_se = [np.nan if s is None else s for s in real["se"]]
    for lab in real.get("rare", []):
        i = labels.index(lab)
        sd = float(TH[:, i].std(ddof=1)) if R > 1 else float("nan")
        rare[lab] = dict(estimate=float(est[i]), boot_sd=sd, sandwich_se=float(real_se[i]),
                         ci95=[float(est[i] - Z95 * sd), float(est[i] + Z95 * sd)],
                         percentile=np.percentile(TH[:, i], [2.5, 97.5]).tolist())
    say("   rare-state coefficients: " + "; ".join(
        f"{k} {v['estimate']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] (sandwich {v['sandwich_se']:.2f})"
        for k, v in rare.items()))
    return dict(replicates=R, gates=gates, quantities=rows, rare=rare,
                fit_seconds=[r["fit"]["fit_seconds"] for r in runs])


# ───────────────────────────── H6 and C3 ─────────────────────────────
def c3(out: Path, pre: str, smoke: bool) -> dict:
    h6 = load(glob.glob(str(out / "h6" / "H6_K6_s*" / "result.json")))
    say(f"\n== H6 fits: {len(h6)} starts")
    if not h6:
        return {}
    best = min(h6, key=lambda r: r["train_nll"])
    r6 = json.loads((R46 / "R6_K6_s1" / "result.json").read_text())
    r4 = json.loads((R46 / "R4" / "result.json").read_text())
    val = {"H6": best["cells"]["val"]["nll"], "R6": r6["cells"]["val"]["nll"], "R4": r4["cells"]["val"]["nll"]}
    say("   starts (train NLL / campaign-27 NLL): " + ", ".join(
        f"{r['tag']} {r['train_nll']:.5f}/{r['cells']['val']['nll']:.5f}" for r in sorted(h6, key=lambda r: r["tag"])))
    say(f"   best {best['tag']}: campaign-27 NLL H6 {val['H6']:.5f}, R6 {val['R6']:.5f}, R4 {val['R4']:.5f}; "
        f"H6 - R6 = {val['H6'] - val['R6']:+.4f}; class weights {best['derived'].get('class_weights')}")
    pred4 = val["H6"] - val["R6"] > 0.1
    _, _, _, q6 = real_fit("R6")
    _, _, _, q4 = real_fit("R4")
    keys = list(IC.SD_KEYS)
    extra = ["hl_dec", "bawa[0]", "runs[0]", "runs[1]", "runs[2]"]
    out_c3 = {"h6": dict(best=best["tag"], val_nll=val, starts=len(h6), pred4=pred4)}
    for null in ("a", "b"):
        fits = {f: load(glob.glob(str(out / "keane" / f"{pre}C3{null}_{f}_r*.json"))) for f in ("R4", "R6")}
        say(f"\n== C3 null ({null}): R4 {len(fits['R4'])} and R6 {len(fits['R6'])} replicates")
        if not fits["R4"] or not fits["R6"]:
            continue
        gates = {"design": all(r["design"]["ok"] for f in fits.values() for r in f),
                 "simulation": all(r["simulation"]["ok"] for f in fits.values() for r in f),
                 "count": all(len(f) == N_EXPECT["C3"] for f in fits.values()) or smoke}
        if null == "a":
            gates["optimum_R6"] = all(r["fit"]["G3_ok"] for r in fits["R6"])
        cs4 = {r["rep"]: r["sim_checksum"] for r in fits["R4"]}
        cs6 = {r["rep"]: r["sim_checksum"] for r in fits["R6"]}
        gates["pairing"] = cs4 == cs6
        say(f"   gates {gates}")
        tab = {}
        say(f"   {'quantity':<10} {'real R4':>8} {'null R4':>16} {'real R6':>8} {'null R6':>16} {'p (R6)':>7}")
        for k in keys + extra:
            v4 = np.array([r["fit"]["quantities"][k]["value"] for r in fits["R4"] if k in r["fit"]["quantities"]])
            v6 = np.array([r["fit"]["quantities"][k]["value"] for r in fits["R6"] if k in r["fit"]["quantities"]])
            real4, real6 = q4[k]["value"], q6[k]["value"]
            sgn = 1.0 if real6 >= 0 else -1.0
            p = (1 + int((sgn * v6 >= sgn * real6).sum())) / (len(v6) + 1)
            s4 = stats(v4, 0.0) if k in keys else stats(v4, float("nan"))
            s6 = stats(v6, 0.0) if k in keys else stats(v6, float("nan"))
            tab[k] = dict(real_R4=real4, real_R6=real6, null_R4=s4, null_R6=s6, p_R6=p,
                          share_R4=float(v4.mean() / real4) if real4 else float("nan"),
                          share_R6=float(v6.mean() / real6) if real6 else float("nan"))
            say(f"   {k:<10} {real4:>8.3f} {v4.mean():>8.3f} ± {s4['sd']:<5.3f} {real6:>8.3f} "
                f"{v6.mean():>8.3f} ± {s6['sd']:<5.3f} {p:>7.3f}")
        I3_null = all(tab[k]["p_R6"] <= 0.048 for k in keys)
        res = dict(gates=gates, table=tab, I3=I3_null)
        if null == "a":
            res["I4"] = all(tab[k]["null_R4"]["bias_z"] > 3 for k in ("loy", "repeat_med"))
            res["pred6"] = all(abs(tab[k]["null_R6"]["bias_z"]) < 3 for k in keys)
        else:
            res["pred7"] = all(np.sign(tab[k]["real_R6"]) * tab[k]["null_R6"]["bias_z"] > 3
                               and abs(tab[k]["share_R6"]) < 0.25 for k in keys)
        say(f"   I3 under null ({null}): {'HOLDS' if I3_null else 'FAILS'}"
            + (f"; I4 {'HOLDS' if res['I4'] else 'FAILS'}; prediction 6 {'holds' if res['pred6'] else 'fails'}"
               if null == "a" else f"; prediction 7 {'holds' if res['pred7'] else 'fails'}"))
        out_c3[null] = res
    if "a" in out_c3 and "b" in out_c3:
        out_c3["I3"] = out_c3["a"]["I3"] and out_c3["b"]["I3"]
    return out_c3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(R46 / "partC"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    pre = "smoke_" if a.smoke else ""
    res = {"C1": c1(out, pre, a.smoke), "C2": c2(out, pre, a.smoke), "C3": c3(out, pre, a.smoke)}
    C1, C2, C3 = res["C1"], res["C2"], res["C3"]
    preds = {}
    if "R6" in C1 and "R4" in C1:
        preds["1 I1 for R4 and R6"] = C1["R6"]["I1"] and C1["R4"]["I1"]
        hl = C1["R6"]["quantities"].get("hl_dec", {})
        others = [k for k in C1["R6"]["I2_fails"] if k != "hl_dec"]
        preds["2 I2 except hl_dec, whose ratio < 0.67"] = (0.90 <= C1["R6"]["pooled_coverage"] <= 0.98
                                                         and not others and hl.get("ratio", 9) < 0.67)
    if C2:
        ci = C2["quantities"]["hl_dec"]["ci95"]
        preds["3 C2 hl_dec interval within 1-3"] = 1.0 <= ci[0] and ci[1] <= 3.0
    if C3:
        preds["4 H6 worse than R6 on campaign 27 by > 0.1"] = C3["h6"]["pred4"]
        if "a" in C3:
            preds["5 I4"] = C3["a"]["I4"]
            preds["6 R6 finds none on null (a)"] = C3["a"]["pred6"]
        if "b" in C3:
            preds["7 R6 finds some on null (b), < 1/4 of real"] = C3["b"]["pred7"]
        if "I3" in C3:
            preds["8 I3"] = C3["I3"]
    res["predictions"] = preds
    say("\n== predictions")
    for k, v in preds.items():
        say(f"   {k}: {'holds' if v else 'FAILS'}")
    (out / "compare.json").write_text(json.dumps(res, indent=1, default=float))
    say(f"\nwrote {out / 'compare.json'}")


if __name__ == "__main__":
    main()
