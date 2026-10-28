"""R43 step 2: read the two generation jobs and apply the pre-registered rule.

    python scripts/r43_step2_read.py \
        --g5 results/r43/step2_g5.json --g6 results/r43/step2_g6.json \
        [--r36i ~/ProductGPT/work_r36i/results/r36i/r36i.json]

Refuses unless both jobs scored the same customers against the same band and
the same real functionals, and neither job logged a problem.

Decision (EXPERIMENTS.md R43 step 2): G6 - G5 c06 free-running energy, paired by
training seed, counts as a difference only if it has the same sign on at least
4 of 5 seeds and |mean| > the band's q95. Everything else is reported with no
decision attached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MIN_SAME_SIGN = 4


def load(p: str):
    js = json.loads(Path(p).read_text(encoding="utf-8"))
    npz = np.load(Path(p).with_name(Path(p).stem + "_functionals.npz"))
    return js, npz


def energies(js: dict, arm: str) -> dict:
    return {int(s): r["energy"] for s, r in js["arms"][arm]["runs"].items()}


def paired(x: dict, y: dict, band: float) -> dict:
    seeds = sorted(set(x) & set(y))
    d = np.array([x[s] - y[s] for s in seeds])
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    counts = bool(max(pos, neg) >= MIN_SAME_SIGN and abs(d.mean()) > band)
    return {"seeds": seeds, "per_seed": d.tolist(), "mean": float(d.mean()),
            "positive": pos, "negative": neg, "counts": counts}


def show(label: str, r: dict) -> None:
    per = ", ".join(f"{v:+.3f}" for v in r["per_seed"])
    print(f"  {label:<26} {per}   mean {r['mean']:+.3f}   +{r['positive']}/-{r['negative']}"
          f"   {'COUNTS as a difference' if r['counts'] else 'no difference'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g5", default="results/r43/step2_g5.json")
    ap.add_argument("--g6", default="results/r43/step2_g6.json")
    ap.add_argument("--r36i", default=None)
    a = ap.parse_args()
    g5, n5 = load(a.g5)
    g6, n6 = load(a.g6)

    # ------------------------------------------------------------- refusals
    bad = []
    if g5["family"] != "g5" or g6["family"] != "g6":
        bad.append(f"families are {g5['family']} / {g6['family']}")
    if g5["smoke"] or g6["smoke"]:
        bad.append("a smoke output was given")
    for k in ("customer_sha256", "n_users", "n_rep", "chunk", "max_events"):
        if g5[k] != g6[k]:
            bad.append(f"{k}: {g5[k]} vs {g6[k]}")
    if g5["null_band"]["q95"] != g6["null_band"]["q95"]:
        bad.append(f"band q95 {g5['null_band']['q95']} vs {g6['null_band']['q95']}")
    if not np.array_equal(n5["real"], n6["real"], equal_nan=True):
        bad.append("the real functionals differ")
    for js in (g5, g6):
        bad += [f"{js['family']}: {p}" for p in js["problems"]]
    if bad:
        raise SystemExit("REFUSING:\n  " + "\n  ".join(bad))

    q = g5["null_band"]["q95"]
    print(f"customers {g5['n_users']} x {g5['n_rep']} replicates (sha {g5['customer_sha256']}), "
          f"band q95 {q:.4f}; code {g5['head']} (g5) / {g6['head']} (g6)")
    print(f"band reproduces R36i's q95: {g5['null_band']['reproduces_r36i']}")

    arms = {**{k: v for k, v in g5["arms"].items()}, **{k: v for k, v in g6["arms"].items()}}
    print(f"\n{'arm':<10}{'n':>3}{'energy':>9}{'sd':>8}{'min':>8}{'max':>8}{'x band':>8}"
          f"{'in':>4}{'AUC':>7}  per seed")
    for nm, v in arms.items():
        s = v["summary"]
        per = ", ".join(f"{r['energy']:+.3f}" for r in v["runs"].values())
        print(f"{nm:<10}{s['n']:>3}{s['energy_mean']:>+9.3f}{s['energy_sd']:>8.3f}"
              f"{s['energy_min']:>+8.3f}{s['energy_max']:>+8.3f}{s['energy_mean'] / q:>8.1f}"
              f"{s['inside_band']:>4}{s['auc_mean']:>7.3f}  {per}")

    E = {k: energies(g5 if k in g5["arms"] else g6, k) for k in arms}

    # ------------------------------------------------------------- decision
    d = paired(E["fr_g6"], E["fr_g5c06"], q)
    print(f"\n=== decision (pre-registered): free running, G6 - G5 c06 ===")
    show("G6 - G5 c06", d)
    if d["counts"] and d["mean"] < 0:
        verdict = "GEN 6 BETTER: adopted as the generator, with the gain reported"
    elif d["counts"] and d["mean"] > 0:
        verdict = "GEN 6 WORSE: not adopted; gen 5 stays the generator"
    else:
        verdict = ("NO DIFFERENCE: gen 6 adopted for what it can do (any history length, "
                   "a fixed cost per occasion), not for generation quality")
    print(f"  verdict: {verdict}")

    # ---------------------------------------------- no decision attached
    print("\n=== reported with no decision attached ===")
    show("A - G5 c06 (free)", paired(E["fr_A"], E["fr_g5c06"], q))
    show("A - G6 (free)", paired(E["fr_A"], E["fr_g6"], q))
    show("G6 - G5 c06 (teacher f.)", paired(E["tf_g6"], E["tf_g5c06"], q))
    for fam, fr, tf in (("G5 c06", "fr_g5c06", "tf_g5c06"), ("G6", "fr_g6", "tf_g6")):
        s = sorted(set(E[fr]) & set(E[tf]))
        share = [1.0 - E[tf][k] / E[fr][k] for k in s]
        print(f"  {fam}: teacher forcing removes "
              + ", ".join(f"{x:.0%}" for x in share) + " of the free-running distance")
    mk = np.mean(list(E["markov1"].values()))
    for fr in ("fr_g5c06", "fr_g6", "fr_A"):
        lo = min(E[fr].values())
        print(f"  {fr}: lowest seed {lo:+.3f} against Markov's mean {mk:+.3f} "
              f"({'Markov ahead on every seed' if lo > max(E['markov1'].values()) else 'not on every seed'})")

    lab = ["Buy1Reg", "Buy10Reg", "Buy1FigA", "Buy10FigA", "Buy1FigB", "Buy10FigB",
           "Buy1Wep", "Buy10Wep", "NotBuy"]
    print(f"\n{'shares':>10}" + "".join(f"{x:>10}" for x in lab))
    print(f"{'real':>10}" + "".join(f"{x:>10.3f}" for x in g5["real"]["shares"]))
    for nm, v in arms.items():
        print(f"{nm:>10}" + "".join(f"{x:>10.3f}" for x in v["summary"]["shares_mean"]))
    print("\ntop functionals and minutes per run:")
    for nm, v in arms.items():
        mins = np.mean([r["seconds"] for r in v["runs"].values()]) / 60
        print(f"  {nm:<10} {', '.join(v['summary']['top_functionals'][:5])}   {mins:.1f} min")

    if a.r36i:
        r = json.loads(Path(a.r36i).expanduser().read_text(encoding="utf-8"))
        print("\ncontinuity with R36i (same draw seeds):")
        for nm in ("armC", "iid", "markov1"):
            old = {s: x["energy"] for s, x in r["arms"][nm]["runs"].items()}
            new = {s: x["energy"] for s, x in g5["arms"][nm]["runs"].items()}
            same = all(abs(old[s] - new[s]) < 1e-9 for s in old if s in new)
            print(f"  {nm:<8} R36i " + ", ".join(f"{old[s]:+.4f}" for s in sorted(old))
                  + "   now " + ", ".join(f"{new[s]:+.4f}" for s in sorted(new))
                  + f"   {'IDENTICAL' if same else 'differs'}")


if __name__ == "__main__":
    main()
