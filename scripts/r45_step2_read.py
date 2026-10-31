"""R45 step 2 -- generation on chopped histories: the pre-registered reading.

Run it once, after scripts/r45_generate.py has finished (EXPERIMENTS.md, R45
step 2):

    python3 scripts/r45_step2_read.py --r43 ~/ProductGPT/work_gen6/results/r43/step2_g6.json

Written on 31 Oct 2026 with the step-2 design, before any step-2 rollout ran.

Energy differences are paired by training seed 1-3 (the rollout seed is 1000 +
training seed in every arm); positive = the first arm is farther from the real
customers. Rule (R43's, at 3 seeds): a difference counts if it has the same sign
on all 3 seeds and |mean| > the band's q95; "worse" or "better" by its sign,
otherwise "no difference".
  1. damage      Z(k) - F(0)   trained and generating on censored histories,
                               against the full data            k = 1, 2, 4
  2. fix         ZF(k) - Z(k)  the censoring flag               k = 1, 2, 4
  3. input only  F(k) - F(0)   a full-history model handed a censored history
Free running decides. DECISION: a generation cost of censoring exists if rule 1
says "worse" at any depth; otherwise none is detectable.
Descriptive, no decision: Z(k) - F(k); every rule on the teacher-forced arms;
the depth curve; AUC, NotBuy share, the share of the distance teacher forcing
removes, top functionals.

Predictions checked:
  1. every free-running run is outside the band
  2. rule 1: no difference at every depth
  3. rule 2: no difference at every depth
  4. rule 3: no difference at every depth
  5. teacher forced, tf F(k) - F(0) is positive on the mean at every depth and
     grows with depth
  6. teacher forced, tf Z(k) < tf F(k) on the mean at every depth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st

SEEDS = ("1", "2", "3")
DEPTHS = (1, 2, 4)
N_USERS, N_REP = 300, 8
CUSTOMERS = "be33f73d819b13eb"          # R36i / R43's 300 customers
ROWS = 22490                            # their holdout rows


def names(pre: str) -> list:
    out = [f"{pre}_F_k0"]
    for k in DEPTHS:
        out += [f"{pre}_F_k{k}", f"{pre}_Z_k{k}", f"{pre}_ZF_k{k}"]
    return out


def load(path: str) -> dict:
    j = json.load(open(path))
    if j.get("smoke"):
        raise SystemExit(f"{path} is a smoke run")
    if j.get("problems"):
        raise SystemExit(f"{path} logged problems: {j['problems']}")
    if (j["n_users"], j["n_rep"], j["customer_sha256"]) != (N_USERS, N_REP, CUSTOMERS):
        raise SystemExit(f"not R43's customers: {j['n_users']} x {j['n_rep']}, "
                         f"sha {j['customer_sha256']}")
    if j["real"]["holdout_rows"] != ROWS:
        raise SystemExit(f"{j['real']['holdout_rows']} holdout rows, not {ROWS}")
    for nm in names("fr") + names("tf"):
        runs = j["arms"].get(nm, {}).get("runs", {})
        if sorted(runs) != list(SEEDS):
            raise SystemExit(f"{nm}: seeds {sorted(runs)}, expected {list(SEEDS)}")
        for s, r in runs.items():
            ig = r["integrity"]
            if (ig["rows_mismatched"] or ig["real_rows_altered"] or ig["rows_outside_frame"]
                    or ig["rows_alive"] != N_REP * ROWS):
                raise SystemExit(f"{nm} s{s}: integrity {ig}")
            if r["null_q95"] != j["null_band"]["q95"]:
                raise SystemExit(f"{nm} s{s}: a different band")
    e = {nm: [j["arms"][nm]["runs"][s]["energy"] for s in SEEDS] for nm in names("fr") + names("tf")}
    seen = {}
    for nm, v in e.items():
        if tuple(v) in seen:
            raise SystemExit(f"{nm} and {seen[tuple(v)]} have identical energies: a bug")
        seen[tuple(v)] = nm
    return j


def energy(j, nm) -> list:
    return [j["arms"][nm]["runs"][s]["energy"] for s in SEEDS]


def rule(d: list, q95: float) -> dict:
    m = st.mean(d)
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    verdict = ("worse" if m > q95 and pos == 3 else
               "better" if m < -q95 and neg == 3 else "no difference")
    return {"d": d, "mean": m, "sd": st.stdev(d), "pos": pos, "neg": neg, "verdict": verdict}


def diff(j, a: str, b: str, q95: float) -> dict:
    return rule([x - y for x, y in zip(energy(j, a), energy(j, b))], q95)


def show(label: str, r: dict) -> None:
    print(f"    {label:<24}" + "  ".join(f"{x:+.3f}" for x in r["d"])
          + f"   mean {r['mean']:+.3f}  sd {r['sd']:.3f}  farther on {r['pos']}/3"
          + f"   -> {r['verdict']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="results/r45/step2.json")
    ap.add_argument("--r43", default=None, help="R43 step 2's step2_g6.json, for continuity")
    ap.add_argument("--out", default="results/r45/step2_read.json")
    a = ap.parse_args()
    print(f"reader sha256 {hashlib.sha256(open(__file__, 'rb').read()).hexdigest()}")
    j = load(a.inp)
    q95 = j["null_band"]["q95"]
    print(f"gates passed: 20 arms x seeds 1-3, R43's {N_USERS} customers x {N_REP} replicates "
          f"({ROWS:,} holdout rows), every run's integrity clean, one band, no two arms equal")
    print(f"band q95 {q95:.7f}: R36i/R43's band "
          + ("REPRODUCED" if j["null_band"]["reproduces_r36i"] else
             "NOT reproduced -- R43's numbers cannot be quoted alongside"))

    cont = None
    if a.r43:
        r43 = json.load(open(a.r43))
        cont = {}
        for mine, theirs in (("fr_F_k0", "fr_g6"), ("tf_F_k0", "tf_g6")):
            d = [energy(j, mine)[i] - r43["arms"][theirs]["runs"][s]["energy"]
                 for i, s in enumerate(SEEDS)]
            cont[mine] = d
            print(f"continuity: {mine} - R43 {theirs}, seeds 1-3: "
                  + "  ".join(f"{x:+.2e}" for x in d)
                  + ("   REPRODUCED" if max(abs(x) for x in d) < 1e-6 else "   differs"))
    print()

    print("===== R45 step 2 -- energy (mean over seeds 1-3, seed sd), x band, AUC, NotBuy")
    table = {}
    for pre in ("fr", "tf"):
        for nm in names(pre):
            runs = j["arms"][nm]["runs"]
            e = energy(j, nm)
            table[nm] = {"energy": st.mean(e), "sd": st.stdev(e),
                         "auc": st.mean(runs[s]["auc"] for s in SEEDS),
                         "notbuy": st.mean(runs[s]["shares"][8] for s in SEEDS),
                         "inside": sum(bool(runs[s]["inside_band"]) for s in SEEDS),
                         "seconds": st.mean(runs[s]["seconds"] for s in SEEDS),
                         "top": j["arms"][nm]["summary"]["top_functionals"][:3]}
            t = table[nm]
            print(f"  {nm:<9} {t['energy']:+.3f} ({t['sd']:.3f})  {t['energy'] / q95:5.1f}x  "
                  f"inside {t['inside']}/3  AUC {t['auc']:.3f}  NotBuy {t['notbuy']:.3f}  "
                  f"{t['seconds'] / 60:.1f} min  top {', '.join(t['top'])}")
        print()
    print(f"  real NotBuy {j['real']['shares'][8]:.3f}")
    print()

    rules = {}
    print("RULES, free running (paired by seed 1-3; positive = the first arm is farther)")
    for k in DEPTHS:
        rules[f"1_k{k}"] = diff(j, f"fr_Z_k{k}", "fr_F_k0", q95)
        rules[f"2_k{k}"] = diff(j, f"fr_ZF_k{k}", f"fr_Z_k{k}", q95)
        rules[f"3_k{k}"] = diff(j, f"fr_F_k{k}", "fr_F_k0", q95)
    for i, label in ((1, "1. Z(k) - F(0)"), (2, "2. ZF(k) - Z(k)"), (3, "3. F(k) - F(0)")):
        for k in DEPTHS:
            show(f"{label}, k={k}", rules[f"{i}_k{k}"])
    cost = [k for k in DEPTHS if rules[f"1_k{k}"]["verdict"] == "worse"]
    print()
    print("DECISION: " + (f"a generation cost of censoring EXISTS at k = {cost}" if cost else
                          "no generation cost of censoring is detectable at any depth"))
    print()

    desc = {}
    print("DESCRIPTIVE, no decision")
    for k in DEPTHS:
        desc[f"fr Z(k) - F(k), k={k}"] = diff(j, f"fr_Z_k{k}", f"fr_F_k{k}", q95)
    for k in DEPTHS:
        desc[f"tf Z(k) - F(0), k={k}"] = diff(j, f"tf_Z_k{k}", "tf_F_k0", q95)
        desc[f"tf ZF(k) - Z(k), k={k}"] = diff(j, f"tf_ZF_k{k}", f"tf_Z_k{k}", q95)
        desc[f"tf F(k) - F(0), k={k}"] = diff(j, f"tf_F_k{k}", "tf_F_k0", q95)
        desc[f"tf Z(k) - F(k), k={k}"] = diff(j, f"tf_Z_k{k}", f"tf_F_k{k}", q95)
    for label, r in desc.items():
        show(label, r)
    removed = {nm[3:]: [1 - t / f for t, f in zip(energy(j, "tf" + nm[2:]), energy(j, nm))]
               for nm in names("fr")}
    print("  share of the free-running distance teacher forcing removes, seeds 1-3:")
    for nm, v in removed.items():
        print(f"    {nm:<7} " + "  ".join(f"{x:.2f}" for x in v))
    print()

    tfF = [table[f"tf_F_k{k}"]["energy"] - table["tf_F_k0"]["energy"] for k in DEPTHS]
    p = {"1": sum(table[nm]["inside"] for nm in names("fr")) == 0,
         "2": all(rules[f"1_k{k}"]["verdict"] == "no difference" for k in DEPTHS),
         "3": all(rules[f"2_k{k}"]["verdict"] == "no difference" for k in DEPTHS),
         "4": all(rules[f"3_k{k}"]["verdict"] == "no difference" for k in DEPTHS),
         "5": all(x > 0 for x in tfF) and tfF[0] < tfF[1] < tfF[2],
         "6": all(table[f"tf_Z_k{k}"]["energy"] < table[f"tf_F_k{k}"]["energy"] for k in DEPTHS)}
    print("PREDICTIONS")
    print(f"  1. every free-running run outside the band: {'holds' if p['1'] else 'fails'}")
    print(f"  2. rule 1 no difference at every depth: {'holds' if p['2'] else 'fails'}")
    print(f"  3. rule 2 no difference at every depth: {'holds' if p['3'] else 'fails'}")
    print(f"  4. rule 3 no difference at every depth: {'holds' if p['4'] else 'fails'}")
    print(f"  5. tf F(k) - F(0) positive and growing with depth: {'holds' if p['5'] else 'fails'} "
          f"({', '.join(f'{x:+.3f}' for x in tfF)})")
    print(f"  6. tf Z(k) < tf F(k) at every depth: {'holds' if p['6'] else 'fails'}")

    out = {"table": table, "rules": rules, "cost_depths": cost, "descriptive": desc,
           "tf_removed": removed, "predictions": p, "continuity": cont, "q95": q95}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
