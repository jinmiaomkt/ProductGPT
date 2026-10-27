"""R46 part B read-out: every arm on the R36i yardstick, and the verdicts.

    python scripts/r46b_compare.py            # after the r46b_rollout array

Reads the array's files (runs/r46/partB/*.json, *_functionals.npz) and R36i's
(work_r36i/results/r36i/r36i.json, r36i_functionals.npz). Nothing is
re-simulated. Computed here, by the rules pre-registered in EXPERIMENTS.md
(R46 part B), so reading the log involves no judgement about thresholds:
  - every gate the elements logged, and that all of them share R36i's real set
  - the R36i-style summary of every arm, ours and R36i's
  - the 27 per-statistic energies (and the satiation block), each against its
    own split-half band from the same 120 splits
  - R36's T2 rule, the verdicts W1-W5 for R6, the ladder steps, and the
    pre-registered predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "simulation", REPO / "scripts"):
    sys.path.insert(0, str(p))

import seq_metrics as sm  # noqa: E402
from r36i_rerun import NULL_SEED, PREFIX_Q95, summarise  # noqa: E402

R46B = Path("/storage/home/jinmiao/ProductGPT/runs/r46/partB")
R36I = Path("/storage/home/jinmiao/ProductGPT/work_r36i/results/r36i")
RUNGS = ("M0", "M1", "R0", "R1", "R1b", "R2", "R3", "R4", "R6")
LAGGED = ("M1", "R1", "R1b", "R2", "R3", "R4", "R6")
STEPS = [("R1", "R0", "last decision"), ("R1b", "R1", "habit"), ("R2", "R1", "run structure"),
         ("R3", "R2", "long memory"), ("R4", "R3", "holdings, pity, guarantee"),
         ("R6", "R4", "persistent taste")]
R36I_ARMS = ("armC", "tf_b33", "fr_b33", "fr_b35_ss015", "fr_b35_ss035", "iid", "markov1")
T2_SINGLE = ("spend", "incidence", "longest_notbuy_run", "switch_rate")
N_BOOT = 120


def differs(x: dict, y: dict, band: float) -> dict:
    """R43's rule: x - y counts only if it has one sign on every seed pair and
    |mean| > band. Seeds pair in order (17-19 against 1-3, or against 17-19)."""
    kx, ky = sorted(x, key=int), sorted(y, key=int)
    if len(kx) != len(ky):
        raise SystemExit(f"cannot pair seeds {kx} with {ky}")
    d = np.array([x[a] - y[b] for a, b in zip(kx, ky)])
    same = bool(len(d) and (np.all(d > 0) or np.all(d < 0)))
    return {"pairs": [f"{a}-{b}" for a, b in zip(kx, ky)], "per_pair": d.tolist(),
            "mean": float(d.mean()), "counts": bool(same and abs(d.mean()) > band)}


def per_statistic(Zr: np.ndarray, Z: np.ndarray, names: list) -> dict:
    """Energy on each functional alone, and on the satiation block, each against
    its own split-half band (same splits: seed NULL_SEED, N_BOOT)."""
    out = {}
    cols = {n: [i] for i, n in enumerate(names)}
    cols["satiation"] = [i for i, n in enumerate(names) if n.startswith("sat_k")]
    for n, c in cols.items():
        r = sm.compare(Zr[:, c], Z[:, c], n_boot=N_BOOT, seed=NULL_SEED)
        out[n] = {"energy": r["stat"], "q95": r["null_q95"], "inside": bool(r["inside_band"])}
    return out


def t2_pass(run: dict) -> dict:
    """R36's rule: inside its own band on >= 4 of the 5 pre-specified functionals,
    and the C2ST AUC CI contains 0.5."""
    ps = run["per_statistic"]
    five = {n: ps[n]["inside"] for n in T2_SINGLE + ("satiation",)}
    lo, hi = run["auc_ci"]
    auc_ok = bool(lo <= 0.5 <= hi)
    return {"inside_of_five": int(sum(five.values())), "five": five, "auc_ci_has_half": auc_ok,
            "pass": bool(sum(five.values()) >= 4 and auc_ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partb", default=str(R46B))
    ap.add_argument("--r36i", default=str(R36I))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pb, r36 = Path(a.partb), Path(a.r36i)
    problems = []

    # ------------------------------------------------------------ R36i's runs
    rj = json.loads((r36 / "r36i.json").read_text())
    with np.load(r36 / "r36i_functionals.npz") as z:
        R = {k: z[k] for k in z.files}
    names = [str(x) for x in R["functionals"]]
    real = R["real"]
    scaler = sm.Scaler.fit(real)
    Zr = scaler(real)
    null = sm.null_band(Zr, n_boot=N_BOOT, seed=NULL_SEED)
    q = float(np.quantile(null, 0.95))
    print(f"[band] q95 {q:.7f} (R36i {PREFIX_Q95:.7f}): "
          f"{'reproduced' if abs(q - PREFIX_Q95) < 1e-9 else 'NOT REPRODUCED'}")
    if abs(q - PREFIX_Q95) >= 1e-9:
        problems.append("band does not reproduce")

    arms = {}
    for nm in R36I_ARMS:
        runs = {}
        for s, rec in rj["arms"][nm]["runs"].items():
            rec = dict(rec)
            rec["per_statistic"] = per_statistic(Zr, scaler(R[f"{nm}__{s}"]), names)
            rec["T2"] = t2_pass(rec)
            runs[s] = rec
        arms[nm] = {"source": "R36i job 42879", "runs": runs}

    # ------------------------------------------------------------ our elements
    files = sorted(p for p in pb.glob("*.json") if p.stem in
                   ["ref"] + [f"{m}_{r}" for m in ("fr", "tf") for r in RUNGS])
    expect = 1 + 2 * len(RUNGS)
    print(f"[elements] {len(files)} of {expect} present")
    if len(files) != expect:
        problems.append(f"{len(files)} of {expect} elements present")
    gates = {}
    for f in files:
        e = json.loads(f.read_text())
        with np.load(f.with_name(f.stem + "_functionals.npz")) as z:
            E = {k: z[k] for k in z.files}
        if not np.array_equal(E["real"], real):
            problems.append(f"{f.stem}: real functionals differ from R36i's")
        if e["problems"]:
            problems += [f"{f.stem}: {p}" for p in e["problems"]]
        gates[f.stem] = e["gates"]
        if f.stem == "ref":
            for nm, runs in e["runs"].items():
                for s, rec in runs.items():
                    d = abs(rec["energy"] - rec.get("r36i_energy", np.nan))
                    print(f"[B3] {nm} {s}: energy {rec['energy']:.9f}  R36i "
                          f"{rec.get('r36i_energy', float('nan')):.9f}  |diff| {d:.1e}")
            continue
        runs = {}
        for s, rec in e["runs"].items():
            rec["per_statistic"] = per_statistic(Zr, scaler(E[f"{f.stem}__{s}"]), names)
            rec["T2"] = t2_pass(rec)
            runs[s] = rec
        arms[f.stem] = {"source": f"R46 part B, {e['tag']} at {e['head']}", "runs": runs}

    for nm, v in arms.items():
        v["summary"] = summarise(v["runs"])
        v["summary"]["T2_pass"] = int(sum(r["T2"]["pass"] for r in v["runs"].values()))
    E = {k: {s: r["energy"] for s, r in v["runs"].items()} for k, v in arms.items()}

    # ------------------------------------------------------------ verdicts
    have = lambda *k: all(x in E for x in k)
    V = {}
    if have("fr_R6", "fr_b33", "markov1", "fr_b35_ss035", "tf_R6"):
        V["W1_beats_free_running_b33"] = differs(E["fr_R6"], E["fr_b33"], q)
        V["W2_beats_markov1"] = differs(E["fr_R6"], E["markov1"], q)
        V["W3_beats_ss035"] = differs(E["fr_R6"], E["fr_b35_ss035"], q)
        for k in ("W1_beats_free_running_b33", "W2_beats_markov1", "W3_beats_ss035"):
            V[k]["holds"] = bool(V[k]["counts"] and V[k]["mean"] < 0)
        V["W4_passes_T2"] = {"draws_passing": arms["fr_R6"]["summary"]["T2_pass"],
                             "holds": arms["fr_R6"]["summary"]["T2_pass"] == len(E["fr_R6"])}
        loop = differs(E["fr_R6"], E["tf_R6"], q)
        loop["holds"] = not loop["counts"]
        loop["fr_over_tf"] = {s: E["fr_R6"][s] / E["tf_R6"][s] for s in E["fr_R6"]}
        V["W5_loop_benign"] = loop
    V["ladder"] = {}
    for hi, lo, what in STEPS:
        if have(f"fr_{hi}", f"fr_{lo}"):
            V["ladder"][f"{hi}-{lo}"] = dict(differs(E[f"fr_{hi}"], E[f"fr_{lo}"], q), information=what)
    if have("fr_M1", "markov1"):
        V["ladder"]["M1-markov1"] = dict(differs(E["fr_M1"], E["markov1"], q),
                                         information="same model class, different fitting sample")
    V["exploratory_beat_markov1"] = {r: differs(E[f"fr_{r}"], E["markov1"], q)
                                     for r in RUNGS if have(f"fr_{r}", "markov1")}
    V["exploratory_beat_markov1_rungs"] = [r for r, d in V["exploratory_beat_markov1"].items()
                                           if d["counts"] and d["mean"] < 0]

    # ------------------------------------------------------------ predictions
    P = {}
    mean = {k: float(np.mean(list(v.values()))) for k, v in E.items()}
    if "W1_beats_free_running_b33" in V:
        P["1_every_lagged_rung_beats_fr_b33"] = all(
            (lambda d: d["counts"] and d["mean"] < 0)(differs(E[f"fr_{r}"], E["fr_b33"], q))
            for r in LAGGED if f"fr_{r}" in E)
        P["2_W2"] = V["W2_beats_markov1"]["holds"]
        P["3_W3"] = V["W3_beats_ss035"]["holds"]
        P["4_W4_fails"] = not V["W4_passes_T2"]["holds"]
        ratio = V["W5_loop_benign"]["fr_over_tf"]
        P["5a_fr_over_tf_below_2_every_draw"] = all(x < 2 for x in ratio.values())
        P["5b_tf_R6_farther_than_tf_b33"] = mean["tf_R6"] > mean["tf_b33"]
    lad = V["ladder"]
    if all(k in lad for k in ("R1-R0", "R2-R1", "R3-R2", "R4-R3", "R6-R4", "R1b-R1")):
        gains = {k: -lad[k]["mean"] for k in ("R2-R1", "R3-R2", "R4-R3", "R6-R4", "R1b-R1")}
        P["6a_R2_largest_after_R1"] = max(gains, key=gains.get) == "R2-R1"
        P["6b_R1b_does_not_count"] = not lad["R1b-R1"]["counts"]
        top3 = arms["fr_R3"]["summary"]["top_functionals"][:5]
        P["6c_R4_beats_R3_with_sat_in_R3_top"] = bool(
            lad["R4-R3"]["counts"] and lad["R4-R3"]["mean"] < 0 and any(t.startswith("sat_k") for t in top3))
    obs_nb = rj["real"]["shares"][8]
    P["7_notbuy_within_0.05"] = {r: abs(arms[f"fr_{r}"]["summary"]["shares_mean"][8] - obs_nb) <= 0.05
                                 for r in LAGGED if f"fr_{r}" in arms}

    # ------------------------------------------------------------ report
    order = ["armC", "tf_b33"] + [f"tf_{r}" for r in RUNGS] + ["fr_b33", "fr_b35_ss015", "fr_b35_ss035",
                                                               "iid", "markov1"] + [f"fr_{r}" for r in RUNGS]
    print(f"\n=== energy on the R36i yardstick: 300 customers x 8 replicates, band q95 {q:.4f} ===")
    print(f"{'arm':<14}{'n':>3}{'energy':>9}{'sd':>8}{'min':>8}{'max':>8}{'x band':>8}{'z':>8}"
          f"{'in':>4}{'AUC':>7}{'T2':>4}  top functionals")
    for nm in order:
        if nm not in arms:
            continue
        s = arms[nm]["summary"]
        print(f"{nm:<14}{s['n']:>3}{s['energy_mean']:>+9.3f}{s['energy_sd']:>8.3f}{s['energy_min']:>+8.3f}"
              f"{s['energy_max']:>+8.3f}{s['energy_mean'] / q:>8.1f}{s['z_mean']:>8.1f}{s['inside_band']:>4}"
              f"{s['auc_mean']:>7.3f}{s['T2_pass']:>4}  {', '.join(s['top_functionals'][:3])}")
    lab = ["Buy1Reg", "Buy10Reg", "Buy1FigA", "Buy10FigA", "Buy1FigB", "Buy10FigB", "Buy1Wep", "Buy10Wep",
           "NotBuy"]
    print("\n=== decision shares over the holdout rows (pooled) ===")
    print(f"{'':>14}" + "".join(f"{x:>10}" for x in lab))
    print(f"{'real':>14}" + "".join(f"{x:>10.3f}" for x in rj["real"]["shares"]))
    for nm in order:
        if nm in arms:
            print(f"{nm:>14}" + "".join(f"{x:>10.3f}" for x in arms[nm]["summary"]["shares_mean"]))
    print("\n=== per-statistic energy, mean over draws (inside its own band: *) ===")
    keys = list(names) + ["satiation"]
    print(f"{'arm':<14}" + "".join(f"{k[:9]:>10}" for k in keys))
    for nm in order:
        if nm not in arms:
            continue
        rr = list(arms[nm]["runs"].values())
        row = ""
        for k in keys:
            e = np.mean([r["per_statistic"][k]["energy"] for r in rr])
            ins = all(r["per_statistic"][k]["inside"] for r in rr)
            row += f"{e:>9.3f}{'*' if ins else ' '}"
        print(f"{nm:<14}" + row)
    print("\n=== pre-registered verdicts and predictions (EXPERIMENTS.md, R46 part B) ===")
    for k, v in V.items():
        print(f"  {k}: {json.dumps(v, default=float)}")
    for k, v in P.items():
        print(f"  prediction {k}: {json.dumps(v, default=float)}")
    print("\n=== gates ===")
    for k, v in gates.items():
        print(f"  {k}: {json.dumps(v, default=float)}")
    res = {"q95": q, "arms": arms, "verdicts": V, "predictions": P, "gates": gates,
           "problems": problems}
    dest = Path(a.out) if a.out else pb / "compare.json"
    dest.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {dest}")
    if problems:
        print("\nPROBLEMS:")
        for p_ in problems:
            print("  " + p_)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
