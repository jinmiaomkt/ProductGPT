"""R36i: the yardstick one statistic at a time -- 27 energy-distance tables.

    python scripts/r36i_per_statistic.py
    python scripts/r36i_per_statistic.py --npz results/r36i/r36i_functionals.npz \
        --tex docs/r36i_per_statistic_tables.tex --csv results/r36i/per_statistic.csv

The joint energy distance says HOW FAR a generator lands from the real
histories; it does not say on WHICH statistic.  This applies the same yardstick
to each of the 27 statistics alone: the same scaler, the same missing-value
rule, and the same band settings (n_boot 120, seed 3), so each statistic's band
is built on the same customer splits as the joint band.  CPU only; it reads the
per-customer matrices r36i_rerun.py saves.

WHAT IT CANNOT SEE.  A one-dimensional distance is blind, by construction, to
how the statistics move together.  A generator can pass all 27 tables and still
fail jointly, which is why the joint distance, the classifier and the variogram
score stay in the report.

TWO BLOCKS.  Arm C and teacher forcing condition on the realised holdout -- the
customer's actual decisions, or actual acquisitions and decisions -- which no
policy simulation has.  They locate a failure; they are not candidate
simulators, so they are reported apart from the generators a manager could use.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "simulation"))

import seq_metrics as sm  # noqa: E402

N_BOOT, NULL_SEED = 120, 3     # r36i_rerun.py's band settings

GENERATORS = [("fr_b33", "ProductGPT, free running"),
              ("fr_b35_ss015", "scheduled sampling, $p = 0.15$"),
              ("fr_b35_ss035", "scheduled sampling, $p = 0.35$"),
              ("markov1", "first-order Markov chain"),
              ("iid", "independent draws")]
CONTROLS = [("armC", "arm C: real decisions, lottery re-drawn"),
            ("tf_b33", "teacher forcing")]

DECISIONS = ["Buy1, regular banner", "Buy10, regular banner",
             "Buy1, figure A", "Buy10, figure A", "Buy1, figure B",
             "Buy10, figure B", "Buy1, weapon banner", "Buy10, weapon banner",
             "NotBuy"]


def describe(name: str) -> tuple:
    """(title, definition), both LaTeX-safe."""
    fixed = {
        "spend": ("Spend", "total pulls bought over the holdout (Buy1 $=1$, Buy10 $=10$)"),
        "incidence": ("Purchase incidence", "share of holdout occasions with a purchase"),
        "longest_notbuy_run": ("Longest NotBuy run",
                               "longest run of consecutive NotBuy occasions"),
        "mean_buy_run": ("Mean buying run", "mean length of runs of consecutive purchases"),
        "max_buy_run": ("Longest buying run", "longest run of consecutive purchases"),
        "switch_rate": ("Banner switch rate",
                        "share of consecutive purchases that change banner"),
        "spend_hhi": ("Spend concentration",
                      "Herfindahl index of spend across the four banners"),
        "p_buy10_given_buy10": ("Escalation",
                                "$P(\\text{Buy10 at } t \\mid \\text{Buy10 at } t-1)$"),
        "p_buy_given_buy": ("Persistence",
                            "$P(\\text{purchase at } t \\mid \\text{purchase at } t-1)$"),
    }
    if name in fixed:
        return fixed[name]
    if name.startswith("share_d"):
        d = int(name[len("share_d"):])
        return (f"Share of {DECISIONS[d - 1]}",
                "share of holdout occasions with this decision")
    if name.startswith("acf_lag"):
        k = name[len("acf_lag"):]
        return (f"Autocorrelation, lag {k}",
                f"autocorrelation of the purchase indicator at lag {k}")
    if name.startswith("sat_k"):
        k = int(name[len("sat_k"):])
        kk = f"{k}+" if k == sm.SAT_BUCKETS - 1 else str(k)
        return (f"Satiation signature, $k = {kk}$",
                f"$P(\\text{{buy on a banner}} \\mid \\text{{holding {kk} copies of "
                f"its featured product}})$, pooled over figure A, figure B and weapon")
    raise KeyError(name)


def runs_of(npz, arm: str) -> dict:
    pre = f"{arm}__"
    return {k[len(pre):]: npz[k] for k in npz.files if k.startswith(pre)}


def fmt_level(v: float) -> str:
    if not np.isfinite(v):
        return "--"
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(REPO / "results/r36i/r36i_functionals.npz"))
    ap.add_argument("--tex", default=str(REPO / "docs/r36i_per_statistic_tables.tex"))
    ap.add_argument("--csv", default=str(REPO / "results/r36i/per_statistic.csv"))
    ap.add_argument("--overview", default=str(REPO / "docs/r36i_per_statistic_overview.tex"),
                    help="one table: every statistic against every arm, energy in multiples "
                         "of that statistic's band")
    a = ap.parse_args()

    npz = np.load(a.npz, allow_pickle=False)
    names = [str(x) for x in npz["functionals"]]
    if names != sm.functional_names():
        raise SystemExit("the npz's functionals are not seq_metrics' 27, in order")
    F_real = npz["real"]
    scaler = sm.Scaler.fit(F_real)
    Zr = scaler(F_real)
    blocks = [("Generators a manager could use", GENERATORS),
              ("Diagnostic controls: condition on the realised holdout", CONTROLS)]
    present = {arm: runs_of(npz, arm) for _, arms in blocks for arm, _ in arms}
    missing = [arm for arm, r in present.items() if not r]
    if missing:
        print(f"[warn] no runs in the npz for: {', '.join(missing)}")
    print(f"[npz] {F_real.shape[0]} real customers; "
          + ", ".join(f"{arm} x{len(r)}" for arm, r in present.items() if r))

    rows, tex = [], []
    tex.append("% Generated by scripts/r36i_per_statistic.py -- do not edit by hand.\n"
               f"% Source: {Path(a.npz).name}; band n_boot {N_BOOT}, seed {NULL_SEED}.\n")
    for k, name in enumerate(names):
        title, definition = describe(name)
        null = sm.null_band(Zr[:, [k]], n_boot=N_BOOT, seed=NULL_SEED)
        q95 = float(np.quantile(null, 0.95))
        r_mean = float(np.nanmean(F_real[:, k]))
        r_undef = float(np.isnan(F_real[:, k]).mean())
        body = []
        for head, arms in blocks:
            body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{head}}}}}\\\\")
            for arm, label in arms:
                runs = present[arm]
                if not runs:
                    continue
                e, z, ins, gm, gu = [], [], [], [], []
                for lab, F in sorted(runs.items()):
                    c = sm.compare(Zr[:, [k]], scaler(F)[:, [k]], n_boot=N_BOOT,
                                   seed=NULL_SEED)
                    e.append(c["stat"]); z.append(c["z"]); ins.append(c["inside_band"])
                    # all-undefined (e.g. no generated history ever holds 3+ copies)
                    col = F[:, k]
                    gm.append(float(np.nanmean(col)) if np.isfinite(col).any() else np.nan)
                    gu.append(float(np.isnan(F[:, k]).mean()))
                    rows.append({"statistic": name, "arm": arm, "run": lab,
                                 "energy": c["stat"], "band_q95": q95, "z": c["z"],
                                 "inside_band": int(c["inside_band"]),
                                 "real_mean": r_mean, "gen_mean": gm[-1],
                                 "real_undefined": r_undef, "gen_undefined": gu[-1]})
                e = np.asarray(e)
                sd = f" $\\pm$ {e.std(ddof=1):.3f}" if len(e) > 1 else ""
                body.append(f"\\quad {label} & ${e.mean():+.3f}${sd} & "
                            f"{np.mean(z):.1f} & {sum(ins)}/{len(ins)} & "
                            f"{fmt_level(float(np.nanmean(gm)))} & "
                            f"{100 * np.mean(gu):.0f}\\% \\\\")
            body.append("\\midrule")
        body = body[:-1]
        tex.append(
            "\\noindent\\begin{minipage}{\\linewidth}\\centering\\small\n"
            f"\\captionof{{table}}{{\\textbf{{{k + 1}. {title}}}: {definition}. "
            f"Real-vs-real band $q_{{0.95}} = {q95:.3f}$; real mean "
            f"{fmt_level(r_mean)}, undefined for {100 * r_undef:.0f}\\% of customers.}}\n"
            "\\begin{tabular}{@{}lrrcrr@{}}\n\\toprule\n"
            " & energy & $z$ & inside & gen.\\ mean & undefined\\\\\n\\midrule\n"
            + "\n".join(body) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{minipage}\\par\\bigskip\n")
        print(f"  {k + 1:>2} {name:<20} band {q95:.3f}")

    Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tex).write_text("\n".join(tex), encoding="utf-8")

    # The overview: mean energy over an arm's runs, in multiples of the
    # statistic's own band, so 27 statistics with different bands share a scale.
    # Grey marks a statistic inside the band on every run of that arm.
    cols = [arm for _, arms in blocks for arm, _ in arms if present[arm]]
    short = {"fr_b33": "free run.", "fr_b35_ss015": "SS 0.15", "fr_b35_ss035": "SS 0.35",
             "markov1": "Markov", "iid": "i.i.d.", "armC": "arm C", "tf_b33": "teacher f."}
    n_gen = sum(1 for arm, _ in GENERATORS if present[arm])
    ov = ["% Generated by scripts/r36i_per_statistic.py -- do not edit by hand.",
          "\\begin{tabular}{@{}l" + "r" * n_gen + "|" + "r" * (len(cols) - n_gen) + "@{}}",
          "\\toprule",
          f" & \\multicolumn{{{n_gen}}}{{c|}}{{generators}} & "
          f"\\multicolumn{{{len(cols) - n_gen}}}{{c}}{{controls}}\\\\",
          "statistic & " + " & ".join(short[c] for c in cols) + "\\\\", "\\midrule"]
    for name in names:
        cells = []
        for arm in cols:
            rs = [r for r in rows if r["statistic"] == name and r["arm"] == arm]
            ratio = np.mean([r["energy"] for r in rs]) / rs[0]["band_q95"]
            txt = f"{ratio:.1f}"
            if all(r["inside_band"] for r in rs):
                txt = f"\\textcolor{{gray}}{{{txt}}}"
            cells.append(txt)
        ov.append(f"{describe(name)[0]} & " + " & ".join(cells) + "\\\\")
    outside = [sum(1 for name in names
                   if not any(r["inside_band"] for r in rows
                              if r["statistic"] == name and r["arm"] == arm))
               for arm in cols]
    ov += ["\\midrule", "outside on every run & " + " & ".join(map(str, outside)) + "\\\\",
           "\\bottomrule", "\\end{tabular}"]
    Path(a.overview).write_text("\n".join(ov) + "\n", encoding="utf-8")
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.tex}, {a.overview} and {a.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
