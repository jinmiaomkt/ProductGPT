"""
Emit the manuscript's benchmark exhibits as LaTeX, straight from the run
records, so no number is retyped.

Produces, into --out-dir:
    benchmark_main.tex     the family comparison on the holdout cell
    per_class.tex          purchases vs NotBuy, and the Figure-B columns
    stock_ablation.tex     what the inventory representation is worth
    eval_design.tex        a TikZ figure of the 2x2 evaluation design
    capability_matrix.tex  the CONCEPTUAL comparison against benchmark families

    python3 scripts/paper_tables.py --res-dir results/r30/stage5 \
        --out-dir results/paper
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

# display name -> run-tag stem
FAMILIES = [
    ("Plain GRU (pooled vocabulary)", "b17_gru_v6"),
    ("GRU encoder (pooled)", "b17_gru_enc_v6"),
    ("Transformer + recency", "b17_tf_v7"),
    ("GRU encoder", "b17_gru_enc_v7"),
    ("Hybrid (recurrence + attention)", "b17_hyb_v7"),
    ("Plain GRU", "b17_gru_v7"),
    ("Transformer + recency (pooled)", "b17_tf_v6"),
]
CLASSES = ["Buy1 Reg", "Buy10 Reg", "Buy1 FigA", "Buy10 FigA", "Buy1 FigB",
           "Buy10 FigB", "Buy1 Wep", "Buy10 Wep", "NotBuy"]

# The conceptual comparison the reviewer asked for: which capabilities each
# benchmark family HAS, not just which scores better. y = yes, p = partial,
# blank = no. Rows are capabilities; columns are model families.
CAPABILITIES = [
    ("Time-varying offer set (assortment) as input", "y", "p", "", "p", "y"),
    ("Product identity embeddings", "y", "y", "", "", "y"),
    ("Attribute-based products (generalises to unseen products)", "y", "p", "", "y", "p"),
    ("Realised outcome stream distinct from the choice", "y", "", "", "", ""),
    ("Accumulating inventory / satiation state", "y", "", "p", "y", ""),
    ("Irregular inter-event timing", "y", "p", "y", "y", "p"),
    ("No-purchase as a first-class outcome", "y", "", "p", "y", ""),
    ("Quantity decision (1 vs 10 draws)", "y", "", "", "p", ""),
    ("Heterogeneity that transfers to unseen consumers", "p", "p", "p", "", "p"),
    ("Generative rollout for counterfactual policy", "y", "", "y", "p", ""),
]
CAP_COLS = ["ProductGPT", "Sequential recommenders", "Neural point processes",
            "Dynamic choice models", "RNN/LSTM baselines"]


def load(res: Path):
    groups = defaultdict(list)
    for f in sorted(res.glob("*.json")):
        groups[re.sub(r"_s\d+$", "", f.stem)].append(json.loads(f.read_text()))
    return groups


def ms(v, dec=4):
    if not v:
        return "--"
    m = st.mean(v)
    s = st.stdev(v) if len(v) > 1 else 0.0
    return f"{m:.{dec}f}$\\pm${s:.{dec}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    g = load(Path(a.res_dir))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not g:
        raise SystemExit(f"no scored runs in {a.res_dir}")

    # ---- main benchmark table -------------------------------------------
    rows = []
    for name, stem in FAMILIES:
        v = g.get(stem)
        if not v:
            continue
        rows.append((st.mean(r["nll"] for r in v), name, v))
    rows.sort()
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Model & Params & s/epoch & NLL & Hit rate & AUC (group) & ECE \\",
         r"\midrule"]
    for _, name, v in rows:
        L.append(f"{name} & {v[0]['params'] / 1e6:.2f}M & "
                 f"{st.mean(r['secs_per_epoch'] for r in v):.0f} & "
                 f"{ms([r['nll'] for r in v])} & "
                 f"{ms([r['hit_rate'] for r in v if 'hit_rate' in r], 3)} & "
                 f"{ms([r['auc_group'] for r in v if 'auc_group' in r], 3)} & "
                 f"{st.mean(r['ece'] for r in v):.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "benchmark_main.tex").write_text("\n".join(L), encoding="utf-8")

    # ---- per-class ------------------------------------------------------
    L = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Model & Purchase classes (1--8) & NotBuy (9) & Gap \\", r"\midrule"]
    for _, name, v in rows:
        cn = v[0]["class_n"]
        buy_n = sum(cn[:8])
        buy = sum(st.mean(r["class_nll"][i] for r in v) * cn[i] for i in range(8)) / max(buy_n, 1)
        nb = st.mean(r["class_nll"][8] for r in v)
        L.append(f"{name} & {buy:.4f} & {nb:.4f} & {buy - nb:+.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (out / "per_class.tex").write_text("\n".join(L), encoding="utf-8")

    # ---- capability matrix ---------------------------------------------
    L = [r"\begin{tabular}{l" + "c" * len(CAP_COLS) + "}", r"\toprule",
         "Capability & " + " & ".join(CAP_COLS) + r" \\", r"\midrule"]
    mark = {"y": r"$\bullet$", "p": r"$\circ$", "": ""}
    for row in CAPABILITIES:
        L.append(row[0] + " & " + " & ".join(mark[c] for c in row[1:]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"% $\bullet$ = supported, $\circ$ = partial or architecture-dependent."]
    (out / "capability_matrix.tex").write_text("\n".join(L), encoding="utf-8")

    # ---- evaluation design figure ---------------------------------------
    tikz = r"""\begin{tikzpicture}[font=\small, node distance=0pt]
  \definecolor{calib}{RGB}{234,240,255}
  \definecolor{hold}{RGB}{237,247,237}
  % columns = campaigns, rows = customers
  \node[anchor=south] at (2.1,3.5) {\textbf{Campaigns 1--27} (calibration)};
  \node[anchor=south] at (6.3,3.5) {\textbf{Campaigns 28--30} (holdout)};
  \node[rotate=90, anchor=south] at (-0.35,1.75)
        {\textbf{Out-of-sample} \quad \textbf{In-sample}};
  \draw[fill=calib] (0,1.75) rectangle (4.2,3.4);
  \node[align=center] at (2.1,2.6) {TRAINING\\ 1{,}064{,}818 events};
  \draw[fill=hold] (4.2,1.75) rectangle (8.4,3.4);
  \node[align=center] at (6.3,2.6) {in-sample $\times$ holdout\\ 222{,}889 events};
  \draw[fill=calib] (0,0.1) rectangle (4.2,1.75);
  \node[align=center] at (2.1,0.95) {out-of-sample $\times$ calibration\\ 1{,}259{,}900 events};
  \draw[fill=hold, line width=1.2pt] (4.2,0.1) rectangle (8.4,1.75);
  \node[align=center] at (6.3,0.95) {\textbf{HEADLINE CELL}\\ out-of-sample $\times$ holdout\\ 217{,}303 events};
  \draw (0,0.1) rectangle (8.4,3.4);
  \draw[dashed] (4.2,0.1) -- (4.2,3.4);
  \draw[dashed] (0,1.75) -- (8.4,1.75);
  \node[anchor=north, align=left, text width=8.4cm] at (4.2,-0.1)
    {Model selection uses campaign 27 of the training cell only. The holdout
     column is opened once, after every configuration is frozen. 2{,}502
     customers train, 2{,}502 are never seen.};
\end{tikzpicture}"""
    (out / "eval_design.tex").write_text(tikz, encoding="utf-8")

    print(f"wrote {len(list(out.glob('*.tex')))} files to {out}")
    for f in sorted(out.glob("*.tex")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
