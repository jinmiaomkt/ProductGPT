"""One figure that explains R36: how the model is run forward, how the result is
measured, and what goes wrong.

    python scripts/fig_generative_failure.py

Writes results/figs/generative_failure.{png,pdf}. Every number is taken from
the ledger (EXPERIMENTS.md R36c/e/f); nothing here is recomputed.

Panels
  0  the generative process: what the model conditions on, and what feeds back
  A  the 2x2 of controls: which feedback breaks it
  B  the yardstick: energy distance against the real-vs-real band
  C  what goes wrong: decision shares, observed vs teacher-forced vs free-running
  D  the remedy: scheduled sampling, a dose response
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED = "#1A1A1A", "#6B7280"
GREEN, GREENBG = "#1E6B34", "#DDEFE0"
RED, REDBG = "#8C1D18", "#F6DADA"
BLUE, AMBER = "#2A4A9C", "#B8710A"
GREY, GREYBG = "#4B5563", "#ECEEF2"

# ---------------------------------------------------------------- numbers
NULL_Q95 = 0.078                       # split-half real-vs-real, 300 customers
ENERGY = [                             # (label, value, colour, note)
    ("arm C: real decisions,\nlottery re-rolled",        -0.010, GREEN, "inside"),
    ("teacher-forced: decisions\nsampled, streams real",  0.307, GREEN, "inside*"),
    ("first-order Markov chain",                          0.711, GREY,  "outside"),
    ("scheduled sampling p = 0.35",                       0.928, AMBER, "outside"),
    ("scheduled sampling p = 0.15",                       1.390, AMBER, "outside"),
    ("independent draws",                                 2.190, GREY,  "outside"),
    ("trained model, free-running",                       2.352, RED,   "outside"),
]
CLASSES = ["Buy1\nReg", "Buy10\nReg", "Buy1\nFigA", "Buy10\nFigA", "Buy1\nFigB",
           "Buy10\nFigB", "Buy1\nWep", "Buy10\nWep", "NotBuy"]
SHARE_REAL = [0.142, 0.008, 0.176, 0.084, 0.031, 0.021, 0.023, 0.012, 0.503]
SHARE_TF   = [0.140, 0.031, 0.169, 0.095, 0.051, 0.048, 0.036, 0.020, 0.410]
SHARE_FREE = [0.089, 0.048, 0.078, 0.087, 0.132, 0.113, 0.134, 0.135, 0.183]
DOSE_P  = [0.00, 0.15, 0.35]
DOSE_E  = [2.352, 1.390, 0.928]
DOSE_SD = [0.198, 0.358, 0.260]
MARKOV = 0.711


def box(ax, xy, w, h, text, fc, ec=INK, fs=9.5, bold=False, tc=INK):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.3)


def arrow(ax, p, q, color=INK, lw=1.6, rad=0.0, label=None, lpos=None, lcolor=None):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16,
                                 lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text(*(lpos or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)), label,
                ha="center", va="center", fontsize=8.5, color=lcolor or color,
                bbox=dict(fc="white", ec="none", pad=1.5))


def main() -> None:
    fig = plt.figure(figsize=(14.5, 13.2))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.9, 1.0, 1.0],
                  hspace=0.55, wspace=0.34, left=0.06, right=0.98, top=0.93, bottom=0.07)

    # ============================================================ panel 0
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("How the model is run forward (“free-running”)",
                 loc="left", fontsize=13, fontweight="bold", color=INK, pad=6)

    # the real-history strip
    for i in range(9):
        ax.add_patch(FancyBboxPatch((0.03 + i * 0.036, 0.50), 0.03, 0.16,
                                    boxstyle="round,pad=0.002", fc=GREYBG, ec=GREY, lw=0.8))
    ax.text(0.19, 0.78, "campaigns 1–27: the customer’s REAL history",
            ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.text(0.19, 0.36, "the model conditions on all of this,\nexactly as in training",
            ha="center", fontsize=8.5, color=MUTED, linespacing=1.3)
    ax.plot([0.375, 0.375], [0.12, 0.95], ls="--", color=MUTED, lw=1.2)
    ax.text(0.375, 0.98, "holdout starts (campaign 28)", ha="center", va="top",
            fontsize=8.5, color=MUTED)

    # the loop
    box(ax, (0.50, 0.58), 0.18, 0.30,
        "STATE at occasion t\n\noffers on the banners (real)\nproducts held, with duplicates\nprevious decision",
        GREYBG, fs=8.8)
    box(ax, (0.70, 0.58), 0.14, 0.22, "MODEL\n\nsamples one of\nnine decisions", "#E6ECFA", ec=BLUE, fs=9, bold=False)
    box(ax, (0.895, 0.58), 0.15, 0.22, "GACHA ENVIRONMENT\n\npity rules draw\nwhat was received", "#FFF1DC", ec=AMBER, fs=9)
    arrow(ax, (0.34, 0.58), (0.41, 0.58), color=GREY)
    arrow(ax, (0.59, 0.58), (0.63, 0.58), color=INK, label="y_t", lpos=(0.61, 0.72))
    arrow(ax, (0.77, 0.58), (0.82, 0.58), color=INK, label="pulls", lpos=(0.795, 0.72))
    # feedback: two red arrows leaving the BOTTOM of the model and environment
    # boxes and looping back beneath them into the state box. Negative rad
    # bends the arc downward for a right-to-left arrow.
    arrow(ax, (0.895, 0.47), (0.55, 0.43), color=RED, lw=2.0, rad=-0.38,
          label="feeds back: NEW holdings", lpos=(0.735, 0.12), lcolor=RED)
    arrow(ax, (0.70, 0.47), (0.47, 0.43), color=RED, lw=2.0, rad=-0.30,
          label="feeds back: the decision just made", lpos=(0.585, 0.25), lcolor=RED)
    ax.text(0.70, -0.02,
            "Repeat for every occasion in campaigns 28–30. Nothing after the dashed line is real: "
            "the model is conditioning on what IT decided and on what the lottery gave it.",
            ha="center", fontsize=9, color=INK, style="italic")

    # ============================================================ panel A
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A.  Which feedback breaks it?  Only closing the loop through BOTH",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=6)
    cells = [  # (col, row, text, fc, ec, tc)
        (0, 0, "the data itself", GREYBG, GREY, MUTED),
        (1, 0, "−0.010\nINSIDE the band\n\narm C", GREENBG, GREEN, GREEN),
        (0, 1, "0.307\nINSIDE the band\n\nteacher-forced", GREENBG, GREEN, GREEN),
        (1, 1, "2.352\nOUTSIDE (30× band)\n\nfree-running", REDBG, RED, RED),
    ]
    x0, y0, cw, ch = 0.30, 0.10, 0.33, 0.33
    for c, r, text, fc, ec, tc in cells:
        y = y0 + (1 - r) * ch
        ax.add_patch(FancyBboxPatch((x0 + c * cw, y), cw - 0.01, ch - 0.01,
                                    boxstyle="round,pad=0.004", fc=fc, ec=ec, lw=1.6))
        ax.text(x0 + c * cw + cw / 2, y + ch / 2, text, ha="center", va="center",
                fontsize=10.5, color=tc, fontweight="bold", linespacing=1.35)
    ax.text(x0 + cw / 2, 0.80, "acquisitions\nREAL", ha="center", fontsize=10, color=INK)
    ax.text(x0 + 1.5 * cw, 0.80, "acquisitions\nSIMULATED", ha="center", fontsize=10, color=INK)
    ax.text(0.27, y0 + 1.5 * ch, "decisions\nREAL", ha="right", va="center", fontsize=10, color=INK)
    ax.text(0.27, y0 + 0.5 * ch, "decisions\nSAMPLED\nby the model", ha="right", va="center",
            fontsize=10, color=INK)
    ax.text(0.5, -0.04,
            "Randomising the lottery alone is fine. Sampling decisions alone is fine.\n"
            "Only when the model’s decisions determine what it receives, AND what it\n"
            "receives informs its next decision, do the sequences stop looking real.",
            ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.35)

    # ============================================================ panel B
    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("B.  The yardstick: how far from real is each generator?",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=6)
    labels = [e[0] for e in ENERGY]; vals = [e[1] for e in ENERGY]
    cols = [e[2] for e in ENERGY]
    ypos = list(range(len(ENERGY)))[::-1]
    ax.axvspan(-0.05, NULL_Q95, color=GREENBG, zorder=0)
    ax.axvline(NULL_Q95, color=GREEN, lw=1.2, ls="--")
    ax.barh(ypos, vals, color=cols, height=0.62, zorder=2)
    for y, v, e in zip(ypos, vals, ENERGY):
        ax.text(max(v, 0) + 0.04, y, f"{v:+.3f}" if v < 0 else f"{v:.3f}",
                va="center", fontsize=9.5, color=e[2], fontweight="bold")
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-0.15, 2.75)
    ax.set_xlabel("energy distance between generated and real sequences  (0 = identical distributions)",
                  fontsize=9.5)
    ax.annotate("two halves of REAL data land\ninside the green band (q95 = 0.078)",
                xy=(NULL_Q95, len(ENERGY) - 1.45), xytext=(0.75, len(ENERGY) - 1.45),
                fontsize=8.5, color=GREEN, va="center",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.0, -0.19,
            "* teacher-forced was measured on 60 customers, where the band is 0.373; inside its own band.\n"
            "Energy distance compares the set of generated histories to the real ones on 27 statistics the\n"
            "model was never fitted to (spend, incidence, run lengths, switching, satiation).",
            transform=ax.transAxes, fontsize=7.8, color=MUTED, va="top", linespacing=1.3)

    # ============================================================ panel C
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("C.  What goes wrong: the model turns customers into compulsive buyers",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=6)
    import numpy as np
    x = np.arange(len(CLASSES)); w = 0.27
    ax.bar(x - w, SHARE_REAL, w, color=GREY, label="observed")
    ax.bar(x,     SHARE_TF,   w, color=BLUE, label="teacher-forced (streams real)")
    ax.bar(x + w, SHARE_FREE, w, color=RED,  label="free-running (own output)")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=8.5)
    ax.set_ylabel("share of occasions", fontsize=9.5)
    ax.set_ylim(0, 0.58)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    # callouts
    ax.annotate("NotBuy collapses\n0.503 → 0.183", xy=(8 + w, 0.183), xytext=(4.3, 0.52),
                fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
    ax.annotate("Buy10 on the weapon banner\n0.012 → 0.135  (11×)", xy=(7 + w, 0.135),
                xytext=(2.6, 0.33), fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))

    # ============================================================ panel D
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("D.  The remedy: scheduled sampling closes 61% of the gap",
                 loc="left", fontsize=12, fontweight="bold", color=INK, pad=6)
    ax.axhspan(-0.05, NULL_Q95, color=GREENBG, zorder=0)
    ax.axhline(NULL_Q95, color=GREEN, lw=1.2, ls="--")
    ax.axhline(MARKOV, color=GREY, lw=1.2, ls=":")
    ax.errorbar(DOSE_P, DOSE_E, yerr=DOSE_SD, fmt="o-", color=AMBER, ecolor=AMBER,
                lw=2, ms=8, capsize=4, zorder=3)
    for p, e in zip(DOSE_P, DOSE_E):
        ax.text(p + 0.012, e + 0.1, f"{e:.3f}", fontsize=9.5, color=AMBER, fontweight="bold")
    ax.text(0.0, MARKOV + 0.06, "first-order Markov chain, 0.711", fontsize=8.5,
            color=GREY, ha="left")
    ax.text(0.355, NULL_Q95 + 0.05, "real-vs-real band, 0.078", fontsize=8.5,
            color=GREEN, ha="right")
    ax.set_xlabel("p = probability the model is fed its OWN previous decision during training",
                  fontsize=9.5)
    ax.set_ylabel("energy distance (lower is better)", fontsize=9.5)
    ax.set_xlim(-0.03, 0.40); ax.set_ylim(-0.05, 2.9)
    ax.set_xticks(DOSE_P)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.0, -0.2,
            "Monotone in p, with the classifier AUC falling in step: a mechanism, not a lucky seed.\n"
            "Costs nothing on one-step accuracy (0.8801 vs 0.8810). Still 12× the band, "
            "still behind a 9×9 transition matrix.",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="top", linespacing=1.3)

    fig.suptitle("Prediction is fine one step at a time; generation fails when the model runs on its own",
                 fontsize=15, fontweight="bold", color=INK, x=0.06, ha="left", y=0.975)
    fig.text(0.06, 0.953,
             "ProductGPT, out-of-sample customers, holdout campaigns 28–30.  "
             "One-step holdout NLL of the same model: 0.8846, calibration error 0.010–0.019.",
             fontsize=10, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"generative_failure.{ext}", dpi=170 if ext == "png" else None,
                    bbox_inches="tight")
    print(f"wrote {OUT / 'generative_failure.png'} and .pdf")


if __name__ == "__main__":
    main()
