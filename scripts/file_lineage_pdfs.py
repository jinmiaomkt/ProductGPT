"""
File the R33/R34 lineage papers into the OneDrive folders, with consistent names.

Downloads from a library come out named "A LOGIT MODEL OF BRAND CHOICE...pdf",
"ertdae574b.pdf" or worse. This identifies each PDF by its title text and moves
it to the right K folder under the agreed convention

    Author[_Author]_Year_topic_Venue.pdf

    python scripts/file_lineage_pdfs.py                 # dry run, says what it would do
    python scripts/file_lineage_pdfs.py --apply
    python scripts/file_lineage_pdfs.py --src "C:/some/folder" --apply

Identification uses the file name first, then the first page's text if a PDF
reader is available. Anything it cannot place is listed and left alone.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

DEST_ROOT = Path(r"C:\Users\jinmiao\OneDrive - Singapore Management University"
                 r"\E2 Genshim Impact\Parametric Response Model\Inventory Kernel Lineage")

# (folder, final name, distinctive phrases -- matched case-insensitively)
PAPERS = [
    ("K1 Decay and smoothing", "Guadagni_Little_1983_logit_brand_choice_scanner_MktSci.pdf",
     ["logit model of brand choice", "guadagni"]),
    ("K1 Decay and smoothing", "Seetharaman_2004_distributed_lag_state_dependence_MktSci.pdf",
     ["multiple sources of state dependence", "distributed lag"]),
    ("K2 Attribute satiation and substitution",
     "McAlister_1982_attribute_satiation_variety_seeking_JCR.pdf",
     ["dynamic attribute satiation", "attribute satiation model"]),
    ("K2 Attribute satiation and substitution",
     "Lattin_McAlister_1985_substitutes_complements_JMR.pdf",
     ["substitute and complementary relationships", "variety-seeking model to identify"]),
    ("K3 Inventory and purchase timing",
     "Ailawadi_Neslin_1998_promotion_consumption_JMR.pdf",
     ["buying more and consuming it faster", "effect of promotion on consumption"]),
    ("K3 Inventory and purchase timing", "Gupta_1988_when_what_how_much_JMR.pdf",
     ["when, what, and how much to buy", "impact of sales promotions on when"]),
    ("K4 State dependence vs heterogeneity",
     "Keane_1997_heterogeneity_state_dependence_JBES.pdf",
     ["modeling heterogeneity and state dependence"]),
    ("K4 State dependence vs heterogeneity",
     "Heckman_1981_heterogeneity_state_dependence_chapter.pdf",
     ["heterogeneity and state dependence"]),
    ("K4 State dependence vs heterogeneity",
     "Dube_Hitsch_Rossi_2009_state_dependence_consumer_inertia_NBER_w14912.pdf",
     ["alternative explanations for consumer inertia"]),
]


def first_page_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader          # older name
        except ImportError:
            return ""
    try:
        return (PdfReader(str(path)).pages[0].extract_text() or "")[:4000]
    except Exception:
        return ""


def identify(path: Path) -> tuple | None:
    hay = re.sub(r"[^a-z0-9 ]+", " ", path.stem.lower())
    for folder, name, phrases in PAPERS:
        if any(re.sub(r"[^a-z0-9 ]+", " ", p) in hay for p in phrases):
            return folder, name
    text = re.sub(r"\s+", " ", first_page_text(path).lower())
    if text:
        for folder, name, phrases in PAPERS:
            if any(p in text for p in phrases):
                return folder, name
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Downloads"))
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    src = Path(a.src)
    pdfs = sorted(p for p in src.glob("*.pdf") if p.is_file())
    if not pdfs:
        print(f"no PDFs in {src}")
        return
    placed, unknown = [], []
    for p in pdfs:
        hit = identify(p)
        if not hit:
            unknown.append(p)
            continue
        dest = DEST_ROOT / hit[0] / hit[1]
        placed.append((p, dest))
        print(f"{'MOVE' if a.apply else 'would move'}  {p.name}\n"
              f"        -> {hit[0]} / {hit[1]}")
        if a.apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                print("        (already there, skipped)")
            else:
                shutil.move(str(p), str(dest))
    for p in unknown:
        print(f"UNMATCHED  {p.name}   (left in place)")
    print(f"\n{len(placed)} filed, {len(unknown)} unmatched"
          + ("" if a.apply else "   -- dry run, pass --apply to move them"))
    missing = [n for folder, n, _ in PAPERS if not (DEST_ROOT / folder / n).exists()]
    if missing:
        print("\nstill missing:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
