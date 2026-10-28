"""R50: R48's bf16 Qwen3-8B against a float32 reference, on a random sample of customers.

Each input is a tag of scored shards from llm_zeroshot.score, as TAG:DIR:NSHARDS:
    --r48    R48's bf16 shards: the rows behind the published headline
    --fp32   the float32 pass on the sampled customers (scripts/r50_fp32_gpu.pbs)
    --rerun  R48's bf16 configuration rerun on the same customers in the same job

Gates (exit non-zero on failure):
    rows       the float32 pass and the rerun hold exactly R48's rows for the
               sampled customers in the two out-of-sample cells: the same
               customers, positions, labels and cells, each once
    reference  the float32 pass's selfcheck shows no leak (t4c <= repeat + 1e-6)
               and its t3 true-letter mean is <= 1e-3

Pre-registered (EXPERIMENTS.md, R50), out-of-sample x holdout cell:
    B = NLL(R48, bf16) - NLL(float32), row-weighted as R48's headline, with a
    95% customer-bootstrap interval, and the verdict band it falls in
Also reported: the same for the calibration cell; the row-level |difference|
and total-variation distance; the spread of per-customer mean differences; the
rerun against R48 (reproduction) and against float32.  Exploratory: B by
position in the 1,024-occasion window.  Aggregates only.

USAGE (HPCC)
    python scripts/r50_fp32_compare.py --r48 qwen3_8b_out:<runs>/r48:2 \
        --fp32 qwen3_8b_fp32_out:<runs>/r50:1 --rerun qwen3_8b_bf16_out:<runs>/r50:1 \
        --out <runs>/r50/r50_fp32_vs_r48.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from llm_zeroshot.score import CELL_NAMES, HEADLINE, compare_logp  # noqa: E402
from r48_pair_gru import CELLS, POS_BINS, Bootstrap, summ  # noqa: E402
from r49_pair_llms import read_tag  # noqa: E402

KEYS = ("uid_hash", "pos", "label", "cell")
BAND, MARGIN = 0.01, 0.02               # verdict bands (R50); MARGIN is the adoption margin
REF_TOL = 1e-3                          # largest float32 t3 true-letter mean


def verdict(b) -> str:
    lo, hi = b["ci95"]
    if -BAND <= lo and hi <= BAND:
        return "a: R48 stands; the interval lies inside +-0.01"
    if abs(b["est"]) < MARGIN:
        return "b: R48 is reported with the float32 correction alongside"
    return "c: bf16 moves R48's NLL by at least the 0.02 adoption margin"


def shard_json(spec: str):
    tag, rest = spec.split(":", 1)
    d, n = rest.rsplit(":", 1)
    return [json.loads((Path(d) / f"{tag}_shard{k}.json").read_text()) for k in range(int(n))]


def main() -> None:
    ap = argparse.ArgumentParser(description="R50: bf16 R48 against a float32 reference")
    ap.add_argument("--r48", required=True, help="TAG:DIR:NSHARDS of R48's bf16 shards")
    ap.add_argument("--fp32", required=True, help="TAG:DIR:NSHARDS of the float32 pass")
    ap.add_argument("--rerun", required=True, help="TAG:DIR:NSHARDS of the bf16 rerun")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    _, R = read_tag(a.r48)
    _, F = read_tag(a.fp32)
    _, Q = read_tag(a.rerun)
    uids = np.unique(F["uid_hash"])
    R = {k: v[np.isin(R["uid_hash"], uids)] for k, v in R.items()}

    # ---- gates
    def same_rows(X):
        return len(X["nll"]) == len(R["nll"]) and all(np.array_equal(X[k], R[k]) for k in KEYS)
    pairs = np.stack([F["uid_hash"].astype(np.int64), F["pos"].astype(np.int64)], 1)
    fj = shard_json(a.fp32)
    checks = [s.get("selfcheck") or {} for s in fj]
    ref_ok = bool(checks) and all(
        c and c["t4c_before_max"] <= c["repeat_max"] + 1e-6 and c["t3_true_mean"] <= REF_TOL
        and min(c["t4_after_max"], c["t4c_after_max"]) >= 1e-3 for c in checks)
    gate = {"customers": int(len(uids)),
            "fp32_rows_equal_r48": bool(same_rows(F)),
            "rerun_rows_equal_r48": bool(same_rows(Q)),
            "duplicate_rows": int(len(pairs) - len(np.unique(pairs, axis=0))),
            "fp32_dtype": [s.get("dtype") for s in fj],
            "rerun_dtype": [s.get("dtype") for s in shard_json(a.rerun)],
            "reference_selfcheck": checks, "reference_ok": ref_ok}
    gate["pass"] = bool(gate["fp32_rows_equal_r48"] and gate["rerun_rows_equal_r48"]
                        and gate["duplicate_rows"] == 0 and ref_ok
                        and set(gate["fp32_dtype"]) == {"float32"}
                        and set(gate["rerun_dtype"]) == {"bfloat16"})
    out = {"gate": gate, "bands": {"inside": BAND, "margin": MARGIN}}
    if not gate["pass"]:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        raise SystemExit("GATE FAILED")

    # ---- statistics
    f64 = lambda x: x.astype(np.float64)          # noqa: E731
    nll_r, nll_f, nll_q = f64(R["nll"]), f64(F["nll"]), f64(Q["nll"])
    lab = R["label"].astype(np.int64)
    cid = np.unique(F["uid_hash"], return_inverse=True)[1]
    bs = Bootstrap(cid, a.draws, a.seed)
    diff = nll_r - nll_f
    pos = R["pos"].astype(np.int64)

    out["cells"] = {}
    for c in CELLS:
        m = R["cell"] == c
        n_c = np.bincount(cid[m], minlength=bs.C)
        per_cust = (np.bincount(cid[m], diff[m], minlength=bs.C)[n_c > 0] / n_c[n_c > 0])
        ad = np.abs(diff[m])
        cmp = compare_logp(f64(R["logp9"][m]), f64(F["logp9"][m]), lab[m])
        out["cells"][CELL_NAMES[c]] = {
            "rows": int(m.sum()), "customers": int((n_c > 0).sum()),
            "B_bf16_minus_fp32": summ(*bs.reps(m, diff)),
            "nll_bf16_r48": summ(*bs.reps(m, nll_r)),
            "nll_fp32": summ(*bs.reps(m, nll_f)),
            "rerun_minus_fp32": summ(*bs.reps(m, nll_q - nll_f)),
            "abs_diff_true_letter": {"mean": float(ad.mean()), "p50": float(np.median(ad)),
                                     "p99": float(np.percentile(ad, 99)), "max": float(ad.max())},
            "tv": {"mean": cmp["tv_mean"], "max": cmp["tv_max"]},
            "largest_over_letters": cmp["max"],
            "per_customer_mean_diff": {
                "sd": float(per_cust.std(ddof=1)), "min": float(per_cust.min()),
                "p05": float(np.percentile(per_cust, 5)), "p95": float(np.percentile(per_cust, 95)),
                "max": float(per_cust.max())},
            "exploratory_B_by_position": [
                dict(pos=[lo, hi], rows=int((m & (pos >= lo) & (pos < hi)).sum()),
                     **summ(*bs.reps(m & (pos >= lo) & (pos < hi), diff)))
                for lo, hi in zip(POS_BINS[:-1], POS_BINS[1:]) if (m & (pos >= lo) & (pos < hi)).any()],
        }
    rep = np.abs(f64(Q["logp9"]) - f64(R["logp9"]))
    out["reproduction_rerun_vs_r48"] = {"max_abs_logp9": float(rep.max()),
                                        "rows_differing": int((rep.max(1) > 0).sum()),
                                        "rows": int(len(rep))}
    B = out["cells"][CELL_NAMES[HEADLINE]]["B_bf16_minus_fp32"]
    out["verdict"] = verdict(B)
    # rule 5: R49's H2 contrast d = NLL(arm) - NLL(8B), both bf16; read conservatively
    # as d + max(0, upper end of B) (EXPERIMENTS.md, R50)
    out["r49_h2_conservative_shift"] = max(0.0, B["ci95"][1])
    out["fp32_run"] = {k: fj[0].get(k) for k in ("seconds", "tokens", "gpu_peak_gb", "numerics",
                                                  "n_gpus", "sample", "sample_seed", "torch",
                                                  "transformers")}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
