"""R49 diagnosis: bf16 rounding or a leak?  Synthetic customers only, no data.

On 28 Oct 2026 shard 0 of each arm stopped at `llm_zeroshot.score --selfcheck`,
before scoring any real data (bound 0.1 nats on the largest |log p| difference
over the nine letters):

    arm B  Qwen3.6-27B (job 44500)  t3 0.55   t4_before 0.26
    arm A  Qwen3-32B   (job 44501)  t3 0.72   t4_before 0.87

Both comparisons change the transcript's length, so every bf16 matrix product
changes shape and may round differently.  This separates rounding from a leak
with the real model, in bf16, on the job's GPUs:

  repeat  the same transcript twice: run-to-run nondeterminism
  t4c     only later decisions changed: the length and every shape stay the same,
          so rounding cannot differ and any change at occasions <= t is
          information flowing backwards (a leak)
  t4      later decisions and items changed, the length changes (selfcheck T4)
  t3      one pass vs per-occasion prefixes (selfcheck T3)

Each comparison is reported three ways: the selfcheck's statistic (the largest
difference over all nine letters), |difference in log p of the true letter| (what
enters the NLL; mean and max), and the total-variation distance between the two
predicted distributions (mean and max).  Everything is repeated with the output
head in float32 (head32), which removes the bf16 rounding of the logits
themselves.  Run for Qwen3-8B too, on one GPU as in R48, so both arms are judged
against the pipeline that produced the R48 result.

    python scripts/r49_numerics_gpu.py --model <dir> [--yarn 2] [--n-gpus 2] --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from llm_zeroshot.prompt import system_prompt, transcript  # noqa: E402
from llm_zeroshot.score import (answer_positions, compare_logp as compare,  # noqa: E402
                                letter_token_ids, load_model, score_ids)
from llm_zeroshot.synthetic import (perturb_decisions_from, perturb_from,  # noqa: E402
                                    synthetic_customer)

# (S, t, seed, per-occasion prefixes too)
CUSTOMERS = [(40, 20, 2, True), (40, 20, 3, True), (40, 20, 4, True), (300, 150, 5, False)]
STATS = ("max", "true_mean", "true_max", "tv_mean", "tv_max")


def make_scorer(model, letter_ids, head32):
    if not head32:
        return lambda ids, tpos: score_ids(model, ids, tpos, letter_ids, "")[0]
    import torch
    W = model.get_output_embeddings().weight.float()            # once: a few GB
    letters = list(letter_ids)

    def score32(ids, tpos):
        x = torch.tensor([list(ids)], device=model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
            q = torch.tensor(list(tpos), device=h.device) - 1
            logits = h[q].to(W.device).float() @ W.T
            l9 = logits[:, letters] - torch.logsumexp(logits, -1, keepdim=True)
            logp9 = l9 - torch.logsumexp(l9, -1, keepdim=True)
        return logp9.double().cpu().numpy()
    return score32


def one_customer(score, tok, system, letter_ids, S, t, seed, prefixes):
    lab, prev, lto, obt, inv = synthetic_customer(S, seed)

    def ids_of(lab_, obt_):
        text, answers = transcript(lab_, prev, lto, obt_, inv, S, system)
        return answer_positions(tok, text, answers, letter_ids)

    ids, tpos = ids_of(lab, obt)
    lp = score(ids, tpos)
    out = {"repeat": compare(lp, score(ids, tpos), lab)}
    ids3, tpos3 = ids_of(perturb_decisions_from(lab, t), obt)
    if len(ids3) != len(ids):
        raise SystemExit("t4c perturbation changed the transcript length")
    lp3 = score(ids3, tpos3)
    out["t4c_before"] = compare(lp[: t + 1], lp3[: t + 1], lab[: t + 1])
    out["t4c_after"] = compare(lp[t + 1:], lp3[t + 1:], lab[t + 1:])
    ids2, tpos2 = ids_of(*perturb_from(lab, obt, t))
    lp2 = score(ids2, tpos2)
    out["t4_before"] = compare(lp[: t + 1], lp2[: t + 1], lab[: t + 1])
    out["t4_after"] = compare(lp[t + 1:], lp2[t + 1:], lab[t + 1:])
    if prefixes:
        lpp = np.stack([score(ids[: p + 1], [p])[0] for p in tpos])
        out["t3"] = compare(lp, lpp, lab)
        k = np.arange(S)
        out["nll_onepass"] = float(-lp[k, lab - 1].mean())
        out["nll_prefix"] = float(-lpp[k, lab - 1].mean())
    return {"S": S, "t": t, "seed": seed, "tokens": len(ids), **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--yarn", type=float, default=0.0)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import transformers
    from parametric.state import product_table
    system = system_prompt(product_table())
    tok, model = load_model(a.model, "cuda", a.yarn, a.n_gpus, strict=True)
    letter_ids = letter_token_ids(tok)
    res = {"model": a.model, "yarn": a.yarn, "n_gpus": a.n_gpus,
           "model_type": model.config.model_type, "transformers": transformers.__version__,
           "torch": torch.__version__,
           "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "dtype": str(next(model.parameters()).dtype)}
    for head32 in (False, True):
        key = "head32" if head32 else "bf16"
        score = make_scorer(model, letter_ids, head32)
        res[key] = [one_customer(score, tok, system, letter_ids, *c) for c in CUSTOMERS]
        worst = {}
        for name in ("repeat", "t4c_before", "t4c_after", "t4_before", "t4_after", "t3"):
            rows = [r[name] for r in res[key] if name in r]
            worst[name] = {s: max(r[s] for r in rows) for s in STATS}
            print(f"[r49 numerics] {key:6s} {name:10s} " +
                  "  ".join(f"{s} {worst[name][s]:.2e}" for s in STATS), flush=True)
        res[key + "_worst"] = worst
        nll = [(r["nll_onepass"], r["nll_prefix"]) for r in res[key] if "nll_prefix" in r]
        print(f"[r49 numerics] {key:6s} NLL one pass vs prefixes: " +
              "; ".join(f"{x:.4f} vs {y:.4f}" for x, y in nll), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[r49 numerics] wrote {a.out}")


if __name__ == "__main__":
    main()
