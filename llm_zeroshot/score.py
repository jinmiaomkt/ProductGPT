"""R48: score every occasion of the R46 occasion table with a zero-shot LLM.

One forward pass per customer.  At each answer position the model's
full-vocabulary log-probabilities are read for the nine answer letters, giving
two numbers per occasion:

    logp9  (9,)  log p(letter), renormalised over the nine letters: the prediction
    mass         log of the total probability the model put on the nine letters
                 (format adherence; reported, never used to select anything)

Outputs, in --out-dir:
    <tag>_shard<k>.npz   one row per scored occasion:
                         uid_hash, pos, label, cell, nll, logp9 (9), mass
    <tag>_shard<k>.json  counts and per-cell means (aggregates only)
    <tag>_system_prompt.txt  the system prompt, for review

`--dry-run` tokenises every transcript without loading the model weights and
reports the token-length distribution: the CPU smoke test.

Nothing about individual customers is printed.

USAGE (HPCC; see scripts/r48_score_gpu.pbs and scripts/r48_dryrun_cpu.pbs)
    python -m llm_zeroshot.score --table <r46>/occasions_v6_S1024.npz \
        --model /storage/home/jinmiao/ProductGPT/models/Qwen3-8B \
        --out-dir <runs>/r48 --tag qwen3_8b --who out --shard 0 --n-shards 2

R49 (larger models; scripts/r49_score_gpu.pbs) adds:
    --n-gpus 2     bf16 weights split layer by layer over two GPUs (no quantisation)
    --selfcheck    before scoring, T3 and T4 of scripts/test_r48_zeroshot.py on a
                   synthetic customer with the real model on the real device;
                   stops if either shows a difference above --selfcheck-tol
The defaults reproduce R48 exactly.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_zeroshot.prompt import LETTERS, system_prompt, transcript  # noqa: E402

CELL_NAMES = ("train", "val", "insample_users_holdout_period",
              "outsample_users_calib_period", "outsample_users_holdout_period")
HEADLINE = 4


def letter_token_ids(tok) -> List[int]:
    ids = []
    for L in LETTERS:
        enc = tok.encode(L, add_special_tokens=False)
        if len(enc) != 1:
            raise SystemExit(f"answer letter {L!r} is not a single token: {enc}")
        ids.append(enc[0])
    return ids


def answer_positions(tok, text: str, answers: Sequence[int], letter_ids: Sequence[int]
                     ) -> Tuple[List[int], List[int]]:
    """Token ids of the transcript and the token index of each answer letter.

    Fails loudly unless every answer letter is exactly one token, starting at its
    character, whose id is that letter's id.
    """
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    start = {s: i for i, (s, e) in enumerate(offs) if e > s}
    tpos = []
    allowed = set(letter_ids)
    for c in answers:
        i = start.get(c)
        if i is None or offs[i][1] != c + 1 or ids[i] not in allowed:
            raise SystemExit("an answer letter does not tokenise as a single letter token")
        tpos.append(i)
    return ids, tpos


def score_ids(model, ids: Sequence[int], tpos: Sequence[int], letter_ids: Sequence[int],
              device: str):
    """(logp9 (K,9), mass (K,)) as float64 numpy, from one causal forward pass."""
    import torch
    # with --n-gpus 2 the embedding sits on the first GPU and the head on the last
    x = torch.tensor([list(ids)], device=model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
        q = torch.tensor(list(tpos), device=h.device) - 1      # state just before each answer
        logits = model.get_output_embeddings()(h[q])          # (K, V)
        if logits.dtype in (torch.bfloat16, torch.float16):
            logits = logits.float()
        l9 = logits[:, list(letter_ids)] - torch.logsumexp(logits, -1, keepdim=True)
        mass = torch.logsumexp(l9, -1)
        logp9 = l9 - mass[:, None]
    return logp9.double().cpu().numpy(), mass.double().cpu().numpy()


def cell_codes(tab: Dict[str, np.ndarray]) -> np.ndarray:
    from parametric.data import roles
    r = roles(tab)
    code = np.full(tab["label"].shape, -1, np.int8)
    for k, name in enumerate(CELL_NAMES):
        code[r[name]] = k
    return code


def load_model(name: str, device: str, yarn: float, n_gpus: int = 1, strict: bool = False):
    """strict (R49): stop unless every weight the text model needs is in the checkpoint."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device.startswith("cuda") and platform.system() == "Windows":
        raise SystemExit("refusing to use the laptop GPU (19 Oct 2026 crashes); run on HPCC")
    tok = AutoTokenizer.from_pretrained(name)
    v5 = int(transformers.__version__.split(".")[0]) >= 5      # 5.x renamed torch_dtype
    kw = {"dtype" if v5 else "torch_dtype":
          torch.bfloat16 if device.startswith("cuda") else torch.float32,
          "attn_implementation": "sdpa"}
    if yarn:
        if v5:
            raise SystemExit("--yarn is wired for transformers 4.x only (R48, R49 arm A)")
        kw["rope_scaling"] = {"rope_type": "yarn", "factor": float(yarn),
                              "original_max_position_embeddings": 32768}
    if n_gpus > 1:
        if not device.startswith("cuda") or torch.cuda.device_count() < n_gpus:
            raise SystemExit(f"--n-gpus {n_gpus} needs that many visible GPUs")
        kw["device_map"] = "balanced"               # whole layers, about half on each GPU
    model = AutoModelForCausalLM.from_pretrained(name, output_loading_info=strict, **kw)
    if strict:
        model, info = model
        missing = list(info.get("missing_keys", []))
        mismatched = list(info.get("mismatched_keys", []))
        prefixes = sorted({".".join(k.split(".")[:2]) for k in info.get("unexpected_keys", [])})
        print(f"[r48] load: {type(model).__name__}, {len(missing)} missing, "
              f"{len(mismatched)} mis-shaped, unexpected prefixes {prefixes}", flush=True)
        if missing or mismatched:
            raise SystemExit(f"strict load FAILED: missing {missing[:5]}, mis-shaped {mismatched[:5]}")
    model = model.eval() if n_gpus > 1 else model.to(device).eval()
    return tok, model


def selfcheck(model, tok, system: str, letter_ids: Sequence[int], S: int = 40, t: int = 20
              ) -> Dict[str, float]:
    """T3 and T4 of scripts/test_r48_zeroshot.py with the real model on its device.

    t3: max |log p| difference between one pass and per-occasion prefixes
    t4_before: max change at occasions <= t when decision t and everything after
               it change (a leak would show here)
    t4_after: the same at later occasions (must be visibly non-zero)
    In bf16 the first two are rounding-level, not exactly zero.
    """
    from llm_zeroshot.synthetic import perturb_from, synthetic_customer
    lab, prev, lto, obt, inv = synthetic_customer(S, 2)
    text, answers = transcript(lab, prev, lto, obt, inv, S, system)
    ids, tpos = answer_positions(tok, text, answers, letter_ids)
    lp, mass = score_ids(model, ids, tpos, letter_ids, "")
    t3 = 0.0
    for k, p in enumerate(tpos):
        lpk, _ = score_ids(model, ids[:p + 1], [p], letter_ids, "")
        t3 = max(t3, float(np.abs(lpk[0] - lp[k]).max()))
    lab2, obt2 = perturb_from(lab, obt, t)
    text2, answers2 = transcript(lab2, prev, lto, obt2, inv, S, system)
    ids2, tpos2 = answer_positions(tok, text2, answers2, letter_ids)
    lp2, _ = score_ids(model, ids2, tpos2, letter_ids, "")
    return {"t3": t3, "t4_before": float(np.abs(lp2[: t + 1] - lp[: t + 1]).max()),
            "t4_after": float(np.abs(lp2[t + 1:] - lp[t + 1:]).max()),
            "letter_mass": float(np.exp(mass).mean()), "tokens": len(ids)}


def main() -> None:
    ap = argparse.ArgumentParser(description="R48: zero-shot LLM scoring of the occasion table")
    ap.add_argument("--table", required=True, help="R46 occasions_v6_S1024.npz")
    ap.add_argument("--model", required=True, help="HF id or local directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--who", choices=("out", "in", "all"), default="out",
                    help="which customers: out-of-sample (headline), in-sample, or all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--yarn", type=float, default=0.0,
                    help="YaRN factor for >32,768-token transcripts (0 = off)")
    ap.add_argument("--n-gpus", type=int, default=1,
                    help="split the bf16 weights over this many GPUs (R49)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="T3/T4 on a synthetic customer with the real model first (R49)")
    ap.add_argument("--selfcheck-tol", type=float, default=0.1,
                    help="largest allowed T3 / T4-before difference, in nats")
    ap.add_argument("--strict-load", action="store_true",
                    help="stop if any weight is missing or mis-shaped (R49)")
    ap.add_argument("--max-customers", type=int, default=0, help="smoke only")
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenise only; report the token-length distribution")
    a = ap.parse_args()

    from parametric.data import load
    from parametric.state import product_table
    tab = load(Path(a.table))
    cells = cell_codes(tab)
    prod = product_table()
    system = system_prompt(prod)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}_system_prompt.txt").write_text(system, encoding="utf-8")

    n = len(tab["length"])
    sel = [i for i in range(n)
           if a.who == "all" or (tab["is_out"][i] == 1) == (a.who == "out")]
    sel = [i for i in sel if i % a.n_shards == a.shard]
    if a.max_customers:
        sel = sel[: a.max_customers]

    stem = out / f"{a.tag}_shard{a.shard}"
    if not a.dry_run and stem.with_suffix(".npz").exists():
        print(f"[r48] {stem.name}.npz exists; nothing to do")
        return

    if a.dry_run:
        from transformers import AutoTokenizer
        tok, model = AutoTokenizer.from_pretrained(a.model), None
    else:
        tok, model = load_model(a.model, a.device, a.yarn, a.n_gpus, a.strict_load)
    letter_ids = letter_token_ids(tok)
    # Qwen3's config says 40,960 but its card gives 32,768 as the native context;
    # Qwen3.5/3.6 are native to their config's 262,144
    native = None if model is None else (
        32768 if model.config.model_type == "qwen3"
        else model.config.get_text_config().max_position_embeddings)
    check = None
    if a.selfcheck and model is not None:
        check = selfcheck(model, tok, system, letter_ids)
        print(f"[r48] selfcheck {json.dumps(check)}", flush=True)
        if (check["t3"] > a.selfcheck_tol or check["t4_before"] > a.selfcheck_tol
                or check["t4_after"] < 1e-3):
            raise SystemExit("selfcheck FAILED: one pass differs from per-occasion queries, "
                             "or earlier predictions respond to later occasions")

    rows = {k: [] for k in ("uid_hash", "pos", "label", "cell", "nll", "logp9", "mass")}
    lengths, t0 = [], time.time()
    for j, i in enumerate(sel):
        L = int(tab["length"][i])
        text, answers = transcript(tab["label"][i], tab["prev"][i], tab["lto"][i],
                                   tab["obtained"][i], tab["inv_init"][i], L, system)
        ids, tpos = answer_positions(tok, text, answers, letter_ids)
        lengths.append(len(ids))
        if a.dry_run:
            continue
        if len(ids) > native and not a.yarn:
            raise SystemExit(f"a transcript has {len(ids):,} tokens > {native:,}; "
                             "rerun with --yarn 2")
        logp9, mass = score_ids(model, ids, tpos, letter_ids, a.device)
        lab = tab["label"][i, :L].astype(np.int64)
        rows["uid_hash"].append(np.full(L, tab["uid_hash"][i], np.int64))
        rows["pos"].append(np.arange(L, dtype=np.int32))
        rows["label"].append(lab.astype(np.int8))
        rows["cell"].append(cells[i, :L])
        rows["nll"].append(-logp9[np.arange(L), lab - 1].astype(np.float32))
        rows["logp9"].append(logp9.astype(np.float32))
        rows["mass"].append(mass.astype(np.float32))
        if (j + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[r48] {j + 1:,}/{len(sel):,} customers, {sum(lengths):,} tokens, "
                  f"{sum(lengths) / el:,.0f} tok/s", flush=True)

    lens = np.array(lengths)
    summary = {"tag": a.tag, "model": a.model, "who": a.who, "shard": a.shard,
               "n_shards": a.n_shards, "yarn": a.yarn, "customers": len(sel),
               "tokens": {"total": int(lens.sum()), "max": int(lens.max()),
                          "p50": int(np.percentile(lens, 50)),
                          "p90": int(np.percentile(lens, 90)),
                          "p99": int(np.percentile(lens, 99)),
                          "over_32768": int((lens > 32768).sum())},
               "seconds": round(time.time() - t0, 1)}
    if model is not None:
        import torch
        import transformers
        summary.update({"n_gpus": a.n_gpus, "model_type": model.config.model_type,
                        "model_class": type(model).__name__,
                        "transformers": transformers.__version__,
                        "torch": torch.__version__, "selfcheck": check})
        if torch.cuda.is_available():
            summary["gpu_peak_gb"] = [round(torch.cuda.max_memory_allocated(i) / 1e9, 1)
                                      for i in range(torch.cuda.device_count())]
    if not a.dry_run:
        arr = {k: np.concatenate(v) for k, v in rows.items()}
        np.savez_compressed(stem.with_suffix(".npz"), **arr)
        by_cell = {}
        for k, name in enumerate(CELL_NAMES):
            m = arr["cell"] == k
            if m.any():
                hit = (arr["logp9"][m].argmax(1) + 1) == arr["label"][m]
                by_cell[name] = {"n": int(m.sum()), "nll": float(arr["nll"][m].mean()),
                                 "hit_rate": float(hit.mean()),
                                 "letter_mass": float(np.exp(arr["mass"][m]).mean())}
        summary["cells"] = by_cell
    stem.with_suffix(".json" if not a.dry_run else ".dryrun.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
