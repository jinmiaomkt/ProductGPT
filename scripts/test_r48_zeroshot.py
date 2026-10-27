"""R48 tests: the transcript, the answer positions and the one-pass scorer.

Runs on CPU with synthetic customers, the real Qwen3 tokenizer and a tiny
randomly initialised Qwen3 model -- no data, no model weights, no GPU.

    python scripts/test_r48_zeroshot.py [--tokenizer Qwen/Qwen3-8B]

R49 adds --arch qwen3_5: a tiny randomly initialised Qwen3.5/3.6 text model
(three Gated DeltaNet linear-attention layers, then one full-attention layer,
as in Qwen3.6-27B), so T3 and T4 also cover the linear-attention path.  It needs
transformers >= 5.

    python scripts/test_r48_zeroshot.py --tokenizer Qwen/Qwen3.6-27B --arch qwen3_5

  T1  the hand-written chat format equals tokenizer.apply_chat_template
      (non-thinking generation prompt) for a single turn
  T2  every answer letter is one token, at its character, with the letter's id
  T3  one forward pass equals asking each occasion separately (prefix up to and
      including that occasion's generation prompt)
  T4  no look-ahead: changing an occasion's decision and everything after it
      leaves every earlier and same-occasion prediction unchanged (T4c: the same
      with only later decisions changed, so the transcript keeps its length)
  T5  the nine renormalised probabilities sum to one; the letter mass is <= 1
  T6  the transcript states offers only when they change and counts duplicates
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never touch the laptop GPU
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from llm_zeroshot.prompt import (ASSISTANT_PREFIX, IM_END, IM_START, LETTERS,  # noqa: E402
                                 system_prompt, transcript, user_turn)
from llm_zeroshot.score import answer_positions, letter_token_ids, score_ids  # noqa: E402
from llm_zeroshot.synthetic import (perturb_decisions_from, perturb_from,  # noqa: E402
                                    synthetic_customer, synthetic_products)

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def tiny_model(vocab, arch="qwen3"):
    import torch
    torch.manual_seed(0)
    if arch == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, max_position_embeddings=65536, tie_word_embeddings=False)
        m = Qwen3ForCausalLM(cfg).eval()
    else:
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        cfg = Qwen3_5TextConfig(
            vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_attention_heads=2, num_key_value_heads=1, head_dim=64,
            linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4,
            rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                             "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2],
                             "mrope_interleaved": True},
            max_position_embeddings=65536, tie_word_embeddings=False)
        m = Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():                                   # spread the letter logits so
        m.lm_head.weight.mul_(30.0)                         # differences are visible
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--arch", choices=("qwen3", "qwen3_5"), default="qwen3",
                    help="architecture of the tiny random model used by T3-T5")
    a = ap.parse_args()
    print(f"tokenizer {a.tokenizer}, tiny model {a.arch}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prod = synthetic_products()
    system = system_prompt(prod)
    letter_ids = letter_token_ids(tok)

    # T1 chat format
    u = "Occasion 1. Featured now: banner A P13; banner B closed; Weapon banner P41."
    ref = tok.apply_chat_template([{"role": "system", "content": system},
                                   {"role": "user", "content": u}],
                                  tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    mine = f"{IM_START}system\n{system}{IM_END}\n" + user_turn(u) + ASSISTANT_PREFIX
    check("T1 chat format matches apply_chat_template", ref == mine)

    # T2 positions
    S = 300
    lab, prev, lto, obt, inv = synthetic_customer(S, 1)
    text, answers = transcript(lab, prev, lto, obt, inv, S, system)
    ids, tpos = answer_positions(tok, text, answers, letter_ids)
    got = [LETTERS.index(tok.decode([ids[p]])) + 1 for p in tpos]
    check("T2 answer tokens decode to the true decisions", got == list(lab),
          f"{len(ids):,} tokens for {S} occasions ({len(ids) / S:.1f}/occasion)")

    # T3 and T4 run in float64 so rounding (~1e-12) cannot pass for a leak.  The
    # Qwen3.5/3.6 Gated DeltaNet reference path casts to float32 whatever the model
    # dtype (transformers torch_chunk_gated_delta_rule), so there the floor is
    # float32 rounding (~1e-7 when the sequence length, and so every matrix shape,
    # changes) and the bound is 1e-6.  A leak moves predictions by ~T4b (> 1e-3).
    # (T3 ran in float32 with a 1e-4 bound until 28 Oct, when CPU rounding on an
    # HPCC GPU node reached 1.66e-4 for the qwen3 tiny model: jobs 44503/5/7.)
    tol = 1e-10 if a.arch == "qwen3" else 1e-6
    model = tiny_model(len(tok), a.arch).double()

    # T3 one pass == per-occasion prefixes
    S3 = 40
    lab, prev, lto, obt, inv = synthetic_customer(S3, 2)
    text, answers = transcript(lab, prev, lto, obt, inv, S3, system)
    ids, tpos = answer_positions(tok, text, answers, letter_ids)
    lp, mass = score_ids(model, ids, tpos, letter_ids, "cpu")
    worst = 0.0
    for k, p in enumerate(tpos):
        lpk, _ = score_ids(model, ids[:p + 1], [p], letter_ids, "cpu")
        worst = max(worst, float(np.abs(lpk[0] - lp[k]).max()))
    check("T3 one pass equals per-occasion queries", worst < tol,
          f"max |diff| {worst:.1e} (bound {tol:.0e})")
    check("T3b the tiny model's predictions are not degenerate",
          float(lp.max(1).mean() - lp.min(1).mean()) > 0.1)

    # T4 no look-ahead
    t = 20
    lab2, obt2 = perturb_from(lab, obt, t)
    text2, answers2 = transcript(lab2, prev, lto, obt2, inv, S3, system)
    ids2, tpos2 = answer_positions(tok, text2, answers2, letter_ids)
    lp2, _ = score_ids(model, ids2, tpos2, letter_ids, "cpu")
    before = float(np.abs(lp2[: t + 1] - lp[: t + 1]).max())
    after = float(np.abs(lp2[t + 1:] - lp[t + 1:]).max())
    check("T4 no look-ahead (occasions <= t unchanged)", before < tol,
          f"max |diff| {before:.1e} (bound {tol:.0e})")
    check("T4b later occasions do respond (the perturbation is real)", after > 1e-3,
          f"max |diff| {after:.2f}")
    # T4c the same with only later decisions changed: same length, same shapes
    lab3 = perturb_decisions_from(lab, t)
    text3, answers3 = transcript(lab3, prev, lto, obt, inv, S3, system)
    ids3, tpos3 = answer_positions(tok, text3, answers3, letter_ids)
    lp3, _ = score_ids(model, ids3, tpos3, letter_ids, "cpu")
    before3 = float(np.abs(lp3[: t + 1] - lp[: t + 1]).max())
    check("T4c no look-ahead, same-length transcript", len(ids3) == len(ids) and before3 < tol,
          f"max |diff| {before3:.1e}; later {float(np.abs(lp3[t + 1:] - lp[t + 1:]).max()):.2f}")

    # T5 normalisation (float64 model)
    s = np.exp(lp).sum(1)
    check("T5 renormalised probabilities sum to one", np.allclose(s, 1.0, atol=1e-9))
    check("T5b letter mass <= 1", bool((mass <= 1e-9).all()))

    # T6 transcript content
    lab, prev, lto, obt, inv = synthetic_customer(30, 3)
    text, _ = transcript(lab, prev, lto, obt, inv, 30, system)
    check("T6 offers stated once per campaign", text.count("Featured now:") == 2)
    check("T6b duplicates counted", " x2" in text)
    check("T6c pre-window holdings stated once", text.count("Owned before this record") == 1)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
