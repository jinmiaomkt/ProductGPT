"""One fresh process: is the FIRST tiny-model forward pass on CPU different from
the next two?  Prints one line.  Synthetic customer only; no data.
    python r49_first_call.py <repo> <tokenizer dir> <sdpa|eager>
"""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO, TOK, IMPL = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from llm_zeroshot.prompt import system_prompt, transcript  # noqa: E402
from llm_zeroshot.score import answer_positions, letter_token_ids, score_ids  # noqa: E402
from llm_zeroshot.synthetic import synthetic_customer, synthetic_products  # noqa: E402
from test_r48_zeroshot import tiny_model  # noqa: E402

tok = AutoTokenizer.from_pretrained(TOK)
letter_ids = letter_token_ids(tok)
lab, prev, lto, obt, inv = synthetic_customer(40, 2)
text, answers = transcript(lab, prev, lto, obt, inv, 40, system_prompt(synthetic_products()))
ids, tpos = answer_positions(tok, text, answers, letter_ids)
model = tiny_model(len(tok)).double()
model.config._attn_implementation = IMPL
f = [score_ids(model, ids, tpos, letter_ids, "cpu")[0] for _ in range(3)]
d12, d23 = float(np.abs(f[0] - f[1]).max()), float(np.abs(f[1] - f[2]).max())
print(f"{IMPL} threads={torch.get_num_threads()} first-vs-second {d12:.1e} second-vs-third {d23:.1e}",
      flush=True)
