# Laptop, CPU, no data: the scorer's real load path end to end on tiny saved checkpoints of each
# family (auto class, strict load, adapter, selfcheck, memcheck), so a wiring fault shows up now.
import importlib.util
import json
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, ".")
import torch
from transformers import AutoTokenizer

from llm_zeroshot.prompt import FORMATS, system_prompt
from llm_zeroshot.score import letter_token_ids, load_model, memcheck, selfcheck, selfcheck_failures
from llm_zeroshot.synthetic import synthetic_products

spec = importlib.util.spec_from_file_location("t", "scripts/test_r48_zeroshot.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
torch.set_num_threads(1)
system = system_prompt(synthetic_products())

for repo, arch, fname, auto, use_adapter in (
        ("meta-models/Muse-Glimmer-30B", "glimmer", "glimmer", "image", False),
        ("google/gemma-4-31B-it", "gemma4", "gemma4", "causal", False),
        ("unsloth/Meta-Llama-3.1-70B-bnb-4bit", "llama", "press", "causal", False),
        ("unsloth/Meta-Llama-3.1-70B-bnb-4bit", "llama", "press", "causal", True)):
    tok = AutoTokenizer.from_pretrained(repo)
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as ad:
        tiny = t.tiny_model(len(tok), arch)
        tiny.save_pretrained(d)
        tok.save_pretrained(d)
        adapter = ""
        if use_adapter:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=2, lora_alpha=2, use_rslora=True, init_lora_weights=False, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            get_peft_model(t.tiny_model(len(tok), arch), cfg).save_pretrained(ad)
            adapter = ad
        tok2, model = load_model(d, "cpu", 0.0, 1, True, "float32", auto, adapter)
        ids = letter_token_ids(tok2)
        fmt = FORMATS[fname]
        chk = selfcheck(model, tok2, system, ids, fmt=fmt)
        mem = memcheck(model, tok2, system, ids, 1500, fmt)
        text = model.config.get_text_config()
        print(repo.split("/")[-1], "adapter" if adapter else "", "|", type(model).__name__, "| native", text.max_position_embeddings,
              "| failures", selfcheck_failures(chk, 0.1), "| mem", mem,
              "| t3", round(chk["t3_true_mean"], 6), "t4c", chk["t4c_before_max"], "repeat", chk["repeat_max"], "after", round(chk["t4_after_max"], 3),
              "| adapter effect", chk.get("adapter_effect_max"))
