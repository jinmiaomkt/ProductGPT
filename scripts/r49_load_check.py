"""R49 step 2: load a model's bf16 weights on CPU as the scorer will, and check
that every weight the text model needs is in the checkpoint.

The Qwen3.6-27B checkpoint is multimodal (Qwen3_5ForConditionalGeneration);
the scorer loads only its text model (Qwen3_5ForCausalLM), so this is where a
mis-mapped weight would show.  No forward pass and no data.

    python scripts/r49_load_check.py --model <dir> [--yarn 2]

Fails unless there are no missing and no shape-mismatched weights.  Prints the
class, parameter count, layer types and the unexpected checkpoint weights
grouped by prefix (the vision tower is expected there).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--yarn", type=float, default=0.0)
    a = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    v5 = int(transformers.__version__.split(".")[0]) >= 5
    kw = {"dtype" if v5 else "torch_dtype": torch.bfloat16, "attn_implementation": "sdpa",
          "output_loading_info": True}
    if not v5:
        kw["low_cpu_mem_usage"] = True
    if a.yarn:
        kw["rope_scaling"] = {"rope_type": "yarn", "factor": a.yarn,
                              "original_max_position_embeddings": 32768}
    model, info = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    cfg = model.config.get_text_config()
    missing = sorted(info.get("missing_keys", []))
    mismatched = [str(x) for x in info.get("mismatched_keys", [])]
    unexpected = Counter(".".join(k.split(".")[:2]) for k in info.get("unexpected_keys", []))
    rep = {"model": a.model, "transformers": transformers.__version__,
           "class": type(model).__name__, "model_type": model.config.model_type,
           "params": int(sum(p.numel() for p in model.parameters())),
           "dtype": str(next(model.parameters()).dtype),
           "layers": int(cfg.num_hidden_layers),
           "layer_types": dict(Counter(getattr(cfg, "layer_types", None) or ["full_attention"]
                                       * cfg.num_hidden_layers)),
           "max_position_embeddings": int(cfg.max_position_embeddings),
           "rope": getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None),
           "missing": len(missing), "missing_first": missing[:10],
           "mismatched": len(mismatched), "mismatched_first": mismatched[:10],
           "unexpected_by_prefix": dict(unexpected)}
    print(json.dumps(rep, indent=2, default=str))
    ok = not missing and not mismatched
    print(f"[r49] load check {'PASS' if ok else 'FAIL'}: {a.model}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
