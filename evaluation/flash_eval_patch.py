#!/usr/bin/env python3
"""
flash_eval_patch.py

Adds FlashAttention adapter to unified_eval_and_compare.py.
Apply by adding these lines near the top of unified_eval_and_compare.py:

    from flash_eval_patch import FlashAdapter, FLASH_AVAILABLE

And in make_adapter(), add:

    if family == "flash": return FlashAdapter(spec, args)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ── Import Flash model ──
try:
    from model4_decoderonly_feature_flash import build_transformer as flash_build_transformer
    from config4 import get_config as flash_get_config
    from dataset4_productgpt import load_json_dataset as flash_load_json_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers
    FLASH_AVAILABLE = True
    FLASH_IMPORT_ERROR = None
except Exception as exc:
    flash_build_transformer = None
    flash_get_config = None
    flash_load_json_dataset = None
    FLASH_AVAILABLE = False
    FLASH_IMPORT_ERROR = exc

# ── Reuse shared helpers from the eval script ──
from unified_model_eval import (
    BaseAdapter,
    VectorScaling,
    load_calibrator_from_path,
    load_feature_tensor,
    flat_uid,
    FEATURE_COLS,
    FIRST_PROD_ID, LAST_PROD_ID, MAX_TOKEN_ID,
)

# ── Constants matching train4_flash_aws.py ──
PAD_ID = 0
SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
EOS_PROD_ID, SOS_PROD_ID, UNK_PROD_ID = 57, 58, 59
SPECIAL_IDS = [PAD_ID, SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID, EOS_PROD_ID, SOS_PROD_ID]


def _build_tokenizer_src():
    vocab = {
        "[PAD]": PAD_ID,
        **{str(i): i for i in range(1, 10)},
        "[SOS]": SOS_DEC_ID, "[EOS]": EOS_DEC_ID, "[UNK]": UNK_DEC_ID,
        **{str(i): i for i in range(13, UNK_PROD_ID + 1)},
    }
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok


def _build_tokenizer_tgt():
    vocab = {
        "[PAD]": PAD_ID,
        **{str(i): i for i in range(1, 10)},
        "[SOS]": SOS_DEC_ID, "[EOS]": EOS_DEC_ID, "[UNK]": UNK_DEC_ID,
    }
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok


# ── Dataset for Flash inference (same format as ProductGPT) ──
class FlashPredictDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, tok_src, seq_len_ai: int, pad_id: int = 0):
        raw = flash_load_json_dataset(data_path)
        self.samples = []
        for rec in raw:
            uid = flat_uid(rec.get("uid", ""))
            agg = rec["AggregateInput"]
            src_txt = " ".join(map(str, agg)) if isinstance(agg, (list, tuple)) else str(agg)
            ids = tok_src.encode(src_txt).ids[:seq_len_ai]
            if len(ids) < seq_len_ai:
                ids = ids + [pad_id] * (seq_len_ai - len(ids))
            self.samples.append({"uid": uid, "x": torch.tensor(ids, dtype=torch.long)})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def flash_collate_fn(batch):
    uids = [b["uid"] for b in batch]
    x = torch.stack([b["x"] for b in batch])
    lens = [len(b["x"]) for b in batch]
    return {"uid": uids, "x": x, "lens": lens}


class FlashAdapter(BaseAdapter):
    def __init__(self, spec: Dict[str, Any], args):
        super().__init__(spec, args)
        if not FLASH_AVAILABLE:
            raise RuntimeError("Flash model dependencies unavailable") from FLASH_IMPORT_ERROR

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt = Path(spec["ckpt"])
        self.feat_path = Path(spec["feat_xlsx"])
        self.calibration = spec.get("calibration", "none")
        self.calibrator = None
        self.logit_bias_9 = None

        # Config
        self.cfg = flash_get_config()
        self.cfg["ai_rate"] = int(spec.get("ai_rate", 15))
        self.cfg["batch_size"] = int(spec.get("batch_size", 2))
        self.cfg["seq_len_ai"] = self.cfg["seq_len_tgt"] * self.cfg["ai_rate"]

        # HP from spec (Flash doesn't encode nb_features in filename)
        self.hp = {
            "d_model": int(spec["d_model"]),
            "num_heads": int(spec["num_heads"]),
            "N": int(spec["N"]),
            "d_ff": int(spec["d_ff"]),
            "nb_features": int(spec.get("nb_features", 0)),
            "weight": 1,
            "fold_id": 0,
        }

        # Tokenizers
        self.tok_src = _build_tokenizer_src()
        self.tok_tgt = _build_tokenizer_tgt()
        self.pad_id = self.tok_tgt.token_to_id("[PAD]")

        # Build and load model
        self.model = self._build_model()
        self._load_calibration()

    def _build_model(self):
        feat_tensor = load_feature_tensor(self.feat_path)
        model = flash_build_transformer(
            vocab_size_tgt=self.cfg["vocab_size_tgt"],
            vocab_size_src=self.cfg["vocab_size_src"],
            max_seq_len=self.cfg["seq_len_ai"],
            d_model=self.hp["d_model"],
            n_layers=self.hp["N"],
            n_heads=self.hp["num_heads"],
            d_ff=self.hp["d_ff"],
            dropout=0.0,
            nb_features=self.hp["nb_features"],
            feature_tensor=feat_tensor,
            special_token_ids=SPECIAL_IDS,
        ).to(self.device).eval()

        state = torch.load(self.ckpt, map_location=self.device)
        raw_sd = state.get("model_state_dict", state.get("module", state))
        clean_sd = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in raw_sd.items()
        }
        model.load_state_dict(clean_sd, strict=True)
        self._state = state
        print(f"[INFO] {self.name}: loaded Flash model (epoch {state.get('epoch', '?')}, "
              f"val_nll={state.get('best_val_nll', '?')})")
        return model

    def _load_calibration(self):
        if self.calibration == "calibrator":
            cal_path = Path(self.spec.get("calibrator_ckpt", ""))
            if not cal_path.exists():
                stem = self.ckpt.stem.replace("FullProductGPT_", "")
                cal_path = self.ckpt.parent / f"calibrator_{stem}.pt"
            self.calibrator = load_calibrator_from_path(cal_path, self.device)
            if self.calibrator is None:
                print(f"[WARN] {self.name}: calibrator not found; falling back to none")
                self.calibration = "none"
        elif self.calibration == "analytic":
            if "logit_bias_9" in self._state:
                self.logit_bias_9 = torch.tensor(
                    self._state["logit_bias_9"], device=self.device, dtype=torch.float32
                )

    def predict_batches(self) -> Iterable[Dict[str, Any]]:
        ds = FlashPredictDataset(
            self.args.data, self.tok_src, self.cfg["seq_len_ai"], self.pad_id
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self.cfg["batch_size"], shuffle=False,
            collate_fn=flash_collate_fn,
        )

        ai_rate = self.cfg["ai_rate"]
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                uids = batch["uid"]
                lens = batch["lens"]

                logits_full = self.model(x)
                pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=self.device)
                logits = logits_full[:, pos, :] if logits_full.size(1) == x.size(1) else logits_full

                # Extract decision logits
                logits_dec = logits if logits.size(-1) == 9 else logits[..., 1:10]

                # Apply calibration
                if self.calibration == "calibrator" and self.calibrator is not None:
                    probs = self.calibrator(logits_dec)
                elif self.calibration == "analytic" and self.logit_bias_9 is not None:
                    probs = torch.softmax(
                        logits_dec - self.logit_bias_9.view(1, 1, 9), dim=-1
                    )
                else:
                    probs = torch.softmax(logits_dec, dim=-1)

                # Convert bf16 → fp32 for numpy
                yield {
                    "uid": uids,
                    "lens": lens,
                    "probs_dec_9": probs.float().detach().cpu().numpy(),
                    "ai_rate": ai_rate,
                }