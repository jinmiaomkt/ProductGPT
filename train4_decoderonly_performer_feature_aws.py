"""ProductGPT – end‑to‑end train / evaluate script
=================================================
✓ Hyper‑parameters come from `config4.py`
✓ Handles feature embeddings, data‑loading, training (DeepSpeed) and test
✓ No duplicated functions, clean imports, full type hints
"""
from __future__ import annotations

# ────────────────────────────── stdlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import warnings

# ────────────────────────────── third‑party
import boto3
import botocore
import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lamb import Lamb # noqa: F401  (used by DeepSpeed JSON)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer, models, pre_tokenizers
from tqdm import tqdm

# ────────────────────────────── project local
from config4 import get_config, get_weights_file_path, latest_weights_file_path
from dataset4_productgpt import TransformerDataset, load_json_dataset
from model4_decoderonly_feature_performer import build_transformer

# ────────────────────────────── global config
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF
warnings.filterwarnings("ignore")
logging.getLogger("deepspeed").setLevel(logging.ERROR)

# ══════════════════════════════ 1. Constants ═══════════════════════════
PAD_ID = 0
DECISION_IDS = list(range(1, 10))  # 1‑9
SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
EOS_PROD_ID, SOS_PROD_ID, UNK_PROD_ID = 57, 58, 59
SPECIAL_IDS = [
    PAD_ID,
    SOS_DEC_ID,
    EOS_DEC_ID,
    UNK_DEC_ID,
    EOS_PROD_ID,
    SOS_PROD_ID,
]
MAX_TOKEN_ID = UNK_PROD_ID  # 59

# ══════════════════════════════ 2. Data helpers ════════════════════════
FEAT_FILE = Path("/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")
FEATURE_COLS: List[str] = [
    # stats
    "Rarity",
    "MaxLife",
    "MaxOffense",
    "MaxDefense",
    # categorical one‑hots
    "WeaponTypeOneHandSword",
    "WeaponTypeTwoHandSword",
    "WeaponTypeArrow",
    "WeaponTypeMagic",
    "WeaponTypePolearm",
    "EthnicityIce",
    "EthnicityRock",
    "EthnicityWater",
    "EthnicityFire",
    "EthnicityThunder",
    "EthnicityWind",
    "GenderFemale",
    "GenderMale",
    "CountryRuiYue",
    "CountryDaoQi",
    "CountryZhiDong",
    "CountryMengDe",
    # misc
    "type_figure",
    "MinimumAttack",
    "MaximumAttack",
    "MinSpecialEffect",
    "MaxSpecialEffect",
    "SpecialEffectEfficiency",
    "SpecialEffectExpertise",
    "SpecialEffectAttack",
    "SpecialEffectSuper",
    "SpecialEffectRatio",
    "SpecialEffectPhysical",
    "SpecialEffectLife",
    "LTO",
]


def load_feature_tensor(xls_path: Path) -> torch.Tensor:
    """Load product‑level feature embeddings – (V, D) FloatTensor."""
    df = pd.read_excel(xls_path, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)


# ══════════════════════════════ 3. Tokenisers ═════════════════════════=

def _base_tokeniser(extra_vocab: Dict[str, int] | None = None) -> Tokenizer:
    """Word‑level tokeniser with a fixed numeric vocabulary."""
    vocab: Dict[str, int] = {
        "[PAD]": PAD_ID,
        **{str(i): i for i in range(1, 10)},  # decisions
        "[SOS]": SOS_DEC_ID,
        "[EOS]": EOS_DEC_ID,
        "[UNK]": UNK_DEC_ID,
    }
    if extra_vocab:
        vocab.update(extra_vocab)
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok


def build_tokenizer_src() -> Tokenizer:  # with product IDs
    prod_vocab = {str(i): i for i in range(FIRST_PROD_ID, UNK_PROD_ID + 1)}
    return _base_tokeniser(prod_vocab)


def build_tokenizer_tgt() -> Tokenizer:  # decisions only
    return _base_tokeniser()


# ══════════════════════════════ 4. Losses ═════════════════════════════
class FocalLoss(nn.Module):
    """Multi‑class focal loss with optional class weights."""

    def __init__(
        self,
        gamma: float = 0.0,
        ignore_index: int = 0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  
        B, T, V = logits.shape
        logits = logits.view(-1, V)
        targets = targets.view(-1)

        # ─── make sure the `weight` tensor matches logits’ dtype/device ──
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(
                dtype=logits.dtype, device=logits.device
            )

        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=weight,
            ignore_index=self.ignore_index,
        )

        mask = targets != self.ignore_index
        ce = ce[mask]
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if loss.numel() else logits.new_tensor(0.0)


# ══════════════════════════════ 5. Utility fns ════════════════════════

def transition_mask(seq: torch.Tensor) -> torch.Tensor:  # (B, T)
    """Mask where *decision* changes wrt previous step."""
    prev = F.pad(seq, (1, 0), value=-1)[:, :-1]
    return seq != prev


def perplexity(logits: torch.Tensor, targets: torch.Tensor, pad: int = PAD_ID) -> float:
    logp = F.log_softmax(logits, dim=-1)
    lp2d, tgt = logp.view(-1, logp.size(-1)), targets.view(-1)
    mask = tgt != pad
    if mask.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2d[mask], tgt[mask], reduction="mean")).item()


# ══════════════════════════════ 6. DataLoaders ═══════════════════════=

def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, Tokenizer]:
    raw = load_json_dataset(cfg["filepath"])
    n = len(raw)
    tr, va = int(0.8 * n), int(0.1 * n)
    g = torch.Generator().manual_seed(33)
    tr_ds, va_ds, te_ds = random_split(raw, [tr, va, n - tr - va], generator=g)

    tok_src = build_tokenizer_src()
    tok_tgt = build_tokenizer_tgt()

    out_dir = Path(cfg["model_folder"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_src.save(str(out_dir / "tokenizer_ai.json"))
    tok_tgt.save(str(out_dir / "tokenizer_tgt.json"))

    def _wrap(split):
        return TransformerDataset(
            split,
            tok_src,
            tok_tgt,
            cfg["seq_len_ai"],
            cfg["seq_len_tgt"],
            cfg["num_heads"],
            cfg["ai_rate"],
            pad_token=PAD_ID,
        )

    make_loader = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)
    return make_loader(_wrap(tr_ds), True), make_loader(_wrap(va_ds), False), make_loader(_wrap(te_ds), False), tok_tgt


# ══════════════════════════════ 7. S3 helpers ═════════════════════════

def _s3_client():
    try:
        return boto3.client("s3")
    except botocore.exceptions.BotoCoreError:
        return None


def _upload(local: Path, bucket: str, key: str, s3) -> bool:
    if s3 is None or not local.exists():
        return False
    try:
        s3.upload_file(str(local), bucket, key)
        print(f"[S3] {local.name} → s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3‑ERR] {e}")
        return False

# ─────────────────── pretty-printer for metric blocks ───────────────────
def _show(tag: str, metrics: Tuple[float, float, dict, dict, dict, dict]) -> None:
    loss, ppl, m_all, m_st, m_af, m_tr = metrics
    print(f"{tag}  Loss={loss:.4f}  PPL={ppl:.4f}")
    for name, d in (
        ("all",          m_all),
        ("cur-STOP",     m_st),
        ("after-STOP",   m_af),
        ("transition",   m_tr),
    ):
        print(f"  {name:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
              f"AUPRC={d['auprc']:.4f}")

# ══════════════════════════════ 8. Model builder ═════════════════════=

def build_model(cfg: Dict[str, Any], feat_tensor: torch.Tensor) -> nn.Module:
    return build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        max_seq_len=cfg["seq_len_ai"],
        d_model=cfg["d_model"],
        n_layers=cfg["N"],
        n_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        nb_features=cfg["nb_features"],
        kernel_type=cfg["kernel_type"],
        block_size_h=cfg["ai_rate"],
        block_size_w=cfg["ai_rate"],
        d_ff=cfg["d_ff"],
        feature_tensor=feat_tensor,
        special_token_ids=SPECIAL_IDS,
    )


# ══════════════════════════════ 9. Evaluation ════════════════════════

def _subset(pred, lbl, probs, mask, classes=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": np.nan, "f1": np.nan, "auprc": np.nan}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    return {
        "hit": accuracy_score(l, p),
        "f1": f1_score(l, p, average="macro"),
        "auprc": average_precision_score(label_binarize(l, classes=classes), pr[:, 1:10], average="macro"),
    }


def evaluate(loader: DataLoader, model: nn.Module, dev: torch.device, loss_fn, pad: int, tok: Tokenizer, ai_rate: int):
    if not loader:
        nan = float("nan")
        emp = {"hit": nan, "f1": nan, "auprc": nan}
        return nan, nan, emp, emp, emp, emp

    special = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}
    tot_loss = tot_ppl = 0.0
    P: List[np.ndarray] = []
    L: List[np.ndarray] = []
    PR: List[np.ndarray] = []
    m_stop: List[np.ndarray] = []
    m_after_stop: List[np.ndarray] = []
    m_tr: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["aggregate_input"].to(dev)
            tgt = batch["label"].to(dev)
            pos = torch.arange(ai_rate - 1, x.size(1), ai_rate, device=dev)
            logits = model(x)[:, pos, :]

            tgt_ = tgt.clone()
            tgt_[transition_mask(tgt)] = pad
            tot_loss += loss_fn(logits, tgt_).item()
            tot_ppl += perplexity(logits, tgt_, pad)

            prob = F.softmax(logits, dim=-1).view(-1, logits.size(-1)).cpu().numpy()
            pred = prob.argmax(1)
            lbl = tgt.view(-1).cpu().numpy()
            keep = ~np.isin(lbl, list(special))
            P.append(pred[keep])
            L.append(lbl[keep])
            PR.append(prob[keep])

            flat = lambda m: m.view(-1).cpu().numpy()[keep]
            m_stop.append(flat(tgt == 9))
            prev = F.pad(tgt, (1, 0), value=-1)[:, :-1]
            m_after_stop.append(flat(prev == 9))
            m_tr.append(flat(transition_mask(tgt)))

    P_, L_, PR_ = map(np.concatenate, (P, L, PR))
    return (
        tot_loss / len(loader),
        tot_ppl / len(loader),
        _subset(P_, L_, PR_, np.ones_like(P_, dtype=bool)),
        _subset(P_, L_, PR_, np.concatenate(m_stop)),
        _subset(P_, L_, PR_, np.concatenate(m_after_stop)),
        _subset(P_, L_, PR_, np.concatenate(m_tr)),
    )


# ══════════════════════════════ 10. Training loop ════════════════════

def train_model(cfg: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- artefact paths ------------------------------------------------
    uid = (
        f"performer_nbfeat{cfg['nb_features']}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_"
        f"N{cfg['N']}_heads{cfg['num_heads']}_lr{cfg['lr']}_w{cfg['weight']}"
    )
    ckpt_path = Path(cfg["model_folder"]) / f"FullProductGPT_{uid}.pt"
    json_path = ckpt_path.with_suffix(".json")

    s3 = _s3_client()
    bucket = cfg["s3_bucket"]
    ck_key = f"FullProductGPT/performer/FeatureBased/checkpoints/{ckpt_path.name}"
    js_key = f"FullProductGPT/performer/FeatureBased/metrics/{json_path.name}"
    print(f"[INFO] artefacts → s3://{bucket}/{ck_key}")

    # --- data ----------------------------------------------------------
    train_dl, val_dl, test_dl, tok_tgt = build_dataloaders(cfg)
    pad_id = tok_tgt.token_to_id("[PAD]")

    # --- model ---------------------------------------------------------
    feat_tensor = load_feature_tensor(FEAT_FILE)
    model = build_model(cfg, feat_tensor).to(device)

    # ---- QAT stub (optional) -----------------------------------------
    # model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    # torch.quantization.prepare_qat(model, inplace=True)

    # ---- criterion ----------------------------------------------------
    weights = torch.ones(cfg["vocab_size_tgt"], device=device)
    weights[9] = cfg["weight"]
    loss_fn = FocalLoss(cfg["gamma"], pad_id, weights)

    # ---- DeepSpeed ----------------------------------------------------
    ds_cfg = {
        "train_micro_batch_size_per_gpu": cfg["batch_size"],
        "zero_allow_untested_optimizer": True,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "Lamb",
            "params": {"lr": cfg["lr"], "eps": cfg["eps"], "weight_decay": cfg["weight_decay"]},
        },
        "zero_optimization": {"stage": 1},
        "fp16": {"enabled": True},
        "lr_scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": cfg["min_lr"],
                "warmup_max_lr": cfg["lr"],
                "warmup_num_steps": cfg["warmup_steps"],
                "total_num_steps": cfg["num_epochs"] * len(train_dl),
                "decay_style": "cosine",
            },
        },
    }

    engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_cfg)

    # ---- (optional) preload ------------------------------------------
    preload = cfg["preload"]
    if preload:
        file = latest_weights_file_path(cfg) if preload == "latest" else get_weights_file_path(cfg, preload)
        if file and Path(file).exists():
            print("Preloading", file)
            state = torch.load(file, map_location="cpu")
            engine.module.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])

    # ---- training -----------------------------------------------------
    best_loss, patience = None, 0
    for ep in range(cfg["num_epochs"]):
        engine.train()
        running = 0.0
        for batch in tqdm(train_dl, desc=f"Ep {ep:02d}"):
            x = batch["aggregate_input"].to(device)
            tgt = batch["label"].to(device)
            pos = torch.arange(cfg["ai_rate"] - 1, cfg["seq_len_ai"], cfg["ai_rate"], device=device)
            logits = engine(x)[:, pos, :]
            tgt_ = tgt.clone()
            tgt_[transition_mask(tgt)] = pad_id
            loss = loss_fn(logits, tgt_)
            engine.backward(loss)
            engine.step()
            running += loss.item()
        print(f"Train loss {running / len(train_dl):.4f}")

        # ---- validation ----------------------------------------------
        v_metrics = evaluate(val_dl, engine, device, loss_fn, pad_id, tok_tgt, cfg["ai_rate"])
        v_loss = v_metrics[0]
        _show(f"[Epoch {ep:02d}] VAL", v_metrics)
        # print(f"Epoch {ep:02d} ValLoss={v_loss:.4f} PPL={v_metrics[1]:.4f}")

        if best_loss is None or v_loss < best_loss:
            best_loss, patience = v_loss, 0
            ckpt = {
                "epoch": ep,
                "model_state_dict": engine.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, ckpt_path)
            # json_path.write_text(json.dumps({"val_loss": best_loss}, indent=2))

            # ─── save full validation metrics ───────────────────────────────
            meta = {
                "best_checkpoint_path": ckpt_path.name,

                # validation
                "val_loss":        v_metrics[0],
                "val_ppl":         v_metrics[1],
                "val_all":         v_metrics[2],
                "val_cur_stop":    v_metrics[3],
                "val_after_stop":  v_metrics[4],
                "val_transition":  v_metrics[5],
            }
            json_path.write_text(json.dumps(meta, indent=2))
            _upload(json_path, bucket, js_key, s3)
            _upload(ckpt_path, bucket, ck_key, s3)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping")
                break

    # ---- test ---------------------------------------------------------
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        engine.module.load_state_dict(state["model_state_dict"])
        test_metrics = evaluate(test_dl, engine, device, loss_fn, pad_id, tok_tgt, cfg["ai_rate"])
        # print(f"TEST Loss={test_metrics[0]:.4f}  PPL={test_metrics[1]:.4f}")
        _show("** TEST **", test_metrics)

        # ─── append test metrics and re-upload ───────────────────────────
    with json_path.open() as f:
        meta = json.load(f)

    meta.update({
        "test_loss":        test_metrics[0],
        "test_ppl":         test_metrics[1],
        "test_all":         test_metrics[2],
        "test_cur_stop":    test_metrics[3],
        "test_after_stop":  test_metrics[4],
        "test_transition":  test_metrics[5],
    })

    json_path.write_text(json.dumps(meta, indent=2))
    _upload(json_path, bucket, js_key, s3)

    return {"uid": uid, "val_loss": best_loss, "checkpoint": str(ckpt_path)}


# ══════════════════════════════ 11. CLI ═══════════════════════════════
if __name__ == "__main__":
    cfg = get_config()
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]
    train_model(cfg)


    # best_val_loss = None
    # best_checkpoint_path = None
    # epochs_no_improve = 0

    # for epoch in range(initial_epoch, config['num_epochs']):
    #     model_engine.train()
    #     torch.cuda.empty_cache()

    #     cumm_ce, cumm_ctr = 0., 0.
        
    #     batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
    #     # total_loss = 0.0

    #     for batch in batch_iterator:
    #         decoder_input = batch['aggregate_input'].to(device)
    #         label         = batch['label'].to(device)

    #         model_engine.zero_grad()
    #         logits, h = model_engine(decoder_input, return_hidden = True)  # (B, seq_len, vocab_size)

    #         B, T, V = logits.shape
    #         decision_positions = torch.arange(config['ai_rate'] - 1, T, step=config['ai_rate'], device=logits.device)  # shape: (N,)
    #         decision_logits = logits[:, decision_positions, :]  # shape: (B, N, V)
            
    #         loss_ce = loss_fn(
    #             decision_logits,  # predict next token
    #             label
    #         )

    #         # ── contrastive regulariser on *all* tokens ───────────────────────
    #         #   every token in `inp` is guaranteed to be a product‑ID (13‑56, 59)
    #         flat_h   = h.reshape(-1, h.size(-1))                # (B*T, D_model)
    #         flat_ids = decoder_input.reshape(-1)                          # (B*T,)

    #         z_mean, uniq_id = unique_by_id(flat_h, flat_ids)
    #         z_proj = F.normalize(model_engine.module.proj_head(z_mean), dim=-1)
    #         loss_ctr    = nt_xent(z_proj, uniq_id)                  

    #         loss = loss_ce + lambda_ctr * loss_ctr

    #         model_engine.backward(loss)
    #         model_engine.step()

    #         cumm_ce  += loss_ce.item()
    #         cumm_ctr += loss_ctr.item()

    #         # total_loss += loss.item()
    #         global_step += 1

    #     loss = loss / len(train_dataloader)
    #     current_lr = model_engine.optimizer.param_groups[0]["lr"]
    #     print(f"\nEpoch {epoch}: LR={current_lr:.6f}  Train Cross-Entropy Loss={cumm_ce:.4f}  Contrastive Loss={cumm_ctr:.4f}  Train Loss={loss:.4f}")

    #     # Evaluate
    #     val_loss, val_conf_mat, val_ppl, val_hit_rate, val_f1_score, val_auprc = evaluate(val_dataloader, model_engine, device, loss_fn, config['ai_rate'])
    #     print(f"Epoch {epoch} Val Loss={val_loss:.4f}  \nVal PPL={val_ppl:.4f} \nVal Hit Rate={val_hit_rate:.4f} \nVal F1 Score={val_f1_score:.4f} \nVal Area Under Precision-Recall Curve={val_auprc:.4f}")
    #     print("Val Confusion Matrix:\n", val_conf_mat)

    #     # Early stopping or checkpoint
    #     if best_val_loss is None or val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         epochs_no_improve = 0
    #         # best_checkpoint_path = f"best_checkpoint_epoch_{epoch}.pt"
            
    #         best_checkpoint_path = get_weights_file_path(config, "best")
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model_engine.state_dict(),
    #             'optimizer_state_dict': model_engine.optimizer.state_dict(),
    #             'global_step': global_step
    #         }, best_checkpoint_path)
    #         print(f"  [*] New best val_loss={val_loss:.4f}, saved => {best_checkpoint_path}")
    #     else:
    #         epochs_no_improve += 1
    #         if epochs_no_improve >= config['patience']:
    #             print("Early stopping!")
    #             break

    # # Test evaluation
    # if best_checkpoint_path:
    #     print(f"\nBest checkpoint: {best_checkpoint_path}")
    #     state = torch.load(best_checkpoint_path, weights_only=False)
    #     model_engine.load_state_dict(state['model_state_dict'])

    # test_loss, test_conf_mat, test_ppl, test_hit_rate, test_f1_score, test_auprc = evaluate(test_dataloader, model_engine, device, loss_fn, config['ai_rate'])
    # print(f"** Test Loss={test_loss:.4f} \nTest PPL={test_ppl:.4f} \nTest Hit Rate={test_hit_rate:.4f} \nTest F1 Score={test_f1_score:.4f} \nTest Area Under Precision-Recall Curve={test_auprc:.4f}")
    # print("Test Confusion Matrix:\n", test_conf_mat)

    # json_out_path = Path(best_checkpoint_path).with_suffix(".json")
    # metadata = {
    #     "best_checkpoint_path": best_checkpoint_path,
    #     "val_loss": val_loss,
    #     "val_ppl": val_ppl,
    #     "val_confusion_matrix": val_conf_mat.tolist() if val_conf_mat is not None else None,
    #     "val_hit_rate": val_hit_rate,
    #     "val_f1_score": val_f1_score,
    #     "val_auprc": val_auprc,
    #     "test_loss": test_loss,
    #     "test_ppl": test_ppl,
    #     "test_confusion_matrix": test_conf_mat.tolist() if test_conf_mat is not None else None,
    #     "test_hit_rate": test_hit_rate,
    #     "test_f1_score": test_f1_score,
    #     "test_auprc": test_auprc
    # }
    # with open(json_out_path, 'w') as f:
    #     json.dump(metadata, f, indent=2)

    # return {
    #     "best_checkpoint_path": best_checkpoint_path,
    #     "val_loss": val_loss,
    #     "val_ppl": val_ppl,
    #     "val_confusion_matrix": val_conf_mat.tolist() if val_conf_mat is not None else None,
    #     "val_hit_rate": val_hit_rate,
    #     "val_f1_score": val_f1_score,
    #     "val_auprc": val_auprc,
    #     "test_loss": test_loss,
    #     "test_ppl": test_ppl,
    #     "test_confusion_matrix": test_conf_mat.tolist() if test_conf_mat is not None else None,
    #     "test_hit_rate": test_hit_rate,
    #     "test_f1_score": test_f1_score,
    #     "test_auprc": test_auprc
    # }

##############################################################################
# The evaluate function, fixed
##############################################################################
# def evaluate(dataloader, model_engine, device, loss_fn, stepsize):
#     total_loss = 0.0
#     total_ppl  = 0.0

#     all_preds      = []  # for confusion matrix & F1
#     all_labels     = []  # for confusion matrix & F1
#     all_probs      = []  # for AUPRC
#     valid_labels   = []  # same as all_labels, but used for AUPRC

#     model_engine.eval()
    
#     if len(dataloader) == 0:
#         # Return 4 values so the caller can unpack
#         return float('nan'), None, float('nan'), float('nan')

#     # For ignoring special tokens in the *labels*
#     tokenizer_tgt = build_tokenizer_tgt()
#     pad_id = tokenizer_tgt.token_to_id("[PAD]")
#     sos_id = tokenizer_tgt.token_to_id("[SOS]")
#     unk_id = tokenizer_tgt.token_to_id("[UNK]")
#     eos_id = tokenizer_tgt.token_to_id("[EOS]")
#     # special_tokens = {pad_id, sos_id, unk_id, eos_id}
#     special_tokens = {pad_id, sos_id, unk_id, eos_id, EOS_PROD_ID, SOS_PROD_ID}

#     with torch.no_grad():
#         for batch in dataloader:
#             dec_inp = batch['aggregate_input'].to(device)
#             label   = batch['label'].to(device)
#             logits  = model_engine(dec_inp)

#             # with torch.no_grad():
#             #     class9_logits = logits[..., 9]  # (B, T)
#             #     print(f"Avg logit for class 9: {class9_logits.mean().item():.4f}")

#             # counts = (label == 9).sum()
#             # print("Class 9 count at decision positions:", counts.item())

#             # Gather logits at decision positions (e.g., first token of every 15-token block)
#             decision_positions = torch.arange(stepsize - 1, logits.size(1), step=stepsize, device=logits.device)
#             decision_logits = logits[:, decision_positions, :]  # shape: (B, N, vocab_size)

#             # SHIFT for loss
#             loss = loss_fn(decision_logits, label)
#             total_loss += loss.item()

#             # Perplexity
#             ppl = calculate_perplexity(decision_logits, label, pad_token=pad_id)
#             total_ppl += ppl

#             # For metrics: get probabilities and predictions
#             # decision_probs shape => (B, N, vocab_size)
#             decision_probs = F.softmax(decision_logits, dim=-1)  # per-class probabilities

#             # Flatten over (B*N)
#             B, N, V = decision_probs.shape
#             probs_2d  = decision_probs.view(-1, V).cpu().numpy()  # shape (B*N, V)
#             preds_2d  = probs_2d.argmax(axis=-1)                  # argmax for confusion matrix
#             labels_1d = label.view(-1).cpu().numpy()              # shape (B*N,)

#             # SHIFT for predictions
#             # preds  = torch.argmax(decision_logits, dim=-1)  # (B, T-1)
#             # labels_2D = label[:, 1:]                             # (B, T-1)

#             # Flatten to 1D
#             # preds_1D  = preds.cpu().numpy().ravel()
#             # labels_1D = label.cpu().numpy().ravel()

#             # Filter out special tokens from labels
#             valid_mask = ~np.isin(labels_1d, list(special_tokens))
#             preds_2d   = preds_2d[valid_mask]
#             labels_1d  = labels_1d[valid_mask]
#             probs_2d   = probs_2d[valid_mask, :]   # keep the same positions

#             # Append
#             all_preds.append(preds_2d)
#             all_labels.append(labels_1d)
#             all_probs.append(probs_2d)
#             valid_labels.append(labels_1d)

#     # Merge all into single 1D arrays
#     all_preds  = np.concatenate(all_preds)   # shape (N_total,)
#     all_labels = np.concatenate(all_labels)  # shape (N_total,)
#     all_probs  = np.concatenate(all_probs, axis=0)   # shape (N_total, vocab_size)
#     valid_labels = np.concatenate(valid_labels)

#     avg_loss = total_loss / len(dataloader)
#     avg_ppl  = total_ppl  / len(dataloader)

#     # Now we can do confusion_matrix and accuracy
#     # unique_labels = np.unique(all_labels)
#     decision_ids = np.arange(1, 10)   
#     # conf_mat = confusion_matrix(all_labels, all_preds, labels=unique_labels)
#     conf_mat = confusion_matrix(all_labels, all_preds, labels=decision_ids)
#     hit_rate = accuracy_score(all_labels, all_preds)
#     macro_f1 = f1_score(all_labels, all_preds, average='macro')
#     # auprc = average_precision_score(all_labels, all_preds, average='macro')

#     classes_for_bin = np.arange(1, 10)
#     y_true_bin = label_binarize(valid_labels, classes=classes_for_bin)
#     probs_for_auprc = all_probs[:, 1:10]  # keep columns 1..9
#     auprc = average_precision_score(y_true_bin, probs_for_auprc, average='macro')

#     label_mapping = {0: "[PAD]", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
#     readable_labels = [label_mapping.get(i, str(i)) for i in decision_ids]
#     print(f"Label IDs: {decision_ids}")
#     print(f"Label meanings: {readable_labels}")
#     print(f"Unique values in predictions: {np.unique(all_preds, return_counts=True)}")
#     print(f"Unique values in labels: {np.unique(all_labels, return_counts=True)}")

#     return avg_loss, conf_mat, avg_ppl, hit_rate, macro_f1, auprc
