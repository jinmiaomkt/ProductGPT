#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hP_tuning_LSTM_embed.py

LSTM baseline with the SAME input representation as ProductGPT:
  - Token ID embedding (id_embed)
  - Product feature projection (feat_proj)
  - Same SpecialPlusFeatureLookup as the transformer

This makes the comparison fair: both models see token IDs,
both have learned embeddings + product features.
The only difference is attention vs recurrence.
"""
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import itertools, json, os, math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import boto3, numpy as np, torch, torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset, random_split
from tokenizers import Tokenizer, models, pre_tokenizers

# ═════════ 0. Hyper-params ═════════════
D_MODEL_SIZES = [64, 128, 256]
HIDDEN_SIZES  = [64, 128, 256]
LR_VALUES     = [1e-3]
BATCH_SIZES   = [4, 8, 16]
HP_GRID = list(itertools.product(D_MODEL_SIZES, HIDDEN_SIZES, LR_VALUES, BATCH_SIZES))

INPUT_DIM    = 15
NUM_CLASSES  = 10
EPOCHS       = 80
AI_RATE      = 15
SEQ_LEN_TGT  = 1024
VOCAB_SIZE_SRC = 68

JSON_PATH = "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json"
FEAT_XLSX = "/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx"
S3_BUCKET = "productgptbucket"
S3_PREFIX = "LSTM_embed"

LOCAL_TMP = Path("/home/ec2-user/tmp_lstm_embed")
LOCAL_TMP.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3")

# ═════════ Feature tensor ═══════════════
FEATURE_COLS = [
    "Rarity", "MaxLife", "MaxOffense", "MaxDefense",
    "WeaponTypeOneHandSword", "WeaponTypeTwoHandSword", "WeaponTypeArrow",
    "WeaponTypeMagic", "WeaponTypePolearm",
    "EthnicityIce", "EthnicityRock", "EthnicityWater", "EthnicityFire",
    "EthnicityThunder", "EthnicityWind",
    "GenderFemale", "GenderMale",
    "CountryRuiYue", "CountryDaoQi", "CountryZhiDong", "CountryMengDe",
    "type_figure", "MinimumAttack", "MaximumAttack",
    "MinSpecialEffect", "MaxSpecialEffect",
    "SpecialEffectEfficiency", "SpecialEffectExpertise",
    "SpecialEffectAttack", "SpecialEffectSuper",
    "SpecialEffectRatio", "SpecialEffectPhysical", "SpecialEffectLife", "LTO",
]
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
UNK_PROD_ID = 59
MAX_TOKEN_ID = 68

def load_feature_tensor():
    df = pd.read_excel(FEAT_XLSX, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)

# ═════════ Tokenizer ════════════════════
PAD_ID = 0
SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12

def build_tokenizer_src():
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

def build_tokenizer_tgt():
    vocab = {
        "[PAD]": PAD_ID,
        **{str(i): i for i in range(1, 10)},
        "[SOS]": SOS_DEC_ID, "[EOS]": EOS_DEC_ID, "[UNK]": UNK_DEC_ID,
    }
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok

# ═════════ 1. Dataset (same as transformer) ═══════
class TokenSequenceDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            rows = json.load(f)

        self.tok_src = build_tokenizer_src()
        self.tok_tgt = build_tokenizer_tgt()
        seq_len_ai = SEQ_LEN_TGT * AI_RATE

        self.x, self.y = [], []
        for row in rows:
            agg = row["AggregateInput"]
            src_txt = " ".join(map(str, agg)) if isinstance(agg, (list, tuple)) else str(agg)
            ai_ids = self.tok_src.encode(src_txt).ids[:seq_len_ai]
            if len(ai_ids) < seq_len_ai:
                ai_ids = ai_ids + [PAD_ID] * (seq_len_ai - len(ai_ids))

            dec = row["Decision"]
            tgt_txt = " ".join(map(str, dec)) if isinstance(dec, (list, tuple)) else str(dec)
            tgt_ids = self.tok_tgt.encode(tgt_txt).ids[:SEQ_LEN_TGT]
            if len(tgt_ids) < SEQ_LEN_TGT:
                tgt_ids = tgt_ids + [PAD_ID] * (SEQ_LEN_TGT - len(tgt_ids))

            self.x.append(torch.tensor(ai_ids, dtype=torch.long))
            self.y.append(torch.tensor(tgt_ids, dtype=torch.long))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

# ═════════ 2. Embedding (same as transformer) ═════
class SpecialPlusFeatureLookup(nn.Module):
    def __init__(self, d_model, feature_tensor, product_ids, vocab_size_src):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_tensor.size(1)
        self.id_embed = nn.Embedding(vocab_size_src, d_model)
        self.feat_proj = nn.Linear(self.feature_dim, d_model, bias=False)
        self.register_buffer("feat_tbl", feature_tensor, persistent=False)
        prod_mask = torch.zeros(vocab_size_src, dtype=torch.bool)
        prod_mask[product_ids] = True
        self.register_buffer("prod_mask", prod_mask, persistent=False)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, ids):
        ids_long = ids.long()
        id_vec = self.id_embed(ids_long)
        raw_feat = self.feat_tbl[ids_long]
        feat_vec = self.feat_proj(raw_feat)
        keep = self.prod_mask[ids_long]
        feat_vec = feat_vec * keep.unsqueeze(-1)
        return id_vec + self.gamma * feat_vec

# ═════════ 3. LSTM model with embedding ═════════
class LSTMEmbedClassifier(nn.Module):
    """
    LSTM that takes the SAME token ID input as the transformer.
    Uses SpecialPlusFeatureLookup for embedding, then LSTM over the sequence.
    Predicts at every ai_rate-th position (same as transformer).
    """
    def __init__(self, d_model, hidden_size, feature_tensor, ai_rate=15):
        super().__init__()
        self.ai_rate = ai_rate

        self.embed = SpecialPlusFeatureLookup(
            d_model=d_model,
            feature_tensor=feature_tensor,
            product_ids=list(range(FIRST_PROD_ID, LAST_PROD_ID + 1)) + [UNK_PROD_ID],
            vocab_size_src=VOCAB_SIZE_SRC,
        )

        self.lstm = nn.LSTM(d_model, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x):
        emb = self.embed(x)              # (B, seq_len_ai, d_model)
        out, _ = self.lstm(emb)          # (B, seq_len_ai, hidden_size)
        logits = self.fc(out)            # (B, seq_len_ai, NUM_CLASSES)

        pos = torch.arange(self.ai_rate - 1, x.size(1), self.ai_rate, device=x.device)
        return logits[:, pos, :]         # (B, n_decisions, NUM_CLASSES)

# ═════════ 4. Metric helpers ═══════════
def _json_safe(o):
    import numpy as _np, torch as _th
    if isinstance(o, (_th.Tensor, _th.nn.Parameter)): return o.cpu().tolist()
    if isinstance(o, _np.ndarray):  return o.tolist()
    if isinstance(o, (_np.floating, _np.integer)): return o.item()
    if isinstance(o, dict):   return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    return o

# ═════════ 5. Evaluation ═══════════════
def evaluate(loader, model, device, loss_fn):
    model.eval()
    P, L, PR = [], [], []
    nll_sum, token_count = 0.0, 0
    tot_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            B, T, C = logits.shape
            flat_logits = logits.reshape(-1, C)
            flat_labels = yb[:, :T].reshape(-1)

            loss = loss_fn(flat_logits, flat_labels)
            tot_loss += loss.item()

            log_probs = F.log_softmax(flat_logits, dim=-1)
            nll_sum += F.nll_loss(log_probs, flat_labels, ignore_index=0, reduction='sum').item()
            token_count += (flat_labels != 0).sum().item()

            probs = log_probs.exp()
            probs_np = probs.cpu().numpy()
            preds_np = probs_np.argmax(1)
            labs_np = flat_labels.cpu().numpy()
            mask = labs_np != 0

            P.append(preds_np[mask])
            L.append(labs_np[mask])
            PR.append(probs_np[mask])

    P = np.concatenate(P)
    L = np.concatenate(L)
    PR = np.concatenate(PR)

    hit = accuracy_score(L, P)
    f1 = f1_score(L, P, average="macro", zero_division=0)
    try:
        auprc = average_precision_score(
            label_binarize(L, classes=np.arange(1, 10)),
            PR[:, 1:10], average="macro")
    except ValueError:
        auprc = float("nan")

    avg_loss = tot_loss / max(1, len(loader))
    avg_ppl = math.exp(nll_sum / max(1, token_count))

    return avg_loss, avg_ppl, {"hit": hit, "f1": f1, "auprc": auprc}

# ═════════ 6. One sweep job ════════════
def run_one(params):
    d_model, hidden, lr, bs = params
    uid = f"dm{d_model}_h{hidden}_lr{lr}_bs{bs}"

    ds = TokenSequenceDataset(JSON_PATH)
    n = len(ds)
    tr_n, va_n = int(.8 * n), int(.1 * n)
    g = torch.Generator().manual_seed(33)
    tr, va, te = random_split(ds, [tr_n, va_n, n - tr_n - va_n], generator=g)
    LD = lambda d, sh: DataLoader(d, bs, shuffle=sh, collate_fn=collate_fn)
    tr_ld, va_ld, te_ld = LD(tr, True), LD(va, False), LD(te, False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_tensor = load_feature_tensor()
    model = LSTMEmbedClassifier(d_model, hidden, feat_tensor, AI_RATE).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{uid}] Parameters: {n_params:,}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = LOCAL_TMP / f"lstm_embed_{uid}.pt"
    json_path = LOCAL_TMP / f"metrics_{uid}.json"

    best_loss, best_metrics = None, {}
    patience = 0
    PATIENCE_LIMIT = 10

    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            logits = model(xb)
            B, T, C = logits.shape
            loss = loss_fn(logits.reshape(-1, C), yb[:, :T].reshape(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            run_loss += loss.item()

        v_loss, v_ppl, v = evaluate(va_ld, model, dev, loss_fn)
        print(f"[{uid}] Ep{ep:02d} Train={run_loss/len(tr_ld):.4f} "
              f"Val={v_loss:.4f} PPL={v_ppl:.4f} Hit={v['hit']:.4f} F1={v['f1']:.4f}")

        if best_loss is None or v_loss < best_loss:
            best_loss, patience = v_loss, 0
            best_metrics = {"val_loss": v_loss, "val_ppl": v_ppl, **v}
            torch.save(model.state_dict(), ckpt_path)
            print("  [*] new best saved")
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                print("  [early-stop]")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    t_loss, t_ppl, t = evaluate(te_ld, model, dev, loss_fn)
    metrics = {
        "d_model": d_model, "hidden_size": hidden, "lr": lr, "batch_size": bs,
        "n_params": n_params,
        **{f"val_{k}": v for k, v in best_metrics.items()},
        "test_loss": t_loss, "test_ppl": t_ppl,
        **{f"test_{k}": v for k, v in t.items()},
    }
    json_path.write_text(json.dumps(_json_safe(metrics), indent=2))

    for local, key in [
        (ckpt_path, f"{S3_PREFIX}/checkpoints/{ckpt_path.name}"),
        (json_path, f"{S3_PREFIX}/metrics/{json_path.name}"),
    ]:
        s3.upload_file(str(local), S3_BUCKET, key)
        local.unlink(missing_ok=True)
        print(f"[S3] {local.name} → s3://{S3_BUCKET}/{key}")

    return uid, metrics

# ═════════ 7. Sweep + select best ══════
def sweep(max_workers=1):
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut = {ex.submit(run_one, p): p for p in HP_GRID}
        for f in as_completed(fut):
            p = fut[f]
            try:
                uid, metrics = f.result()
                all_results.append(metrics)
                print(f"[Done] {uid}")
            except Exception as e:
                print(f"[Error] params={p} → {e}")

    all_results.sort(key=lambda m: m.get("val_loss", float("inf")))

    print("\n" + "=" * 110)
    print(f"{'d_model':>8} {'Hidden':>8} {'BS':>4} {'Params':>10} "
          f"{'ValLoss':>9} {'ValHit':>8} {'ValF1':>8} "
          f"{'TstLoss':>9} {'TstHit':>8} {'TstF1':>8}")
    print("=" * 110)
    for m in all_results:
        print(f"{m['d_model']:>8} {m['hidden_size']:>8} {m['batch_size']:>4} {m['n_params']:>10,} "
              f"{m.get('val_val_loss', m.get('val_loss', 0)):>9.4f} "
              f"{m.get('val_hit', 0):>8.4f} {m.get('val_f1', 0):>8.4f} "
              f"{m['test_loss']:>9.4f} {m.get('test_hit', 0):>8.4f} {m.get('test_f1', 0):>8.4f}")

    best = all_results[0]
    print(f"\nBEST: d_model={best['d_model']}, hidden={best['hidden_size']}, "
          f"bs={best['batch_size']}, params={best['n_params']:,}")

    summary = {"all_results": _json_safe(all_results), "best": _json_safe(best)}
    summary_path = LOCAL_TMP / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    s3.upload_file(str(summary_path), S3_BUCKET, f"{S3_PREFIX}/sweep_summary.json")
    summary_path.unlink(missing_ok=True)

if __name__ == "__main__":
    sweep(max_workers=1)