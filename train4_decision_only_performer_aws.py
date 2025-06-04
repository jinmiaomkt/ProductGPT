# train4_decision_only_aws.py
# ================================================================
# Decision-Only trainer (quiet) – PairwiseRevenueLoss objective
# ---------------------------------------------------------------
# • DeepSpeed ZeRO-1 + FusedLAMB   (all DS chatter muted)
# • Validation metrics on three subsets:
#     ① main        – non-transition positions
#     ② STOP        – positions whose *current* token is 9
#     ③ transition  – token ≠ previous token
# • On each val-improvement we save
#       DecisionOnly_<uid>.pt   →  s3://<bucket>/DecisionOnly/checkpoints/
#       DecisionOnly_<uid>.json →  s3://<bucket>/DecisionOnly/metrics/
#   and delete the local copies after a successful upload.
# ================================================================

# ─────────────── environment & logging hygiene ────────────────
import os, warnings, logging, json, numpy as np, boto3, botocore
from pathlib import Path
os.environ.update({"TF_CPP_MIN_LOG_LEVEL": "3", "DEEPSPEED_LOG_LEVEL": "error", "DS_DISABLE_LOGS": "1"})
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
for name in ("deepspeed", "torch_checkpoint_engine", "engine"):
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).propagate = False

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb

from model4_decoderonly_index_performer import build_transformer
from dataset4_decision_only import TransformerDataset
from tokenizers import Tokenizer, models, pre_tokenizers
from config4 import get_config
from typing import Tuple 

# ───────────────────── tokenizer ──────────────────────

def _build_tok():
    """Gap‑free vocabulary (adds token 11 to close the hole)."""
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.model = models.WordLevel({**{str(i): i for i in range(1, 10)}, "[PAD]": 0, "[SOS]": 10, "[MASK]": 11, "[UNK]": 12}, unk_token="[UNK]")
    return t

# ─────────────── robust streaming dataset ───────────────

class JsonLineDataset(torch.utils.data.Dataset):
    """Read newline‑delimited JSON – tolerant of trailing commas & list brackets."""

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.offsets: list[int] = []
        with self.filepath.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                txt = line.strip()
                if txt.startswith(b"{"):                # only real objects
                    self.offsets.append(pos)
        if not self.offsets:
            raise ValueError(f"{filepath} appears empty or not NDJSON.")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx: int):
        with self.filepath.open("rb") as f:
            f.seek(self.offsets[idx])
            raw = f.readline().decode("utf-8").strip()
            # strip trailing comma (from old list syntax)
            if raw.endswith(","):
                raw = raw[:-1].rstrip()
            return json.loads(raw)

# ─────────── helper: convert legacy JSON → NDJSON ───────────

def convert_to_jsonl(src: str | Path, dst: str | Path | None = None):
    """Convert monolithic list/dict JSON to newline JSON."""
    src = Path(src)
    dst = Path(dst or src).with_suffix(".jsonl")
    if dst.exists():
        return dst
    with src.open() as fin, dst.open("w") as fout:
        data = json.load(fin)
        if isinstance(data, list):
            for obj in data:
                fout.write(json.dumps(obj) + "\n")
        elif isinstance(data, dict):
            keys = list(data)
            for row in zip(*(data[k] for k in keys)):
                fout.write(json.dumps(dict(zip(keys, row))) + "\n")
        else:
            raise ValueError("Unsupported JSON root for conversion")
    return dst

# ───────── ensure dataset is NDJSON ─────────

def _ensure_jsonl(path: str | Path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return p
    with p.open() as f:
        first_non_space = ""
        while first_non_space.isspace() or not first_non_space:
            first_non_space = f.read(1)
    # If file starts with '[' it's a list → convert
    if first_non_space == "[":
        return convert_to_jsonl(p)
    return p  # assume already NDJSON

# ─────────────── data loaders ───────────────

def _make_loaders(cfg):
    filepath = _ensure_jsonl(cfg["filepath"])
    raw = JsonLineDataset(filepath)
    n = len(raw)
    tr, va = int(0.8 * n), int(0.1 * n)
    gen = torch.Generator().manual_seed(33)
    tr_ds, va_ds, te_ds = random_split(raw, [tr, va, n - tr - va], generator=gen)

    tok_ai = tok_tgt = _build_tok()
    out = Path(cfg["model_folder"]); out.mkdir(parents=True, exist_ok=True)
    # ✔ cast Path → str to satisfy tokenizer.save()
    tok_ai.save(str(out / "tokenizer_ai.json"))
    tok_tgt.save(str(out / "tokenizer_tgt.json"))

    def mk(split):
        return TransformerDataset(split, tok_ai, tok_tgt, cfg["seq_len_ai"], cfg["seq_len_tgt"], cfg["num_heads"], cfg["ai_rate"], pad_token=0)

    loader = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)
    return loader(mk(tr_ds), True), loader(mk(va_ds), False), loader(mk(te_ds), False), tok_tgt


# # --- loss -----------------------------------------------------
# class PairwiseRevenueLoss(nn.Module):
#     def __init__(self, revenue, vocab_size, ignore_index=0):
#         super().__init__()
#         if len(revenue) < vocab_size:
#             revenue = revenue + [0.] * (vocab_size - len(revenue))
#         rev = torch.as_tensor(revenue, dtype=torch.float32)
#         self.register_buffer("penalty", -torch.abs(rev[:, None] - rev[None, :]))
#         self.ignore = ignore_index

#     def forward(self, logits, tgt):
#         V = logits.size(-1)
#         probs = F.softmax(logits.view(-1, V), dim=-1)
#         tgt   = tgt.view(-1)
#         keep  = tgt != self.ignore
#         if keep.sum() == 0:
#             return logits.sum() * 0.0
#         pen = self.penalty.to(probs)
#         return -(probs[keep] * pen[tgt[keep]]).sum(dim=-1).mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0.0, ignore_index=0, class_weights=None):
        """
        Args:
            gamma (float): Focal loss exponent, default=2.
            ignore_index (int): Token ID to ignore in the loss.
            class_weights (Tensor): 1D tensor of shape [num_classes],
                                    e.g. to upweight rare classes.
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        # Register the weights as a buffer so they move to GPU with the model.
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, inputs, targets):
        """
        inputs: (B, T, V) => raw logits
        targets: (B, T)   => integer class IDs
        """
        B, T, V = inputs.shape

        # Flatten to 1D
        inputs_2d = inputs.reshape(-1, V)         # (B*T, V)
        targets_1d = targets.reshape(-1)          # (B*T,)

        # --- make sure `weight` matches logits’ dtype/device -------------
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights
            if weight.dtype != inputs_2d.dtype or weight.device != inputs_2d.device:
                weight = weight.to(device=inputs_2d.device, dtype=inputs_2d.dtype)

        # Use cross_entropy with 'none' reduction so we can apply focal transform ourselves
        ce_loss = F.cross_entropy(
            inputs_2d,
            targets_1d,
            reduction='none',
            weight=weight,  # <---- the magic: per-class weighting
            ignore_index=self.ignore_index
        )

        # Mask out tokens == ignore_index
        valid_mask = (targets_1d != self.ignore_index)
        ce_loss = ce_loss[valid_mask]

        # Focal transform
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        # If everything got masked, return 0
        if focal.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)

        return focal.mean()
    
# --- tiny utils -------------------------------------------------
def _transition_mask(lbl: torch.Tensor):
    return lbl != F.pad(lbl, (1, 0), value=-1)[:, :-1]

def _ppl(logits, tgt, pad=0):
    lp  = F.log_softmax(logits, dim=-1)
    lp2 = lp.view(-1, lp.size(-1)); t = tgt.view(-1)
    m   = t != pad
    if m.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2[m], t[m], reduction="mean")).item()

def _subset(pred, lbl, probs, rev_err, mask, cls=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": np.nan, "f1": np.nan, "auprc": np.nan, "rev_mae": np.nan}
    p, l, pr, re = pred[mask], lbl[mask], probs[mask], rev_err[mask]
    hit = accuracy_score(l, p)
    f1  = f1_score(l, p, average="macro")
    try:
        au = average_precision_score(
                label_binarize(l, classes=cls), pr[:, 1:10], average="macro")
    except ValueError:
        au = np.nan
    return {"hit": hit, "f1": f1, "auprc": au, "rev_mae": re.mean()}

def _json_safe(o):
    import numpy as _np, torch as _th
    if isinstance(o, (_th.Tensor, _th.nn.Parameter)):
        return o.detach().cpu().tolist()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (_np.generic,)):
        return o.item()
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o

# --- S3 helpers -------------------------------------------------
def _s3_client():
    try: return boto3.client("s3")
    except botocore.exceptions.BotoCoreError: return None

def _upload(local: Path, bucket: str, key: str, s3) -> bool:
    if s3 is None or not local.exists():
        return False
    try:
        s3.upload_file(str(local), bucket, key)
        print(f"[S3] {local.name}  →  s3://{bucket}/{key}")
        return True
    except botocore.exceptions.BotoCoreError as e:
        print(f"[S3-ERR] {e}")
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

# ═════════════════ data & model ═════════════════════════════════
# def _make_loaders(cfg):
#     # raw = load_json_dataset(cfg["filepath"])
#     raw = JsonLineDataset(cfg["filepath"])

#     n   = len(raw); tr, va = int(.8*n), int(.1*n)
#     gen = torch.Generator().manual_seed(33)
#     tr_ds, va_ds, te_ds = random_split(raw, [tr, va, n-tr-va], generator=gen)

#     tok_ai = tok_tgt = _build_tok()
#     out = Path(cfg["model_folder"]); out.mkdir(parents=True, exist_ok=True)
#     tok_ai .save(str(out / "tokenizer_ai.json"))
#     tok_tgt.save(str(out / "tokenizer_tgt.json"))

#     def mk(split):
#         return TransformerDataset(split, tok_ai, tok_tgt,
#                                   cfg["seq_len_ai"], cfg["seq_len_tgt"],
#                                   cfg["num_heads"], cfg["ai_rate"], pad_token=0)
#     loader = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)
#     return (loader(mk(tr_ds), True),
#             loader(mk(va_ds), False),
#             loader(mk(te_ds), False),
#             tok_tgt)

def _build_model(cfg):
    return build_transformer(
        vocab_size  = cfg["vocab_size_tgt"], 
        d_model     = cfg["d_model"],
        n_layers    = cfg["N"],
        n_heads     = cfg["num_heads"],
        d_ff        = cfg["d_ff"],
        dropout     = cfg["dropout"],
        max_seq_len = cfg["seq_len_ai"],
        nb_features = cfg["nb_features"], 
        kernel_type = cfg["kernel_type"]
    )

# ═════════════════ evaluation ═══════════════════════════════════
# ───────── evaluation on three requested subsets ─────────
# ───────────────────────── evaluation (adds "all" and after-STOP) ──────────────────────
def _evaluate(loader, eng, dev, loss_fn, step, pad, tok):
    """
    Returns:
        loss, ppl,
        metrics_all, metrics_cur_stop, metrics_after_stop, metrics_transition
    """
    if len(loader) == 0:
        nan = float("nan")
        empty = {"hit": nan, "f1": nan, "auprc": nan}
        return nan, nan, empty, empty, empty, empty

    special = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}

    tloss = tppl = 0.0
    P, L, PR, RE = [], [], [], []
    cur_stop_mask_all   = []
    after_stop_mask_all = []
    transition_mask_all = []

    REV_VEC = torch.tensor([1, 10, 1, 10, 1, 10, 1, 10, 0],
                       dtype=torch.float32)                      # shape (9,)
    rev_vec = REV_VEC.to(dev)

    eng.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["aggregate_input"].to(dev)
            y = batch["label"].to(dev)

            pos = torch.arange(step - 1, x.size(1), step, device=dev)

            logits = eng(x)[:, pos, :]              # (B, N, V)
            tgt    = y[:, pos].clone()              # (B, N)

            # mask transitions for the loss only
            tgt_masked = tgt.clone()
            tgt_masked[_transition_mask(y)[:, pos]] = pad

            tloss += loss_fn(logits, tgt_masked).item()
            tppl  += _ppl(logits, tgt_masked, pad)

            # ---- metrics --------------------------------------------------
            prob_t = F.softmax(logits, dim=-1)
            rev_vec = rev_vec.to(dtype=prob_t.dtype)

            # ----- revenue error (all torch) --------------------------------
            exp_rev  = (prob_t[..., 1:10] * rev_vec).sum(-1)              # (B, n_slots)
            true_rev = rev_vec[(tgt - 1).clamp(min=0, max=8)]             # same shape
            rev_err  = torch.abs(exp_rev - true_rev).view(-1).cpu().numpy()

            # ─── flatten predictions & labels (after skipping specials) ───
            prob = prob_t.view(-1, logits.size(-1)).cpu().numpy()
            pred = prob.argmax(1)
            lbl  = tgt.view(-1).cpu().numpy()

            keep = ~np.isin(lbl, list(special))      # drop [PAD] [SOS] [UNK]
            P.append(pred[keep])
            L.append(lbl [keep])
            PR.append(prob[keep])
            RE.append(rev_err[keep])

            # derived masks
            flat = lambda m: m.view(-1).cpu().numpy()[keep]
            cur_stop_mask_all   .append(flat(tgt == 9))
            transition_mask_all .append(flat(_transition_mask(y)[:, pos]))
            prev_tok = F.pad(tgt, (1, 0), value=-1)[:, :-1]
            after_stop_mask_all .append(flat(prev_tok == 9))

    # concat everything
    P = np.concatenate(P)
    L = np.concatenate(L)
    PR = np.concatenate(PR)
    RE = np.concatenate(RE)
    cur_stop_mask    = np.concatenate(cur_stop_mask_all)
    after_stop_mask  = np.concatenate(after_stop_mask_all)
    transition_mask  = np.concatenate(transition_mask_all)

    all_mask = np.ones_like(P, dtype=bool)

    return (tloss / len(loader), tppl / len(loader),
            _subset(P, L, PR, RE, all_mask),
            _subset(P, L, PR, RE, cur_stop_mask),
            _subset(P, L, PR, RE, after_stop_mask),
            _subset(P, L, PR, RE, transition_mask))

# ═════════════════ training loop ════════════════════════════════
def train_model(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- unique run-id & file names -----------------------
    uid   = (f"decisiononly_performerfeatures{cfg['nb_features']}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
             f"heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}")
    ckpt_local = Path(cfg["model_folder"]) / f"DecisionOnly_{uid}.pt"
    json_local = ckpt_local.with_suffix(".json")

    s3      = _s3_client()
    bucket  = cfg["s3_bucket"]
    ck_key  = f"DecisionOnly/checkpoints/{ckpt_local.name}"
    js_key  = f"DecisionOnly/metrics/{json_local.name}"
    print(f"[INFO] artefacts will be saved to\n"
          f"  • s3://{bucket}/{ck_key}\n"
          f"  • s3://{bucket}/{js_key}\n")

    # ---------- data / model / loss -----------------------------
    tr, va, te, tok = _make_loaders(cfg)
    pad_id = tok.token_to_id("[PAD]")

    model   = _build_model(cfg)
    # loss_fn = PairwiseRevenueLoss([0,1,10,1,10,1,10,1,10,0], cfg["vocab_size_tgt"], pad_id)

    tokenizer_tgt = _build_tok()

    weights = torch.ones(cfg['vocab_size_tgt'])
    weights[9] = cfg['weight']
    weights = weights.to(dev)
    loss_fn = FocalLoss(gamma=cfg['gamma'], ignore_index=tokenizer_tgt.token_to_id('[PAD]'), class_weights=weights).to(dev)
    
    eng, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": cfg["batch_size"],
            "zero_allow_untested_optimizer": True,
            "optimizer": {
                "type": "Lamb",
                "params": { "lr": cfg["lr"], "eps": cfg["eps"], "weight_decay": cfg["weight_decay"] }
            },
            "zero_optimization": { "stage": 2 }, 
            "gradient_accumulation_steps": 2,
            "fp16": {"enabled": True}
        }
    )

    # ---------- epochs ------------------------------------------
    best_val_loss, patience = None, 0
    for ep in range(cfg["num_epochs"]):
        eng.train(); run = 0.0
        for b in tqdm(tr, desc=f"Ep {ep:02d}", leave=False):
            x, y = b["aggregate_input"].to(dev), b["label"].to(dev)
            pos  = torch.arange(cfg["ai_rate"]-1, x.size(1),
                                cfg["ai_rate"], device=dev)

            logits = eng(x)[:, pos, :]             # (B, N, V)
            tgt    = y[:, pos].clone()
            tgt[_transition_mask(y)[:, pos]] = pad_id

            loss = loss_fn(logits, tgt)
            eng.zero_grad(); eng.backward(loss); eng.step()
            run += loss.item()
        print(f"\nTrain loss {run/len(tr):.4f}")

        v_loss,v_ppl,v_all,v_stop,v_after,v_tr = _evaluate(
            va, eng, dev, loss_fn, cfg["ai_rate"], pad_id, tok)
        
        print(f"Valudation Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag, d in (
                ("all",        v_all),
                ("cur-STOP",   v_stop),
                ("after-STOP", v_after),
                ("transition", v_tr)):
            print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                f"AUPRC={d['auprc']:.4f}   RevMAE={d['rev_mae']:.4f} ")
            
        print(f"[INFO] artefacts will be saved to\n"
          f"  • s3://{bucket}/{ck_key}\n"
          f"  • s3://{bucket}/{js_key}\n")

        # ----- store best ----------------------------------------
        if best_val_loss is None or v_loss < best_val_loss:
            best_val_loss, patience = v_loss, 0

            best_val_metrics = {
                "val_loss"             : v_loss,
                "val_ppl"              : v_ppl,
                "val_all_hit_rate"     : v_all["hit"],
                "val_all_f1_score"     : v_all["f1"],
                "val_all_auprc"        : v_all["auprc"],
                "val_all_rev_mae"      : v_all["rev_mae"],
                "val_stop_hit_rate"    : v_stop["hit"],
                "val_stop_f1_score"    : v_stop["f1"],
                "val_stop_auprc"       : v_stop["auprc"],
                "val_stop_rev_mae"     : v_stop["rev_mae"],
                "val_after_hit_rate"     : v_after["hit"],
                "val_after_f1_score"     : v_after["f1"],
                "val_after_auprc"        : v_after["auprc"],
                "val_after_rev_mae"      : v_after["rev_mae"],
                "val_transition_hit_rate"     : v_tr["hit"],
                "val_transition_f1_score"     : v_tr["f1"],
                "val_transition_auprc"        : v_tr["auprc"],
                "val_transition_rev_mae"      : v_tr["rev_mae"],
            }
            torch.save({"epoch": ep,
                         "best_val_loss": best_val_loss,
                         "model_state_dict": eng.module.state_dict()}, ckpt_local)

        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping."); break

    # ---------- test --------------------------------------------
    # (will only run if ckpt wasn't deleted, e.g. during local tests)
    if ckpt_local.exists():
        state = torch.load(ckpt_local, map_location=dev)
        eng.module.load_state_dict(state["model_state_dict"])

        t_loss,t_ppl,t_all,t_stop,t_after,t_tr = _evaluate(te, eng, dev, loss_fn, cfg["ai_rate"], pad_id, tok)

        print(f"Valudation Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag, d in (
            ("all",        v_all),
            ("cur-STOP",   t_stop),
            ("after-STOP", t_after),
            ("transition", v_tr)):
            print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                f"AUPRC={d['auprc']:.4f}   RevMAE={d['rev_mae']:.4f} ")

        print(f"[INFO] artefacts will be saved to\n"
          f"  • s3://{bucket}/{ck_key}\n"
          f"  • s3://{bucket}/{js_key}\n")

    metadata = {
        "best_checkpoint_path": ckpt_local.name,
        **best_val_metrics,
        "test_loss"            : t_loss,
        "test_ppl"             : t_ppl,
        "test_all_hit_rate"     : t_all["hit"],
        "test_all_f1_score"     : t_all["f1"],
        "test_all_auprc"        : t_all["auprc"],
        "test_all_rev_mae"      : t_all["rev_mae"],
        "test_stop_hit_rate"    : t_stop["hit"],
        "test_stop_f1_score"    : t_stop["f1"],
        "test_stop_auprc"       : t_stop["auprc"],
        "test_stop_rev_mae"     : t_stop["rev_mae"],
        "test_after_hit_rate"     : t_after["hit"],
        "test_after_f1_score"     : t_after["f1"],
        "test_after_auprc"        : t_after["auprc"],
        "test_after_rev_mae"      : t_after["rev_mae"],
        "test_transition_hit_rate"     : t_tr["hit"],
        "test_transition_f1_score"     : t_tr["f1"],
        "test_transition_auprc"        : t_tr["auprc"],
        "test_transition_rev_mae"      : t_tr["rev_mae"],
    }

    json_local.write_text(json.dumps(_json_safe(metadata), indent=2))
    print(f"[INFO] Metrics written → {json_local}")    

    if _upload(ckpt_local, bucket,
               f"DecisionOnly/checkpoints/{ckpt_local.name}", s3):
        ckpt_local.unlink(missing_ok=True)
    
    if _upload(json_local, bucket,
               f"DecisionOnly/metrics/{json_local.name}", s3):
        json_local.unlink(missing_ok=True)

    return {"uid": uid, "val_loss": best_val_loss, "best_checkpoint_path": str(ckpt_local)}
   
# ──────────────── CLI (for quick manual run) ────────────────────
if __name__ == "__main__":
    cfg = get_config()
    cfg["ai_rate"] = 1
    train_model(cfg)
