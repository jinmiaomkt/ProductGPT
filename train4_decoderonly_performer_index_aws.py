import os, json, warnings, logging, numpy as np, boto3, botocore
from pathlib import Path
from typing import Dict, Any

# ─────────────────────────── silence clutter ────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("deepspeed").setLevel(logging.ERROR)

# ─────────────────────────── third-party deps ───────────────────────────
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb

# ─────────────────────────── project modules ────────────────────────────
from config4    import get_config
from model4_decoderonly_index_performer   import build_transformer
from dataset4_productgpt import TransformerDataset, load_json_dataset
from tokenizers           import Tokenizer, models, pre_tokenizers

# ══════════════════════ 1.  TOKENISERS ══════════════════════════════════
def _tok_base(extra: Dict[str, int] | None = None) -> Tokenizer:
    """
    Same as before, but `extra` is optional.  If you call `_tok_base()`
    you now get the 13-to-60 mapping as default.
    """
    if extra is None:                           # <- NEW
        extra = {str(i): i for i in range(13, 61)}
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(
        vocab={
            "[PAD]": 0,
            **{str(i): i for i in range(1, 10)},   # 1…9
            "[SOS]": 10,
            "[EOS]": 11,
            "[UNK]": 12,
            **extra
        },
        unk_token="[UNK]"
    )
    return tok


build_tokenizer_src = lambda: _tok_base({str(i): i for i in range(13, 61)})
build_tokenizer_tgt = lambda: _tok_base({})                       # 1…9 only

# ══════════════════════ 2.  REVENUE-GAP LOSS ════════════════════════════
# class PairwiseRevenueLoss(nn.Module):
#     def __init__(self, revenue, vocab_size, ignore_index=0):
#         super().__init__()
#         if len(revenue) < vocab_size:
#             revenue = revenue + [0.] * (vocab_size - len(revenue))
#         rev = torch.tensor(revenue, dtype=torch.float32)
#         self.register_buffer("penalty",
#                              -torch.abs(rev[:, None] - rev[None, :]))  # V×V
#         self.ignore_index = ignore_index

#     def forward(self, logits, targets):
#         V = logits.size(-1)
#         probs = F.softmax(logits.view(-1, V), dim=-1)   # (B*N, V)
#         tgt   = targets.view(-1)                        # (B*N,)
#         keep  = tgt != self.ignore_index
#         if keep.sum() == 0:
#             return logits.new_tensor(0.0)
#         pen = self.penalty.to(probs)
#         return -(probs[keep] * pen[tgt[keep]]).sum(1).mean()

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

        # --- bring the weight vector onto the right device & dtype -------
        if self.w is not None:
            if self.w.numel() != V:
                raise ValueError(
                    f"`weight` length {self.w.numel()} ≠ #classes {V}"
                )
            w = self.w.to(device=inputs_2d.device, dtype=inputs_2d.dtype)
        else:
            w = None

        # Use cross_entropy with 'none' reduction so we can apply focal transform ourselves
        ce_loss = F.cross_entropy(
            inputs_2d,
            targets_1d,
            reduction='none',
            weight=w,  # <---- the magic: per-class weighting
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

# ══════════════════════ 3.  HELPER FUNCTIONS ════════════════════════════
def transition_mask(seq: torch.Tensor) -> torch.Tensor:
    """True at decision-t where the decision differs from t-1."""
    prev = F.pad(seq, (1, 0), value=-1)[:, :-1]
    return seq != prev

def _perplexity(logits, targets, pad=0):
    logp = F.log_softmax(logits, dim=-1)
    lp2d, tgt = logp.view(-1, logp.size(-1)), targets.view(-1)
    m = tgt != pad
    if m.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2d[m], tgt[m], reduction="mean")).item()

def _subset(pred, lbl, probs, mask, classes=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": np.nan, "f1": np.nan, "auprc": np.nan}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    hit = accuracy_score(l, p)
    f1  = f1_score(l, p, average="macro")
    try:
        auprc = average_precision_score(
            label_binarize(l, classes=classes), pr[:, 1:10], average="macro")
    except ValueError:
        auprc = np.nan
    return {"hit": hit, "f1": f1, "auprc": auprc}

def _json_safe(o: Any):
    import numpy as _np, torch as _th
    if isinstance(o, (_th.Tensor, _th.nn.Parameter)): return o.cpu().tolist()
    if isinstance(o, _np.ndarray):  return o.tolist()
    if isinstance(o, (_np.floating, _np.integer)):   return o.item()
    if isinstance(o, dict):   return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
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
    
# ══════════════════════ 4.  DATA LOADERS ════════════════════════════════
def _make_loaders(cfg):
    raw = load_json_dataset(cfg["filepath"])
    n   = len(raw); tr, va = int(.8*n), int(.1*n)
    g   = torch.Generator().manual_seed(33)
    tr_ds, va_ds, te_ds = random_split(raw, [tr, va, n-tr-va], generator=g)

    tok_src = build_tokenizer_src()
    tok_tgt = build_tokenizer_tgt()

    out_dir = Path(cfg["model_folder"]); out_dir.mkdir(parents=True, exist_ok=True)
    tok_src.save(str(out_dir / "tokenizer_ai.json"))
    tok_tgt.save(str(out_dir / "tokenizer_tgt.json"))

    def mk(split):
        return TransformerDataset(
            split, tok_src, tok_tgt,
            cfg["seq_len_ai"], cfg["seq_len_tgt"],
            cfg["num_heads"], cfg["ai_rate"], pad_token=0)

    LD = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)
    return (LD(mk(tr_ds), True),
            LD(mk(va_ds), False),
            LD(mk(te_ds), False),
            tok_tgt)

# ══════════════════════ 5.  MODEL ══════════════════════════════════════
def _build_model(cfg):
    return build_transformer(
        vocab_size  = cfg["vocab_size_src"],      # logits over *src* vocab
        d_model     = cfg["d_model"],
        n_layers    = cfg["N"],
        n_heads     = cfg["num_heads"],
        d_ff        = cfg["d_ff"],
        dropout     = cfg["dropout"],
        max_seq_len = cfg["seq_len_ai"],
        nb_features = cfg["nb_features"], 
        block_size_h = cfg["ai_rate"],
        block_size_w = cfg["ai_rate"],
        kernel_type = cfg["kernel_type"]
)

# ══════════════════════ 6.  EVALUATION ══════════════════════════════════
def _evaluate(loader, eng, dev, loss_fn, pad, tok, ai_rate):
    if not loader:
        nan = float("nan"); emp = {"hit":nan,"f1":nan,"auprc":nan}
        return nan, nan, emp, emp, emp, emp

    special = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}

    tot_loss = tot_ppl = 0.0
    P, L, PR = [], [], []
    m_stop, m_after_stop, m_tr = [], [], []

    eng.eval()
    with torch.no_grad():
        for b in loader:
            x   = b["aggregate_input"].to(dev)       # (B, seq_len_ai)
            tgt = b["label"].to(dev)                 # (B, seq_len_tgt)

            seq_len = x.size(1)
            pos = torch.arange(ai_rate -1,
                            seq_len, 
                            ai_rate,
                            device=dev)
            assert pos.numel() == tgt.size(1), (
                f"expected {tgt.size(1)} slots but got {pos.numel()}"
            )
            # n_slots = tgt.size(1)
            # pos = torch.arange(ai_rate-1,
            #                    ai_rate*n_slots,
            #                    ai_rate, device=dev)   # len == n_slots
            logits = eng(x)[:, pos, :]               # (B, n_slots, V)

            # ---- loss: ignore transitions --------------------------------
            tgt_loss = tgt.clone()
            tgt_loss[transition_mask(tgt)] = pad
            tot_loss += loss_fn(logits, tgt_loss).item()
            tot_ppl  += _perplexity(logits, tgt_loss, pad)

            # ---- metrics --------------------------------------------------
            prob = F.softmax(logits, dim=-1).view(-1, logits.size(-1)).cpu().numpy()
            pred = prob.argmax(1)
            lbl  = tgt.view(-1).cpu().numpy()
            keep = ~np.isin(lbl, list(special))

            P.append(pred[keep]); L.append(lbl[keep]); PR.append(prob[keep])

            flat = lambda m: m.view(-1).cpu().numpy()[keep]
            m_stop.append(flat(tgt == 9))
            prev = F.pad(tgt, (1,0), value=-1)[:, :-1]
            m_after_stop.append(flat(prev == 9))
            m_tr.append(flat(transition_mask(tgt)))

    P,L,PR = map(np.concatenate, (P,L,PR))
    m_stop_cur   = np.concatenate(m_stop)
    m_after_stop = np.concatenate(m_after_stop)
    m_transition = np.concatenate(m_tr)
    all_mask     = np.ones_like(P, dtype=bool)

    return (tot_loss/len(loader), tot_ppl/len(loader),
            _subset(P,L,PR, all_mask),
            _subset(P,L,PR, m_stop_cur),
            _subset(P,L,PR, m_after_stop),
            _subset(P,L,PR, m_transition))

# ══════════════════════ 7.  TRAINING LOOP ═══════════════════════════════
def train_model(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uid   = (f"performer_nb_features{cfg['nb_features']}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
             f"heads{cfg['num_heads']}_lr{cfg['lr']}_weight{cfg['weight']}")
    ckpt_path = Path(cfg["model_folder"]) / f"FullProductGPT_{uid}.pt"
    json_path = ckpt_path.with_suffix(".json")

    s3      = _s3_client()
    bucket  = cfg["s3_bucket"]
    ck_key  = f"FullProductGPT/performer/checkpoints/{ckpt_path.name}"
    js_key  = f"FullProductGPT/performer/metrics/{json_path.name}"
    
    print(f"[INFO] artefacts will be saved to\n"
          f"  • s3://{bucket}/{ck_key}\n"
          f"  • s3://{bucket}/{js_key}\n")
    
    tr, va, te, tok_tgt = _make_loaders(cfg)
    pad_id = tok_tgt.token_to_id("[PAD]")

    model   = _build_model(cfg)
    # loss_fn = PairwiseRevenueLoss(
    #     revenue=[0,1,10,1,10,1,10,1,10,0],    # customise as you wish
    #     vocab_size=cfg["vocab_size_src"],      # MUST match logits.size(-1)
    #     ignore_index=pad_id)

    tokenizer_tgt = _tok_base()

    weights = torch.ones(cfg['vocab_size_src'])
    weights[9] = cfg['weight']
    weights = weights.to(dev)
    loss_fn = FocalLoss(gamma=cfg['gamma'], ignore_index=tokenizer_tgt.token_to_id('[PAD]'), class_weights=weights).to(dev)

    eng, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": cfg["batch_size"],
            "zero_allow_untested_optimizer": True,
            "optimizer":{"type":"Lamb",
                         "params":{"lr":cfg["lr"],
                                   "eps":cfg["eps"],
                                   "weight_decay":cfg["weight_decay"]}},
            "zero_optimization":{"stage":1},
            "gradient_accumulation_steps": 2,
            "fp16": {"enabled": True}
        }
    )

    best, patience = None, 0
    for ep in range(cfg["num_epochs"]):
        # ---------------- training -------------------------------------
        eng.train(); running = 0.0
        for b in tqdm(tr, desc=f"Ep {ep:02d}", leave=False):
            x   = b["aggregate_input"].to(dev)
            tgt = b["label"].to(dev)               # (B, n_slots)

            # seq_len = logits.size(1)
            pos = torch.arange(cfg["ai_rate"]-1,
                            cfg["seq_len_ai"],
                            cfg["ai_rate"],
                            device=dev)
            assert pos.numel() == tgt.size(1), (
                f"expected {tgt.size(1)} slots but got {pos.numel()}"
            )
            # n_slots = tgt.size(1)
            # pos = torch.arange(cfg["ai_rate"]-1,
            #                    cfg["ai_rate"]*n_slots,
            #                    cfg["ai_rate"], device=dev)
            logits = eng(x)[:, pos, :]             # (B, n_slots, V)

            tgt_train = tgt.clone()
            tgt_train[transition_mask(tgt)] = pad_id
            loss = loss_fn(logits, tgt_train)

            eng.zero_grad(); eng.backward(loss); eng.step()
            running += loss.item()
        print(f"\nTrain loss {running/len(tr):.4f}")

        # ---------------- validation ------------------------------------
        v_loss,v_ppl,v_all,v_stop,v_after,v_tr = _evaluate(
            va, eng, dev, loss_fn, pad_id, tok_tgt, cfg["ai_rate"])

        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag,d in (("all",v_all),("STOP_cur",v_stop),
                      ("after_STOP",v_after),("transition",v_tr)):
            print(f"  {tag:<12} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

        # -------------- checkpoint logic --------------------------------
        if best is None or v_loss < best:
            best, patience = v_loss, 0
            torch.save({"epoch": ep,
                        "model_state_dict": eng.module.state_dict()}, ckpt_path)
            json_path.write_text(json.dumps(_json_safe({
                "best_checkpoint_path": ckpt_path.name,
                "val_loss": best, "val_ppl": v_ppl,
                "val_all": v_all, "val_stop_cur": v_stop,
                "val_after_stop": v_after, "val_transition": v_tr
            }), indent=2))
            print(f"  [*] new best saved → {ckpt_path.name}")

            if _upload(ckpt_path, bucket, ck_key, s3):
                ckpt_path.unlink(missing_ok=True)
            if _upload(json_path, bucket, js_key, s3):
                json_path.unlink(missing_ok=True)

        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping."); break

    # -------------- test on best weights --------------------------------
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=dev)
        eng.module.load_state_dict(state["model_state_dict"])

        t_loss,t_ppl,t_all,t_stop,t_after,t_tr = _evaluate(
            te, eng, dev, loss_fn, pad_id, tok_tgt, cfg["ai_rate"])

        print(f"\n** TEST ** Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
        for tag,d in (("all",t_all),("STOP_cur",t_stop),
                      ("after_STOP",t_after),("transition",t_tr)):
            print(f"  {tag:<12} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

    # return {"uid": uid, "val_loss": best}
    return {"uid": uid, "val_loss": best, "best_checkpoint_path": str(ckpt_path)}

# ══════════════════════ 8.  CLI ENTRY ═══════════════════════════════════
if __name__ == "__main__":
    cfg = get_config()
    cfg["ai_rate"] = 15
    cfg["seq_len_ai"]  = cfg["ai_rate"] * cfg["seq_len_tgt"]   # = 480
    train_model(cfg)
