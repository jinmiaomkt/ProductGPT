# train4_decision_only_aws.py
# ===================================================================
# Decision-Only trainer (quiet)  ◇  PairwiseRevenueLoss objective
# -------------------------------------------------------------------
# • DeepSpeed ZeRO-1 + FusedLAMB, all DS chatter muted
# • Validation metrics reported on three subsets:
#       ① main (non-transition tokens)
#       ② STOP (current slot == 9)
#       ③ transition (token != previous token)
# • Whenever val-loss improves:
#       – DecisionOnly_<uid>.pt   ➜  s3://<bucket>/DecisionOnly/checkpoints/
#       – DecisionOnly_<uid>.json ➜  s3://<bucket>/DecisionOnly/metrics/
#   Local copies are removed after a successful upload.
# • First console lines show the exact S3 targets.
# ===================================================================

# ───────────────────── env & logging hygiene ────────────────────────
import os, warnings, logging, json, numpy as np, boto3, botocore
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["DEEPSPEED_LOG_LEVEL"]   = "error"
os.environ["DS_DISABLE_LOGS"]       = "1"

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
for n in ("deepspeed", "torch_checkpoint_engine", "engine"):
    logging.getLogger(n).setLevel(logging.ERROR)
    logging.getLogger(n).propagate = False

# ───────────────────── std / 3rd-party imports ──────────────────────
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb

from model4_decoderonly     import build_transformer
from dataset4_decision_only import TransformerDataset, load_json_dataset
from tokenizers             import Tokenizer, models, pre_tokenizers
from config4_decision_only_git import get_config

# ───────────────────── tokenizer (fixed vocab) ──────────────────────
def _build_tok():
    t = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.model = models.WordLevel(
        {**{str(i): i for i in range(1, 10)}, "[PAD]": 0,
         "[SOS]": 10, "[UNK]": 12},
        unk_token="[UNK]")
    return t

# ───────────────────── Pair-wise revenue loss ───────────────────────
class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue = revenue + [0.] * (vocab_size - len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("penalty",
                             -torch.abs(rev[:, None] - rev[None, :]))
        self.ignore = ignore_index

    def forward(self, logits, targets):
        B, T, V = logits.shape
        probs = F.softmax(logits.view(-1, V), dim=-1)
        tgt   = targets.view(-1)
        keep  = tgt != self.ignore
        if keep.sum() == 0:
            return logits.new_tensor(0.0)
        pen = self.penalty.to(probs)
        return -(probs[keep] * pen[tgt[keep]]).sum(dim=-1).mean()

# ───────────────────── misc helpers ─────────────────────────────────
def _transition_mask(y):
    return y != F.pad(y, (1, 0), value=-1)[:, :-1]

def _ppl(logits, tgt, pad=0):
    lp = F.log_softmax(logits, dim=-1)
    lp2d, t = lp.view(-1, lp.size(-1)), tgt.view(-1)
    keep = t != pad
    if keep.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2d[keep], t[keep], reduction="mean")).item()

def _subset(pred, lbl, probs, mask, cls=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": np.nan, "f1": np.nan, "auprc": np.nan}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    hit = accuracy_score(l, p)
    f1  = f1_score(l, p, average="macro")
    try:
        au = average_precision_score(
                label_binarize(l, classes=cls), pr[:, 1:10], average="macro")
    except ValueError:
        au = np.nan
    return {"hit": hit, "f1": f1, "auprc": au}

def _json_safe(obj):
    import numpy as _np, torch as _th
    if isinstance(obj, (_th.Tensor, _th.nn.Parameter)):
        return obj.detach().cpu().tolist()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

# ───────────────────── S3 helpers ────────────────────────────────────
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

# ───────────────────── dataloaders ───────────────────────────────────
def _make_loaders(cfg):
    data = load_json_dataset(cfg["filepath"])
    n = len(data); tr, va = int(.8*n), int(.1*n)
    s = torch.Generator().manual_seed(33)
    tr_ds, va_ds, te_ds = random_split(data, [tr, va, n-tr-va], generator=s)

    tok_ai = tok_tgt = _build_tok()
    out = Path(cfg["model_folder"]); out.mkdir(parents=True, exist_ok=True)
    tok_ai .save(str(out/"tokenizer_ai.json"))
    tok_tgt.save(str(out/"tokenizer_tgt.json"))

    def mk(split):
        return TransformerDataset(split, tok_ai, tok_tgt,
                                  cfg["seq_len_ai"], cfg["seq_len_tgt"],
                                  cfg["num_heads"], cfg["ai_rate"], pad_token=0)
    loader = lambda ds, sh: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=sh)
    return (loader(mk(tr_ds), True),
            loader(mk(va_ds), False),
            loader(mk(te_ds), False),
            tok_tgt)

# ───────────────────── model factory ─────────────────────────────────
def _build_model(cfg):
    return build_transformer(cfg["vocab_size_tgt"], cfg["seq_len_ai"],
                             cfg["d_model"], cfg["N"], cfg["num_heads"],
                             cfg["d_ff"], cfg["dropout"])

# ───────────────────── evaluation (3 subsets) ───────────────────────
def _evaluate(loader, eng, dev, loss_fn, step, pad, tok):
    if not len(loader):
        nan = float("nan"); return nan, nan, {}, {}, {}
    sp = {pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")}

    tloss = tppl = 0.0
    P, L, PR = [], [], []
    m_stop, m_trans = [], []

    eng.eval()
    with torch.no_grad():
        for b in loader:
            x, y = b["aggregate_input"].to(dev), b["label"].to(dev)
            g = eng(x)[:, torch.arange(step-1, gsize := y.size(1), step, device=dev), :]

            y_eval = y.clone(); y_eval[_transition_mask(y)] = pad
            tloss += loss_fn(g, y_eval).item()
            tppl  += _ppl(g, y_eval, pad)

            pr  = F.softmax(g, dim=-1).view(-1, g.size(-1)).cpu().numpy()
            pd  = pr.argmax(1); lb = y.view(-1).cpu().numpy()
            keep = ~np.isin(lb, list(sp))

            P.append(pd[keep]); L.append(lb[keep]); PR.append(pr[keep])
            m_stop .append((y == 9).view(-1).cpu().numpy()[keep])
            m_trans.append(_transition_mask(y).view(-1).cpu().numpy()[keep])

    P, L, PR = map(np.concatenate, (P, L, PR))
    m_stop, m_trans = map(np.concatenate, (m_stop, m_trans))
    main_mask = ~m_trans               # non-transition tokens

    return (tloss/len(loader), tppl/len(loader),
            _subset(P, L, PR, main_mask),
            _subset(P, L, PR, m_stop),
            _subset(P, L, PR, m_trans))

# ───────────────────── training loop ─────────────────────────────────
def train_model(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- run-specific ids / paths ------------------------------
    slots = cfg["seq_len_ai"] // cfg["ai_rate"]
    uid   = (f"ctx{slots}_dmodel{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
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

    # ---------- data & model ------------------------------------------
    tr, va, te, tok = _make_loaders(cfg)
    pad_id = tok.token_to_id("[PAD]")

    model   = _build_model(cfg)
    loss_fn = PairwiseRevenueLoss([0,1,10,1,10,1,10,1,10,0],
                                  cfg["vocab_size_tgt"], pad_id)

    eng, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": cfg["batch_size"],
            "optimizer": {"type": "Lamb",
                          "params": {"lr": cfg["lr"],
                                     "eps": cfg["eps"],
                                     "weight_decay": cfg["weight_decay"]}},
            "zero_optimization": {"stage": 1},
            "fp16": {"enabled": False},
            "zero_allow_untested_optimizer": True})

    # ---------- training ---------------------------------------------
    best, patience = None, 0
    for ep in range(cfg["num_epochs"]):
        eng.train(); running = 0.0
        for b in tqdm(tr, desc=f"Ep {ep:02d}", leave=False):
            x, y = b["aggregate_input"].to(dev), b["label"].to(dev)
            pos  = torch.arange(cfg["ai_rate"]-1, x.size(1),
                                cfg["ai_rate"], device=dev)
            logits = eng(x)[:, pos, :]
            y_tr = y.clone(); y_tr[_transition_mask(y)] = pad_id

            loss = loss_fn(logits, y_tr)
            eng.zero_grad(); eng.backward(loss); eng.step()
            running += loss.item()
        print(f"\nTrain loss {running/len(tr):.4f}")

        v_loss, v_ppl, v_main, v_stop, v_trans = _evaluate(
            va, eng, dev, loss_fn, cfg["ai_rate"], pad_id, tok)
        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag, d in (("main", v_main),
                       ("STOP", v_stop),
                       ("transition", v_trans)):
            print(f"  {tag:<10} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
                  f"AUPRC={d['auprc']:.4f}")

        # ---------- checkpoint on improvement -------------------------
        if best is None or v_loss < best:
            best, patience = v_loss, 0
            torch.save({"epoch": ep,
                        "model_state_dict": eng.module.state_dict()}, ckpt_local)

            meta = _json_safe({
                "best_checkpoint_path": ckpt_local.name,   # file name only
                "val_loss": best, "val_ppl": v_ppl,
                "val_main": v_main, "val_stop": v_stop,
                "val_transition": v_trans})
            json_local.write_text(json.dumps(meta, indent=2))

            if _upload(ckpt_local, bucket, ck_key, s3):
                ckpt_local.unlink(missing_ok=True)
            if _upload(json_local, bucket, js_key, s3):
                json_local.unlink(missing_ok=True)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping."); break

    # ---------- test --------------------------------------------------
    if ckpt_local.exists():                                       # not deleted?
        state = torch.load(ckpt_local, map_location=dev)
        eng.module.load_state_dict(state["model_state_dict"])

    t_loss, t_ppl, t_main, t_stop, t_trans = _evaluate(
        te, eng, dev, loss_fn, cfg["ai_rate"], pad_id, tok)

    print(f"\n** TEST ** Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
    for tag, d in (("main", t_main),
                   ("STOP", t_stop),
                   ("transition", t_trans)):
        print(f"  {tag:<10} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  "
              f"AUPRC={d['auprc']:.4f}")

    return {"uid": uid, "val_loss": best}

# ───────────────────── CLI entry-point ───────────────────────────────
if __name__ == "__main__":
    train_model(get_config())
