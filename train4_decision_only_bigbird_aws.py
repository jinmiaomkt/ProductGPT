import os, warnings, logging, json, numpy as np, boto3, botocore
from pathlib import Path
os.environ.update({
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "DEEPSPEED_LOG_LEVEL" : "error",
    "DS_DISABLE_LOGS"     : "1",
})
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Sampler
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import deepspeed
from pytorch_lamb import Lamb

from model4_bigbird import build_transformer
from dataset4 import TransformerDataset, load_json_dataset
from config4 import get_config

class PairwiseRevenueLoss(nn.Module):
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue = revenue + [0.] * (vocab_size - len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("penalty",
                             -torch.abs(rev[:, None] - rev[None, :]))
        self.ignore = ignore_index

    def forward(self, logits, tgt):
        B, N, V = logits.shape
        probs = F.softmax(logits.view(-1, V), dim=-1)
        tgt = tgt.view(-1)
        keep = tgt != self.ignore
        pen = self.penalty.to(probs)
        return -(probs[keep] * pen[tgt[keep]]).sum(dim=-1).mean()

def _transition_mask(lbl):
    return lbl != F.pad(lbl, (1, 0), value=-1)[:, :-1]

def _ppl(logits, tgt, pad=0):
    lp2, t = F.log_softmax(logits, -1).view(-1, logits.size(-1)), tgt.view(-1)
    mask = t != pad
    if mask.sum() == 0:
        return float("nan")
    return torch.exp(F.nll_loss(lp2[mask], t[mask], reduction="mean")).item()

def _subset(pred, lbl, probs, mask, cls=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": np.nan, "f1": np.nan, "auprc": np.nan}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    hit = accuracy_score(l, p)
    f1 = f1_score(l, p, average="macro")
    try:
        au = average_precision_score(
            label_binarize(l, classes=cls), pr[:, 1:10], average="macro")
    except ValueError:
        au = np.nan
    return {"hit": hit, "f1": f1, "auprc": au}

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

def _s3():
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
        print(f"[S3-ERR] {e}")
        return False

class BucketSampler(Sampler):
    def __init__(self, lengths, batch_size):
        self.batch = batch_size
        order = np.argsort(lengths)
        buckets = [order[i:i + batch_size]
                   for i in range(0, len(order), batch_size)]
        np.random.shuffle(buckets)
        self.flat = [i for b in buckets for i in b]

    def __iter__(self):
        return iter(self.flat)

    def __len__(self):
        return len(self.flat)

def _make_loaders(cfg, tokenizer):
    raw = load_json_dataset(cfg["filepath"])
    n = len(raw); tr, va = int(.8 * n), int(.1 * n)
    g = torch.Generator().manual_seed(33)
    tr_data, va_data, te_data = random_split(raw, [tr, va, n - tr - va], generator=g)

    def wrap(ds):
        return TransformerDataset(ds, tokenizer,
                                  input_key="AggregateInput",
                                  pad_token=0,
                                  ctx_window=cfg["ctx_window"],
                                  ai_rate=cfg["ai_rate"])

    tr_ds, va_ds, te_ds = map(wrap, (tr_data, va_data, te_data))
    lengths = [len(x["aggregate_input"]) for x in tr_ds]
    sampler = BucketSampler(lengths, cfg["batch_size"])

    collate = lambda b: TransformerDataset.collate_fn(
        b, pad_id=0,
        ctx_window=cfg["ctx_window"],
        ai_rate=cfg["ai_rate"])

    # LD = lambda ds, smpl=None: DataLoader(
    #     ds, batch_size=None if smpl else cfg["batch_size"],
    #     sampler=smpl, collate_fn=collate, pin_memory=True)
    
    # ── replace the LD (= DataLoader) helper in _make_loaders ──────────
    LD = lambda ds, smpl=None: DataLoader(
            ds,
            batch_size=cfg["batch_size"],   # ← always use a batch size
            sampler=smpl,                   # BucketSampler gives the *order*
            shuffle=False,                  # no conflict with sampler
            collate_fn=collate,
            pin_memory=True)


    return LD(tr_ds, sampler), LD(va_ds), LD(te_ds)

def _build_model(cfg):
    cfg["ai_rate"] = 1
    return build_transformer(
        vocab_size=cfg["vocab_size_tgt"],
        max_seq_len=cfg["seq_len_ai"],
        d_model=cfg["d_model"],
        n_layers=cfg["N"],
        n_heads=cfg["num_heads"],
        window_size=cfg["window_size"],
        block_size=cfg["block_size"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"]
    )

def _eval(loader, eng, dev, loss_fn, step, pad, tok):
    if len(loader) == 0:
        nan = float("nan")
        e = {"hit": nan, "f1": nan, "auprc": nan}
        return nan, nan, e, e, e, e

    tloss = tppl = 0.0
    P, L, PR = [], [], []
    ms, ma, mt = [], [], []

    special = [pad, tok.token_to_id("[SOS]"), tok.token_to_id("[UNK]")]

    eng.eval()
    with torch.no_grad():
        for b in loader:
            x = b["aggregate_input"].to(dev, non_blocking=True)
            y = b["label"].to(dev)

            # pos = torch.arange(step - 1, x.size(1), step, device=dev)
            # logits = eng(x)[:, pos, :]
            # tgt = y[:, pos].clone()

            logits = eng(x)[:, -1, :]                # (B, V) ← only last position
            tgt    = y.view(-1)                      # (B,)

            tgt_mask = tgt.clone()
            tgt_mask[_transition_mask(y)[:, pos]] = pad
            tloss += loss_fn(logits, tgt_mask).item()
            tppl += _ppl(logits, tgt_mask, pad)

            prob = F.softmax(logits, -1).view(-1, logits.size(-1)).cpu().numpy()
            pred = prob.argmax(1)
            lbl = tgt.view(-1).cpu().numpy()

            keep = ~np.isin(lbl, special)
            P.append(pred[keep]); L.append(lbl[keep]); PR.append(prob[keep])

            flat = lambda m: m.view(-1).cpu().numpy()[keep]
            ms.append(flat(tgt == 9))
            mt.append(flat(_transition_mask(y)[:, pos]))
            prev = F.pad(tgt, (1, 0), value=-1)[:, :-1]
            ma.append(flat(prev == 9))

    P, L, PR = map(np.concatenate, (P, L, PR))
    ms, ma, mt = map(np.concatenate, (ms, ma, mt))
    all_m = np.ones_like(P, dtype=bool)

    return (tloss / len(loader), tppl / len(loader),
            _subset(P, L, PR, all_m),
            _subset(P, L, PR, ms),
            _subset(P, L, PR, ma),
            _subset(P, L, PR, mt))

def train_model(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(
        {**{str(i): i for i in range(1, 10)},
         "[PAD]": 0, "[SOS]": 10, "[UNK]": 12},
        unk_token="[UNK]")
    pad_id = tok.token_to_id("[PAD]")

    slots = cfg["ctx_window"] // cfg["ai_rate"]
    uid = (f"ai_rate{cfg['ai_rate']}_ctx{slots}_d{cfg['d_model']}_ff{cfg['d_ff']}_N{cfg['N']}_"
           f"h{cfg['num_heads']}_lr{cfg['lr']}_wt{cfg['weight']}")
    ckpt = Path(cfg["model_folder"]) / f"DecisionOnly_{uid}.pt"
    meta = ckpt.with_suffix(".json")

    s3 = _s3()
    bucket = cfg["s3_bucket"]
    ck_key = f"DecisionOnly/checkpoints/{ckpt.name}"
    js_key = f"DecisionOnly/metrics/{meta.name}"
    print(f"[INFO] artefacts →  s3://{bucket}/{ck_key}")
    print(f"[INFO] artefacts →  s3://{bucket}/{js_key}")

    tr, va, te = _make_loaders(cfg, tok)
    model = _build_model(cfg)
    loss_fn = PairwiseRevenueLoss(
        [0,1,10,1,10,1,10,1,10,0], cfg["vocab_size_tgt"], pad_id)

    eng, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(),
        config={
            "train_micro_batch_size_per_gpu": cfg["batch_size"],
            "zero_allow_untested_optimizer": True,
            "optimizer": {"type": "Lamb", "params": {
                "lr": cfg["lr"], "weight_decay": cfg["weight_decay"],
                "eps": cfg["eps"]}},
            "zero_optimization": {"stage": 1},
            "fp16": {"enabled": True}})
    # scaler = torch.cuda.amp.GradScaler()

    best = patience = None
    for ep in range(cfg["num_epochs"]):
        eng.train(); running = 0.0
        for b in tqdm(tr, desc=f"Ep {ep:02d}", leave=False):
            x = b["aggregate_input"].to(dev, non_blocking=True)
            y = b["label"].to(dev)
            # pos = torch.arange(cfg["ai_rate"]-1, x.size(1), cfg["ai_rate"], device=dev)

            logits = eng(x)[:, -1, :]                # (B, V)
            tgt    = y.view(-1)                      # (B,)
            # ---------------------------------------------------------

            # adjust downstream shapes once:
            # prob = F.softmax(logits, -1).cpu().numpy()     # (B, V)
            prob = F.softmax(logits, -1).detach().cpu().numpy()   # (B, V)
            # pred = prob.argmax(1)                           # (B,)
            # lbl  = tgt.cpu().numpy()                        # (B,)

            # loss
            loss = loss_fn(logits.unsqueeze(1), tgt.unsqueeze(1))   # keep Pairwise API

            eng.backward(loss)                 # DeepSpeed scales & back-props
            eng.step()                         # optimizer + zero
            # eng.zero_grad(set_to_none=True)
            eng.zero_grad()          # ← no kwargs

            running += loss.item()

            # with torch.cuda.amp.autocast():
            #     logits = eng(x)[:, pos, :]
            #     tgt = y[:, pos].clone()
            #     tgt[_transition_mask(y)[:, pos]] = pad_id
            #     loss = loss_fn(logits, tgt)

            # scaler.scale(loss).backward()
            # scaler.step(eng.optimizer); scaler.update()
            
        print(f"\nTrain loss {running/len(tr):.4f}")

        v_loss, v_ppl, v_all, v_stop, v_after, v_tr = _eval(
            va, eng, dev, loss_fn, cfg["ai_rate"], pad_id, tok)
        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        for tag, d in (("all", v_all), ("cur-STOP", v_stop),
                       ("after-STOP", v_after), ("transition", v_tr)):
            print(f"  {tag:<12} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

        if best is None or v_loss < best:
            best, patience = v_loss, 0
            torch.save({"epoch": ep, "model_state_dict": eng.module.state_dict()}, ckpt)
            meta.write_text(json.dumps(_json_safe({
                "val_loss": best, "val_ppl": v_ppl,
                "val_all": v_all, "val_cur_stop": v_stop,
                "val_after_stop": v_after, "val_transition": v_tr}), indent=2))
            if _upload(ckpt, bucket, ck_key, s3): ckpt.unlink(missing_ok=True)
            if _upload(meta, bucket, js_key, s3): meta.unlink(missing_ok=True)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print("Early stopping."); break

    if ckpt.exists():
        state = torch.load(ckpt, map_location=dev)
        eng.module.load_state_dict(state["model_state_dict"])

    t_loss, t_ppl, t_all, t_st, t_af, t_tr = _eval(
        te, eng, dev, loss_fn, cfg["ai_rate"], pad_id, tok)
    print(f"\n** TEST **  Loss={t_loss:.4f}  PPL={t_ppl:.4f}")
    for tag, d in (("all", t_all), ("cur-STOP", t_st),
                   ("after-STOP", t_af), ("transition", t_tr)):
        print(f"  {tag:<12} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

    return {"uid": uid, "val_loss": best}

if __name__ == "__main__":
    train_model(get_config())
