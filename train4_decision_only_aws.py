import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide most TF log messages

import warnings
warnings.filterwarnings("ignore")

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

from tqdm import tqdm
from pathlib import Path

# Deepspeed & LAMB
import deepspeed
from pytorch_lamb import Lamb
import logging
logging.getLogger("deepspeed").setLevel(logging.ERROR)

from model4_decoderonly import build_transformer
from dataset4_decision_only import TransformerDataset, load_json_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from config4_decision_only_git import get_config, get_weights_file_path, latest_weights_file_path

# from google.cloud import storage

# def upload_to_gcs(local_path: str, bucket_name: str, destination_blob_name: str):
#     """Uploads a local file to GCS bucket."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(local_path)
#     print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")

##############################################################################
# Tokenizer-building functions
##############################################################################
def build_tokenizer_tgt():
    """
    Build a 'target' tokenizer for decisions with a fixed vocab.
    e.g. 0..8, plus [SOS],[EOS],[UNK],[PAD], etc.
    """
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fixed_vocab = {
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, 
        "[PAD]": 0,  
        "[SOS]": 10,
        # "[EOS]": 11,
        "[UNK]": 12
    }
    tokenizer.model = models.WordLevel(vocab=fixed_vocab, unk_token="[UNK]")
    return tokenizer
class PairwiseRevenueLoss(nn.Module):
    """L = –E[ |R_true – R_pred| ]  (minimise –L ⇒ maximise revenue)"""
    def __init__(self, revenue, vocab_size, ignore_index=0):
        super().__init__()
        if len(revenue) < vocab_size:
            revenue = revenue + [0.] * (vocab_size - len(revenue))
        rev = torch.tensor(revenue, dtype=torch.float32)
        self.register_buffer("penalty",
                             -torch.abs(rev[:, None] - rev[None, :]))  # V×V
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits.reshape(-1, logits.size(-1)), dim=-1)
        tgt   = targets.reshape(-1)
        m     = tgt != self.ignore_index
        if m.sum() == 0:
            return logits.new_tensor(0.0)
        exp_gap = (probs[m] * self.penalty[tgt[m]]).sum(dim=-1)
        return (-exp_gap).mean()

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, ignore_index=0, class_weights=None):
#         """
#         Args:
#             gamma (float): Focal loss exponent, default=2.
#             ignore_index (int): Token ID to ignore in the loss.
#             class_weights (Tensor): 1D tensor of shape [num_classes],
#                                     e.g. to upweight rare classes.
#         """
#         super().__init__()
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         # Register the weights as a buffer so they move to GPU with the model.
#         if class_weights is not None:
#             self.register_buffer('class_weights', class_weights)
#         else:
#             self.class_weights = None

#     def forward(self, inputs, targets):
#         """
#         inputs: (B, T, V) => raw logits
#         targets: (B, T)   => integer class IDs
#         """
#         B, T, V = inputs.shape

#         # Flatten to 1D
#         inputs_2d = inputs.reshape(-1, V)         # (B*T, V)
#         targets_1d = targets.reshape(-1)          # (B*T,)

#         # Use cross_entropy with 'none' reduction so we can apply focal transform ourselves
#         ce_loss = F.cross_entropy(
#             inputs_2d,
#             targets_1d,
#             reduction='none',
#             weight=self.class_weights  # <---- the magic: per-class weighting
#         )

#         # Mask out tokens == ignore_index
#         valid_mask = (targets_1d != self.ignore_index)
#         ce_loss = ce_loss[valid_mask]

#         # Focal transform
#         pt = torch.exp(-ce_loss)
#         focal = (1 - pt) ** self.gamma * ce_loss

#         # If everything got masked, return 0
#         if focal.numel() == 0:
#             return torch.tensor(0.0, device=inputs.device)

#         return focal.mean()

def transition_mask(lbl: torch.Tensor):
    prev = F.pad(lbl, (1,0), value=-1)[:, :-1]
    return lbl != prev

def calculate_perplexity(logits, targets, pad_token=0):
    """
    logits: (B, T, vocab_size)
    targets: (B, T)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    B, T, V = logits.shape

    log_probs_2d = log_probs.reshape(-1, V)
    targets_1d   = targets.reshape(-1)

    # mask out pad
    mask = (targets_1d != pad_token)
    log_probs_2d = log_probs_2d[mask]
    targets_1d   = targets_1d[mask]

    nll = F.nll_loss(log_probs_2d, targets_1d, reduction='mean')
    return torch.exp(nll).item()

def _pp_subset(tag, d):
    print(f"  {tag:<11} Hit={d['hit']:.4f}  F1={d['f1']:.4f}  AUPRC={d['auprc']:.4f}")

##############################################################################
# get_dataloaders
##############################################################################
def get_dataloaders(config):
    data = load_json_dataset(config['filepath'])
    
    train_size = int(0.8 * len(data))
    val_size   = int(0.1 * len(data))
    test_size  = len(data) - train_size - val_size

    seed_value = 33
    torch.manual_seed(seed_value)

    train_data, val_data, test_data = random_split(
        data, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed_value)
    )

    # Build tokenizers
    tokenizer_tgt = build_tokenizer_tgt()  
    tokenizer_ai = build_tokenizer_tgt()

    pad_id_src = tokenizer_ai.token_to_id("[PAD]")
    print("Product tokenizer's [PAD] ID:", pad_id_src)

    tokenizer_tgt.save(os.path.join(config["model_folder"], "tokenizer_tgt.json"))
    tokenizer_ai.save(os.path.join(config["model_folder"], "tokenizer_ai.json"))

    train_dataset = TransformerDataset(train_data, tokenizer_ai, tokenizer_tgt,  config['seq_len_ai'], config['seq_len_tgt'], config['num_heads'], config['ai_rate'], pad_token=0)
    val_dataset   = TransformerDataset(val_data, tokenizer_ai, tokenizer_tgt,  config['seq_len_ai'], config['seq_len_tgt'], config['num_heads'], config['ai_rate'], pad_token=0)
    test_dataset  = TransformerDataset(test_data, tokenizer_ai, tokenizer_tgt,  config['seq_len_ai'], config['seq_len_tgt'], config['num_heads'], config['ai_rate'], pad_token=0)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader, test_loader, tokenizer_tgt

##############################################################################
# build_model
##############################################################################
def get_model(config):
    model = build_transformer(
        vocab_size   = config['vocab_size_tgt'],
        d_model      = config['d_model'],
        n_layers     = config['N'],
        n_heads      = config['num_heads'],
        d_ff         = config['d_ff'],
        dropout      = config['dropout'],
        max_seq_len  = config['seq_len_ai']
    )
    return model

def _subset_metrics(pred, lbl, probs, mask, cls=np.arange(1, 10)):
    if mask.sum() == 0:
        return {"hit": float("nan"), "f1": float("nan"),
                "auprc": float("nan"), "conf": None}
    p, l, pr = pred[mask], lbl[mask], probs[mask]
    conf = confusion_matrix(l, p, labels=np.unique(l))
    hit  = accuracy_score(l, p)
    f1   = f1_score(l, p, average='macro')
    try:
        y_true = label_binarize(l, classes=cls)
        auprc  = average_precision_score(y_true, pr[:,1:10], average='macro')
    except ValueError:
        auprc = float("nan")
    return {"hit": hit, "f1": f1, "auprc": auprc, "conf": conf}

def evaluate(loader, engine, device, loss_fn, step, pad_id, tok):
    if len(loader) == 0:
        nan=float('nan'); return nan,nan,{}, {}, {}, {}
    special = {pad_id, tok.token_to_id('[SOS]'), tok.token_to_id('[UNK]')}
    tloss = tppl = 0.0
    P,L,PR = [],[],[]
    m_stop,m_prev,m_tr = [],[],[]

    engine.eval()
    with torch.no_grad():
        for b in loader:
            x,y = b['aggregate_input'].to(device), b['label'].to(device)
            g   = engine(x)
            pos = torch.arange(step-1, g.size(1), step, device=device)
            g   = g[:,pos,:]
            y_eval = y.clone(); y_eval[transition_mask(y)] = pad_id
            tloss += loss_fn(g, y_eval).item()
            tppl  += calculate_perplexity(g, y_eval, pad_id)

            probs = F.softmax(g,dim=-1).view(-1,g.size(-1)).cpu().numpy()
            pred  = probs.argmax(1)
            lbl   = y.view(-1).cpu().numpy()
            valid = ~np.isin(lbl, list(special))
            P.append(pred[valid]); L.append(lbl[valid]); PR.append(probs[valid])
            m_stop.append((y==9).view(-1).cpu().numpy()[valid])
            m_prev.append((F.pad(y,(1,0),value=-1)[:,:-1]==9).view(-1).cpu().numpy()[valid])
            m_tr  .append(transition_mask(y).view(-1).cpu().numpy()[valid])

    P,L,PR = np.concatenate(P),np.concatenate(L),np.concatenate(PR)
    m_stop,m_prev,m_tr = map(np.concatenate,(m_stop,m_prev,m_tr))
    main_mask = ~m_tr
    return (tloss/len(loader), tppl/len(loader),
            _subset_metrics(P,L,PR,main_mask),
            _subset_metrics(P,L,PR,m_stop),
            _subset_metrics(P,L,PR,m_prev),
            _subset_metrics(P,L,PR,m_tr))

# ─────────────────────────── 7 ─ train ────────────────────────────────────
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_dl, va_dl, te_dl, tok = get_dataloaders(cfg)
    pad = tok.token_to_id("[PAD]")

    model = get_model(cfg).to(device)
    loss_fn = PairwiseRevenueLoss([0,1,10,1,10,1,10,1,10,0],
                                  cfg['vocab_size_tgt'], pad).to(device)

    steps_total = len(tr_dl)*cfg['num_epochs']
    ds_cfg = {"train_micro_batch_size_per_gpu": cfg['batch_size'],
              "zero_allow_untested_optimizer": True,
              "optimizer": {"type": "Lamb",
                            "params": {"lr": cfg['lr'],
                                       "eps": cfg['eps'],
                                       "weight_decay": cfg['weight_decay']}},
              "lr_scheduler": {"type":"WarmupDecayLR",
                               "params":{"warmup_min_lr":cfg['min_lr'],
                                         "warmup_max_lr":cfg['lr'],
                                         "warmup_num_steps":cfg['warmup_steps'],
                                         "total_num_steps":steps_total,
                                         "decay_style":"cosine"}},
              "fp16":{"enabled":False},
              "zero_optimization":{"stage":1}}
    engine,_,_,_ = deepspeed.initialize(model=model,
                                        model_parameters=model.parameters(),
                                        config=ds_cfg)

    best,pat,ckpt=None,0,None
    for ep in range(cfg['num_epochs']):
        # --- train --------------------------------------------------------
        engine.train(); run=0.0
        for b in tqdm(tr_dl, desc=f"Ep {ep:02d}"):
            x,y=b['aggregate_input'].to(device),b['label'].to(device)
            pos = torch.arange(cfg['ai_rate']-1,x.size(1),cfg['ai_rate'],device=device)
            g   = engine(x)[:,pos,:]
            y_tr=y.clone(); y_tr[transition_mask(y)]=pad
            loss=loss_fn(g,y_tr)
            engine.zero_grad(); engine.backward(loss); engine.step()
            run+=loss.item()
        print(f"\nTrain loss {run/len(tr_dl):.4f}")

        # --- validate -----------------------------------------------------
        v_loss,v_ppl,v_main,v_stop,v_after,v_tr = evaluate(
            va_dl,engine,device,loss_fn,cfg['ai_rate'],pad,tok)
        print(f"Epoch {ep:02d}  ValLoss={v_loss:.4f}  PPL={v_ppl:.4f}")
        _pp_subset("main",v_main); _pp_subset("STOP cur",v_stop)
        _pp_subset("afterSTOP",v_after); _pp_subset("transition",v_tr)

        if best is None or v_loss<best:
            best=v_loss; pat=0
            ckpt=get_weights_file_path(cfg,'best')
            engine.save_checkpoint(str(Path(ckpt).parent),tag='best')
            print("  [*] checkpoint saved")
        else:
            pat+=1
            if pat>=cfg['patience']:
                print("Early stop."); break

    # --- test -------------------------------------------------------------
    engine.load_checkpoint(str(Path(ckpt).parent),tag='best')
    t_loss,t_ppl,t_main,t_stop,t_after,t_tr = evaluate(
        te_dl,engine,device,loss_fn,cfg['ai_rate'],pad,tok)
    print("\n** TEST ** Loss={:.4f}  PPL={:.4f}".format(t_loss,t_ppl))
    _pp_subset("main",t_main); _pp_subset("STOP cur",t_stop)
    _pp_subset("afterSTOP",t_after); _pp_subset("transition",t_tr)

    meta = {
        "best_checkpoint_path": ckpt,
        # val
        "val_loss": best,
        "val_ppl":  v_ppl,
        "val_main": v_main,
        "val_stop_cur": v_stop,
        "val_after_stop": v_after,
        "val_transition": v_tr,
        # test
        "test_loss": t_loss,
        "test_ppl":  t_ppl,
        "test_main": t_main,
        "test_stop_cur": t_stop,
        "test_after_stop": t_after,
        "test_transition": t_tr
    }
    with open(Path(ckpt).with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta

# ─────────────────────────── 8 ─ main ─────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config()
    res = train(cfg)
    print("\nSaved →", res["best_checkpoint_path"])