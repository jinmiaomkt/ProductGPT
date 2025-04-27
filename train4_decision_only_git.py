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
from tokenizers.models import WordLevel
from config4_decision_only_git import get_config, get_weights_file_path, latest_weights_file_path

from google.cloud import storage

def upload_to_gcs(local_path: str, bucket_name: str, destination_blob_name: str):
    """Uploads a local file to GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")

##############################################################################
# Tokenizer-building functions
##############################################################################
def build_tokenizer_src():
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
    for i in range(13, 61):
        fixed_vocab[str(i)] = i
    tokenizer.model = models.WordLevel(vocab=fixed_vocab, unk_token="[UNK]")
    return tokenizer

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

##############################################################################
# FocalLoss
##############################################################################
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=5.0, ignore_index=0):
#         """
#         We'll do the per-element masking by ourselves.
#         """
#         super().__init__()
#         self.gamma = gamma
#         self.ignore_index = ignore_index

#     def forward(self, inputs, targets):
#         """
#         inputs: (B, T, V)
#         targets: (B, T)
#         """
#         B, T, V = inputs.shape

#         # Flatten
#         inputs_2d = inputs.reshape(-1, V)      # (B*T, V)
#         targets_1d = targets.reshape(-1)       # (B*T,)

#         # "none" reduction so we can manually mask
#         ce_loss = F.cross_entropy(inputs_2d, targets_1d, reduction='none')  # shape (B*T,)

#         # Build mask for "ignore_index"
#         mask = (targets_1d != self.ignore_index)  # True where we keep the token

#         # Keep only non-ignored tokens
#         ce_loss = ce_loss[mask]

#         # Standard focal loss formula
#         pt = torch.exp(-ce_loss)   # shape (N_nonignored,)
#         focal = (1 - pt)**self.gamma * ce_loss

#         # Return average over non-ignored
#         if focal.numel() == 0:
#             return torch.tensor(0.0, device=inputs.device)
#         return focal.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=0, class_weights=None):
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

        # Use cross_entropy with 'none' reduction so we can apply focal transform ourselves
        ce_loss = F.cross_entropy(
            inputs_2d,
            targets_1d,
            reduction='none',
            weight=self.class_weights  # <---- the magic: per-class weighting
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

##############################################################################
# Perplexity helper
##############################################################################
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

    # Save them
    # tokenizer_lto.save(os.path.join(config["model_folder"], "tokenizer_lto.json"))
    # tokenizer_src.save(os.path.join(config["model_folder"], "tokenizer_src.json"))
    tokenizer_tgt.save(os.path.join(config["model_folder"], "tokenizer_tgt.json"))
    tokenizer_ai.save(os.path.join(config["model_folder"], "tokenizer_ai.json"))

    train_dataset = TransformerDataset(train_data, tokenizer_ai, tokenizer_tgt,  config['seq_len_ai'], config['seq_len_tgt'], config['num_heads'], config['ai_rate'], pad_token=0)
    val_dataset   = TransformerDataset(val_data, tokenizer_ai, tokenizer_tgt,  config['seq_len_ai'], config['seq_len_tgt'], config['num_heads'], config['ai_rate'], pad_token=0)
    test_dataset  = TransformerDataset(test_data, tokenizer_ai, tokenizer_tgt,  config['seq_len_ai'], config['seq_len_tgt'], config['num_heads'], config['ai_rate'], pad_token=0)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'], shuffle=False)
    return train_loader, val_loader, test_loader

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

##############################################################################
# Training loop
##############################################################################
def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("Using device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = get_model(config).to(device)

    # QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    tokenizer_tgt = build_tokenizer_tgt()
    
    # Focal Loss
    # loss_fn = FocalLoss(gamma=config['gamma'], ignore_index=tokenizer_tgt.token_to_id('[PAD]')).to(device)

    weights = torch.ones(config['vocab_size_ai'])
    weights[9] = config['weight']
    weights = weights.to(device)
    loss_fn = FocalLoss(gamma=config['gamma'], ignore_index=tokenizer_tgt.token_to_id('[PAD]'), class_weights=weights).to(device)

    # DeepSpeed config
    ds_config = {
        "train_micro_batch_size_per_gpu": config['batch_size'],
        "gradient_accumulation_steps": 1,
        "zero_allow_untested_optimizer": True,
        "gradient_clipping": 1.0,
        "use_lr_scheduler": True,
        "optimizer": {
            "type": "Lamb",
            "params": {
                "lr": config['lr'],
                "eps": config['eps'],
                "weight_decay": config['weight_decay']
            }
        },
        "lr_scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": config['min_lr'],
                "warmup_max_lr": config['lr'],
                "warmup_num_steps": config['warmup_steps'],
                "total_num_steps": None,
                "decay_style": "cosine"
            }
        },
        "fp16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": 1
        },
        "wall_clock_breakdown": False
    }

    total_steps = config["num_epochs"] * len(train_loader)
    ds_config["lr_scheduler"]["params"]["total_num_steps"] = total_steps

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    initial_epoch = 0
    global_step   = 0

    if config.get("preload") == "latest":
        latest_ckpt_path = latest_weights_file_path(config)
        if latest_ckpt_path is not None and Path(latest_ckpt_path).exists():
            print(f"[INFO] Loading checkpoint from {latest_ckpt_path} ...")
            checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
            model_engine.load_state_dict(checkpoint['model_state_dict'])
            
            initial_epoch = checkpoint.get('epoch', 0) + 1
            global_step   = checkpoint.get('global_step', 0)

            print(f"[INFO] Successfully loaded. Resuming from epoch={initial_epoch}, global_step={global_step}.")
        else:
            print("[INFO] No previous checkpoint found. Starting fresh.")
    else:
        print("[INFO] Starting from a fresh model initialization (no preload).")

    best_val_loss = None
    best_checkpoint_path = None
    epochs_no_improve = 0

    for epoch in range(initial_epoch, config['num_epochs']):
        model_engine.train()
        torch.cuda.empty_cache()

        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        total_loss = 0.0

        for batch in batch_iterator:
            decoder_input = batch['aggregate_input'].to(device)
            label         = batch['label'].to(device)

            model_engine.zero_grad()
            logits = model_engine(decoder_input)  # (B, seq_len, vocab_size)

            B, T, V = logits.shape
            decision_positions = torch.arange(config['ai_rate'] - 1, T, step=config['ai_rate'], device=logits.device)  # shape: (N,)
            decision_logits = logits[:, decision_positions, :]  # shape: (B, N, V)

            loss = loss_fn(
                decision_logits,  # predict next token
                label
            )

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            global_step += 1

        train_loss = total_loss / len(train_loader)
        current_lr = model_engine.optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}: LR={current_lr:.6f}  TrainLoss={train_loss:.4f}")

        # Evaluate
        val_loss, val_conf_mat, val_ppl, val_hit_rate, val_f1_score, val_auprc = evaluate(val_loader, model_engine, device, loss_fn, stepsize=config['ai_rate'])
        print(f"Epoch {epoch} Val Loss={val_loss:.4f}  \nVal PPL={val_ppl:.4f} \nVal Hit Rate={val_hit_rate:.4f} \nVal F1 Score={val_f1_score:.4f} \nVal Area Under Precision-Recall Curve={val_auprc:.4f}")
        print("Val Confusion Matrix:\n", val_conf_mat)

        # Early stopping or checkpoint
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # best_checkpoint_path = f"best_checkpoint_epoch_{epoch}.pt"
            
            best_checkpoint_path = get_weights_file_path(config, "best")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_engine.state_dict(),
                'optimizer_state_dict': model_engine.optimizer.state_dict(),
                'global_step': global_step
            }, best_checkpoint_path)

            # Upload best checkpoint to GCS
            gcs_bucket_name = config['gcp_bucket']  # <-- your bucket
            destination_blob = f"checkpoints/{Path(best_checkpoint_path).name}"  
            # upload inside 'checkpoints/' folder
            upload_to_gcs(best_checkpoint_path, gcs_bucket_name, destination_blob)

            print(f"  [*] New best val_loss={val_loss:.4f}, saved => {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['patience']:
                print("Early stopping!")
                break

    # Test evaluation
    if best_checkpoint_path:
        print(f"\nBest checkpoint: {best_checkpoint_path}")
        state = torch.load(best_checkpoint_path, weights_only=False)
        model_engine.load_state_dict(state['model_state_dict'])

    test_loss, test_conf_mat, test_ppl, test_hit_rate, test_f1_score, test_auprc = evaluate(test_loader, model_engine, device, loss_fn, stepsize=config['ai_rate'])
    print(f"** Test Loss={test_loss:.4f} \nTest PPL={test_ppl:.4f} \nTest Hit Rate={test_hit_rate:.4f} \nTest F1 Score={test_f1_score:.4f} \nTest Area Under Precision-Recall Curve={test_auprc:.4f}")
    print("Test Confusion Matrix:\n", test_conf_mat)

    json_out_path = Path(best_checkpoint_path).with_suffix(".json")
    metadata = {
        "best_checkpoint_path": best_checkpoint_path,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "val_confusion_matrix": val_conf_mat.tolist() if val_conf_mat is not None else None,
        "val_hit_rate": val_hit_rate,
        "val_f1_score": val_f1_score,
        "val_auprc": val_auprc,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "test_confusion_matrix": test_conf_mat.tolist() if test_conf_mat is not None else None,
        "test_hit_rate": test_hit_rate,
        "test_f1_score": test_f1_score,
        "test_auprc": test_auprc
    }
    with open(json_out_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        "best_checkpoint_path": best_checkpoint_path,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "val_confusion_matrix": val_conf_mat.tolist() if val_conf_mat is not None else None,
        "val_hit_rate": val_hit_rate,
        "val_f1_score": val_f1_score,
        "val_auprc": val_auprc,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "test_confusion_matrix": test_conf_mat.tolist() if test_conf_mat is not None else None,
        "test_hit_rate": test_hit_rate,
        "test_f1_score": test_f1_score,
        "test_auprc": test_auprc
    }

##############################################################################
# The evaluate function, fixed
##############################################################################
def evaluate(dataloader, model_engine, device, loss_fn, stepsize):
    total_loss = 0.0
    total_ppl  = 0.0

    all_preds      = []  # for confusion matrix & F1
    all_labels     = []  # for confusion matrix & F1
    all_probs      = []  # for AUPRC
    valid_labels   = []  # same as all_labels, but used for AUPRC

    model_engine.eval()
    
    if len(dataloader) == 0:
        # Return 4 values so the caller can unpack
        return float('nan'), None, float('nan'), float('nan')

    # For ignoring special tokens in the *labels*
    tokenizer_tgt = build_tokenizer_tgt()
    pad_id = tokenizer_tgt.token_to_id("[PAD]")
    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    unk_id = tokenizer_tgt.token_to_id("[UNK]")
    # eos_id = tokenizer_tgt.token_to_id("[EOS]")
    # special_tokens = {pad_id, sos_id, unk_id, eos_id}
    special_tokens = {pad_id, sos_id, unk_id}

    with torch.no_grad():
        for batch in dataloader:
            dec_inp = batch['aggregate_input'].to(device)
            label   = batch['label'].to(device)
            logits  = model_engine(dec_inp)

            # with torch.no_grad():
            #     class9_logits = logits[..., 9]  # (B, T)
            #     print(f"Avg logit for class 9: {class9_logits.mean().item():.4f}")

            # counts = (label == 9).sum()
            # print("Class 9 count at decision positions:", counts.item())

            # Gather logits at decision positions (e.g., first token of every 15-token block)
            decision_positions = torch.arange(stepsize - 1, logits.size(1), step=stepsize, device=logits.device)
            decision_logits = logits[:, decision_positions, :]  # shape: (B, N, vocab_size)

            # SHIFT for loss
            loss = loss_fn(decision_logits, label)
            total_loss += loss.item()

            # Perplexity
            ppl = calculate_perplexity(decision_logits, label, pad_token=pad_id)
            total_ppl += ppl

            # For metrics: get probabilities and predictions
            # decision_probs shape => (B, N, vocab_size)
            decision_probs = F.softmax(decision_logits, dim=-1)  # per-class probabilities

            # Flatten over (B*N)
            B, N, V = decision_probs.shape
            probs_2d  = decision_probs.view(-1, V).cpu().numpy()  # shape (B*N, V)
            preds_2d  = probs_2d.argmax(axis=-1)                  # argmax for confusion matrix
            labels_1d = label.view(-1).cpu().numpy()              # shape (B*N,)

            # SHIFT for predictions
            # preds  = torch.argmax(decision_logits, dim=-1)  # (B, T-1)
            # labels_2D = label[:, 1:]                             # (B, T-1)

            # Flatten to 1D
            # preds_1D  = preds.cpu().numpy().ravel()
            # labels_1D = label.cpu().numpy().ravel()

            # Filter out special tokens from labels
            valid_mask = ~np.isin(labels_1d, list(special_tokens))
            preds_2d   = preds_2d[valid_mask]
            labels_1d  = labels_1d[valid_mask]
            probs_2d   = probs_2d[valid_mask, :]   # keep the same positions

            # Append
            all_preds.append(preds_2d)
            all_labels.append(labels_1d)
            all_probs.append(probs_2d)
            valid_labels.append(labels_1d)

    # Merge all into single 1D arrays
    all_preds  = np.concatenate(all_preds)   # shape (N_total,)
    all_labels = np.concatenate(all_labels)  # shape (N_total,)
    all_probs  = np.concatenate(all_probs, axis=0)   # shape (N_total, vocab_size)
    valid_labels = np.concatenate(valid_labels)

    avg_loss = total_loss / len(dataloader)
    avg_ppl  = total_ppl  / len(dataloader)

    # Now we can do confusion_matrix and accuracy
    unique_labels = np.unique(all_labels)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    hit_rate = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    # auprc = average_precision_score(all_labels, all_preds, average='macro')

    classes_for_bin = np.arange(1, 10)
    y_true_bin = label_binarize(valid_labels, classes=classes_for_bin)
    probs_for_auprc = all_probs[:, 1:10]  # keep columns 1..9
    auprc = average_precision_score(y_true_bin, probs_for_auprc, average='macro')

    label_mapping = {0: "[PAD]", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
    readable_labels = [label_mapping.get(i, str(i)) for i in unique_labels]
    print(f"Label IDs: {unique_labels}")
    print(f"Label meanings: {readable_labels}")
    print(f"Unique values in predictions: {np.unique(all_preds, return_counts=True)}")
    print(f"Unique values in labels: {np.unique(all_labels, return_counts=True)}")

    return avg_loss, conf_mat, avg_ppl, hit_rate, macro_f1, auprc

##############################################################################
# MAIN
##############################################################################
if __name__ == "__main__":
    config = get_config()
    train_model(config)