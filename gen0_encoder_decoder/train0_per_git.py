import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")

import json
import torch
import re
import math
import torch.nn as nn
import torch.nn.functional as F

import torch.quantization
import numpy as np

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

import deepspeed
from deepspeed.runtime.lr_schedules import WarmupLR

from pytorch_lamb import Lamb
# from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import confusion_matrix, accuracy_score

# Tokenizers & Data
from tokenizers import Tokenizer, pre_tokenizers, trainers, models
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel

# Local modules
from dataset0_per import TransformerDataset, load_json_dataset
from model0_per import build_transformer
from config0git import get_config, get_weights_file_path, latest_weights_file_path

# Logging
import logging
logging.getLogger("deepspeed").setLevel(logging.ERROR)
        
##############################################################################
# Compute Perplexity
##############################################################################
def calculate_perplexity(logits, targets, pad_token=9):
    """
    Computes Perplexity (PPL) for a given batch of logits and target tokens.
    
    :param logits: Tensor of shape (batch_size, seq_len, vocab_size) containing model output logits.
    :param targets: Tensor of shape (batch_size, seq_len) containing ground-truth token indices.
    :param pad_token: Token ID for padding, ignored in loss calculation.
    :return: Perplexity score.
    """
    log_probs = F.log_softmax(logits, dim=-1)  # Convert logits to log probabilities
    batch_size, seq_len, vocab_size = logits.shape
    log_probs = log_probs.view(-1, vocab_size)  # Flatten batch and seq_len
    targets = targets.view(-1)  # Flatten targets
    
    # Ignore padding tokens if specified
    if pad_token is not None:
        mask = targets != pad_token
        log_probs = log_probs[mask]
        targets = targets[mask]

    # Compute Negative Log-Likelihood Loss
    nll_loss = F.nll_loss(log_probs, targets, reduction="mean")

    # Convert to Perplexity
    perplexity = torch.exp(nll_loss)

    return perplexity.item()

##############################################################################
# Focal Loss Implementation
##############################################################################
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean', ignore_index=None):
        """
        Focal Loss for handling class imbalance with an option to ignore specific tokens.

        :param gamma: Focusing parameter (higher gamma reduces loss for easy examples).
        :param reduction: 'sum' (default), 'mean', or 'none'.
        :param ignore_index: Token ID to ignore in loss computation (e.g., [PAD] token).
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index  # Store ignore_index for masking

    def forward(self, inputs, targets):
        """
        Computes Focal Loss while properly ignoring specified token indices.

        :param inputs: Logits (before softmax), shape (batch_size, seq_len, vocab_size).
        :param targets: True token indices, shape (batch_size, seq_len).
        :return: Loss value.
        """
        # Create a mask for valid tokens (i.e., ignore_index tokens will be masked out)
        if self.ignore_index is not None:
            mask = targets != self.ignore_index  # Mask for valid tokens
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)  # No ignored tokens

        # Compute per-token cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='sum')  # Shape: (batch_size, seq_len)

        # Mask out loss for ignored tokens
        ce_loss = ce_loss * mask  # Zero out ignored losses

        # Compute probability of the correct class (p_t)
        pt = torch.exp(-ce_loss)  # Shape: (batch_size, seq_len)

        # Apply Focal Loss formula: (1 - pt) ^ gamma * CE loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Compute final loss based on the specified reduction method
        if self.reduction == "mean":
            # Avoid division by zero in case all tokens are ignored
            return focal_loss.sum() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0, device=inputs.device)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss  # Return per-token loss if `reduction='none'`
    
##############################################################################
# Functions for building tokenizers
##############################################################################
def build_tokenizer_src(data, key, vocab_size=1000):
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
    )
    tokenizer.train_from_iterator(get_all_sentences(data, key), trainer=trainer)
    return tokenizer

def build_tokenizer_tgt():
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fixed_vocab = {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
        "[SOS]": 9,
        "[EOS]": 0,
        "[UNK]": 10,
        "[PAD]": 11  
    }
    tokenizer.model = models.WordLevel(vocab=fixed_vocab, unk_token="[UNK]")
    return tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

##############################################################################
# DataLoader Preparation
##############################################################################
def get_dataloaders(config):
    data = load_json_dataset(config['filepath'])
    
    train_ds_size = int(0.8 * len(data))
    val_ds_size = int(0.1 * len(data))
    test_ds_size = len(data) - train_ds_size - val_ds_size
    
    # Set a random seed for reproducibility
    seed_value = 33  
    # You can change this to any integer for different splits
    torch.manual_seed(seed_value)

    # Perform the split with deterministic randomness
    train_data, val_data, test_data = random_split(
        data, [train_ds_size, val_ds_size, test_ds_size], generator=torch.Generator().manual_seed(seed_value)
    )

    tokenizer_src = build_tokenizer_src(train_data, "Item", vocab_size=config['vocab_size_src'])
    tokenizer_tgt = build_tokenizer_tgt()

    train_dataset = TransformerDataset(
        train_data, tokenizer_src, tokenizer_tgt,
        config['seq_len_src'], config['seq_len_tgt'],
        config['num_heads'], config['source_rate']
    )
    val_dataset = TransformerDataset(
        val_data, tokenizer_src, tokenizer_tgt,
        config['seq_len_src'], config['seq_len_tgt'],
        config['num_heads'], config['source_rate']
    )
    test_dataset = TransformerDataset(
        test_data, tokenizer_src, tokenizer_tgt,
        config['seq_len_src'], config['seq_len_tgt'],
        config['num_heads'], config['source_rate']
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=False)
    val_dataloader   = DataLoader(val_dataset,   batch_size=config['batch_size'], num_workers=4, shuffle=False)
    test_dataloader  = DataLoader(test_dataset,  batch_size=config['batch_size'], num_workers=4, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt

##############################################################################
# Model
##############################################################################
def get_model(config, vocab_src_len, vocab_tgt_len):
    return build_transformer(
        vocab_src_len, vocab_tgt_len,
        config["seq_len_src"], config['seq_len_tgt'],
        d_model=config['d_model'],
        N=config['N'],
        h=config['num_heads'],
        dropout=config['dropout'],
        kernel_type=config['kernel_type'],
        d_ff=config['d_ff']
    )

##############################################################################
# Training Loop
##############################################################################
def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
    device = torch.device(device)
    print("Using device:", device)

    train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataloaders(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # QAT Setup
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)

    # DeepSpeed config
    ds_config = {
        "train_micro_batch_size_per_gpu": config['batch_size'],
        "gradient_accumulation_steps": 1,
        "zero_allow_untested_optimizer": True,
        "gradient_clipping": 1.0,
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
                "total_num_steps": config['num_epochs'],    
                "decay_style": "cosine"      
            }
        },        
        "fp16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": 1
        },
        # "steps_per_print": 200,
        "wall_clock_breakdown": False
    }

    # Dynamically set total_num_steps
    total_num_steps = config["num_epochs"] * len(train_dataloader)
    ds_config["lr_scheduler"]["params"]["total_num_steps"] = total_num_steps
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Optional preload
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = (
        latest_weights_file_path(config) if preload == 'latest'
        else get_weights_file_path(config, preload) if preload
        else None
    )
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=False)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')


   # -------------------------
    # **Learning Rate Scheduler**
    # -------------------------
    # warmup_steps = config.get('warmup_steps', 500)
    # total_steps = config.get('num_epochs') * len(train_dataloader)
    # peak_lr = config['lr']
    # min_lr = config.get('min_lr', 1e-6)

    # scheduler = WarmupCosineDecayLR(optimizer, warmup_steps, total_steps, peak_lr, min_lr)

    loss_fn = FocalLoss(gamma=config['gamma'], reduction='mean',ignore_index=tokenizer_tgt.token_to_id('[PAD]')).to(device)

    # -------------------------
    # Validation function with Focal Loss
    # -------------------------
    def cal_loss_loader(dataloader, model_engine, device, loss_fn, tokenizer_tgt, num_batches=None):
        model_engine.eval()
        total_loss = 0
        total_perplexity = 0
        all_preds = []
        all_labels = []

        if len(dataloader) == 0:
            return float('nan'), float('nan'), None

        if num_batches is None:
            num_batches = len(dataloader)

        pad_id = tokenizer_tgt.token_to_id("[PAD]")
        sos_id = tokenizer_tgt.token_to_id("[SOS]")
        # eos_id = tokenizer_tgt.token_to_id("[EOS]")
        unk_id = tokenizer_tgt.token_to_id("[UNK]")
        special_tokens = {pad_id, sos_id, unk_id}

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    print(f"Break at batch {i}")
                    break

                source_input = batch['source_input'].to(device)
                target_input = batch['target_input'].to(device)
                label = batch['label'].to(device)

                with torch.cuda.amp.autocast(enabled=False):  
                    encoder_output = model_engine.module.encode(source_input)
                    decoder_output = model_engine.module.decode(encoder_output, target_input)
                    proj_output = model_engine.module.project(decoder_output)

                    loss = loss_fn(proj_output.view(-1, proj_output.shape[-1]),
                               label.view(-1))
                total_loss += loss.item()

                # Compute Perplexity
                perplexity = calculate_perplexity(proj_output, label, pad_id)
                total_perplexity += perplexity

                preds = torch.argmax(proj_output, dim=-1).cpu().numpy().flatten()
                labels = label.cpu().numpy().flatten()

                valid_mask = ~np.isin(labels, list(special_tokens))
                preds = preds[valid_mask]
                labels = labels[valid_mask]

                all_preds.append(preds)
                all_labels.append(labels)

        avg_loss = total_loss / len(dataloader)
        avg_perplexity = total_perplexity / num_batches

        if not all_preds:
            return avg_loss, float('nan'), None, avg_perplexity

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # print("Unique predictions (LTO=4) B:", np.unique(all_preds))

        # def count_classes(dataset):
        #   all_labels = []
        #   for item in dataset:
        #       all_labels.extend(item['label'])  # Assuming 'label' is a list of tokens
        #   all_labels = np.array(all_labels)
          
        #   unique, counts = np.unique(all_labels, return_counts=True)
        #   return dict(zip(unique, counts))

        # print("Validation Class distribution (LTO=4):", count_classes(dataloader))

        unique_labels = np.unique(all_labels)  

        # hit_rate = accuracy_score(all_labels, all_preds)
        conf_mat = confusion_matrix(all_labels, all_preds, labels = unique_labels)
        return avg_loss, conf_mat, avg_perplexity

    # Hyperparams for plateau + early stop
    patience = config.get('patience', 2)
    # lr_reduce_factor = config.get('lr_reduce_factor', 0.5)
    # min_lr = config.get('min_lr', 1e-6)

    best_val_loss = None
    best_val_conf = None
    best_val_ppl  = None
    best_checkpoint_path = None
    epochs_no_improve = 0
    patience = config.get('patience', 2)

    # If you want early stop after hitting min_lr and no improvement, set:
    # early_stop_after_min_lr_patience = True

    unique_id = config['model_basename'].rstrip("_")

    for epoch in range(initial_epoch, config['num_epochs']):
        model_engine.train()
        torch.cuda.empty_cache()

        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:
            source_input = batch['source_input'].to(device)
            target_input = batch['target_input'].to(device)
            label = batch['label'].to(device)

            model_engine.zero_grad()

            with torch.cuda.amp.autocast(enabled=False):
                encoder_output = model_engine.module.encode(source_input)
                decoder_output = model_engine.module.decode(encoder_output, target_input)
                proj_output = model_engine.module.project(decoder_output)

                loss = loss_fn(proj_output.view(-1, proj_output.shape[-1]), label.view(-1))

            model_engine.backward(loss)
            model_engine.step()
            global_step += 1

        # Print learning rate for tracking
        current_lr = model_engine.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} - Current LR: {current_lr:.6f}")

        train_loss, train_conf_mat, train_ppl = cal_loss_loader(train_dataloader, model_engine, device, loss_fn, tokenizer_tgt)
        print(f"\n[Epoch {epoch:02d}] Train Loss={train_loss:.3f}, Train Perplexity={train_ppl:.3f}")
        
        if train_conf_mat is not None:
            print("Train Confusion Matrix:\n", train_conf_mat)

        val_loss, val_conf_mat, val_ppl = cal_loss_loader(val_dataloader, model_engine, device, loss_fn, tokenizer_tgt)
        print(f"\n[Epoch {epoch:02d}] Val Loss={val_loss:.3f}, Val Perplexity={val_ppl:.3f}")
        
        if val_conf_mat is not None:
            print("Val Confusion Matrix:\n", val_conf_mat)

        # Check improvement
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_val_conf_mat = val_conf_mat
            epochs_no_improve = 0

            best_checkpoint_path = get_weights_file_path(config, f"{unique_id}_best")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_engine.state_dict(),
                'optimizer_state_dict': model_engine.optimizer.state_dict(),
                'global_step': global_step,
                'focal_loss': best_val_loss,
                'confusion_matrix': best_val_conf_mat.tolist() if val_conf_mat is not None else None
            }, best_checkpoint_path)

            print(f"New best focal loss={best_val_loss:.4f}. Saved checkpoint to {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

            # If we exceed patience, reduce LR or early stop
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    if best_checkpoint_path:
        print(f"\nTraining complete. Best checkpoint was {best_checkpoint_path} with focal loss = {best_val_loss:.4f}")
        final_metrics = {
            "val_loss": best_val_loss,
            "val_ppl": best_val_ppl,
            "confusion_matrix": best_val_conf_mat.tolist() if best_val_conf_mat is not None else None
        }
        return final_metrics
    else:
        print("\nTraining complete, but no improvements were ever recorded.")
        return {
            "val_loss": float('inf'),
            "val_ppl": None,
            "confusion_matrix": None
        }

if __name__ == "__main__":
    config = get_config()
    train_model(config)