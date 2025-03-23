import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide most TF log messages

import warnings
warnings.filterwarnings("ignore")

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Huggingface Tokenizers
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import Whitespace

# Torch utilities
from torch.utils.data import Dataset, DataLoader, random_split

# Metrics
from sklearn.metrics import confusion_matrix

# DeepSpeed & LAMB optimizer
import deepspeed
from deepspeed.runtime.lr_schedules import WarmupLR
from pytorch_lamb import Lamb

# Local modules
from dataset4_per import TransformerDataset, load_json_dataset
from model4_per import build_transformer
from config4git import get_config, get_weights_file_path, latest_weights_file_path

import logging
logging.getLogger("deepspeed").setLevel(logging.ERROR)


##############################################################################
# Compute Perplexity
##############################################################################
def calculate_perplexity(logits, targets, pad_token=9):
    """
    Computes Perplexity (PPL) for a given batch of logits and target tokens.
    Ignores the pad_token in the calculation.
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

    # Negative Log-Likelihood
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
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Computes Focal Loss while ignoring specified token indices.
        inputs: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        """
        # Create a mask for valid tokens
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)

        # Cross-entropy loss (mean across all tokens)
        ce_loss = F.cross_entropy(inputs, targets, reduction='mean')

        # Mask out ignored tokens from the loss
        ce_loss = ce_loss * mask

        # Probability of correct class
        pt = torch.exp(-ce_loss)

        # Focal loss formula
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.sum() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0, device=inputs.device)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss  # 'none'


##############################################################################
# Tokenizer Builders
##############################################################################
def build_tokenizer_src(data, key, vocab_size=1000):
    """
    Build a WordLevel tokenizer that splits only on whitespace.
    data: list of items
    key: field name (e.g., "Item") containing the text or list
    """
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[SOS]", "[UNK]"]
    )
    tokenizer.train_from_iterator(get_all_sentences(data, key), trainer=trainer)
    return tokenizer

def build_tokenizer_tgt():
    """
    Build a fixed-vocab WordLevel tokenizer for the target side, including [UNK], [PAD], etc.
    """
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    fixed_vocab = {
        "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "[SOS]": 7,
        "[EOS]": 0,      # Caution: [EOS] shares ID=0 with "0" unless intentionally used
        "[UNK]": 8,
        "[PAD]": 9
    }
    tokenizer.model = models.WordLevel(vocab=fixed_vocab, unk_token="[UNK]")
    return tokenizer


def get_all_sentences(data, key):
    """
    Generator to feed lines of text/numbers to the tokenizer.
    Converts integers or list-of-integers to strings.
    """
    for row in data:
        val = row[key]
        if isinstance(val, int):
            yield str(val)
        elif isinstance(val, list):
            yield " ".join(str(x) for x in val)
        elif isinstance(val, str):
            yield val
        else:
            yield str(val)


##############################################################################
# DataLoader Preparation
##############################################################################
def get_dataloaders(config):
    data = load_json_dataset(config['filepath'])
    
    # Split data into train/val/test
    train_ds_size = int(0.8 * len(data))
    val_ds_size = int(0.1 * len(data))
    test_ds_size = len(data) - train_ds_size - val_ds_size

    # Seed for reproducibility
    seed_value = 33
    torch.manual_seed(seed_value)

    train_data, val_data, test_data = random_split(
        data, [train_ds_size, val_ds_size, test_ds_size],
        generator=torch.Generator().manual_seed(seed_value)
    )

    # Build tokenizers from the training set
    tokenizer_src = build_tokenizer_src(train_data, "Item", vocab_size=config['vocab_size_src'])
    tokenizer_tgt = build_tokenizer_tgt()
    tokenizer_lto = build_tokenizer_src(train_data, "Item", vocab_size=config['vocab_size_src'])  # example usage

    # Create datasets
    train_dataset = TransformerDataset(
        train_data,
        tokenizer_src,
        tokenizer_tgt,
        tokenizer_lto,
        config['seq_len_src'],
        config['seq_len_tgt'],
        config['seq_len_lto'],
        config['num_heads'],
        config['source_rate'],
        config['lto_rate']
    )
    val_dataset = TransformerDataset(
        val_data,
        tokenizer_src,
        tokenizer_tgt,
        tokenizer_lto,
        config['seq_len_src'],
        config['seq_len_tgt'],
        config['seq_len_lto'],
        config['num_heads'],
        config['source_rate'],
        config['lto_rate']
    )
    test_dataset = TransformerDataset(
        test_data,
        tokenizer_src,
        tokenizer_tgt,
        tokenizer_lto,
        config['seq_len_src'],
        config['seq_len_tgt'],
        config['seq_len_lto'],
        config['num_heads'],
        config['source_rate'],
        config['lto_rate']
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=4, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], num_workers=4, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'], num_workers=4, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer_src, tokenizer_tgt, tokenizer_lto


##############################################################################
# Build the Model
##############################################################################
def get_model(config, vocab_src_len, vocab_tgt_len, vocab_lto_len):
    """
    Create the Transformer model with QAT config attached.
    """
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        vocab_lto_len,
        config["seq_len_src"],
        config['seq_len_tgt'],
        config['seq_len_lto'],
        d_model=config['d_model'],
        N=config['N'],
        h=config['num_heads'],
        dropout=config['dropout'],
        kernel_type=config['kernel_type'],
        d_ff=config['d_ff']
    )
    return model


##############################################################################
# Validation / Evaluation Function
##############################################################################
def cal_loss_loader(dataloader, model_engine, device, loss_fn, tokenizer_tgt, num_batches=None):
    """
    Compute average loss, confusion matrix, and perplexity on a given DataLoader.
    Mask out special tokens [PAD], [SOS], [UNK].
    """
    model_engine.eval()
    total_loss = 0
    total_perplexity = 0
    all_preds = []
    all_labels = []

    if len(dataloader) == 0:
        return float('nan'), None, float('nan')

    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    # Identify special tokens
    pad_id = tokenizer_tgt.token_to_id("[PAD]")
    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    unk_id = tokenizer_tgt.token_to_id("[UNK]")
    special_tokens = {pad_id, sos_id, unk_id}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            source_input = batch['source_input'].to(device)
            target_input = batch['target_input'].to(device)
            lto_input    = batch['lto_input'].to(device)
            label        = batch['label'].to(device)

            # Forward pass
            encoder_output = model_engine.module.encode(source_input, lto_input)
            decoder_output = model_engine.module.decode(encoder_output, target_input)
            proj_output    = model_engine.module.project(decoder_output)

            # Loss
            loss = loss_fn(proj_output.view(-1, proj_output.shape[-1]), label.view(-1))
            total_loss += loss.item()

            # Perplexity
            perplexity = calculate_perplexity(proj_output, label, pad_id)
            total_perplexity += perplexity

            # Predictions
            preds = torch.argmax(proj_output, dim=-1).cpu().numpy().flatten()
            labels = label.cpu().numpy().flatten()

            # Filter out special tokens
            valid_mask = ~np.isin(labels, list(special_tokens))
            preds  = preds[valid_mask]
            labels = labels[valid_mask]

            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = total_loss / num_batches
    avg_perplexity = total_perplexity / num_batches

    if not all_preds:
        # No valid tokens found
        return avg_loss, None, avg_perplexity

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    unique_labels = np.unique(all_labels)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=unique_labels)

    return avg_loss, conf_mat, avg_perplexity


##############################################################################
# Training Loop
##############################################################################
def train_model(config):
    """
    Train the Transformer model with DeepSpeed, QAT, and Focal Loss.
    Returns a dictionary of final (best) metrics for external logging.
    """

    # Setup device
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if hasattr(torch, 'has_mps') and torch.has_mps else "cpu"
    device = torch.device(device)
    print("Using device:", device)

    # Get data
    train_loader, val_loader, test_loader, tokenizer_src, tokenizer_tgt, tokenizer_lto = get_dataloaders(config)

    # Build model
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        tokenizer_lto.get_vocab_size()
    ).to(device)

    # Setup QAT
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

    total_num_steps = config["num_epochs"] * len(train_loader)
    ds_config["lr_scheduler"]["params"]["total_num_steps"] = total_num_steps

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Preload existing checkpoint if specified
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = (
        latest_weights_file_path(config) if preload == 'latest'
        else get_weights_file_path(config, preload) if preload
        else None
    )

    if model_filename and os.path.exists(model_filename):
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, weights_only=False)
        model.load_state_dict(state['model_state_dict'], strict=False)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    # Define loss function (Focal Loss)
    loss_fn = FocalLoss(
        gamma=config['gamma'],
        reduction='mean',
        ignore_index=tokenizer_tgt.token_to_id('[PAD]')
    ).to(device)

    # For early stopping
    patience = config.get('patience', 2)
    best_val_loss = None
    best_val_conf = None
    best_val_ppl = None
    best_checkpoint_path = None
    epochs_no_improve = 0

    # ----------------
    # Training
    # ----------------
    for epoch in range(initial_epoch, config['num_epochs']):
        model_engine.train()
        torch.cuda.empty_cache()

        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}")

        for batch in batch_iterator:
            source_input = batch['source_input'].to(device)
            target_input = batch['target_input'].to(device)
            lto_input    = batch['lto_input'].to(device)
            label        = batch['label'].to(device)

            model_engine.zero_grad()

            # Forward pass
            encoder_output = model_engine.module.encode(source_input, lto_input)
            decoder_output = model_engine.module.decode(encoder_output, target_input)
            proj_output    = model_engine.module.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, proj_output.shape[-1]), label.view(-1))

            # Backprop + update
            model_engine.backward(loss)
            model_engine.step()
            global_step += 1

        current_lr = model_engine.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d} - LR: {current_lr:.6f}")

        # Evaluate on Training set
        train_loss, train_conf_mat, train_ppl = cal_loss_loader(train_loader, model_engine, device, loss_fn, tokenizer_tgt)
        print(f"[Epoch {epoch:02d}] Train Loss={train_loss:.3f}, Train PPL={train_ppl:.3f}")
        if train_conf_mat is not None:
            print("Train Confusion Matrix:\n", train_conf_mat)

        # Evaluate on Validation set
        val_loss, val_conf_mat, val_ppl = cal_loss_loader(val_loader, model_engine, device, loss_fn, tokenizer_tgt)
        print(f"[Epoch {epoch:02d}] Val   Loss={val_loss:.3f}, Val   PPL={val_ppl:.3f}")
        if val_conf_mat is not None:
            print("Val Confusion Matrix:\n", val_conf_mat)

        # Track best model
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_conf = val_conf_mat
            best_val_ppl = val_ppl
            epochs_no_improve = 0

            unique_id = config['model_basename'].rstrip("_")
            best_checkpoint_path = get_weights_file_path(config, f"{unique_id}_best")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_engine.state_dict(),
                'optimizer_state_dict': model_engine.optimizer.state_dict(),
                'global_step': global_step,
                'val_loss': val_loss,
                'val_confusion_matrix': val_conf_mat.tolist() if val_conf_mat is not None else None
            }, best_checkpoint_path)

            print(f"New best val_loss={val_loss:.4f}. Saved checkpoint to {best_checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    # ----------------
    # End of Training
    # ----------------
    if best_checkpoint_path:
        print(f"\nTraining complete. Best checkpoint: {best_checkpoint_path} with val_loss={best_val_loss:.4f}")
    else:
        print("\nTraining complete, but no 'best' checkpoint was saved.")

    # Optionally return final metrics
    final_metrics = {
        "val_loss": best_val_loss if best_val_loss is not None else float('inf'),
        "val_ppl": best_val_ppl,
        "confusion_matrix": best_val_conf.tolist() if best_val_conf is not None else None
    }
    return final_metrics


if __name__ == "__main__":
    config = get_config()
    train_model(config)
