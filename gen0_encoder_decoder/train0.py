import warnings
import json
import torch
import re

from tokenizers import Tokenizer, pre_tokenizers, trainers, models
from tokenizers.pre_tokenizers import Split, Sequence

from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from dataset0 import TransformerDataset, causal_mask_square, load_json_dataset, causal_mask_rectangular

from model0 import build_transformer
from config0 import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

# import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, lto, long_long_self_mask, short_long_cross_mask, short_short_cross_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # short_long_cross_mask = short_long_cross_mask.unsqueeze(1) 
    # short_long_cross_mask = short_long_cross_mask.repeat(1, 8, 1, 1)

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, long_long_self_mask, lto, short_long_cross_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build a causal (triangular) mask for the current decoder_input length
        cur_len = decoder_input.size(1)
        causal_mask = torch.triu(torch.ones(cur_len, cur_len, device=device), diagonal=1)
        causal_mask = (causal_mask == 0).unsqueeze(0).unsqueeze(1)

        # calculate output
        out = model.decode(
            encoder_output, 
            causal_mask, 
            decoder_input, 
            short_short_cross_mask)
        # out = model.decode(encoder_output, short_short_self_mask, decoder_input, short_short_cross_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def get_all_sentences(ds, lang):
    for item in ds:
        yield item[lang]

def build_tokenizer(data, key, vocab_size=1000):
    """
    Build a WordLevel tokenizer that splits only on whitespace.
    
    data: list of dicts
        Each dict must have a string `record[key]`.
    key: string
        The dict key where the text is stored.
    vocab_size: int
        Maximum vocabulary size (including special tokens).
    """

    # 1) Use WordLevel model (no subword merges)
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

    # 2) Whitespace pre-tokenizer => splits on whitespace only
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 3) WordLevelTrainer
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
    )

    # 4) Train from iterator
    tokenizer.train_from_iterator(get_all_sentences(data, key), trainer=trainer)

    return tokenizer


def get_dataloaders(config):
    data = load_json_dataset(config['filepath'])
    
    train_data = []
    val_data = []

    train_ds_size = int(0.8 * len(data))
    val_ds_size = int(0.1 * len(data))
    test_ds_size = len(data) - train_ds_size - val_ds_size
    
    train_data, val_data, test_data = random_split(data, [train_ds_size, val_ds_size, test_ds_size])

    # for user_record in data:
    #     decisions = user_record["Decision"]
    #     num_decisions = len(decisions)

    #     items = user_record["Item"]
    #     num_items = len(items)

    #     f5as = user_record["F5A"]
    #     num_f5as = len(f5as)

    #     # Calculate index boundary for 80/20
    #     train_size_decisions = int(num_decisions * 0.8)
    #     train_size_items = int(num_items * 0.8)
    #     train_size_f5as = int(num_f5as * 0.8)
        
    #     # Slice out training and validation for this user
    #     user_train_decisions = decisions[:train_size_decisions]
    #     user_val_decisions   = decisions[train_size_decisions:]
        
    #     user_train_items = decisions[:train_size_items]
    #     user_val_items = decisions[train_size_items:]
        
    #     user_train_f5as = decisions[:train_size_f5as]
    #     user_val_f5as   = decisions[train_size_f5as:]
        
    #     # Append a new user record to train_data if not empty
    #     if len(user_train_decisions) > 0:
    #         train_data.append({
    #             "user_id": user_record["uid"],
    #             "Item": user_train_items,
    #             "Decision": user_train_decisions,
    #             "F5A": user_train_f5as
    #         })
        
    #     # Append a new user record to val_data if not empty
    #     if len(user_val_decisions) > 0:
    #         val_data.append({
    #             "user_id": user_record["uid"],
    #             "Item": user_val_items,
    #             "Decision": user_val_decisions,
    #             "F5A": user_val_f5as
    #         })

    # Build tokenizers using the train_data (recommended)
    tokenizer_src = build_tokenizer(train_data, "Item", vocab_size=config['vocab_size_src'])
    tokenizer_tgt = build_tokenizer(train_data, "Decision", vocab_size=config['vocab_size_tgt'])
    # tokenizer_lto = build_tokenizer(train_data, "Item", vocab_size=config['vocab_size_lto'])
    
    # Create datasets
    # Make sure your TransformerDataset is designed to handle a list of user records, each containing a 'decisions' list.
    train_dataset = TransformerDataset(train_data, tokenizer_src, tokenizer_tgt, config['seq_len_src'], config['seq_len_tgt'], config['num_heads'], config['source_rate'])
    val_dataset   = TransformerDataset(val_data, tokenizer_src, tokenizer_tgt, config['seq_len_src'], config['seq_len_tgt'], config['num_heads'], config['source_rate'])
    test_dataset  = TransformerDataset(test_data, tokenizer_src, tokenizer_tgt, config['seq_len_src'], config['seq_len_tgt'], config['num_heads'], config['source_rate'])

    # Create DataLoaders
    # For training, you can shuffle across users if that’s appropriate for your scenario.
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_dataloader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False)
    test_dataloader  = DataLoader(test_dataset,  batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len_src"], config['seq_len_tgt'], d_model=config['d_model'], N = config['N'], h=config['num_heads'], dropout=config['dropout'], d_ff=config['d_ff'])
    return model

def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataloaders(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
#     writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    def cal_loss_loader(dataloader, model, device, loss_fn, num_batches = None):
        total_loss = 0
        if len(dataloader) == 0:
            return float('nan')
        elif num_batches is None:
            num_batches = len(dataloader)
        else:
            num_batches = min(len(dataloader), num_batches)

        # batch_iterator = tqdm(dataloader, desc="Calculating loss")  
        for i, batch in enumerate(dataloader):
            if i < num_batches:
                source_input = batch['source_input'].to(device) # (b, seq_len_long)
                target_input = batch['target_input'].to(device) # (B, seq_len_short)
                # lto_input = batch['lto_input'].to(device) # (B, seq_len_short)

                long_long_self_mask = batch['long_long_self_mask'].to(device) # (B, 1, seq_len_long, seq_len_long)
                short_short_self_mask = batch['short_short_self_mask'].to(device) # (B, 1, seq_len_short, seq_len_short)
                short_long_cross_mask = batch['short_long_cross_mask'].to(device) # (B, 1, seq_len_short, seq_len_long)
                # short_short_cross_mask = batch['short_short_cross_mask'].to(device) # (B, 1, seq_len_short, seq_len_short)

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(source_input, long_long_self_mask) # (B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, short_short_self_mask, target_input, short_long_cross_mask) # (B, seq_len, d_model)
                proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                total_loss += loss.item()               
            else:
                break

        return total_loss / num_batches


    train_losses, val_losses, track_tokens_seen = [], [], []

    # Initialize tokens_seen
    tokens_seen = 0
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            source_input = batch['source_input'].to(device) # (b, seq_len_long)
            target_input = batch['target_input'].to(device) # (B, seq_len_short)
            # lto_input = batch['lto_input'].to(device) # (B, seq_len_short)
            
            long_long_self_mask = batch['long_long_self_mask'].to(device) # (B, 1, seq_len_long, seq_len_long)
            short_short_self_mask = batch['short_short_self_mask'].to(device) # (B, 1, seq_len_short, seq_len_short)
            short_long_cross_mask = batch['short_long_cross_mask'].to(device) # (B, 1, seq_len_short, seq_len_long)
            # short_short_cross_mask = batch['short_short_cross_mask'].to(device) # (B, 1, seq_len_short, seq_len_short)

            # print(f"Batch {batch} -- min: {source_input.min()}, max: {source_input.max()}")
            # print(f"Batch {batch} -- min: {target_input.min()}, max: {target_input.max()}")
            # print(f"Batch {batch} -- min: {lto_input.min()}, max: {lto_input.max()}")

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(source_input, long_long_self_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, short_short_self_mask, target_input, short_long_cross_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            # writer.add_scalar('train loss', loss.item(), global_step)
            # writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            tokens_seen += source_input.numel()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % config['eval_freq'] == 0:
                model.eval()
                with torch.no_grad():
                    train_loss = cal_loss_loader(train_dataloader, model, device, loss_fn)
                    val_loss = cal_loss_loader(val_dataloader, model, device, loss_fn)
                    # writer.add_scalar('train loss', train_loss, global_step)
                    # writer.add_scalar('val loss', val_loss, global_step)
                    # writer.flush()
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f'Epoch {epoch+1:02d} - Global Step {global_step:06d} - Train Loss {train_loss:6.3f} - Val Loss {val_loss:6.3f}')


        # Run validation at the end of every epoch
        # run_validation(model, val_dataloader, tokenizer_tgt, config['seq_len_tgt'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
