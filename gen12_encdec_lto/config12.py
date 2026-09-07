from pathlib import Path

def get_config():
    return {
        "filepath": "drive/MyDrive/ProductGPT/clean_list_int_wide12_simple3.json",
        "vocab_size_src": 68,
        "vocab_size_tgt": 10,
        "vocab_size_lto": 68,
        "seq_len_src": 10240,
        "seq_len_tgt": 1024,
        "seq_len_lto": 12288,
        "batch_size": 4,
        "num_epochs": 200,
        "warmup_steps": 50,
        "lr": 10**-4,
        "min_lr": 10**-6,
        "d_model": 64,
        "N": 6, 
        "num_heads": 8,
        "dropout": 0.1,
        "kernel_type": "exp",
        "d_ff": 32,
        "eval_freq": 40,
        "source_rate": 10,
        "lto_rate": 12,
        "eps": 1e-6,
        "weight_decay": 0.01,
        "patience": 30, 
        # "lr_reduce_factor": 0.5,
        # "max_lr_reductions": 3,
        "gamma": 2.0,
        # "mem_len_src": 640,
        # "mem_len_tgt": 64,
        "datasource": 'ProductGPT',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    full_path = Path('.') / model_folder
    full_path.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
