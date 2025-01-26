from pathlib import Path

def get_config():
    return {
        "filepath": "drive/MyDrive/ProductGPT/clean_list_int_wide12.json",
        "vocab_size_src": 122,
        "vocab_size_tgt": 10,
        "vocab_size_lto": 122,
        "seq_len_src": 1280,
        "seq_len_tgt": 128,
        "seq_len_lto": 1536,
        "batch_size": 8,
        "num_epochs": 60,
        "lr": 10**-4,
        "d_model": 64,
        "N": 6, 
        "num_heads": 8,
        "dropout": 0.1,
        "d_ff": 64,
        "eval_freq": 50,
        "source_rate": 10,
        "lto_rate": 12,
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
