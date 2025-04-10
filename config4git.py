from pathlib import Path

def get_config():
    return {
        # Data
        "filepath": "/home/ec2-user/data/clean_list_int_wide4_simple6_IndexBasedTrain.json",
        "vocab_size_src": 68,
        "vocab_size_tgt": 18,
        "vocab_size_lto": 68,
        "vocab_size_ai": 68,
        "seq_len_src": 10240,
        "seq_len_tgt": 1024,
        "seq_len_lto": 4096,
        "seq_len_ai": 15360,
        "batch_size": 8,
        "num_epochs": 1000,
        "warmup_steps": 5,
        "lr": 10**-4,
        "min_lr": 10**-6,
        "d_model": 32,
        "N": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "kernel_type": "exp",
        "d_ff": 32,
        "eval_freq": 40,
        "source_rate": 10,
        "lto_rate": 4,
        "ai_rate": 15,
        "weight_decay": 0.01,
        "patience": 6,
        "gamma": 3,
        "eps": 10**-6,

        # Logging and paths
        "datasource": "ProductGPT",
        "model_folder": "/home/ec2-user/output",
        "model_basename": "MyProductGPT_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str) -> str:
    """
    Build the path for saving/loading a model checkpoint, based on:
      - datasource
      - model_folder
      - model_basename + epoch
    """
    # e.g. "ProductGPT_weights" 
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    # e.g. "MyProductGPT_epoch5.pt" or "MyProductGPT_best.pt"
    model_filename = f"{config['model_basename']}{epoch}.pt"

    full_path = Path('.') / model_folder
    full_path.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

    return str(full_path / model_filename)

def latest_weights_file_path(config) -> str:
    """
    Finds the newest checkpoint file matching `config['model_basename']*.pt`
    inside the folder <datasource>_<model_folder>.
    Returns the file path if found, otherwise None.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    # Match e.g. "MyProductGPT_*.pt"
    pattern = f"{config['model_basename']}*.pt"

    folder_path = Path('.') / model_folder
    if not folder_path.exists():
        return None

    weights_files = list(folder_path.glob(pattern))
    if len(weights_files) == 0:
        return None

    # Sort by filename (or by modified time if preferred)
    weights_files.sort()
    return str(weights_files[-1])  # return the last (newest if naming is chronological)
