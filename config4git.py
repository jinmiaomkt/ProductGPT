from pathlib import Path

def get_config():
    """
    Returns a dictionary with hyperparameters and paths.
    Update these as needed for your training.
    """
    return {
        # Data
        "filepath": "/home/ec2-user/Data/tmp/clean_list_int_wide4_simple4_IndexBasedTrain.json",
        
        # Vocabulary sizes
        "vocab_size_src": 48,
        "vocab_size_tgt": 10,
        "vocab_size_lto": 48,

        # Sequence lengths
        "seq_len_src": 10240,
        "seq_len_tgt": 1024,
        "seq_len_lto": 4096,

        # Training parameters
        "batch_size": 32,
        "num_epochs": 200,
        "warmup_steps": 5,
        "lr": 1e-4,         # peak LR
        "min_lr": 1e-6,
        "weight_decay": 0.01,
        "patience": 30,
        "gamma": 2.0,       # focal loss gamma
        "eps": 1e-6,        # LAMB optimizer epsilon

        # Model architecture
        "d_model": 64,
        "N": 6,             # number of layers
        "num_heads": 8,
        "dropout": 0.1,
        "kernel_type": "exp",
        "d_ff": 32,

        # Rates
        "source_rate": 10,
        "lto_rate": 4,

        # Logging and paths
        "datasource": "ProductGPT",
        "model_folder": "weights",
        "model_basename": "MyProductGPT_",  # prefix for model checkpoint files
        "preload": "latest",               # can be 'latest', a specific file suffix, or None
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
