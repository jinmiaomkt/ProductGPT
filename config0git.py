from pathlib import Path

def get_config():
    """
    Returns a dictionary of hyperparameters and file paths.
    Adjust the 'filepath', 'vocab_size_src', etc., as needed.
    """
    return {
        # Data
        "filepath": "/home/ec2-user/Data/tmp/clean_list_int_wide4_simple4_IndexBasedTrain.json",
        
        # Model / Tokenizer Sizes
        "vocab_size_src": 48,
        "vocab_size_tgt": 12,
        "seq_len_src": 10240,
        "seq_len_tgt": 1024,

        # Transformer Hyperparameters
        "d_model": 64,
        "N": 6,                # number of layers
        "num_heads": 8,
        "dropout": 0.1,
        "kernel_type": "exp",
        "d_ff": 64,
        "source_rate": 10,

        # Training Hyperparameters
        "batch_size": 16,
        "num_epochs": 200,
        "warmup_steps": 10,
        "lr": 1e-4,            # peak LR
        "min_lr": 1e-6,
        "eps": 1e-6,           # optimizer epsilon
        "weight_decay": 0.01,
        "patience": 30,
        "gamma": 2.0,          # Focal Loss gamma

        # Logging / Paths
        "datasource": "ProductGPT",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str) -> str:
    """
    Build a full checkpoint path given a config and a suffix (epoch).
    Example output: "./ProductGPT_weights/tmodel_epoch3.pt"
    """
    # Build the model folder name
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    
    # Example filename: "tmodel_epoch3.pt"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    
    full_path = Path('.') / model_folder
    full_path.mkdir(parents=True, exist_ok=True)  # Create folder if needed
    
    return str(full_path / model_filename)

def latest_weights_file_path(config) -> str:
    """
    Find the newest file matching 'model_basename*' in the folder '<datasource>_<model_folder>'.
    Returns the file path if found, otherwise None.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    pattern = f"{config['model_basename']}*"

    folder_path = Path('.') / model_folder
    if not folder_path.exists():
        return None

    weights_files = list(folder_path.glob(pattern))
    if not weights_files:
        return None

    weights_files.sort()
    return str(weights_files[-1])  # Return the last (alphabetically greatest) match
