from pathlib import Path

def get_config():
    return {
        # Data
        "filepath": "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json",
        "vocab_size_src": 68,
        "vocab_size_tgt": 18,
        "vocab_size_lto": 68,
        "vocab_size_ai": 18,
        "seq_len_src": 10240,
        "seq_len_tgt": 1024,
        "seq_len_lto": 4096,
        "seq_len_ai": 1024,
        "batch_size": 64,
        "num_epochs": 200,
        "warmup_steps": 5,
        "lr": 10**-5,
        "min_lr": 10**-6,
        "d_model": 16,
        "N": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "kernel_type": "exp",
        "d_ff": 16,
        "eval_freq": 40,
        "source_rate": 10,
        "lto_rate": 4,
        "ai_rate": 1,
        "weight_decay": 0.01,
        "patience": 6,
        "gamma": 0,
        "eps": 10**-6,
        "weight": 8,

        # Logging and paths
        # "datasource": "ProductGPT",
        "model_folder": "/home/ec2-user/output",
        "model_basename": "DecisionOnly_",
        # "preload": "latest",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel", 

        "s3_bucket": "productgptbucket",  # <- replace with your real bucket name
        "gcp_bucket": "productgptbucket",          # <- replace with your real region
        "s3_prefix": "my_runs",            # optional folder name in bucket
    }


def get_weights_file_path(config, epoch: str) -> str:
    """
    Build the path for saving/loading a model checkpoint, based on:
      - datasource
      - model_folder
      - model_basename + epoch
    """
    # e.g. "ProductGPT_weights" 
    model_folder = f"{config['model_folder']}"

    unique_id = f"dmodel{config['d_model']}_ff{config['d_ff']}_N{config['N']}_heads{config['num_heads']}_lr{config['lr']}_weight{config['weight']}"
    basename = f"DecisionOnly_{unique_id}"
    model_filename = f"{basename}.pt"
    
    full_path = Path('.') / model_folder
    full_path.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

    return str(full_path / model_filename)

def latest_weights_file_path(config) -> str:
    """
    Finds the newest checkpoint file matching `config['model_basename']*.pt`
    inside the folder <datasource>_<model_folder>.
    Returns the file path if found, otherwise None.
    """
    model_folder = f"{config['model_folder']}"
    
    unique_id = f"dmodel{config['d_model']}_ff{config['d_ff']}_N{config['N']}_heads{config['num_heads']}_ lr{config['lr']}_weight{config['weight']}"
    basename = f"DecisionOnly_{unique_id}"
    model_filename = f"{basename}.pt"

    folder_path = Path('.') / model_folder
    if not folder_path.exists():
        return None

    weights_files = list(folder_path.glob(model_filename))
    if len(weights_files) == 0:
        return None

    # Sort by filename (or by modified time if preferred)
    weights_files.sort()
    return str(weights_files[-1])  # return the last (newest if naming is chronological)
