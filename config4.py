from pathlib import Path

def get_config():
    return {
        "filepath": "drive/MyDrive/ProductGPT/clean_list_int_wide4_simple6_IndexBasedTrain.json",
        "vocab_size_src": 68,
        "vocab_size_tgt": 18,
        "vocab_size_lto": 68,
        "vocab_size_ai": 68,
        "seq_len_src": 10240,
        "seq_len_tgt": 1024,
        "seq_len_lto": 4096,
        "seq_len_ai": 15360,
        "batch_size": 32,
        "num_epochs": 200,
        "warmup_steps": 5,
        "lr": 10**-2,
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
        "patience": 3, 
        # "lr_reduce_factor": 0.5,
        # "max_lr_reductions": 3,
        "gamma": 3,
        # "mem_len_src": 640,
        # "mem_len_tgt": 64,
        "eps": 10**-6,
        "datasource": 'ProductGPT',
        "model_folder": "drive/MyDrive/ProductGPT/ProductGPT_weights",
        "model_basename": "MyProductGPT_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    
    model_folder = Path(config["model_folder"])
    unique_id = f"dmodel{config['d_model']}_ff{config['d_ff']}_N{config['N']}_heads{config['num_heads']}_gamma{config['gamma']}"
    # config['model_basename'] = f"MyProductGPT_{unique_id}_"
    
    basename = f"MyProductGPT_{unique_id}_"
    model_filename = f"{basename}{epoch}.pt"
    # full_path = Path('.') / model_folder
    # full_path.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist
    
    model_folder.mkdir(parents=True, exist_ok=True)
    return str(model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    # model_folder = f"{config['datasource']}_{config['model_folder']}"
    # model_filename = f"{config['model_basename']}*"

    model_folder = Path(config["model_folder"])
    # weights_files = list(Path(model_folder).glob(model_filename))
    
    weights_files = list(model_folder.glob("MyProductGPT_*[0-9].pt"))
    if not weights_files:
        return None
    weights_files.sort()
    return str(weights_files[-1])
