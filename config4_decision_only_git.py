# config4_decision_only_git.py
from pathlib import Path


def get_config():
    """
    Central config dictionary.
    The only change you’re likely to tweak again is `seq_len_ai`
    (context-window size) and its derived `seq_len_tgt`.
    """
    # ------------------------------------------------------------------ #
    #  Context-window settings                                           #
    # ------------------------------------------------------------------ #
    seq_len_ai   = 64          # ← *** 64-token context window ***
    ai_rate      = 1           # decision every 15th token  (positions 14,29,…)
    seq_len_tgt  = seq_len_ai // ai_rate      # 64 // 15 = 4 decision slots

    return {
        # ------------------------------ data paths --------------------- #
        "filepath": "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json",

        # ------------------------------ vocabulary sizes --------------- #
        "vocab_size_src": 68,
        "vocab_size_tgt": 18,
        "vocab_size_lto": 68,
        "vocab_size_ai": 18,

        # ------------------------------ sequence lengths --------------- #
        # src / lto kept large because they’re unused in Decision-Only.
        "seq_len_src": 10240,
        "seq_len_lto": 4096,
        "seq_len_ai":  seq_len_ai,
        "seq_len_tgt": seq_len_tgt,

        # ------------------------------ training hyper-params ---------- #
        "batch_size": 64,
        "num_epochs": 200,
        "warmup_steps": 5,
        "lr": 1e-5,
        "min_lr": 1e-6,
        "d_model": 16,
        "N": 2,
        "num_heads": 2,
        "dropout": 0.1,
        "kernel_type": "exp",
        "d_ff": 16,
        "eval_freq": 40,
        "source_rate": 10,
        "lto_rate": 4,
        "ai_rate":  ai_rate,     # ← must match decision stride everywhere
        "weight_decay": 0.01,
        "patience": 6,
        "gamma": 0,
        "eps": 1e-6,
        "weight": 8,

        # ------------------------------ logging / paths ---------------- #
        "model_folder": "/home/ec2-user/output",
        "model_basename": "DecisionOnly_",
        "preload": None,               # or "latest"
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",

        # ------------------------------ cloud storage ------------------ #
        "s3_bucket": "productgptbucket",
        "gcp_bucket": "productgptbucket",
        # "s3_prefix": "my_runs",
    }


# ---------------------------------------------------------------------------#
# helper functions (unchanged)                                               #
# ---------------------------------------------------------------------------#
def get_weights_file_path(config, epoch: str) -> str:
    model_folder = f"{config['model_folder']}"
    uid = (f"dmodel{config['d_model']}_ff{config['d_ff']}_N{config['N']}_"
           f"heads{config['num_heads']}_lr{config['lr']}_weight{config['weight']}")
    fname = f"DecisionOnly_{uid}.pt"
    full = Path(model_folder)
    full.mkdir(parents=True, exist_ok=True)
    return str(full / fname)


def latest_weights_file_path(config) -> str:
    model_folder = f"{config['model_folder']}"
    uid = (f"dmodel{config['d_model']}_ff{config['d_ff']}_N{config['N']}_"
           f"heads{config['num_heads']}_ lr{config['lr']}_weight{config['weight']}")
    fname = f"DecisionOnly_{uid}.pt"
    folder = Path(model_folder)
    if not folder.exists():
        return None
    files = list(folder.glob(fname))
    if not files:
        return None
    files.sort()
    return str(files[-1])
