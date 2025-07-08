# config4_index_git.py
# ————————————————————————————————————————————————————————————
# Central place for *all* hyper-parameters.  The only knob you need to
# touch to change the context length is  `ctx_window`  below.
# seq_len_ai  and  seq_len_tgt  are derived automatically so that:
#     seq_len_ai            == ctx_window
#     seq_len_tgt (labels)  == ctx_window // ai_rate
# ————————————————————————————————————————————————————————————
from __future__ import annotations  # ← add this line
from pathlib import Path

# ─────────────────────────── core config ───────────────────────────
def _raw_config():
    return {
        # ---------- data ----------
        "filepath": "/home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json",
        "test_filepath": "/home/ec2-user/data/clean_list_int_wide4_simple6.json",
        "vocab_size_src": 68,
        "vocab_size_tgt": 18,
        "vocab_size_lto": 68,
        "vocab_size_ai" : 68,

        # ---------- context window ----------
        "ctx_window": 64,
        "nb_features": 8,
        "window_size": None,          # ← edit 64 * 15
        "seq_len_lp":  5120,       # filled in below
        "seq_len_ai":  None,       # filled in below
        "seq_len_tgt": 1024,

        # ---------- training ----------
        "k": 4096,
        "batch_size": 4,
        "num_epochs": 200,
        "warmup_steps": 5,
        "lr": 1e-4,
        "min_lr": 1e-6,
        "d_model": 16,
        "N": 1,
        "num_heads": 1,
        "dropout": 0.1,
        "kernel_type": "exp",
        "d_ff": 16,
        "eval_freq": 40,

        # ---------- rates ----------
        "source_rate": 10,
        "lp_rate":     5,
        "lto_rate":    4,
        "ai_rate":     1,          # decision every 15 tokens in aggregate input

        # ---------- optimisation ----------
        "weight_decay": 0.01,
        "patience": 3,
        "gamma": 0,
        "eps": 1e-6,
        "weight": 2,

        # ---------- logging / paths ----------
        "model_folder":   "/home/ec2-user/output",
        "model_basename": "MyProductGPT_",
        "preload": None,               # "latest" or explicit checkpoint
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",

        # ---------- cloud ----------
        "s3_bucket":  "productgptbucket",
        "gcp_bucket": "productgptbucket",
        "s3_prefix":  "my_runs",
    }

# ─────────────────────────── public helper ─────────────────────────

def get_config():
    cfg = _raw_config()
    cfg["window_size"] = cfg["ctx_window"] * cfg["lp_rate"]
    cfg["block_size"] = 16 * cfg["lp_rate"]
    cfg["seq_len_lp"]  = cfg["seq_len_tgt"] * cfg["lp_rate"]
    return cfg

# ─────────────────────────── checkpoint helpers ───────────────────
def get_weights_file_path(config, tag: str) -> str:
    folder = Path(config['model_folder'])
    folder.mkdir(parents=True, exist_ok=True)

    uid = (f"lp_rate{config['lp_rate']}_ctx{config['ctx_window']}_dmodel{config['d_model']}_ff{config['d_ff']}"
           f"_N{config['N']}_heads{config['num_heads']}_gamma{config['gamma']}"
           f"_lr{config['lr']}_weight{config['weight']}")
    filename = f"ProductGPT_{uid}_{tag}.pt"
    return str(folder / filename)

def latest_weights_file_path(config) -> str | None:
    folder = Path(config['model_folder'])
    if not folder.exists():
        return None

    pattern = f"ProductGPT_*_ctx{config['ctx_window']}.pt"
    files   = sorted(folder.glob(pattern))
    return str(files[-1]) if files else None
