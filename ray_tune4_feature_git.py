# ray_tune4_feature_git.py
from __future__ import annotations

import os, json, socket
from pathlib import Path
import boto3
import torch

from config4 import get_config
from train4_decoderonly_performer_feature_aws import train_model

import ray
from ray import tune
from ray.air import session
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

# -------------------- your fold logic --------------------
FOLD_ID  = 0
SPEC_URI = "s3://productgptbucket/folds/productgptfolds.json"

def load_fold_spec(uri: str):
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    with open(uri, "r") as f:
        return json.load(f)

spec = load_fold_spec(SPEC_URI)
uids_test     = [u for u, f in spec["assignment"].items() if f == FOLD_ID]
uids_trainval = [u for u in spec["assignment"] if u not in uids_test]
assert uids_test and uids_trainval

def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# -------------------- Trainable --------------------
def trainable_ray(config: dict):
    """
    Ray Tune trainable: called once per trial in its own process.
    `config` contains HP sampled by Ray, plus any fixed fields you inject.
    """
    # ---- ensure single-GPU deepspeed init is isolated per trial ----
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    # Ray sets CUDA_VISIBLE_DEVICES automatically according to resources.
    # So DON’T override it here unless you know what you’re doing.

    base_cfg = get_config()
    cfg = dict(base_cfg)

    # ---- fixed experiment settings ----
    cfg.update({
        "mode": "train",
        "fold_id": FOLD_ID,
        "uids_test": uids_test,
        "uids_trainval": uids_trainval,
        "ai_rate": 15,
        "do_infer": config.get("do_infer", False),   
        # IMPORTANT: disable inference during tuning
        "num_epochs": config.get("num_epochs", 120),
        # optional: speed-up stage-A tuning on a small dataset:
        "data_frac": config.get("data_frac", 1.0),          # requires your build_dataloaders patch
        "subsample_seed": 33,
        # optional: disable expensive shuffling augmentation in tuning:
        "augment_train": config.get("augment_train", False), # requires your build_dataloaders patch
        "permute_repeat": config.get("permute_repeat", 1),
    })
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]

    # ---- unpack coupled params cleanly ----
    d_model, num_heads = config["dm_heads"]
    cfg["d_model"] = d_model
    cfg["num_heads"] = num_heads
    cfg["d_ff"] = min(int(d_model * config["dff_mult"]), 512)

    # ---- rest of HP ----
    cfg.update({
        "nb_features": config["nb_features"],
        # "d_ff": config["d_ff"],
        "N": config["N"],
        "dropout": config["dropout"],
        "lr": config["lr"],
        "weight": config["weight"],
        "gamma": config["gamma"],
        "warmup_steps": config["warmup_steps"],
        # optional keys your trainer may read:
        "dropout_attn": config["dropout"],
        "dropout_ffn":  config["dropout"],
        "label_smoothing": config.get("label_smoothing", 0.0),
    })

    # # include trial name in model_basename to avoid collisions
    # trial_name = session.get_trial_name()
    # cfg["model_basename"] = f"MyProductGPT_RT_{trial_name}"

    # # ---- report_fn to Ray ----
    # def report_fn(m: dict):
    #     # Ray expects numbers; keep it flat
    #     session.report(m)

    # Detect whether we are running inside a Ray Tune session
    try:
        trial_name = session.get_trial_name()
        in_tune = True
    except Exception:
        trial_name = "manual"
        in_tune = False

    cfg["model_basename"] = f"MyProductGPT_RT_{trial_name}"

    def report_fn(m: dict):
        # Only report when inside Tune; Stage-B retrain is outside Tune
        if in_tune:
            session.report(m)

    # ---- optional stop function: Ray can signal stop via should_checkpoint / etc.
    # ASHA will stop a trial by raising a TuneError internally after report.
    # You usually don't need stop_check_fn, but here’s a safe placeholder:
    def stop_check_fn() -> bool:
        return False
    train_model(cfg, report_fn=report_fn if in_tune else None, stop_check_fn=stop_check_fn)

    # Run training; MUST call report_fn each epoch inside train_model
    # train_model(cfg, report_fn=report_fn, stop_check_fn=stop_check_fn)


def main():
    ray.init(ignore_reinit_error=True)

    # ---- Valid (d_model, heads) combos to satisfy divisibility + head_dim>=16 ----
    valid_dm_heads = []
    for dm in [64, 96, 128]:
        for h in [4, 6, 8]:
            if dm % h == 0 and (dm // h) >= 16:
                valid_dm_heads.append((dm, h))

    # ---- Search space (Ray handles sampling) ----
    param_space = {
        # stage-A knobs (optional)
        "num_epochs": 120,
        "data_frac": 0.05,          # << cheap tuning (requires build_dataloaders patch)
        "augment_train": False,     # << disable expensive permutation augmentation during tuning
        "permute_repeat": 1,

        # hyperparams
        "nb_features": tune.choice([32, 48, 64]),
        "dm_heads": tune.choice(valid_dm_heads),   # couples d_model and num_heads safely
        "N": tune.randint(2, 4),                   # 4..8
        "dropout": tune.uniform(0.0, 0.2),
        "lr": tune.loguniform(1e-4, 1e-3),
        "weight": tune.choice([1, 2, 4, 6, 8]),
        "gamma": tune.uniform(0.8, 1.5),
        "warmup_steps": tune.choice([500, 1000, 2000]),
        "label_smoothing": tune.uniform(0.0, 0.1),
        "do_infer": False,

        # You can make d_ff depend on d_model inside trainable if you want.
        # For simplicity: sample a multiplier then compute d_ff = min(d_model*m, 512)
        "dff_mult": tune.choice([2, 3]),
        "d_ff": 256,  # placeholder; we’ll override in trainable below if desired
    }

    # ---- ASHA scheduler: early stop bad trials ----
    asha = ASHAScheduler(
        time_attr="epoch",
        metric="val_nll",
        mode="min",
        max_t=120,
        grace_period=10,
        reduction_factor=3,
    )

    algo = HyperOptSearch(
        metric="val_nll",
        mode="min",
    )

    # ---- Run config ----
    tuner = tune.Tuner(
        tune.with_resources(trainable_ray, resources={"cpu": 4, "gpu": 1}),
        tune_config=tune.TuneConfig(
            num_samples=300,     # like your JMR example
            max_concurrent_trials=1,
            search_alg=algo,
            scheduler=asha,
        ),
        run_config=ray.air.RunConfig(
            name="ProductGPT_RayTune",
            storage_path=str(Path("./ray_results").resolve()),
        ),
        param_space=param_space,
    )

    results = tuner.fit()

    best = results.get_best_result(metric="val_nll", mode="min")
    print("\n===== BEST CONFIG =====")
    print(best.config)
    print("best val_nll:", best.metrics.get("val_nll"))
    print("val_hit:", best.metrics.get("val_hit"))
    print("val_f1_macro:", best.metrics.get("val_f1_macro"))
    print("val_auprc_macro:", best.metrics.get("val_auprc_macro"))

    # ---- Stage B: retrain best on full data with expensive shuffling ----
    best_cfg = best.config
    final_cfg = best_cfg.copy()
    final_cfg.update({
        "data_frac": 1.0,
        "augment_train": True,
        "permute_repeat": 1,     # optionally >1 if you implemented sample_index correctly
        "num_epochs": 200,
        "do_infer": True,
    })

    print("\n===== RETRAIN BEST ON FULL DATA WITH AUGMENT =====")
    trainable_ray(final_cfg)

if __name__ == "__main__":
    main()