# two_stage_search_then_shuffle.py
from __future__ import annotations
import os, json, math, random, socket
from dataclasses import dataclass
from pathlib import Path
import boto3, botocore, torch

from config4 import get_config
from train4_decoderonly_performer_feature_aws import train_model

# -------------------- user knobs --------------------
FOLD_ID = 0
SPEC_URI = "s3://productgptbucket/CV/folds.json"

# Stage A: cheap search
STAGEA_TRIALS = 32
STAGEA_MAX_EPOCHS = 40
STAGEA_DATA_FRAC = 0.05         # e.g., 5% of records
STAGEA_AUGMENT_TRAIN = False    # IMPORTANT: no permutation augmentation

# Stage B: final train
STAGEB_MAX_EPOCHS = 200
STAGEB_DATA_FRAC = 1.0          # full
STAGEB_AUGMENT_TRAIN = True     # enable permutation augmentation
STAGEB_PERMUTE_REPEAT = 1       # optionally >1 (but see note below)

# Selection criterion
SELECT_METRIC = "val_all_auprc"  # or "val_all_f1_score", etc.

random.seed(1337)

# -------------------- utilities --------------------
def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def load_fold_spec(uri: str):
    if uri.startswith("s3://"):
        bucket, key = uri[5:].split("/", 1)
        body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    with open(uri, "r") as f:
        return json.load(f)

def sample_hp():
    while True:
        nb_features = random.choice([32, 48, 64])
        d_model     = random.choice([96, 128, 160, 256])
        num_heads   = random.choice([4, 6, 8])
        if d_model % num_heads != 0:
            continue
        head_dim = d_model // num_heads
        if head_dim < 16:
            continue

        N         = random.randint(4, 8)
        dff_mult  = random.choice([2, 3, 4])
        d_ff      = min(d_model * dff_mult, 512)

        dropout   = round(random.uniform(0.0, 0.2), 3)
        lr        = random.choice([1e-3, 5e-4, 1e-4])
        lr        = float(f"{lr:.6g}")

        weight    = random.choice([1, 2, 4, 6, 8])
        gamma     = round(random.uniform(0.8, 1.5), 3)
        warmup_steps = random.choice([500, 1000, 2000])

        return {
            "nb_features": nb_features,
            "d_model": d_model,
            "d_ff": d_ff,
            "N": N,
            "num_heads": num_heads,
            "dropout": dropout,
            "lr": lr,
            "weight": weight,
            "gamma": gamma,
            "warmup_steps": warmup_steps,
        }

def build_cfg(base_cfg: dict, hp: dict, *, fold_id: int, uids_test, uids_trainval,
              max_epochs: int, data_frac: float, augment_train: bool, permute_repeat: int):
    cfg = dict(base_cfg)

    cfg.update({
        "mode": "train",
        "fold_id": fold_id,
        "uids_test": uids_test,
        "uids_trainval": uids_trainval,
        "num_epochs": max_epochs,

        # --------- NEW knobs consumed by your patched trainer ----------
        "data_frac": data_frac,
        "subsample_seed": 33,
        "augment_train": augment_train,
        "permute_repeat": permute_repeat,
    })

    # keep your ai_rate/seq_len_ai logic
    cfg["ai_rate"] = int(cfg.get("ai_rate", 10))
    cfg["seq_len_ai"] = cfg["ai_rate"] * cfg["seq_len_tgt"]

    cfg.update(hp)

    uid = (
        f"feat{hp['nb_features']}_dm{hp['d_model']}_ff{hp['d_ff']}_"
        f"N{hp['N']}_h{hp['num_heads']}_do{hp['dropout']}_"
        f"lr{hp['lr']}_w{hp['weight']}_g{hp['gamma']}_wu{hp['warmup_steps']}_"
        f"frac{data_frac}_aug{int(augment_train)}_rep{permute_repeat}"
    )
    cfg["model_basename"] = f"MyProductGPT_FeatureBased_{uid}"
    return cfg, uid

@dataclass
class TrialResult:
    uid: str
    hp: dict
    metric: float
    full_metrics: dict

def run_one(cfg: dict):
    # single-GPU, avoid DS port collisions
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(free_port())

    return train_model(cfg)  # should return dict; we’ll rely on returned val metrics if present

def two_stage():
    spec = load_fold_spec(SPEC_URI)
    uids_test     = [u for u, f in spec["assignment"].items() if f == FOLD_ID]
    uids_trainval = [u for u in spec["assignment"] if u not in uids_test]
    assert uids_test and uids_trainval

    base_cfg = get_config()

    # ---------------- Stage A: cheap random search ----------------
    best: TrialResult | None = None
    results: list[TrialResult] = []

    for t in range(STAGEA_TRIALS):
        hp = sample_hp()
        cfg, uid = build_cfg(
            base_cfg, hp,
            fold_id=FOLD_ID,
            uids_test=uids_test,
            uids_trainval=uids_trainval,
            max_epochs=STAGEA_MAX_EPOCHS,
            data_frac=STAGEA_DATA_FRAC,
            augment_train=STAGEA_AUGMENT_TRAIN,
            permute_repeat=1,
        )

        print(f"\n[Stage A Trial {t+1}/{STAGEA_TRIALS}] {uid}\nHP={hp}")
        out = run_one(cfg)

        # Prefer reading metric directly from returned object, fallback to NaN
        # Your train_model currently returns: val_auprc, val_f1, val_loss etc.
        metric = float(out.get("val_auprc", float("nan")))
        tr = TrialResult(uid=uid, hp=hp, metric=metric, full_metrics=out)
        results.append(tr)

        if best is None or (not math.isnan(metric) and metric > best.metric):
            best = tr
            print(f"[Stage A] New best: {best.uid} {SELECT_METRIC}={best.metric}")

    assert best is not None, "Stage A produced no results."

    print("\n====================")
    print(f"[Stage A BEST] {best.uid}")
    print(f"HP: {best.hp}")
    print(f"{SELECT_METRIC}: {best.metric}")
    print("====================\n")

    # ---------------- Stage B: retrain best on full data with shuffling augmentation ----------------
    cfgB, uidB = build_cfg(
        base_cfg, best.hp,
        fold_id=FOLD_ID,
        uids_test=uids_test,
        uids_trainval=uids_trainval,
        max_epochs=STAGEB_MAX_EPOCHS,
        data_frac=STAGEB_DATA_FRAC,
        augment_train=STAGEB_AUGMENT_TRAIN,
        permute_repeat=STAGEB_PERMUTE_REPEAT,
    )

    # Make it obvious in artifact names this is the FINAL run
    cfgB["model_basename"] = f"MyProductGPT_FeatureBased_FINAL_{uidB}"
    print(f"[Stage B] Training best HP on full data with augmentation. UID={cfgB['model_basename']}")

    outB = run_one(cfgB)

    # save a local summary for convenience
    summary = {
        "stageA_best_uid": best.uid,
        "stageA_best_hp": best.hp,
        "stageA_best_metric": best.metric,
        "stageB_uid": cfgB["model_basename"],
        "stageB_results": outB,
    }
    Path("two_stage_summary.json").write_text(json.dumps(summary, indent=2))
    print("[DONE] wrote two_stage_summary.json")

if __name__ == "__main__":
    two_stage()
