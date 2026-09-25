"""
R30 stage 5: diagnostics on the FROZEN stage-4 models. No training, no selection.

Stage 4 said the four families tie on aggregate NLL. Three questions follow, and
none of them is answerable from a single number:

  per-class   do they tie everywhere, or trade rare classes against the majority
              one? NotBuy is ~38% of occasions, so an aggregate tie can hide a
              large difference on the eight purchase classes.
  calibration are the predicted probabilities usable as probabilities? For a
              revenue simulation they have to be, and NLL alone does not say so
              (reliability and sharpness trade off inside it).
  efficiency  parameters and seconds per epoch beside the loss. After stage 4
              this is a headline exhibit, not a footnote.

Reads each run's best.pt, scores the out-of-sample x holdout cell, and writes
ONLY aggregates: per-class sums, a 9x9 confusion matrix, calibration bins, and
timing. No per-customer row ever leaves this script.

USAGE (on HPCC, in a GPU job)
    python scripts/stage5_eval.py --tags b17_hyb_v7 b17_tf_v7 --out-dir results/r30/stage5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config5
import dataset_multistream
import model_multistream_state_space
from eval_per_occasion import CELL, N_CLASSES, build_model
from shared.features import load_feature_tensor
from train5_multistream import (amp_dtype, build_loaders, inventory_kwargs, pick_device,
                                set_seed, set_unknown_user_to_mean)

N_BINS = 15


def score(ckpt: Path, device) -> dict:
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    for k, v in (("vocab_level", 6), ("first_prod_id", 13), ("last_prod_id", 56),
                 ("unk_prod_id", 59)):
        cfg.setdefault(k, v)
    dataset_multistream.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    model_multistream_state_space.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"],
                                                    cfg["unk_prod_id"])
    set_seed(cfg["seed"])
    _, _, tests, num_users = build_loaders(cfg)
    feat = load_feature_tensor(config5.feature_file_path(cfg),
                               id_column=cfg.get("feature_id_column"),
                               first_prod_id=cfg["first_prod_id"], last_prod_id=cfg["last_prod_id"],
                               max_token_id=cfg["vocab_size_src"] - 1)
    model = build_model(cfg, feat, num_users, device)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch for {ckpt}")
    set_unknown_user_to_mean(model, cfg.get("_trained_user_indices", []))
    model.eval()
    adtype = amp_dtype(device) if cfg.get("amp") else None

    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)     # true x predicted
    cls_nll = np.zeros(N_CLASSES)
    cls_n = np.zeros(N_CLASSES, dtype=np.int64)
    bin_n = np.zeros(N_BINS, dtype=np.int64)
    bin_conf = np.zeros(N_BINS)
    bin_acc = np.zeros(N_BINS)
    brier = 0.0
    with torch.no_grad():
        for batch in tests[CELL]:
            lto, obt = batch["lto"].to(device), batch["obtained"].to(device)
            prev, uid = batch["prev_decision"].to(device), batch["user_id"].to(device)
            tgt = batch["label"].to(device)
            ipt = batch["ipt"].to(device) if "ipt" in batch else None
            inv = inventory_kwargs(model, batch, device)
            ctx = (torch.autocast("cuda", dtype=adtype) if adtype is not None
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                logits = model(lto, obt, prev, uid, ipt, **inv)
            p = F.softmax(logits.float()[..., 1:1 + N_CLASSES], dim=-1)
            mask = (tgt >= 1) & (tgt <= N_CLASSES)
            if not mask.any():
                continue
            pm = p[mask]                                    # (M,9)
            y = (tgt[mask] - 1).long()                      # 0..8
            nll = -torch.log(pm.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12))
            pred = pm.argmax(1)
            top = pm.max(1).values
            onehot = F.one_hot(y, N_CLASSES).float()
            brier += float(((pm - onehot) ** 2).sum())
            idx = np.ravel_multi_index((y.cpu().numpy(), pred.cpu().numpy()),
                                       (N_CLASSES, N_CLASSES))
            conf += np.bincount(idx, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
            cls_nll += np.bincount(y.cpu().numpy(), weights=nll.cpu().numpy(),
                                   minlength=N_CLASSES)
            cls_n += np.bincount(y.cpu().numpy(), minlength=N_CLASSES)
            b = np.clip((top.cpu().numpy() * N_BINS).astype(int), 0, N_BINS - 1)
            bin_n += np.bincount(b, minlength=N_BINS)
            bin_conf += np.bincount(b, weights=top.cpu().numpy(), minlength=N_BINS)
            bin_acc += np.bincount(b, weights=(pred == y).cpu().numpy().astype(float),
                                   minlength=N_BINS)

    n = int(cls_n.sum())
    hist = ckpt.parent / "history.json"
    secs = 0.0
    if hist.exists():
        try:
            rows = json.loads(hist.read_text())
            rows = rows if isinstance(rows, list) else rows.get("epochs", [])
            vals = [r.get("secs", 0) for r in rows if isinstance(r, dict)]
            secs = float(np.mean(vals)) if vals else 0.0
        except json.JSONDecodeError:
            pass
    ece = float(np.sum(np.abs(bin_acc - bin_conf)) / max(n, 1))
    return {
        "n_occasions": n,
        "nll": float(cls_nll.sum() / max(n, 1)),
        "class_nll": (cls_nll / np.maximum(cls_n, 1)).tolist(),
        "class_n": cls_n.tolist(),
        "confusion": conf.tolist(),
        "bin_n": bin_n.tolist(), "bin_conf": bin_conf.tolist(), "bin_acc": bin_acc.tolist(),
        "ece": ece,
        "brier": float(brier / max(n, 1)),
        "params": int(sum(p.numel() for p in model.parameters())),
        "secs_per_epoch": secs,
        "vocab_level": int(cfg["vocab_level"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="run tag stems, e.g. b17_hyb_v7")
    ap.add_argument("--runs-root", default="/storage/home/jinmiao/ProductGPT/runs")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    device = pick_device()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stem in a.tags:
        for ck in sorted(Path(a.runs_root).glob(f"*_b4_{stem}_s[0-9]/**/best.pt")):
            tag = ck.parent.parent.parent.name.split("_b4_")[-1]
            dest = out / f"{tag}.json"
            if dest.exists():
                print(f"[stage5] {tag} already scored")
                continue
            res = score(ck, device)
            dest.write_text(json.dumps(res), encoding="utf-8")
            print(f"[stage5] {tag}: nll {res['nll']:.4f}  ECE {res['ece']:.4f}  "
                  f"params {res['params']:,}  {res['secs_per_epoch']:.0f}s/epoch", flush=True)


if __name__ == "__main__":
    main()
