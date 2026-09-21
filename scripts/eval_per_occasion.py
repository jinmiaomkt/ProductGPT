"""
Score every held-out occasion of a finished gen-5 run, for R31.

R31 asks two things the aggregate test cells cannot answer:

  A1  Is the R28 crossover real at the level of CUSTOMERS? All models are
      scored on the same out-of-sample customers, so their per-customer
      differences can be paired and bootstrapped: thousands of units instead
      of three seeds.
  C   Does it appear WHERE product retrieval should matter -- on occasions that
      offer a product the customer already owns -- and not elsewhere?

This script loads best.pt, rebuilds the exact model (either family, either
vocabulary level), rebuilds the same split, and writes one row per scored
occasion of the out-of-sample x holdout cell:

    uid_hash   int64   stable hash of the customer id (pairs models; never printed)
    pos        int32   position of the occasion in the customer's window
    label      int8    decision 1..9
    nll        float32 -log p(label)
    owned_offer int8   1 = at least one offered limited 5-star is already owned
                       0 = limited 5-stars offered, none owned (the placebo)
                      -1 = no individually identified limited product on offer
    n_ltd      int16   distinct limited 5-stars owned before the occasion

Subgroup features are computed in the LEVEL-6 id space for every model (level-7
tokens are collapsed with ProductVocab7.xlsx), so a level-6 and a level-7 model
assign every occasion to exactly the same subgroup. Ownership before occasion t
uses the pre-window counts plus the shifted obtained stream up to row t, which
is what the model itself could see.

Outputs go to --out-dir as <run tag>.npz. Nothing about individual customers is
printed; only counts.

USAGE (on HPCC, in a GPU job)
    python scripts/eval_per_occasion.py --ckpt <run>/gen5_multistream/hpcc/best.pt \
        --out-dir ~/ProductGPT/work/results/r31
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config5
import dataset_multistream
import model_multistream_state_space
from model_multistream_state_space import build_transformer
from shared.features import load_feature_tensor
from train5_multistream import (
    amp_dtype,
    build_loaders,
    inventory_kwargs,
    pick_device,
    set_seed,
    set_unknown_user_to_mean,
)

CELL = "outsample_users_holdout_period"
N_CLASSES = 9
LTD_LO, LTD_HI = 18, 56          # individually identified limited 5-stars, level-6 ids
LEVEL6_TOP = 60


def uid_hash(u) -> int:
    u = u[0] if isinstance(u, (list, tuple)) else u
    return int.from_bytes(hashlib.sha1(str(u).encode()).digest()[:8], "little", signed=True)


def build_model(cfg, feat, num_users, device):
    arch = cfg.get("arch", "transformer")
    if arch != "transformer":
        from model_recurrent_baseline import build_recurrent_baseline
        return build_recurrent_baseline(
            cell=arch, vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
            d_model=cfg["d_model"], n_layers=cfg["N"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
            feature_tensor=feat, num_users=num_users, use_user_embedding=cfg["use_user_embedding"],
            product_id_embed=cfg.get("product_id_embed", True)).to(device)
    return build_transformer(
        vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
        max_seq_len=cfg["max_events"], d_model=cfg["d_model"], n_layers=cfg["N"],
        n_heads=cfg["num_heads"], d_ff=cfg["d_ff"], dropout=cfg["dropout"], feature_tensor=feat,
        ai_rate=cfg["ai_rate"], num_users=num_users, lto_len=cfg["lto_len"],
        obtained_len=cfg["obtained_len"], prev_dec_len=cfg["prev_dec_len"],
        use_user_embedding=cfg["use_user_embedding"], num_mix_heads=cfg.get("num_mix_heads", 0),
        encoder=cfg.get("encoder", "transformer"),
        use_offer_inventory_attn=cfg.get("use_offer_inventory_attn", True),
        attn_recency_bias=cfg.get("attn_recency_bias", False),
        attn_time_bias=cfg.get("attn_time_bias", "none"),
        product_id_embed=cfg.get("product_id_embed", True),
        time_bias_lag_ipt=cfg.get("time_bias_lag_ipt", True),
        inventory=cfg.get("inventory", "tokens"), sat_layers=cfg.get("sat_layers", 0),
        fuse=cfg.get("fuse", "gate")).to(device)


def collapse_table(cfg) -> torch.Tensor:
    """Native token id -> level-6 token id (identity at level 6)."""
    top = cfg["vocab_size_src"]
    table = torch.arange(max(top, LEVEL6_TOP), dtype=torch.long)
    if int(cfg.get("vocab_level", 6)) == 7:
        tab = pd.read_excel(config5.feature_file_path(cfg))
        for i7, i6 in zip(tab["NewProductIndex7"].astype(int), tab["NewProductIndex6"].astype(int)):
            table[i7] = i6
        last = int(tab["NewProductIndex7"].max())
        table[last + 1], table[last + 2], table[last + 3] = 57, 58, 59
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test only")
    a = ap.parse_args()

    ckpt = Path(a.ckpt)
    tag = ckpt.parent.parent.parent.name.split("_b4_")[-1]
    device = pick_device()
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    for k, v in (("vocab_level", 6), ("first_prod_id", 13), ("last_prod_id", 56), ("unk_prod_id", 59)):
        cfg.setdefault(k, v)
    dataset_multistream.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    model_multistream_state_space.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"],
                                                    cfg["unk_prod_id"])
    print(f"[r31] {tag}: arch={cfg.get('arch')} encoder={cfg.get('encoder')} "
          f"vocab={cfg['vocab_level']} inventory={cfg.get('inventory', 'tokens')} "
          f"cross_attn={cfg.get('use_offer_inventory_attn', True)} epoch={state.get('epoch')}")

    set_seed(cfg["seed"])
    _, _, tests, num_users = build_loaders(cfg)
    feat = load_feature_tensor(config5.feature_file_path(cfg), id_column=cfg.get("feature_id_column"),
                               first_prod_id=cfg["first_prod_id"], last_prod_id=cfg["last_prod_id"],
                               max_token_id=cfg["vocab_size_src"] - 1)
    model = build_model(cfg, feat, num_users, device)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    set_unknown_user_to_mean(model, cfg.get("_trained_user_indices", []))
    model.eval()
    adtype = amp_dtype(device) if cfg.get("amp") else None
    to6 = collapse_table(cfg).to(device)
    first = cfg["first_prod_id"]

    out = {k: [] for k in ("uid_hash", "pos", "label", "nll", "owned_offer", "n_ltd")}
    with torch.no_grad():
        for bi, batch in enumerate(tests[CELL]):
            if a.max_batches and bi >= a.max_batches:
                break
            lto = batch["lto"].to(device)
            obt = batch["obtained"].to(device)
            prev = batch["prev_decision"].to(device)
            uid = batch["user_id"].to(device)
            tgt = batch["label"].to(device)
            ipt = batch["ipt"].to(device) if "ipt" in batch else None
            inv = inventory_kwargs(model, batch, device)
            ctx = (torch.autocast("cuda", dtype=adtype) if adtype is not None
                   else torch.autocast("cpu", enabled=False))
            with ctx:
                logits = model(lto, obt, prev, uid, ipt, **inv)
            p = F.softmax(logits.float()[..., 1:1 + N_CLASSES], dim=-1)          # (B,S,9)

            B, S = tgt.shape
            # ownership before each occasion, in level-6 id space
            obt6 = to6[obt.clamp(0, to6.numel() - 1)]                             # (B,S,10)
            per_row = torch.zeros(B, S, LEVEL6_TOP, device=device)
            per_row.scatter_add_(2, obt6.clamp(0, LEVEL6_TOP - 1),
                                 (obt6 > 0).float())
            owned = per_row.cumsum(dim=1)                                         # rows <= t hold o_(t-1)
            if "inv_init_count" in batch:
                init = batch["inv_init_count"].to(device).float()                 # (B, n_native)
                ids6 = to6[torch.arange(first, first + init.size(1), device=device)]
                init6 = torch.zeros(B, LEVEL6_TOP, device=device)
                init6.scatter_add_(1, ids6.unsqueeze(0).expand(B, -1), init)
                owned = owned + init6.unsqueeze(1)
            held = owned > 0                                                      # (B,S,60)
            lto6 = to6[lto.clamp(0, to6.numel() - 1)]                             # (B,S,4)
            is_ltd = (lto6 >= LTD_LO) & (lto6 <= LTD_HI)
            own_off = torch.gather(held, 2, lto6.clamp(0, LEVEL6_TOP - 1)) & is_ltd
            owned_offer = torch.where(is_ltd.any(-1), own_off.any(-1).to(torch.int8),
                                      torch.full((B, S), -1, dtype=torch.int8, device=device))
            n_ltd = held[..., LTD_LO:LTD_HI + 1].sum(-1)

            mask = (tgt >= 1) & (tgt <= N_CLASSES)
            nll = -torch.log(torch.gather(p, 2, (tgt.clamp(1, N_CLASSES) - 1).unsqueeze(-1))
                             .squeeze(-1).clamp_min(1e-12))
            hashes = torch.tensor([uid_hash(u) for u in batch["uid"]], device=device)
            pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
            out["uid_hash"].append(hashes.unsqueeze(1).expand(B, S)[mask].cpu().numpy())
            out["pos"].append(pos[mask].cpu().numpy().astype(np.int32))
            out["label"].append(tgt[mask].cpu().numpy().astype(np.int8))
            out["nll"].append(nll[mask].cpu().numpy().astype(np.float32))
            out["owned_offer"].append(owned_offer[mask].cpu().numpy())
            out["n_ltd"].append(n_ltd[mask].cpu().numpy().astype(np.int16))

    arr = {k: np.concatenate(v) for k, v in out.items()}
    od = Path(a.out_dir).expanduser()
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(od / f"{tag}.npz", **arr)
    oo = arr["owned_offer"]
    print(f"[r31] {tag}: {len(arr['nll']):,} occasions from {len(np.unique(arr['uid_hash'])):,} customers; "
          f"mean NLL {arr['nll'].mean():.4f}; owned-offer {int((oo == 1).sum()):,} / "
          f"not-owned {int((oo == 0).sum()):,} / no limited offer {int((oo == -1).sum()):,}")
    print(f"[r31] wrote {od / (tag + '.npz')}")


if __name__ == "__main__":
    main()
