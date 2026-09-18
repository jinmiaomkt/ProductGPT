"""
Correctness checks for the per-product vocabulary (R28, vocab level 7).

Level 6 gives 118 products only 44 tokens: every 3-star weapon shares one id,
as do the 4-stars and the standard 5-stars. Level 7 gives each product its own
id (13-130) and its own embedding row. This test checks the switch end to end
WITHOUT touching customer data:

  1. the vocabulary table is contiguous, complete and numeric;
  2. config5.apply_vocab_level rewires ids, vocab size, data and feature files;
  3. the feature tensor has a filled row for every product id and zeros elsewhere;
  4. both models build and run a forward pass at level 7, with the inventory
     tokens and slots paths;
  5. additive_inventory counts in the wider id range;
  6. level 6 still behaves exactly as before (no silent regression).

USAGE
    python scripts/test_vocab7.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gen5_multistream"))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
import model_multistream_state_space as mss  # noqa: E402
from model_multistream_state_space import build_transformer  # noqa: E402
from model_recurrent_baseline import build_recurrent_baseline  # noqa: E402
from shared.features import FEATURE_COLS, load_feature_tensor  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def main() -> None:
    cfg = config5.get_config("pilot")
    config5.apply_vocab_level(cfg, 7)
    feat_path = config5.feature_file_path(cfg)
    tab = pd.read_excel(feat_path)

    print("1. vocabulary table")
    ids = sorted(int(i) for i in tab["NewProductIndex7"])
    check("ids contiguous from 13", ids == list(range(13, 13 + len(ids))), f"{len(ids)} products, 13-{ids[-1]}")
    check("one row per product id", len(set(ids)) == len(ids))
    check("all feature cells numeric",
          int(tab[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").isna().sum().sum()) == 0)
    check("last product id matches config", ids[-1] == cfg["last_prod_id"], f"{ids[-1]} vs {cfg['last_prod_id']}")

    print("2. config")
    check("vocab size above unk id", cfg["vocab_size_src"] > cfg["unk_prod_id"])
    check("data file switched", "simple7" in cfg["data_file"], cfg["data_file"])
    check("feature id column switched", cfg["feature_id_column"] == "NewProductIndex7")

    print("3. feature tensor")
    feat = load_feature_tensor(feat_path, id_column=cfg["feature_id_column"],
                              first_prod_id=cfg["first_prod_id"], last_prod_id=cfg["last_prod_id"],
                              max_token_id=cfg["vocab_size_src"] - 1)
    filled = [i for i in range(feat.size(0)) if bool(feat[i].abs().sum() > 0)]
    check("shape (vocab, 34)", tuple(feat.shape) == (cfg["vocab_size_src"], len(FEATURE_COLS)), str(tuple(feat.shape)))
    check("every product row filled", filled and min(filled) >= cfg["first_prod_id"] and max(filled) <= cfg["last_prod_id"],
          f"{len(filled)} non-zero rows")
    check("special rows all zero", bool(feat[:cfg["first_prod_id"]].abs().sum() == 0)
          and bool(feat[cfg["last_prod_id"] + 1:].abs().sum() == 0))

    print("4. models build and run at level 7")
    dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
    mss.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"], cfg["unk_prod_id"])
    B, S, d = 2, 6, 32
    lto = torch.randint(cfg["first_prod_id"], cfg["last_prod_id"] + 1, (B, S, 4))
    obt = torch.randint(cfg["first_prod_id"], cfg["last_prod_id"] + 1, (B, S, 10))
    prev = torch.randint(1, 10, (B, S))
    common = dict(vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
                  max_seq_len=S, d_model=d, n_layers=1, n_heads=4, d_ff=64, dropout=0.0,
                  feature_tensor=feat)
    for name, inv in (("transformer, inventory=tokens", "tokens"), ("transformer, inventory=slots", "slots")):
        m = build_transformer(**common, inventory=inv)
        kw = {}
        if inv == "slots":
            n_prod = cfg["last_prod_id"] - cfg["first_prod_id"] + 1
            kw = dict(inv_init_count=torch.zeros(B, n_prod), inv_init_last=torch.full((B, n_prod), -1e6))
        out = m(lto, obt, prev, **kw)
        check(name, out.shape == (B, S, cfg["vocab_size_tgt"]) and bool(torch.isfinite(out).all()), str(tuple(out.shape)))
    gru = build_recurrent_baseline("gru", vocab_size_src=cfg["vocab_size_src"],
                                   vocab_size_tgt=cfg["vocab_size_tgt"], d_model=d, n_layers=1,
                                   d_ff=64, dropout=0.0, feature_tensor=feat)
    out = gru(lto, obt, prev)
    check("gru baseline", out.shape == (B, S, cfg["vocab_size_tgt"]) and bool(torch.isfinite(out).all()))
    check("embedding covers the widest id", gru.product_embed.id_embed.num_embeddings > cfg["last_prod_id"])

    print("5. additive inventory over the wider range")
    ids_t = torch.full((1, 3, 10), 0, dtype=torch.long)
    ids_t[0, 0, 0] = cfg["last_prod_id"]        # highest product id
    ids_t[0, 1, 0] = cfg["last_prod_id"]
    ids_t[0, 2, 0] = cfg["first_prod_id"]
    count, since = mss.additive_inventory(ids_t)
    n_prod = cfg["last_prod_id"] - cfg["first_prod_id"] + 1
    check("counts the highest product id", count.shape[-1] == n_prod and float(count[0, 2, -1]) == 2.0,
          f"count={float(count[0, 2, -1])}")
    check("counts the lowest product id", float(count[0, 2, 0]) == 1.0)

    print("6. level 6 unchanged")
    cfg6 = config5.apply_vocab_level(config5.get_config("pilot"), 6)
    dsm.set_product_range(cfg6["first_prod_id"], cfg6["last_prod_id"])
    mss.set_product_range(cfg6["first_prod_id"], cfg6["last_prod_id"], cfg6["unk_prod_id"])
    feat6 = load_feature_tensor(config5.feature_file_path(cfg6))
    check("level 6 ids 13-56, unk 59",
          (cfg6["first_prod_id"], cfg6["last_prod_id"], cfg6["unk_prod_id"]) == (13, 56, 59))
    check("level 6 feature tensor is (60, 34)", tuple(feat6.shape) == (60, 34), str(tuple(feat6.shape)))
    check("level 6 data file", cfg6["data_file"].endswith("simple6_IPT.json"), cfg6["data_file"])
    check("module constants restored", (mss.FIRST_PROD_ID, mss.LAST_PROD_ID, mss.N_PRODUCTS) == (13, 56, 44))

    print()
    if fails:
        print(f"FAILED: {len(fails)} check(s): {fails}")
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
