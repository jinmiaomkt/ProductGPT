"""
Does a trained time-bias model actually USE its own row's gap? (R21)

scripts/ipt_leak_check.py shows the channel exists in the data: IPT_t carries
0.40 nats about y_t. This asks whether a batch-5 model exploits it. Load one
best.pt trained with the leaky clock, and score the same holdout cells twice:

    as trained  row t's clock includes IPT_t        (what batch 5 reported)
    lagged      row t's clock stops at row t-1      (information legitimately
                                                     available before deciding)

If the lagged NLL collapses to or past the ordinal-ALiBi level (0.886), the
batch-5 gain was the leak. The lagged number is NOT a fair estimate of a
properly trained lagged model -- the weights were fitted to the leaky clock --
so it bounds reliance, not the fixed model's accuracy. That needs retraining.

Aggregate metrics only.

USAGE
    python scripts/ipt_leak_counterfactual.py path/to/best.pt [more.pt ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen5_multistream"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train5_multistream as T  # noqa: E402
from model_multistream_state_space import build_transformer  # noqa: E402
from shared.features import load_feature_tensor  # noqa: E402
import config5  # noqa: E402

CELLS = ("outsample_users_holdout_period", "insample_users_holdout_period")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    device = T.pick_device()
    adtype = T.amp_dtype(device)
    loaders = None
    for p in paths:
        state = torch.load(p, map_location=device, weights_only=False)
        cfg = dict(state["cfg"])
        cfg["batch_size"] = 1
        if loaders is None:
            _, _, loaders, num_users = T.build_loaders(cfg)
        feat = load_feature_tensor(config5.feature_path())
        model = build_transformer(
            vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
            max_seq_len=cfg["max_events"], d_model=cfg["d_model"], n_layers=cfg["N"],
            n_heads=cfg["num_heads"], d_ff=cfg["d_ff"], dropout=cfg["dropout"],
            feature_tensor=feat, ai_rate=cfg["ai_rate"], num_users=state["num_users"],
            lto_len=cfg["lto_len"], obtained_len=cfg["obtained_len"],
            prev_dec_len=cfg["prev_dec_len"], use_user_embedding=cfg["use_user_embedding"],
            num_mix_heads=cfg.get("num_mix_heads", 0),
            encoder=cfg.get("encoder", "transformer"),
            use_offer_inventory_attn=cfg.get("use_offer_inventory_attn", True),
            attn_recency_bias=cfg.get("attn_recency_bias", False),
            attn_time_bias=cfg.get("attn_time_bias", "none"),
            product_id_embed=cfg.get("product_id_embed", True),
        ).to(device)
        model.load_state_dict(state["model_state_dict"])
        print(f"\n=== {p.parent.parent.parent.name if p.parent.name == 'hpcc' else p}"
              f"  (epoch {state['epoch']}, time_bias={cfg.get('attn_time_bias')})")
        for cell in CELLS:
            row = []
            for lag in (False, True):
                model.time_bias_lag_ipt = lag
                with torch.no_grad():
                    m = T.evaluate(model, loaders[cell], device, adtype)
                row.append(m)
            a, b = row
            print(f"  {cell:<32} as-trained nll={a['nll']:.4f} hit={a['hit']:.4f}"
                  f"  | lagged nll={b['nll']:.4f} hit={b['hit']:.4f}"
                  f"  | delta nll={b['nll'] - a['nll']:+.4f}")


if __name__ == "__main__":
    main()
