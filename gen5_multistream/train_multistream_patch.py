"""
SUPERSEDED (Sep 2026) -- kept only for reference.

This was a patch guide: a set of blocks to copy by hand into one of the gen-4
trainers. That is no longer necessary. gen5_multistream/train5_multistream.py
is now a real, runnable trainer for this model:

    python gen5_multistream/train5_multistream.py --profile pilot

Differences worth knowing if you read the blocks below:
  - The real trainer uses plain PyTorch, NOT DeepSpeed, so it runs on Windows
    as well as on HPCC. The gen-4 trainers this guide targeted are DeepSpeed-only.
  - dataset_multistream.py no longer needs LTO_ObtainedProducts /
    LTO_PreviousDecision; it derives those streams from AggregateInput, so it
    works on the IPT files that dropped them.
  - projection_gate_mode is not used by MultiStreamStateSpaceTransformer; it is
    accepted and ignored for backwards compatibility. The blocks below pass it.

Do not copy from this file. Read train5_multistream.py instead.
"""

# ---------------------------------------------------------------------
# 1. Replace the model import near the top of your training script.
# ---------------------------------------------------------------------

# OLD:
# from model4_mixture2_decoderonly_feature_performer import build_transformer

# NEW:
from model_multistream_state_space import build_transformer


# ---------------------------------------------------------------------
# 2. Replace build_model.
# ---------------------------------------------------------------------

def build_model(cfg, feat_tensor):
    return build_transformer(
        vocab_size_tgt=cfg["vocab_size_tgt"],
        vocab_size_src=cfg["vocab_size_src"],
        max_seq_len=cfg["seq_len_ai"],
        ai_rate=cfg["ai_rate"],
        d_model=cfg["d_model"],
        n_layers=cfg["N"],
        n_heads=cfg["num_heads"],
        num_users=cfg["num_users"],
        dropout=cfg["dropout"],
        nb_features=cfg.get("nb_features", None),
        kernel_type=cfg.get("kernel_type", "exp"),
        d_ff=cfg["d_ff"],
        feature_tensor=feat_tensor,
        special_token_ids=SPECIAL_IDS,
        projection_mix_space="logit",
    )


# ---------------------------------------------------------------------
# 3. In __main__, change the default projection space.
# ---------------------------------------------------------------------

# OLD:
# cfg.setdefault("projection_mix_space", "prob")

# NEW:
# cfg.setdefault("projection_mix_space", "logit")


# ---------------------------------------------------------------------
# 4. Replace the batch forward block inside the training loop.
# ---------------------------------------------------------------------

"""
OLD pattern:

    x = batch["aggregate_input"].to(device)
    tgt_full = batch["label"].to(device)
    u = batch["user_id"].to(device) if "user_id" in batch else None
    pos = torch.arange(cfg["ai_rate"] - 1, x.size(1), cfg["ai_rate"], device=device)

    raw_full = engine(x, u, projection_gate_mode=train_proj_gate_mode) ...

    logits_like_full = _to_logits_like(raw_full, mix_space=mix_space)

    if logits_like_full.size(1) == x.size(1):
        logits = logits_like_full[:, pos, :]
    else:
        logits = logits_like_full

    tgt = _align_labels_to_slots(...)
"""

# NEW pattern:

def training_step_block(engine, batch, device, loss_fn, train_proj_gate_mode, PAD_ID):
    lto = batch["lto"].to(device)                         # (B,S,4)
    obtained = batch["obtained"].to(device)               # (B,S,10)
    prev_dec = batch["prev_decision"].to(device)          # (B,S)
    tgt = batch["label"].to(device)                       # (B,S)
    u = batch["user_id"].to(device) if "user_id" in batch else None

    raw = engine(
        lto,
        obtained,
        prev_dec,
        u,
        projection_gate_mode=train_proj_gate_mode,
    )

    logits = raw                                            # model returns logits: (B,S,V)

    if not (tgt != PAD_ID).any():
        return None

    loss = loss_fn(logits, tgt)
    engine.backward(loss)
    engine.step()
    return loss.detach()


# ---------------------------------------------------------------------
# 5. Replace collect_val_logits.
# ---------------------------------------------------------------------

@torch.no_grad()
def collect_val_logits(val_loader, model, device, ai_rate=None, projection_gate_mode="mean"):
    model.eval()
    logits_chunks, label_chunks = [], []

    for batch in val_loader:
        lto = batch["lto"].to(device)
        obtained = batch["obtained"].to(device)
        prev_dec = batch["prev_decision"].to(device)
        tgt = batch["label"].to(device)
        u = batch["user_id"].to(device) if "user_id" in batch else None

        logits = model(
            lto,
            obtained,
            prev_dec,
            u,
            projection_gate_mode=projection_gate_mode,
        )

        logits_dec = logits[..., 1:10]                     # decisions 1..9
        mask = (tgt >= 1) & (tgt <= 9)
        if mask.sum() == 0:
            continue

        logits_chunks.append(logits_dec[mask])
        label_chunks.append(tgt[mask].long())

    return torch.cat(logits_chunks), torch.cat(label_chunks)


# ---------------------------------------------------------------------
# 6. Replace the model forward inside evaluate().
# Keep your metric code, but change the data extraction and forward pass.
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_forward_block(batch, model, dev, projection_gate_mode="mean"):
    lto = batch["lto"].to(dev)
    obtained = batch["obtained"].to(dev)
    prev_dec = batch["prev_decision"].to(dev)
    tgt = batch["label"].to(dev)
    u = batch["user_id"].to(dev) if "user_id" in batch else None

    logits = model(
        lto,
        obtained,
        prev_dec,
        u,
        projection_gate_mode=projection_gate_mode,
    )

    logits_dec = logits[..., 1:10]
    prob_dec = F.softmax(logits_dec, dim=-1)
    mask = (tgt >= 1) & (tgt <= 9)

    return logits, logits_dec, prob_dec, tgt, mask


# ---------------------------------------------------------------------
# 7. Replace the inference forward block.
# ---------------------------------------------------------------------

@torch.no_grad()
def infer_forward_block(batch, engine, device, calibrator=None):
    lto = batch["lto"].to(device)
    obtained = batch["obtained"].to(device)
    prev_dec = batch["prev_decision"].to(device)
    uids = batch["uid"]
    u = batch["user_id"].to(device) if "user_id" in batch else None

    logits = engine(
        lto,
        obtained,
        prev_dec,
        u,
        projection_gate_mode="mean",
    )

    logits_dec = logits[..., 1:10]

    if calibrator is not None:
        probs_dec = calibrator(logits_dec)
    else:
        probs_dec = F.softmax(logits_dec, dim=-1)

    return uids, probs_dec


# ---------------------------------------------------------------------
# 8. Optional diagnostic: get attention for interpretation.
# ---------------------------------------------------------------------

@torch.no_grad()
def extract_offer_inventory_attention(model, batch, device):
    """
    Returns:
        logits:   (B,S,V)
        sat_attn: (B,S,4,S*10)

    sat_attn[b,t,l,m] tells which historical obtained-product token m the
    current LTO slot l attends to when predicting event t.
    """
    model.eval()

    lto = batch["lto"].to(device)
    obtained = batch["obtained"].to(device)
    prev_dec = batch["prev_decision"].to(device)
    u = batch["user_id"].to(device) if "user_id" in batch else None

    logits, sat_attn = model(
        lto,
        obtained,
        prev_dec,
        u,
        projection_gate_mode="mean",
        return_attention=True,
    )

    return logits, sat_attn
