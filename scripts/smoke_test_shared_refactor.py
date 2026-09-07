"""
Smoke test for the shared/ refactor (see the "Extract duplicated model blocks
into shared/" commit).

This does NOT need real data or a real checkpoint. It builds each of the six
refactored models with small random dimensions, runs one forward pass, and
checks a backward pass produces gradients. That proves the pieces pulled out
into shared/ (embeddings, attention, decoder stack, projection) actually wire
together end-to-end after the refactor -- something a static diff can't show.

It does NOT prove the refactor preserved exact numerics against a
pre-refactor checkpoint (no real checkpoint exists on this machine to compare
against). Run this from the repo root with env.ps1 / PYTHONPATH set:

    python scripts\\smoke_test_shared_refactor.py
"""
from __future__ import annotations

import sys
import traceback

import torch

VOCAB_SRC = 68
VOCAB_TGT = 18
D_MODEL = 32
N_LAYERS = 2
N_HEADS = 4
D_FF = 64
NB_FEATURES = 8
DROPOUT = 0.1
FEATURE_DIM = 34
SEQ_LEN = 30
BATCH = 2


def fake_feature_tensor():
    return torch.randn(VOCAB_SRC, FEATURE_DIM)


def run_one(name, build_fn, kwargs, expected_vocab_out):
    """
    expected_vocab_out: the projection width this variant should produce.
    - two-vocab (src/tgt split) models project to VOCAB_TGT (18).
    - single-vocab "index" models share one vocab for embedding AND
      projection, so they correctly project to VOCAB_SRC (68), not 18.
      That's real architecture, not a bug -- see model4_decoderonly_index_performer.py
      lines ~43 (InputEmbeddings) and ~59 (ProjectionLayer), both given
      the same `vocab_size`.
    """
    print(f"--- {name} ---")
    try:
        model = build_fn(**kwargs)

        x = torch.randint(0, VOCAB_SRC, (BATCH, SEQ_LEN))
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out

        expected_shape = (BATCH, SEQ_LEN, expected_vocab_out)
        assert logits.shape == expected_shape, (
            f"unexpected output shape {logits.shape}, expected {expected_shape}"
        )

        loss = logits.sum()
        loss.backward()

        n_params = sum(p.numel() for p in model.parameters())
        n_with_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is not None
        )
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)

        print(f"    output shape OK: {tuple(logits.shape)}")
        print(f"    backward OK: {n_with_grad}/{n_trainable} trainable params got a gradient")
        print(f"    total params: {n_params:,}")
        print("    PASS")
        return True
    except Exception:
        print("    FAIL")
        traceback.print_exc()
        return False


def main():
    results = {}

    special_ids = torch.tensor([0, 10, 11, 12, 57, 58])

    # (vocab_size_src, vocab_size_tgt) style: feature-based, non-mixture.
    two_vocab_feature = dict(
        vocab_size_src=VOCAB_SRC, vocab_size_tgt=VOCAB_TGT,
        max_seq_len=SEQ_LEN, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT, nb_features=NB_FEATURES,
        feature_tensor=fake_feature_tensor(), special_token_ids=special_ids,
    )

    # single `vocab_size` style: index-only, no feature branch.
    single_vocab_index = dict(
        vocab_size=VOCAB_SRC, max_seq_len=SEQ_LEN, d_model=D_MODEL,
        n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT,
        nb_features=NB_FEATURES,
    )

    # mixture variants additionally require num_users.
    mixture_feature = dict(two_vocab_feature, num_users=5)

    from model4_decoderonly_feature_performer import build_transformer as bt1
    results["model4_decoderonly_feature_performer"] = run_one(
        "model4_decoderonly_feature_performer", bt1, two_vocab_feature, VOCAB_TGT
    )

    from model4_decoderonly_index_performer import build_transformer as bt2
    results["model4_decoderonly_index_performer"] = run_one(
        "model4_decoderonly_index_performer", bt2, single_vocab_index, VOCAB_SRC
    )

    from model4_mixture_decoderonly_feature_performer import build_transformer as bt3
    results["model4_mixture_decoderonly_feature_performer"] = run_one(
        "model4_mixture_decoderonly_feature_performer", bt3, mixture_feature, VOCAB_TGT
    )

    from model4_mixture2_decoderonly_feature_performer import build_transformer as bt4
    results["model4_mixture2_decoderonly_feature_performer"] = run_one(
        "model4_mixture2_decoderonly_feature_performer", bt4, mixture_feature, VOCAB_TGT
    )

    from model2_decoderonly_feature_performer import build_transformer as bt5
    results["model2_decoderonly_feature_performer"] = run_one(
        "model2_decoderonly_feature_performer", bt5, two_vocab_feature, VOCAB_TGT
    )

    from model2_decoderonly_index_performer import build_transformer as bt6
    results["model2_decoderonly_index_performer"] = run_one(
        "model2_decoderonly_index_performer", bt6, single_vocab_index, VOCAB_SRC
    )

    print("\n=== SUMMARY ===")
    n_pass = sum(results.values())
    n_total = len(results)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n{n_pass}/{n_total} passed")

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
