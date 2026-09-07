"""
Find the largest event window (S) the gen-5 model can train at on this GPU.

The offer-inventory cross-attention builds a (B, H, S, lto_len, S*obtained_len)
score tensor, so memory grows with S**2. This runs one real forward+backward at
each setting with synthetic tensors and reports peak allocation, so the ceiling
is measured rather than guessed.

    python gen5_multistream/probe_memory.py
    python gen5_multistream/probe_memory.py --d-model 128 --heads 8 --layers 4
"""
from __future__ import annotations

import argparse

import torch

import config5
from model_multistream_state_space import build_transformer
from shared.features import load_feature_tensor


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} TB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--events", type=int, nargs="*",
                    default=[64, 128, 256, 384, 512, 768, 1024])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device; nothing to probe")
        return

    dev = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {torch.cuda.get_device_name(0)}  total={human(total)}")
    print(f"model: d_model={args.d_model} heads={args.heads} layers={args.layers} "
          f"d_ff={args.d_ff} batch={args.batch}\n")

    feat = load_feature_tensor(config5.feature_path())
    cfg = config5.get_config("pilot")
    bf16 = torch.cuda.is_bf16_supported()

    print(f"{'events S':>9}  {'peak GPU':>11}  {'% of card':>9}  status")
    print("-" * 48)

    for S in args.events:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_transformer(
                vocab_size_src=cfg["vocab_size_src"],
                vocab_size_tgt=cfg["vocab_size_tgt"],
                max_seq_len=S,
                d_model=args.d_model, n_layers=args.layers,
                n_heads=args.heads, d_ff=args.d_ff, dropout=0.1,
                feature_tensor=feat, ai_rate=cfg["ai_rate"],
                num_users=64,
                lto_len=cfg["lto_len"], obtained_len=cfg["obtained_len"],
                prev_dec_len=cfg["prev_dec_len"],
            ).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

            B = args.batch
            lto = torch.randint(13, 57, (B, S, cfg["lto_len"]), device=dev)
            obt = torch.randint(13, 57, (B, S, cfg["obtained_len"]), device=dev)
            prev = torch.randint(1, 10, (B, S), device=dev)
            uid = torch.randint(0, 64, (B,), device=dev)
            tgt = torch.randint(1, 10, (B, S), device=dev)

            ctx = torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16)
            with ctx:
                logits = model(lto, obt, prev, uid)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            peak = torch.cuda.max_memory_allocated()
            print(f"{S:>9}  {human(peak):>11}  {100.0*peak/total:>8.1f}%  ok")
            del model, opt, lto, obt, prev, uid, tgt, logits, loss
        except torch.cuda.OutOfMemoryError:
            print(f"{S:>9}  {'--':>11}  {'--':>9}  OUT OF MEMORY")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{S:>9}  {'--':>11}  {'--':>9}  OUT OF MEMORY")
                break
            raise
        finally:
            torch.cuda.empty_cache()

    print("\nNote: peak scales ~S**2. Pick max_events with headroom -- real batches "
          "vary in length and the optimiser state grows with model size.")


if __name__ == "__main__":
    main()
