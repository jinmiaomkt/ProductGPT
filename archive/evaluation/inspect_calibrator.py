#!/usr/bin/env python3
"""
Inspect a calibrator checkpoint to understand its structure.

Usage:
    python3 inspect_calibrator.py --ckpt /path/to/calibrator_....pt
"""
import argparse
import torch
import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to calibrator .pt file")
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")

    print("=" * 60)
    print(f"Calibrator checkpoint: {args.ckpt}")
    print("=" * 60)

    # Case 1: it's a raw dict (state_dict or config dict)
    if isinstance(state, dict):
        print(f"\nTop-level type: dict")
        print(f"Top-level keys: {list(state.keys())}\n")

        for k, v in state.items():
            if torch.is_tensor(v):
                print(f"  '{k}': Tensor shape={v.shape} dtype={v.dtype} value={v}")
            elif isinstance(v, dict):
                print(f"  '{k}': dict with keys={list(v.keys())}")
                for k2, v2 in v.items():
                    if torch.is_tensor(v2):
                        print(f"      '{k2}': Tensor shape={v2.shape} dtype={v2.dtype} value={v2}")
                    else:
                        print(f"      '{k2}': {type(v2).__name__} = {v2}")
            else:
                print(f"  '{k}': {type(v).__name__} = {v}")

    # Case 2: it's an nn.Module saved directly
    elif hasattr(state, 'state_dict'):
        print(f"\nTop-level type: {type(state).__name__} (nn.Module)")
        sd = state.state_dict()
        print(f"state_dict keys: {list(sd.keys())}")
        for k, v in sd.items():
            print(f"  '{k}': Tensor shape={v.shape} dtype={v.dtype} value={v}")

        # Check for common calibrator attributes
        for attr in ['temperature', 'temp', 'scale', 'bias', 'weight']:
            if hasattr(state, attr):
                val = getattr(state, attr)
                print(f"\nAttribute '{attr}': {val}")

    else:
        print(f"\nTop-level type: {type(state).__name__}")
        print(f"Value: {state}")

    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)

    # Try to identify the calibrator type
    if isinstance(state, dict):
        keys_lower = {k.lower() for k in state.keys()}

        if 'temperature' in keys_lower or 'temp' in keys_lower:
            print("  -> Likely TEMPERATURE SCALING calibrator")
            print("     Use: --calibrator_type temperature")

        if 'model_state_dict' in state:
            sd = state['model_state_dict']
            sd_keys = list(sd.keys())
            print(f"  -> Has 'model_state_dict' with keys: {sd_keys}")
            if any('temperature' in k.lower() for k in sd_keys):
                print("     Use: --calibrator_type temperature")
            elif any('weight' in k.lower() for k in sd_keys) and any('bias' in k.lower() for k in sd_keys):
                print("     Use: --calibrator_type platt")
            elif any('scale' in k.lower() for k in sd_keys):
                print("     Use: --calibrator_type vector")

        if any('weight' in k.lower() for k in state.keys()) and any('bias' in k.lower() for k in state.keys()):
            print("  -> Likely PLATT SCALING calibrator (weight + bias)")
            print("     Use: --calibrator_type platt")

    print("\nRun this script first, then use the output to set --calibrator_type")
    print("in the modified inference script.")


if __name__ == "__main__":
    main()