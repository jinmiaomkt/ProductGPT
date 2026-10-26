"""The occasion table: exactly the rows, window and partition gen 5 scores.

`build_loaders` (gen5_multistream/train5_multistream.py) under split_mode=both
does four things this module repeats line for line:

  1. builds ONE TransformerDataset over every record, in JSON record order,
     with max_events=1024, truncate="tail" and shift_obtained=True;
  2. shuffles range(n_users) with random.Random(split_seed=33);
  3. takes the first int(0.5 * n) as out-of-sample customers;
  4. under val_mode="late" trains in-sample customers on campaigns < 27 and
     validates them on campaign 27, and scores three test cells.

The uid files in eval_inputs/gen5 are NOT used: under split_mode=both the
trainer never reads them, so using them here would score different customers.

Nothing is printed except counts and a checksum.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# The cell counts of b17_gru_v6_s1 (final.json, read 26 Oct 2026). Gate G1.
REFERENCE_CELLS = {
    "insample_users_holdout_period": 222_889,
    "outsample_users_calib_period": 1_259_900,
    "outsample_users_holdout_period": 217_303,
}
CELLS = ("insample_users_holdout_period", "outsample_users_calib_period",
         "outsample_users_holdout_period")

DATA_FILE = "clean_list_int_wide4_simple6_IPT.json"


def uid_hash(uid: str) -> int:
    """Stable 63-bit hash of a customer id, so models can be paired. Never printed."""
    return int.from_bytes(hashlib.sha1(uid.encode("utf-8")).digest()[:8], "little") >> 1


def export(out_path: Path, *, max_events: int = 1024, split_seed: int = 33,
           user_holdout_frac: float = 0.5, val_from: int = 27,
           holdout_from: int = 28, max_users: Optional[int] = None) -> Dict[str, int]:
    import paths
    from dataset_multistream import TransformerDataset, load_json_dataset

    raw = load_json_dataset(str(paths.data_file(DATA_FILE)))
    if max_users:
        raw = raw[:max_users]
    base = TransformerDataset(raw, ai_rate=15, lto_len=4, obtained_len=10,
                              prev_dec_len=1, max_events=max_events,
                              base_seed=1, shift_obtained=True, truncate="tail")
    if not (base.has_holdout and base.has_campaign and base.has_ipt):
        raise SystemExit("data file lacks holdout flags, CampaignID or IPT")
    n = len(base)

    order = list(range(n))
    random.Random(split_seed).shuffle(order)
    n_out = max(1, int(user_holdout_frac * n))
    is_out = np.zeros(n, dtype=np.int8)
    is_out[order[:n_out]] = 1

    S = max(int(base._label[i].numel()) for i in range(n))
    lab = np.zeros((n, S), np.int8)
    prev = np.zeros((n, S), np.int8)
    lto = np.zeros((n, S, 4), np.int16)
    obt = np.zeros((n, S, 10), np.int16)
    ipt = np.zeros((n, S), np.float32)
    camp = np.zeros((n, S), np.int8)
    hflag = np.zeros((n, S), np.int8)
    length = np.zeros(n, np.int32)
    inv0 = np.zeros((n, base._inv_count[0].numel()), np.float32)
    uh = np.zeros(n, np.int64)
    for i in range(n):
        L = int(base._label[i].numel())
        length[i] = L
        lab[i, :L] = base._label[i].numpy()
        prev[i, :L] = base._prev[i].numpy()
        lto[i, :L] = base._lto[i].numpy()
        obt[i, :L] = base._obt[i].numpy()
        ipt[i, :L] = base._ipt[i].numpy()
        camp[i, :L] = base._camp[i].numpy()
        hflag[i, :L] = base._hf[i].numpy()
        inv0[i] = base._inv_count[i].numpy()
        uh[i] = uid_hash(base._uid_cache[i])

    valid = (lab >= 1) & (lab <= 9)
    hol = hflag == 1
    out = is_out[:, None] == 1
    roles = {
        "train": valid & ~out & ~hol & (camp < val_from),
        "val": valid & ~out & ~hol & (camp >= val_from),
        "insample_users_holdout_period": valid & ~out & hol,
        "outsample_users_calib_period": valid & out & ~hol,
        "outsample_users_holdout_period": valid & out & hol,
    }
    # campaign vs flag consistency: the feature flag IS campaign >= 28
    bad_flag = int(((camp >= holdout_from) != hol)[valid].sum())
    counts = {k: int(v.sum()) for k, v in roles.items()}
    counts["users"] = n
    counts["out_users"] = int(is_out.sum())
    counts["rows"] = int(valid.sum())
    counts["flag_campaign_mismatch"] = bad_flag

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, label=lab, prev=prev, lto=lto, obtained=obt,
                        ipt=ipt, campaign=camp, holdout=hflag, length=length,
                        is_out=is_out, inv_init=inv0, uid_hash=uh)
    meta = dict(counts=counts, max_events=max_events, split_seed=split_seed,
                user_holdout_frac=user_holdout_frac, val_from=val_from,
                holdout_from=holdout_from, data_file=DATA_FILE,
                checksum=hashlib.sha1(uh.tobytes() + is_out.tobytes()).hexdigest()[:16])
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return counts


def load(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def roles(tab: Dict[str, np.ndarray], *, val_from: int = 27) -> Dict[str, np.ndarray]:
    """Boolean (N, S) masks for the fit, selection and test roles."""
    lab, camp = tab["label"], tab["campaign"]
    valid = (lab >= 1) & (lab <= 9)
    hol = tab["holdout"] == 1
    out = tab["is_out"][:, None] == 1
    return {
        "valid": valid,
        "train": valid & ~out & ~hol & (camp < val_from),
        "val": valid & ~out & ~hol & (camp >= val_from),
        "insample_users_holdout_period": valid & ~out & hol,
        "outsample_users_calib_period": valid & out & ~hol,
        "outsample_users_holdout_period": valid & out & hol,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="R46: export the occasion table (G1)")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--max-users", type=int, default=None,
                    help="smoke only: first N records (G1 is then not checked)")
    a = ap.parse_args()
    counts = export(Path(a.out), max_users=a.max_users)
    print(json.dumps(counts, indent=2))
    if a.max_users:
        print("[G1] not checked (--max-users)")
        return
    ok = all(counts[k] == v for k, v in REFERENCE_CELLS.items())
    for k, v in REFERENCE_CELLS.items():
        print(f"[G1] {k:<32} {counts[k]:>10,} vs reference {v:>10,}  "
              f"{'OK' if counts[k] == v else 'MISMATCH'}")
    print(f"[G1] {'PASS' if ok and counts['flag_campaign_mismatch'] == 0 else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
