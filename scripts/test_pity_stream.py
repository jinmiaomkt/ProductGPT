"""R38: correctness and leak checks for the pity-depth stream.

  A. CORRECTNESS  the vectorised cumsum/cummax must equal a brute-force replay.
  B. CAUSALITY    rows after t must not move pity at t, and must not move the
                  logits at t.
  C. NO LEAK      pity depth is a deterministic function of the obtained stream,
                  which the model already has, so it cannot add information the
                  model lacked -- but the project rule (two leaks so far) is that
                  any field produced by the row-generating process is tested
                  against the label before a model consumes it. We check that no
                  pity value determines the label, the way an all-zero obtained
                  block determined NotBuy with probability 1.000 (R20).

Run:  python scripts/test_pity_stream.py
"""
from __future__ import annotations

import os
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
import model_multistream_state_space as mm  # noqa: E402
from model_multistream_state_space import (BANNER_OF_DEC, HARD_PITY4,  # noqa: E402
                                           HARD_PITY5, pity_depth)
from shared.features import load_feature_tensor  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def brute_force(obt, prev, rarity, first, last):
    """Row-by-row replay of the same definition, written independently."""
    B, S, L = obt.shape
    out = np.zeros((B, S, 8), dtype=np.float64)
    for b in range(B):
        cum = [0.0] * 4
        at5 = [None] * 4
        at4 = [None] * 4
        cum_at5 = [0.0] * 4
        cum_at4 = [0.0] * 4
        for t in range(S):
            d = int(prev[b, t])
            ban = BANNER_OF_DEC[d] if 0 <= d <= 9 else -1
            if ban >= 0:
                ids = [int(x) for x in obt[b, t] if first <= int(x) <= last]
                cum[ban] += len(ids)
                rr = [rarity[i] for i in ids]
                if any(r >= 4.5 for r in rr):
                    at5[ban] = t
                    cum_at5[ban] = cum[ban]
                if any(r >= 3.5 for r in rr):
                    at4[ban] = t
                    cum_at4[ban] = cum[ban]
            for bb in range(4):
                p5 = (cum[bb] - cum_at5[bb]) / HARD_PITY5[bb]
                p4 = (cum[bb] - cum_at4[bb]) / HARD_PITY4[bb]
                out[b, t, 2 * bb] = min(max(p5, 0.0), 4.0)
                out[b, t, 2 * bb + 1] = min(max(p4, 0.0), 4.0)
    return out


print("building data ...")
cfg = config5.get_config("pilot")
config5.apply_vocab_level(cfg, 7)
dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
mm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"], cfg["unk_prod_id"])
DATA = pathlib.Path(os.environ["PRODUCTGPT_DATA"])
recs = dsm.load_json_dataset(str(DATA / cfg["data_file"]))[:200]
ds = dsm.TransformerDataset(recs, max_events=256)
items = [ds[i] for i in range(len(ds))]
batch = dsm.collate_multistream(items[:16])
feat = load_feature_tensor(config5.feature_file_path(cfg),
                           id_column=cfg["feature_id_column"],
                           first_prod_id=cfg["first_prod_id"],
                           last_prod_id=cfg["last_prod_id"],
                           max_token_id=cfg["vocab_size_src"] - 1)
rarity = feat[:, 0].numpy()

# ───────────────────────────── A. correctness ─────────────────────────────
print("\nA. the vectorised form equals a brute-force replay")
got = pity_depth(batch["obtained"], batch["prev_decision"], feat[:, 0].float()).numpy()
want = brute_force(batch["obtained"].numpy(), batch["prev_decision"].numpy(),
                   rarity, cfg["first_prod_id"], cfg["last_prod_id"])
err = np.abs(got - want).max()
check("cumsum/cummax matches the replay", err < 1e-5, f"max|d|={err:.2e}")
check("the feature is non-trivial", got.std() > 1e-3 and got.max() > 0.05,
      f"mean={got.mean():.3f} max={got.max():.3f} sd={got.std():.3f}")

# ───────────────────────────── B. causality ───────────────────────────────
print("\nB. causality")
S = batch["label"].shape[1]
t0 = min(60, S - 2)
pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
g = torch.Generator().manual_seed(3)
pert["obtained"][:, t0 + 1:] = torch.randint(13, 131, pert["obtained"][:, t0 + 1:].shape,
                                             generator=g)
pert["prev_decision"][:, t0 + 1:] = torch.randint(1, 10,
                                                  pert["prev_decision"][:, t0 + 1:].shape,
                                                  generator=g)
p2 = pity_depth(pert["obtained"], pert["prev_decision"], feat[:, 0].float()).numpy()
d_at = np.abs(got[:, :t0 + 1] - p2[:, :t0 + 1]).max()
d_after = np.abs(got[:, t0 + 1:] - p2[:, t0 + 1:]).max()
check(f"future rows do not change pity at or before row {t0}", d_at < 1e-6,
      f"max|d|={d_at:.2e}")
check("future rows DO change pity after it (the test can fail)", d_after > 1e-3,
      f"max|d|={d_after:.2e}")

torch.manual_seed(0)
model = mm.build_transformer(
    vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
    max_seq_len=256, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0,
    feature_tensor=feat, ai_rate=cfg["ai_rate"], num_users=4,
    lto_len=cfg["lto_len"], obtained_len=cfg["obtained_len"],
    prev_dec_len=cfg["prev_dec_len"], use_user_embedding=False,
    encoder="gru_attn", fuse="stack", inventory="slots", sat_layers=2,
    pity_stream=True).eval()
with torch.no_grad():
    kw = {"ipt": batch["ipt"]} if batch.get("ipt") is not None else {}
    a = model(batch["lto"], batch["obtained"], batch["prev_decision"], None, **kw)
    kw2 = {"ipt": pert["ipt"]} if pert.get("ipt") is not None else {}
    c = model(pert["lto"], pert["obtained"], pert["prev_decision"], None, **kw2)
    f = lambda x: (x[0] if isinstance(x, (tuple, list)) else x)[:, t0, 1:10]
    dl = (f(a) - f(c)).abs().max().item()
check(f"future rows do not change the logits at row {t0}", dl < 1e-5, f"max|d|={dl:.2e}")

# ───────────────────────────── C. no leak ─────────────────────────────────
print("\nC. the label is not readable off pity depth")
allp, ally = [], []
for it in items:
    p = pity_depth(it["obtained"].unsqueeze(0), it["prev_decision"].unsqueeze(0),
                   feat[:, 0].float())[0].numpy()
    lab = it["label"].numpy()
    live = lab != 0
    allp.append(p[live])
    ally.append(lab[live])
P = np.vstack(allp)
Y = np.concatenate(ally)
print(f"     {len(Y):,} scored rows")

# bin each feature coarsely; look for any cell that pins the label
worst, worst_at = 0.0, ""
for j in range(P.shape[1]):
    b = np.digitize(P[:, j], [0.001, 0.2, 0.4, 0.6, 0.8, 1.0])
    for cell in np.unique(b):
        m = b == cell
        if m.sum() < 200:
            continue
        share = np.bincount(Y[m], minlength=10)[1:10].max() / m.sum()
        if share > worst:
            worst, worst_at = share, f"feature {j}, bin {cell}, n={m.sum():,}"
base = np.bincount(Y, minlength=10)[1:10].max() / len(Y)
check("no pity cell determines the label", worst < 0.995,
      f"max cell share {worst:.3f} ({worst_at}); marginal majority {base:.3f}")
check("pity is informative but not deterministic", worst > base,
      f"{worst:.3f} > {base:.3f}")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for x in FAIL:
        print(f"   FAILED: {x}")
    sys.exit(1)
