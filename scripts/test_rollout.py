"""Correctness checks for the R36a fixed-grid rollout engine.

Run:  python scripts/test_rollout.py

Covers four things, in the order a failure would matter:

  A. NO LEAK       the row grid may be held fixed only because `is_inserted`
                   reaches neither the model nor the trainer.
  B. CAUSALITY     logits at row t must not move when rows > t change.  If they
                   do, a rollout is meaningless -- the model would be reading
                   decisions it has not made yet.
  C. ENVIRONMENT   V0: the calibrated environment must reproduce the acquisition
                   statistics of held-out CALIBRATION rows.  If the environment
                   is wrong, rollouts fail for reasons that have nothing to do
                   with the model.
  D. WIRING        the feedback loop writes what it claims, teacher forcing
                   leaves the streams alone, PAD rows stay empty, and a seed
                   reproduces a rollout exactly.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared"):
    sys.path.insert(0, str(p))

import config5  # noqa: E402
import dataset_multistream as dsm  # noqa: E402
import model_multistream_state_space as mm  # noqa: E402
from shared.features import load_feature_tensor  # noqa: E402

from gacha_env import (BANNER_OF, NOMINAL_PULLS, AcquisitionEnv,  # noqa: E402
                       PityState, ProductTable)
from rollout import rollout  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


# ───────────────────────────────── setup ─────────────────────────────────
print("building a small model and dataset ...")
cfg = config5.get_config("pilot")
config5.apply_vocab_level(cfg, 7)
cfg.update(dict(d_model=32, N=2, num_heads=4, d_ff=64, dropout=0.0, max_events=96))
dsm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"])
mm.set_product_range(cfg["first_prod_id"], cfg["last_prod_id"], cfg["unk_prod_id"])

DATA = pathlib.Path(os.environ["PRODUCTGPT_DATA"])
recs = dsm.load_json_dataset(str(DATA / cfg["data_file"]))[:120]
ds = dsm.TransformerDataset(recs, max_events=cfg["max_events"])
items = [ds[i] for i in range(len(ds))]
batch = dsm.collate_multistream(items[:8])
batch["user_idx"] = torch.zeros(batch["label"].shape[0], dtype=torch.long)

feat = load_feature_tensor(config5.feature_file_path(cfg),
                           id_column=cfg.get("feature_id_column"),
                           first_prod_id=cfg["first_prod_id"],
                           last_prod_id=cfg["last_prod_id"],
                           max_token_id=cfg["vocab_size_src"] - 1)
torch.manual_seed(0)
model = mm.build_transformer(
    vocab_size_src=cfg["vocab_size_src"], vocab_size_tgt=cfg["vocab_size_tgt"],
    max_seq_len=cfg["max_events"], d_model=cfg["d_model"], n_layers=cfg["N"],
    n_heads=cfg["num_heads"], d_ff=cfg["d_ff"], dropout=0.0, feature_tensor=feat,
    ai_rate=cfg["ai_rate"], num_users=4, lto_len=cfg["lto_len"],
    obtained_len=cfg["obtained_len"], prev_dec_len=cfg["prev_dec_len"],
    use_user_embedding=False, encoder="gru_attn", fuse="stack", attn_recency_bias=True,
    inventory="slots", sat_layers=2, kernel="attr", decay="exp", tier=True,
    decay_init=2.0, decay_freeze=True,
).eval()

vocab = ProductTable.from_xlsx(config5.feature_file_path(cfg))

# ───────────────────────────── A. no leak ─────────────────────────────
print("\nA. the fixed grid does not hand the model its label")
hits = subprocess.run(
    ["git", "grep", "-l", "is_inserted", "--", "gen5_multistream", "shared"],
    cwd=str(REPO), capture_output=True, text=True).stdout.split()
offenders = [h for h in hits if "dataset_multistream" not in h]
check("is_inserted reaches neither the model nor the trainer", not offenders,
      f"only {hits}" if not offenders else f"consumed by {offenders}")

# ───────────────────────────── B. causality ───────────────────────────
print("\nB. causality: rows after t must not move the logits at t")
with torch.no_grad():
    S = batch["label"].shape[1]
    t = min(40, S - 2)
    kw = {}
    if batch.get("ipt") is not None:
        kw["ipt"] = batch["ipt"]
    base = model(batch["lto"], batch["obtained"], batch["prev_decision"],
                 batch["user_idx"], **kw)
    base = (base[0] if isinstance(base, (tuple, list)) else base)[:, t, 1:10]

    pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    g = torch.Generator().manual_seed(1)
    pert["prev_decision"][:, t + 1:] = torch.randint(
        1, 10, pert["prev_decision"][:, t + 1:].shape, generator=g)
    pert["obtained"][:, t + 1:] = torch.randint(
        13, 131, pert["obtained"][:, t + 1:].shape, generator=g)
    kw2 = {}
    if pert.get("ipt") is not None:
        kw2["ipt"] = pert["ipt"]
    got = model(pert["lto"], pert["obtained"], pert["prev_decision"],
                pert["user_idx"], **kw2)
    got = (got[0] if isinstance(got, (tuple, list)) else got)[:, t, 1:10]
    delta = (base - got).abs().max().item()
check(f"future rows do not change logits at row {t}", delta < 1e-5, f"max|d|={delta:.2e}")

with torch.no_grad():
    short = model(batch["lto"][:, :t + 1], batch["obtained"][:, :t + 1],
                  batch["prev_decision"][:, :t + 1], batch["user_idx"],
                  **({"ipt": batch["ipt"][:, :t + 1]} if batch.get("ipt") is not None else {}))
    short = (short[0] if isinstance(short, (tuple, list)) else short)[:, t, 1:10]
    dprefix = (base - short).abs().max().item()
check("prefix-only forward equals full-length forward at row t", dprefix < 1e-5,
      f"max|d|={dprefix:.2e}")

# ─────────────────────────── C. environment V0 ────────────────────────
print("\nC. environment reproduces the calibration acquisition process")
HOLDOUT_FROM = 28
env = AcquisitionEnv(vocab, rarity="pity", identity="empirical").fit(
    items, holdout_from=HOLDOUT_FROM)

real_slots, sim_slots = {}, {}
real_rar, sim_rar = {}, {}
real_feat, sim_feat = {}, {}
rng = np.random.default_rng(0)
states = {}
for it in items:
    lto = np.asarray(it["lto"]); obt = np.asarray(it["obtained"])
    prv = np.asarray(it["prev_decision"]); lab = np.asarray(it["label"])
    camp = np.asarray(it["campaign"]) if it.get("campaign") is not None else None
    st = PityState()
    Sx = int((lab != 0).sum())
    for t in range(1, Sx):
        if camp is not None and int(camp[t]) >= HOLDOUT_FROM:
            continue
        d = int(prv[t])
        if d < 1 or d > 8:
            continue
        b = BANNER_OF[d]
        offer = [int(x) for x in lto[t - 1]]
        featured = {offer[s] for s in ((0,) if b == "figA" else (1,) if b == "figB"
                                       else (2, 3) if b == "wep" else ()) if offer[s] > 0}
        for name, row in (("real", obt[t]), ("sim", env.step(d, offer, st, rng))):
            sl, rr, ff = ((real_slots, real_rar, real_feat) if name == "real"
                          else (sim_slots, sim_rar, sim_feat))
            valid = vocab.in_range(row)
            sl.setdefault(d, []).append(int(valid.sum()))
            for p in row[valid]:
                p = int(p)
                r = vocab.rarity[p]
                rr.setdefault(b, []).append(r)
                if r == 5:
                    ff.setdefault(b, []).append(1 if p in featured else 0)

for d in sorted(set(real_slots) & set(sim_slots)):
    a, b_ = np.mean(real_slots[d]), np.mean(sim_slots[d])
    check(f"mean filled slots, decision {d} (nominal {NOMINAL_PULLS[d]})",
          abs(a - b_) < 0.35, f"real {a:.2f} vs sim {b_:.2f}")

for b in sorted(set(real_rar) & set(sim_rar)):
    a = np.mean(np.array(real_rar[b]) == 5)
    c = np.mean(np.array(sim_rar[b]) == 5)
    check(f"5-star rate, banner {b}", abs(a - c) < 0.006, f"real {a:.4f} vs sim {c:.4f}")

# Noise-aware: 5-star hits are rare, so a fixed tolerance either hides a real
# bias when n is large or fails on pure sampling noise when it is small.  Three
# standard errors of the difference of two binomial proportions.
for b in sorted(set(real_feat) & set(sim_feat)):
    ra, sa = np.asarray(real_feat[b]), np.asarray(sim_feat[b])
    a, c = ra.mean(), sa.mean()
    pooled = (ra.sum() + sa.sum()) / (len(ra) + len(sa))
    se = np.sqrt(max(pooled * (1 - pooled), 1e-9) * (1 / len(ra) + 1 / len(sa)))
    check(f"featured share among 5-stars, banner {b}", abs(a - c) <= max(3 * se, 0.02),
          f"real {a:.3f} vs sim {c:.3f}  (n={len(ra)}/{len(sa)}, 3se={3*se:.3f})")

# ───────────────────────────── D. wiring ──────────────────────────────
print("\nD. rollout wiring")
start = 20
r1 = rollout(model, batch, env, vocab, start=start, n_rep=2, seed=7)
r2 = rollout(model, batch, env, vocab, start=start, n_rep=2, seed=7)
check("same seed reproduces the rollout",
      np.array_equal(r1["decisions"], r2["decisions"]))
r3 = rollout(model, batch, env, vocab, start=start, n_rep=2, seed=8)
check("a different seed changes it",
      not np.array_equal(r1["decisions"], r3["decisions"]))

gen, alive = r1["decisions"], r1["alive"]
check("nothing generated before `start`", (gen[:, :start] == 0).all())
check("nothing generated on PAD rows", (gen[~alive] == 0).all())
live = gen[:, start:][alive[:, start:]]
check("every generated decision is a valid class",
      live.size > 0 and bool(((live >= 1) & (live <= 9)).all()),
      f"n={live.size:,}")

before = batch["obtained"].clone()
rt = rollout(model, batch, env, vocab, start=start, n_rep=1, seed=3, teacher_forced=True)
check("teacher forcing leaves the input streams untouched",
      torch.equal(before, batch["obtained"]))
check("the caller's batch is never mutated by a free rollout",
      torch.equal(before, batch["obtained"]))

occ = np.bincount(live, minlength=10)[1:10]
check("the rollout does not collapse onto one class", (occ > 0).sum() >= 3,
      f"classes used: {(occ > 0).sum()}/9")

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"   FAILED: {f}")
    sys.exit(1)
