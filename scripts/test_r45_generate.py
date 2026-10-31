"""
Tests for R45 step 2, generation on chopped histories (scripts/r45_generate.py,
the censoring flag in the gen-6 generation path, rollout's pity_from). CPU only,
synthetic customers.

  G1  the flag reaches generation: with non-zero flag parameters, the streaming
      rollout (advance from init_stream) draws exactly what the full-prefix loop
      (stream_forward) draws, free running and teacher forced, in a batch that
      mixes censored and uncensored customers; zeroing the flag changes the draws.
  G2  a model without the flag ignores it: identical draws with and without the
      batch's `censored` field.
  G3  pity_from: the environment is warmed on the full history's real prefix,
      call for call, while the model reads the chopped one; a batch whose live
      rows differ is refused.
  G4  unchop: at k = 0 the chop path reproduces the plain rollout draw for draw;
      at k = 2 every generated row lands on a live row of the full history, the
      real rows before each holdout are untouched and nothing is lost.
  G5  training after the init_stream refactor: censor_vec and censor_h0 get
      finite, non-zero gradients through stream_forward.

    python scripts/test_r45_generate.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "simulation",
          REPO / "shared", REPO / "scripts"):
    sys.path.insert(0, str(p))

import rollout as ro  # noqa: E402
from censor6 import ChopView, collate_prefix  # noqa: E402
from dataset_multistream import collate_multistream  # noqa: E402
from r45_generate import chop_batch, unchop  # noqa: E402
from test_r45_censor import item, model  # noqa: E402

FAIL = []
torch.set_num_threads(1)
HOLD = 28           # first holdout campaign, as in the data
VAL = 27


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAIL.append(name)


class ToyEnv:
    """Deterministic given the rng: what each decision yields. Records every
    warm-up call, in order."""

    def __init__(self):
        self.seen = []

    def step(self, y, offer, state, rng):
        n = 0 if y == 9 else (10 if y % 2 == 0 else 1)
        out = np.zeros(10, dtype=np.int64)
        if n:
            out[:n] = rng.integers(13, 57, n)
        return out

    def observe(self, prev, offer, obt, state):
        self.seen.append((int(prev), tuple(int(x) for x in obt)))


def customers():
    """Three histories of different lengths reaching campaign 28 (campaign runs of
    2 rows from campaign 3: campaign 27 starts at row 48, 28 at row 50), scored
    on the holdout only, as in the test cell."""
    out = []
    for S, seed in ((64, 1), (58, 2), (70, 3)):
        it = item(S, seed, camp_len=2)
        lab = it["label"].clone()
        lab[it["campaign"] < HOLD] = 0
        it["label"] = lab
        out.append(it)
    return out


def chopped(items, ks):
    """One customer per entry of ks, each chopped at its own depth."""
    views = [ChopView([it], "calendar" if k else "none", k, "zero", VAL)[0]
             for it, k in zip(items, ks)]
    b = collate_prefix(views, collate_multistream(views))
    b["user_idx"] = torch.zeros(len(views), dtype=torch.long)
    return b


def full(items):
    b = collate_multistream(items)
    b["user_idx"] = torch.zeros(len(items), dtype=torch.long)
    return b


def start_of(b):
    return int(ro.first_row_of_campaign(b["campaign"], HOLD).min())


def same_draws(a, b):
    return np.array_equal(a["decisions"], b["decisions"]) and np.array_equal(a["obtained"], b["obtained"])


def g1():
    print("G1 the flag reaches generation")
    m = model(True)
    with torch.no_grad():
        m.censor_vec.fill_(0.5)
        m.censor_h0.fill_(0.3)
    b = chopped(customers(), [2, 1, 0])
    check(f"batch flags {b['censored'].tolist()}", b["censored"].tolist() == [1, 1, 0])
    st = start_of(b)
    for tf in (False, True):
        m.streaming = False                          # full prefix through stream_forward
        x = ro.rollout(m, b, ToyEnv(), None, start=st, n_rep=2, seed=7, teacher_forced=tf)
        m.streaming = True                           # advance from init_stream
        y = ro.rollout(m, b, ToyEnv(), None, start=st, n_rep=2, seed=7, teacher_forced=tf)
        n = int((y["decisions"] > 0).sum())
        check(f"{'teacher forced' if tf else 'free running'}: streaming = full prefix "
              f"({n} generated occasions)", same_draws(x, y) and n > 0)
    b0 = dict(b, censored=torch.zeros_like(b["censored"]))
    x = ro.rollout(m, b, ToyEnv(), None, start=st, n_rep=2, seed=7)
    z = ro.rollout(m, b0, ToyEnv(), None, start=st, n_rep=2, seed=7)
    check("zeroing the flag changes the draws", not np.array_equal(x["decisions"], z["decisions"]))


def g2():
    print("G2 a model without the flag ignores it")
    m = model(False)
    b = chopped(customers(), [2, 1, 0])
    st = start_of(b)
    nb = {k: v for k, v in b.items() if k not in ("censored", "chopped")}
    for streaming in (True, False):
        m.streaming = streaming
        x = ro.rollout(m, b, ToyEnv(), None, start=st, n_rep=2, seed=3)
        y = ro.rollout(m, nb, ToyEnv(), None, start=st, n_rep=2, seed=3)
        check(f"{'streaming' if streaming else 'full prefix'}: identical draws", same_draws(x, y))
    m.streaming = True


def g3():
    print("G3 pity_from: the game keeps the full history")
    m = model(False)
    items = customers()
    b6 = full(items)
    bz = chopped(items, [2, 2, 2])
    e_full, e_from, e_own = ToyEnv(), ToyEnv(), ToyEnv()
    ro.rollout(m, b6, e_full, None, start=start_of(b6), n_rep=2, seed=5)
    ro.rollout(m, bz, e_from, None, start=start_of(bz), n_rep=2, seed=5,
               pity_from=(b6, start_of(b6)))
    ro.rollout(m, bz, e_own, None, start=start_of(bz), n_rep=2, seed=5)
    check(f"warm-up replays the full prefix, call for call ({len(e_full.seen)} calls)",
          e_from.seen == e_full.seen and len(e_full.seen) > 0)
    check(f"without pity_from it replays only the chopped rows ({len(e_own.seen)} calls)",
          len(e_own.seen) < len(e_full.seen))
    bad = dict(b6, label=b6["label"].clone())
    bad["label"][0, -1] = 0 if int(bad["label"][0, -1]) else 1
    try:
        ro.rollout(m, bz, ToyEnv(), None, start=start_of(bz), n_rep=2, seed=5,
                   pity_from=(bad, start_of(b6)))
        refused = False
    except ValueError:
        refused = True
    check("a pity_from batch with different live rows is refused", refused)
    tf = ToyEnv()
    ro.rollout(m, bz, tf, None, start=start_of(bz), n_rep=2, seed=5, teacher_forced=True,
               pity_from=(b6, start_of(b6)))
    check("teacher forced: no warm-up, as before", tf.seen == [])


def g4():
    print("G4 unchop")
    m = model(False)
    items = customers()
    b6 = full(items)
    o6 = ro.first_row_of_campaign(b6["campaign"], HOLD)
    st6 = int(o6.min())
    plain = ro.rollout(m, b6, ToyEnv(), None, start=st6, n_rep=2, seed=9)
    bz, s, lz = chop_batch(items, [0, 1, 2], 0, VAL)
    r = ro.rollout(m, bz, ToyEnv(), None, start=start_of(bz), n_rep=2, seed=9,
                   pity_from=(b6, st6))
    back, bad = unchop(r, b6, s, lz, 2)
    # unchop keeps each history's own rows; the plain rollout also writes the last
    # decision's outcome into the padding row after it, which nothing scores
    real = np.repeat(np.arange(b6["label"].shape[1])[None, :] < lz[:, None], 2, axis=0)
    check("k = 0: the chop path is the plain rollout, draw for draw",
          bad == 0 and int(s.sum()) == 0 and same_draws(r, plain)
          and np.array_equal(back["decisions"], plain["decisions"])
          and np.array_equal(back["obtained"][real], plain["obtained"][real]))

    bz, s, lz = chop_batch(items, [0, 1, 2], 2, VAL)
    r = ro.rollout(m, bz, ToyEnv(), None, start=start_of(bz), n_rep=2, seed=9,
                   pity_from=(b6, st6))
    back, bad = unchop(r, b6, s, lz, 2)
    alive = np.repeat(b6["label"].numpy() != 0, 2, axis=0)
    real_obt = np.repeat(b6["obtained"].numpy(), 2, axis=0)
    keep = np.arange(alive.shape[1])[None, :] <= np.repeat(o6, 2)[:, None]
    lens = np.array([int(it["lto"].size(0)) for it in items])
    check(f"k = 2: chops {s.tolist()}, chopped + kept = full lengths {lens.tolist()}",
          bool((s > 0).all()) and np.array_equal(s + lz, lens))
    check(f"every generated row lands on a live row of the full history "
          f"({int((back['decisions'] > 0).sum())} rows), nothing lost (bad {bad})",
          bad == 0 and np.array_equal(back["decisions"] > 0, alive))
    check("real acquisitions untouched up to each holdout start",
          int((back["obtained"] != real_obt)[keep].sum()) == 0)
    check("the generated acquisitions are the chopped rollout's",
          all(np.array_equal(back["obtained"][j, s[j // 2]:s[j // 2] + lz[j // 2]],
                             r["obtained"][j, :lz[j // 2]]) for j in range(6)))


def g5():
    print("G5 training through the flag after the refactor")
    m = model(True)
    m.train()
    b = chopped(customers(), [2, 1, 0])
    m.zero_grad()

    def cb(c0, c1, logits):
        logits.float().pow(2).mean().backward()

    m.stream_forward(b["lto"], b["obtained"], b["prev_decision"], b["user_id"],
                     censored=b["censored"], on_chunk=cb)
    g = {n: p.grad for n, p in m.named_parameters() if n in ("censor_vec", "censor_h0")}
    check(f"censor_vec and censor_h0 get finite non-zero gradients ({', '.join(g)})",
          len(g) == 2 and all(v is not None and bool(torch.isfinite(v).all())
                              and float(v.abs().sum()) > 0 for v in g.values()))


def main() -> None:
    g1()
    g2()
    g3()
    g4()
    g5()
    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all R45 step 2 checks passed")


if __name__ == "__main__":
    main()
