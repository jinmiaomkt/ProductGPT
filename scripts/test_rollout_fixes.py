"""
Regression tests for four rollout-engine defects found in the October 2026 review.

1. Pre-window inventory. With a context window shorter than the history, the
   slots inventory needs inv_init_count / inv_init_last for everything acquired
   before the window. The rollout's forward pass dropped them, so every rollout
   started with an empty inventory at the window edge.
2. Item pools. A 4-star (or 3-star) draw always came from the 4-star CHARACTER
   pool, even on the weapon banner, and a 5-star on the regular banner was always
   a character. Products are now drawn from what each banner actually yielded in
   calibration.
3. Early loop exit. The generation loop stopped at the first row where no
   sequence in the batch was alive, so a sequence whose holdout starts after
   such a gap was never generated. Same loop in baselines.simulate and arm C.
4. Pity warm-up. The real prefix was replayed only up to the batch's earliest
   start, so (a) the row just before a sequence's first generated decision -- the
   real outcome of its last real decision -- was skipped, (b) sequences entering
   the holdout later lost the real rows in between, and (c) the 50/50 guarantee
   state was never warmed at all.

Synthetic inputs only; no data needed.

    python scripts/test_rollout_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "gen5_multistream", REPO / "simulation", REPO / "shared", REPO / "scripts"):
    sys.path.insert(0, str(p))

import model_multistream_state_space as mm  # noqa: E402
from baselines import simulate  # noqa: E402
from gacha_env import AcquisitionEnv, PityState, ProductTable  # noqa: E402
from rollout import _forward_logits, _warm_pity, rollout  # noqa: E402

FAILED = []
NONE = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


# Products: id -> (rarity, is_figure, is_limited)
PRODUCTS = {13: (5, 1, 1),   # featured figure (figure A banner)
            14: (5, 1, 0),   # standard 5-star figure
            15: (4, 1, 0),   # 4-star figure
            16: (4, 0, 0),   # 4-star weapon
            17: (3, 0, 0),   # 3-star weapon
            18: (5, 0, 0),   # standard 5-star weapon
            19: (5, 0, 1),   # featured weapon
            20: (5, 0, 1)}   # second featured weapon
ids = np.array(sorted(PRODUCTS))
VOCAB = ProductTable(ids, np.array([PRODUCTS[i][0] for i in ids]),
                     np.array([PRODUCTS[i][1] for i in ids]),
                     np.array([PRODUCTS[i][2] for i in ids]), 13, 20)
OFFER = [13, 0, 19, 20]


def item(decisions, products, campaign=1):
    """One calibration sequence. decisions[t] is made at row t; its acquisitions
    land in row t+1, which is how the loader stores them (the R1 shift)."""
    S = len(decisions) + 1
    lto = np.tile(np.array(OFFER), (S, 1))
    obt = np.zeros((S, 10), dtype=np.int64)
    prv = np.zeros(S, dtype=np.int64)
    lab = np.zeros(S, dtype=np.int64)
    for t, (d, prods) in enumerate(zip(decisions, products)):
        lab[t] = d
        prv[t + 1] = d
        obt[t + 1, :len(prods)] = prods
    lab[S - 1] = 9
    return {"lto": lto, "obtained": obt, "prev_decision": prv, "label": lab,
            "campaign": np.full(S, campaign)}


class ZeroEnv:
    """Environment stand-in: every purchase yields nothing; observe is a no-op."""

    def step(self, decision, offer, state, rng):
        return np.zeros(10, dtype=np.int64)

    def observe(self, decision, offer, row, state):
        return None


class Stub(torch.nn.Module):
    """Always prefers decision 3 (Buy1 figure A)."""

    def forward(self, lto, obt, prev, user_idx=None, ipt=None, **kw):
        out = torch.full((*prev.shape, 18), -20.0)
        out[..., 3] = 20.0
        return out


def gap_batch():
    """Sequence 0 is alive on rows 2-3, sequence 1 on rows 6-7: nobody on 4-5."""
    B, S = 2, 9
    lab = torch.zeros(B, S, dtype=torch.long)
    lab[0, 2:4] = 9
    lab[1, 6:8] = 9
    return {"lto": torch.tensor(OFFER).repeat(B, S, 1), "obtained": torch.zeros(B, S, 10, dtype=torch.long),
            "prev_decision": torch.full((B, S), 9, dtype=torch.long), "label": lab,
            "user_idx": torch.zeros(B, dtype=torch.long)}


def main() -> None:
    print("1. the rollout's forward pass carries the pre-window inventory")
    feat = torch.zeros(60, 34)
    feat[13:57] = (torch.rand(44, 34, generator=torch.Generator().manual_seed(1)) > 0.5).float()
    torch.manual_seed(0)
    model = mm.build_transformer(
        vocab_size_src=60, vocab_size_tgt=18, max_seq_len=12, d_model=32, n_layers=1, n_heads=4,
        d_ff=64, dropout=0.0, feature_tensor=feat, ai_rate=15, num_users=2, lto_len=4,
        obtained_len=10, prev_dec_len=1, use_user_embedding=False, encoder="gru",
        inventory="slots", sat_layers=0).eval()
    B, S, t = 2, 12, 8
    g = torch.Generator().manual_seed(2)
    buf = {"lto": torch.randint(13, 57, (B, S, 4), generator=g),
           "obtained": torch.randint(13, 57, (B, S, 10), generator=g) * (torch.rand(B, S, 10, generator=g) < 0.3),
           "prev_decision": torch.randint(1, 10, (B, S), generator=g),
           "ipt": torch.rand(B, S, generator=g) * 24, "user_idx": torch.zeros(B, dtype=torch.long),
           "label": torch.full((B, S), 9),
           "inv_init_count": torch.zeros(B, 44), "inv_init_last": torch.full((B, 44), -1.0e6)}
    buf["inv_init_count"][:, :5] = 3.0
    buf["inv_init_last"][:, :5] = -4.0
    s = slice(0, t + 1)
    with torch.no_grad():
        want = model(buf["lto"][:, s], buf["obtained"][:, s], buf["prev_decision"][:, s], buf["user_idx"],
                     ipt=buf["ipt"][:, s], inv_init_count=buf["inv_init_count"],
                     inv_init_last=buf["inv_init_last"])[:, t, 1:10]
        without = model(buf["lto"][:, s], buf["obtained"][:, s], buf["prev_decision"][:, s],
                        buf["user_idx"], ipt=buf["ipt"][:, s])[:, t, 1:10]
        got = _forward_logits(model, buf, t)
    check("pre-window inventory changes this model's logits (the test can fail)",
          not torch.allclose(want, without), f"max|d| {(want - without).abs().max():.2e}")
    check("the rollout forward equals the model given the pre-window inventory",
          torch.allclose(got, want, atol=1e-6), f"max|d| {(got - want).abs().max():.2e}")

    print("\n2. purchases yield the products each banner actually yielded in calibration")
    items = []
    for k in range(400):
        # weapon banner: 4-stars are 70% weapons; regular banner: 5-stars split
        # evenly between a standard figure and a standard weapon
        items.append(item([7, 7, 1, 3],
                          [[16 if k % 10 < 7 else 15], [17], [14 if k % 2 else 18], [15]]))
    env = AcquisitionEnv(VOCAB).fit(items, holdout_from=28)
    rng = np.random.default_rng(0)
    st = PityState()
    wep4 = [env._draw_identity("wep", 4, OFFER, st, rng) for _ in range(2000)]
    reg5 = [env._draw_identity("reg", 5, OFFER, st, rng) for _ in range(2000)]
    fig4 = [env._draw_identity("figA", 4, OFFER, st, rng) for _ in range(2000)]
    share = np.mean(np.array(wep4) == 16)
    check("4-stars on the weapon banner include 4-star weapons at the calibrated rate",
          abs(share - 0.7) < 0.04, f"4-star weapon share {share:.3f}, calibrated 0.70")
    share = np.mean(np.array(reg5) == 18)
    check("5-stars on the regular banner include standard weapons at the calibrated rate",
          abs(share - 0.5) < 0.04, f"5-star weapon share {share:.3f}, calibrated 0.50")
    check("4-stars on the figure banner stay figures, as calibrated",
          np.mean(np.array(fig4) == 15) > 0.99, f"figure share {np.mean(np.array(fig4) == 15):.3f}")

    print("\n3. sequences whose holdout starts after an all-dead gap are still generated")
    gb = gap_batch()
    r = rollout(Stub(), gb, None, VOCAB, start=2, n_rep=1, seed=0, teacher_forced=True)
    check("rollout: the late sequence is generated", (r["decisions"][1, 6:8] > 0).all(),
          f"rows 6-7 of sequence 1: {r['decisions'][1, 6:8].tolist()}")
    r = simulate(gap_batch(), ZeroEnv(), VOCAB, lambda prev, t, rng: np.full(prev.shape[0], 3),
                 start=2, n_rep=1, seed=0)
    check("baseline simulator: the late sequence is generated", (r["decisions"][1, 6:8] > 0).all(),
          f"rows 6-7 of sequence 1: {r['decisions'][1, 6:8].tolist()}")
    from r36f_env_arm import arm_c
    gb = gap_batch()
    gb["label"][1, 6:8] = 3
    r = arm_c(gb, ZeroEnv(), VOCAB, start=2, n_rep=1, seed=0)
    check("arm C replay: the late sequence is replayed", (r["decisions"][1, 6:8] > 0).all(),
          f"rows 6-7 of sequence 1: {r['decisions'][1, 6:8].tolist()}")

    print("\n4. pity warm-up replays each sequence's real prefix up to its own first decision")
    # Both sequences pull Buy1 on figure A at rows 0-3: 3-star, 3-star, a STANDARD
    # 5-star (a lost 50/50), 3-star -> outcomes land in rows 1-4. Sequence 0
    # generates from row 4; sequence 1 has two more real Buy1 3-stars (rows 5-6)
    # and generates from row 6.
    seqs = [item([3, 3, 3, 3, 9, 9, 9, 9], [[17], [17], [14], [17], [], [], [], []]),
            item([3, 3, 3, 3, 3, 3, 9, 9], [[17], [17], [14], [17], [17], [17], [], []])]
    batch = {k: torch.as_tensor(np.stack([s_[k] for s_ in seqs])) for k in seqs[0]}
    batch["label"][:] = 0
    batch["label"][0, 4:7] = 9
    batch["label"][1, 6:8] = 9
    env = AcquisitionEnv(VOCAB)
    env.guarantee_q, env.guarantee_L = {"figA": 0.5}, {"figA": 1}
    states = [PityState(), PityState()]
    _warm_pity(states, batch, VOCAB, env, 4)
    check("sequence 0: pulls since its last 5-star include the row before its first decision",
          states[0].since5["figA"] == 1, f"since5 {states[0].since5['figA']}, want 1")
    check("sequence 1: its later real rows are replayed up to its own start",
          states[1].since5["figA"] == 3, f"since5 {states[1].since5['figA']}, want 3")
    check("the lost 50/50 is carried into the rollout (guarantee armed)",
          states[0].losses["figA"] == 1 and states[1].losses["figA"] == 1,
          f"losses {states[0].losses['figA']}, {states[1].losses['figA']}, want 1, 1")

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {', '.join(FAILED)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
