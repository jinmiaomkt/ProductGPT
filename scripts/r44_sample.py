"""
R44 stage 2: the R30 stage-2 random search, drawn once and shared by both
families, gen 5 (rev-2 stack incumbent) and gen 6 (streaming hybrid).

Prints one line per run: "G5 <qsub -v string>" or "G6 <qsub -v string>".
scripts/submit_batch40.sh submits them. Pre-registered in EXPERIMENTS.md, R44.

Symmetry. Both families are the stacked hybrid, so they share R30 stage 1's
capacity frontier for it: best d96 N6, runner-up d128 N4. Configuration k is
the SAME draw for both families (paired search). Gen 6 also draws a window
from a separate RNG stream, so adding it does not change the shared draws.

Sampling (R30 stage 2's space):
    (d, N)       d96 N6 for even k, d128 N4 for odd k (12 each)
    lr           log-uniform [1e-4, 1.2e-3]
    dropout      {0.05, 0.1, 0.2, 0.3}
    weight decay {0, 0.01, 0.05, 0.1}
    grad accum   {2, 4, 8}  (batch 4 -> effective batch 8, 16, 32)
    d_ff ratio   {2, 3, 4}
    warmup frac  {0.02, 0.05, 0.1}
    window       {128, 384, 768, unlimited}, gen 6 only, 3 configurations each per cell
Frozen for both: level 7, campaign-27 validation, 40 epochs / patience 5,
class weights, the b36 stock (attribute kernel, half-life 2 frozen, duplicate
tiers, campaign control), ALiBi, 8 heads, 2 satiation layers, seed 1.
Gen 5 keeps max_events 1024; gen 6 reads full histories in chunks of 512.

USAGE
    python3 scripts/r44_sample.py          # print the 48 run strings
"""
from __future__ import annotations

import math
import random

N_CONFIGS = 24
CELLS = [(96, 6), (128, 4)]          # R30 stage 1: the hybrid's best and runner-up
SHARED_SEED, WINDOW_SEED = 4401, 4402
WINDOWS = [128, 384, 768, 0]         # 0 = unlimited

G5 = ("PROFILE=hpcc,MAX_EVENTS=1024,EPOCHS=40,BATCH_SIZE=4,VOCAB_LEVEL=7,VAL_MODE=late,"
      "VAL_FROM=27,USER_EMB=0,SEED=1,ENCODER=gru_attn,FUSE=stack,ALIBI=1,INVENTORY=slots,"
      "SAT_LAYERS=2,N_HEADS=8,KERNEL=attr,DECAY=exp,TIER=1,DECAY_FREEZE=1,DECAY_INIT=2,CAMP_FE=1")
G6 = "SEED=1,BATCH_SIZE=4,N_HEADS=8,CHUNK=512"   # the rest are the trainer's frozen defaults


def draws():
    rng = random.Random(SHARED_SEED)
    # Stratified, not independent: each capacity cell gets every window exactly
    # 3 times, in a seeded shuffle (independent draws gave 11 unlimited and 3 at
    # 128, and stratifying only overall left the window confounded with the cell).
    wrng = random.Random(WINDOW_SEED)
    per_cell = []
    for _ in CELLS:
        w = WINDOWS * (N_CONFIGS // len(CELLS) // len(WINDOWS))
        wrng.shuffle(w)
        per_cell.append(w)
    windows = [per_cell[k % 2][k // 2] for k in range(N_CONFIGS)]
    for k in range(N_CONFIGS):
        d, n = CELLS[k % 2]
        yield {
            "k": k, "d": d, "n": n,
            "lr": math.exp(rng.uniform(math.log(1e-4), math.log(1.2e-3))),
            "drop": rng.choice([0.05, 0.1, 0.2, 0.3]),
            "wd": rng.choice([0, 0.01, 0.05, 0.1]),
            "acc": rng.choice([2, 4, 8]),
            "ffr": rng.choice([2, 3, 4]),
            "warm": rng.choice([0.02, 0.05, 0.1]),
            "window": windows[k],
        }


def main() -> None:
    for c in draws():
        hp = (f"D_MODEL={c['d']},D_FF={c['ffr'] * c['d']},N_LAYERS={c['n']},LR={c['lr']:.2e},"
              f"DROPOUT={c['drop']},WD={c['wd']},GRAD_ACCUM={c['acc']},WARMUP={c['warm']}")
        print(f"G5 {G5},{hp},TAG=b40_g5_c{c['k']:02d}_s1")
        print(f"G6 {G6},WINDOW={c['window']},{hp},TAG=b40_g6_c{c['k']:02d}")


if __name__ == "__main__":
    main()
