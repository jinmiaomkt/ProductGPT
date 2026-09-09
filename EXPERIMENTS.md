# Experiment ledger

One row per run or diagnostic. This file is the handoff between the two work
streams (see "Session streams" in CLAUDE.md): the architecture stream decides
what to test and writes the hypothesis; the HPCC stream runs it and writes the
result.

**Rules**

- A result enters as a *conclusion*, never as a pasted log. "Job 12345 OOMed at
  S=1536" — not the traceback. Debugging belongs in the architecture stream.
- Every row names the hypothesis it tests. A run without one is a run whose
  result you cannot interpret three weeks later.
- Numbers that were never recorded are marked `not recorded`. Do not
  reconstruct them from memory; re-run or read the run directory.
- Run directories on HPCC follow the pattern built by `scripts/gen5_train_hpcc.pbs`:
  `/storage/home/jinmiao/ProductGPT/runs/gen5_<profile>_S<max_events>_b<batch>[_<tag>]`
  and are STABLE per configuration, so a resubmitted job resumes its own `last.pt`.

---

## Live queue

Refresh this from an actual `qstat -u $USER` at the start of each HPCC
session. Never write a job id from memory — a wrong id is worse than none.

| Job id | Run dir / tag | Submitted | Hypothesis | Status |
|---|---|---|---|---|
| 40755 | `v2_emb` | 2026-09-09 | Baseline **with** the user embedding, under the R5 design | R |
| 40756 | `v2_noemb` | 2026-09-09 | Does dropping the embedding still help once the split is fixed? | Q |
| 40757 | `v2_noemb_aug` | 2026-09-09 | **Augmentation alone** — the arm that never produced output | Q |
| 40758 | `v2_noemb_aug_do25` | 2026-09-09 | Augmentation + dropout 0.25, to de-confound against 40757 | Q |

All four at S=1024 / batch 4, deliberately matching `jmr_flat` and `jmr_mix8`
so the whole table becomes comparable. **S was NOT raised at the same time** —
changing the representation and the regularisation together would confound
both. Tags carry a `v2_` prefix so the pre-redesign run directories, which are
the historical record for R6, are not overwritten.

Only two GPUs run per user at once, so 40756–40758 queue behind 40755.

Check with `bash scripts/hpcc_status.sh --once --force`. Note that `qstat`
is NOT on the PATH of a non-interactive SSH session — it lives in
`/opt/pbs/bin`, and a bare `qstat` over `ssh` fails with "command not found",
which looks exactly like an empty queue. The script handles this and reports
being blind rather than reporting silence as good news.

Site facts worth remembering (verified from `qstat -Q -f GPU`, Sep 2026):
the GPU queue sets no `resources_max.walltime`, and `max_run_res.ngpus` is
2 per user — a third job queues. Plan sweeps in batches of two.

---

## Ledger

| # | Date | What | Hypothesis | Outcome | Decision |
|---|---|---|---|---|---|
| R1 | Sep 2026 | Label-leak diagnostic | The `obtained` stream carries o_t, not o_(t-1), making the label readable | **Confirmed** on the IPT file; simple6 files clean | `shift_obtained=True` by default |
| R2 | Sep 2026 | Post-fix re-measure | Fixing the leak should collapse NotBuy metrics | NotBuy F1 1.000 → ~0.67 | All pre-fix numbers void |
| R3 | Sep 2026 | Truncation direction | Head vs tail truncation is cosmetic | **Wrong** — head truncation deleted the holdout | `truncate="tail"` |
| R4 | Sep 2026 | Memory probe | Find the largest feasible `max_events` | S=1024 fits HPCC at batch 4; S=1536 OOMs **at batch 4** | Superseded — see R10 |
| R10 | Sep 9 2026 | Cap sweep + memory arithmetic | S=1024 is a hardware ceiling | **Wrong** — it is a batching artefact. Memory ∝ B·S², so S=1536 at batch 1 needs ~11 GB, half of what S=1024 at batch 4 already uses | Build a token-budget sampler; do **not** request more GPU memory |
| R5 | Sep 2026 | Split redesign | Four holdout conventions existed; none matched the benchmark | Adopted Lu & Kannan's 2×2 | `split_mode="both"` default |
| R6 | Sep 8 2026 | Regularisation sweep | Dropping the user embedding hurts; augmentation is a free win | **Both wrong.** Dropping the embedding *helped* (val NLL 1.013 → 0.951); aug+dropout was worse (0.957) | Keep `use_user_embedding=False`; re-run augmentation alone |
| R7 | Sep 8 2026 | Mixture-head port | Per-customer mixture beats a flat head | Better on the holdout period (NLL 1.121 vs 1.247, macro F1 0.484 vs 0.447), slightly worse on validation NLL and revenue MAE | Keep mix8; heterogeneity reading pending HP tuning |
| R8 | Sep 2026 | Laptop pilot | Smoke-test the full gen-5 path end to end | Runs; collapses to 2 classes | Baseline to beat |
| R9 | Sep 8 2026 | Event-stream measurement | Is a continuous-time formulation feasible? | Yes, with two data caveats | Proceed to a scoring harness |

---

## Detail

### R1 — Label leakage in the `obtained` stream

`scripts/check_obtained_leak.py`. In `clean_list_int_wide4_simple6_IPT.json`,
P(y_t = 9 | obtained block all zero) = **1.0000 over 83,664 events**, with no
counter-examples: a user who did not draw obtained nothing, so an all-zero
block identified NotBuy exactly. The `simple6` (non-IPT) files contain **zero**
all-zero blocks, so gen 2 and gen 4 were never affected — an earlier suspicion
that they were is retracted.

Fixed in the loader, not the data: `dataset_multistream.py` prepends a PAD row
and drops the last, so event t sees o_(t-1). Commits `42467ad`, `6a06da4`.

> Open: the R generator still emits the unshifted stream. If it is ever fixed
> upstream, `shift_obtained` must be turned off or the data is shifted twice.

### R3 — Truncation direction

With `truncate="head"` the train/val/test event counts were
**49,195 / 948 / 488**; with `"tail"`, **30,735 / 9,583 / 10,313**. Head
truncation keeps each user's *first* events, which deletes the late campaigns
that constitute the holdout. The tiny val/test counts were the symptom.

### R4 — Memory

Attention over the offer–inventory cross product builds a `(B,H,S,4,S*10)`
tensor, so cost grows as S².

| S | Laptop (RTX PRO 500, 6 GB) | HPCC (~44.5 GB) |
|---|---|---|
| 256 | 0.35 GB | — |
| 512 | 1.29 GB | — |
| 768 | 2.87 GB | — |
| 1024 | 5.08 GB (85%) | 20.23 GB (45.5%) |
| 1536 | — | **OOM** |
| 2048 | — | ~81 GB projected |

Use `scripts/probe_memory.py` before raising `max_events`.

### R10 — S=1024 is a batching artefact, not a hardware ceiling

Cost of each candidate `max_events`, measured over the full cohort by
`scripts/measure_event_stream.py --caps ...`:

| cap | users truncated | events lost |
|---|---|---|
| 1024 | **9.1%** | 2.09% |
| 1280 | 1.7% | 0.17% |
| **1536** | **0.0%** | **0.00%** |

The longest user is 1,512 events, so 1536 covers everyone and there is no
reason to go to 1600 or 2048. The 2% of lost events is not spread evenly — it
falls entirely on the 9% *longest* users, i.e. the heaviest spenders, which is
the worst place to lose data.

Memory scales as B·S², since the cross-attention builds `(B,H,S,4,S*10)`.
Anchoring on the measured 1024@b4 = 20.23 GB:

| config | B·S² | projected |
|---|---|---|
| 1024 @ b4 | 4.19e6 | 20.2 GB (measured) |
| 1536 @ b4 | 9.44e6 | ~45.5 GB → OOM, matching what was observed |
| 1536 @ b2 | 4.72e6 | ~22.8 GB |
| **1536 @ b1** | 2.36e6 | **~11.4 GB** |
| 2048 @ b1 | 4.19e6 | ~20.2 GB |

So the fix is a **token-budget sampler**: bucket users by length and set batch
size to `floor(budget / S²)` with the budget pinned at the current 4.2e6.
Median length is 556, so most batches get 13–16 users and average batch size
*rises*; only the few longest users drop to batch 1.

Note `collate_multistream` already pads to the batch max, not a global max, so
the waste comes from random batch composition — one long user setting the
length for three short ones — not from global padding.

Two caveats before implementing. With variable batch sizes the loss must be
normalised by **valid token count**, not by batch, or gradients skew toward
long-sequence batches. And the projections above are a linear extrapolation
from a single measured point; verify (1536, b1) and (2048, b1) with
`probe_memory.py` first.

**Do not request more GPU memory from IITS on account of this.** The ceiling
is a batching choice, and ~44.5 GB is ample once the sampler exists.

### R5 — Evaluation design

Now matches Lu & Kannan (JMR 2026, 63:1) Table 4: customers split in two,
periods split in two, all four cells reported. Three changes were needed —
customer holdout 10% → 50%; validation drawn from held-out *customers* rather
than a period; out-of-sample customers receive the population-mean parameter
(ω̄) rather than an untrained row. Campaign 28 moved back into the holdout,
since `FeatureBasedHoldout` starts there and gen 5 is a feature-based model —
validating on it meant selecting checkpoints on test data.

Commits `0c549c2`, `1ad79d1`, `652c687`, `e55a2ab`.

### R6 — Regularisation sweep

Flags added in `ff452ae`: `USER_EMB=0`, `AUGMENT=1`, `DROPOUT`, `PATIENCE`,
each with its own `TAG` so run directories do not collide. All at S=1024,
batch 4, d_model=128, N=4, full cohort.

| Run tag | user emb | augment | dropout | epochs (best) | best val NLL |
|---|---|---|---|---|---|
| `gen5_hpcc_S1024_b4` | **yes** | no | 0.10 | 11 (5) | 1.0130 |
| `gen5_hpcc_S1024_b4_noemb` | no | no | 0.10 | 52 (41) | **0.9510** |
| `gen5_hpcc_S1024_b4_noemb_aug_do25` | no | yes | 0.25 | 60 (49) | 0.9567 |

Two results. **Dropping the per-customer embedding helped** — 1.0130 → 0.9510,
and the embedded model early-stopped at epoch 11 having peaked at epoch 5,
which is the signature of memorisation rather than learning. **Augmentation did
not help**, but that arm changed dropout at the same time, so it does not
isolate augmentation.

The reason the confound was never resolved: `gen5_hpcc_S1024_b4_noemb_aug` —
the augmentation-only arm — **produced no output at all**. Its run directory is
empty, so the clean comparison was never available. Re-run that arm alone.

> These three predate the R5 evaluation redesign, so their validation NLL is
> computed on a *different* validation set from the R7 runs below. Comparable
> within this table, **not** across to R7.

### R7 — Mixture head (Lu & Kannan mechanism)

`UserMixtureOutputHead`: H output projections with per-customer weights
α_n = softmax(user_mix_logits[n]); unseen customers get ᾱ, the population mean.
H parameters per customer instead of a 128-dim embedding — roughly 32× less
memorisation capacity, and interpretable as soft segment membership.

Two arms, both under the R5 2×2 design, both at S=1024 / batch 4 /
d_model=128 / N=4 / `use_user_embedding=False`, full cohort.

| | `jmr_flat` | `jmr_mix8` |
|---|---|---|
| mix heads | 0 | 8 |
| params | 1,215,382 | 1,271,676 |
| epochs (best) | 45 (35) | 40 (29) |
| best val NLL | **0.9908** | 0.9947 |
| in-sample × holdout — NLL | 1.2473 | **1.1205** |
| in-sample × holdout — hit | 0.5841 | **0.5919** |
| in-sample × holdout — macro F1 | 0.4469 | **0.4844** |
| in-sample × holdout — macro AUPRC | 0.4933 | **0.5120** |
| in-sample × holdout — rev MAE | **1.074** | 1.175 |
| out-sample × holdout — NLL | 1.2330 | **1.1109** |
| out-sample × calibration — NLL | **0.9858** | 0.9895 |

Both now predict **all 9 classes**, unlike the R8 pilot's 2 of 9. NotBuy F1
sits at 0.66–0.71 across cells, consistent with the post-leak-fix level.

Three things worth carrying forward:

1. **Validation NLL mis-ranks the two models.** `jmr_flat` wins on validation
   (0.9908 vs 0.9947) but loses clearly on the holdout period (1.2473 vs
   1.1205, macro F1 0.4469 vs 0.4844). This is direct evidence for the open
   "NLL or AUPRC as the selection metric?" question — here they disagree, and
   validation NLL picks the worse model.
2. **The metrics disagree with each other.** `jmr_mix8` wins on NLL, hit rate,
   macro F1 and AUPRC; `jmr_flat` wins on revenue MAE (1.074 vs 1.175). If
   revenue is the managerial target, the ranking flips. Do not report a single
   winner without saying by which metric.
3. **Customer heterogeneity buys nothing out of sample — this is the big one.**
   In *both* models, customers the model never trained on score slightly
   *better* than customers it did (flat: 1.2330 vs 1.2473; mix8: 1.1109 vs
   1.1205). The in-sample/out-of-sample gap that the Lu & Kannan design exists
   to measure is not merely small here, it runs the wrong way. Whatever the
   mixture head is buying, it is not customer-specific knowledge that
   generalises. This bears directly on the open "does the paper need user
   heterogeneity?" question and deserves its own diagnostic before the deck
   claims otherwise.

Commits `e406d05`, `fa045e2`.

### R8 — Laptop pilot (reference point, not a result)

`--profile pilot`: 300 users, S=128, batch 2, 2 epochs, d_model=64, N=2,
188,182 params, `use_user_embedding=false`, `num_mix_heads=0`,
`split_mode=both`, `shift_obtained=true`.

| Cell | NLL | Hit | macro F1 | macro AUPRC | rev MAE | n |
|---|---|---|---|---|---|---|
| in-sample × holdout | 1.507 | 0.442 | 0.099 | 0.204 | 2.38 | 10,104 |
| out-sample × calibration | 1.459 | 0.515 | 0.113 | 0.182 | 2.25 | 8,090 |
| out-sample × holdout | 1.517 | 0.452 | 0.098 | 0.190 | 2.34 | 11,110 |

**The diagnostic that matters:** the model predicts only class 1 (Buy1_Reg) and
class 9 (NotBuy). Classes 2–8 have `predicted = 0` in every cell — macro F1 of
0.099 is that, not noise. Expected at this size, but it is the floor any real
run must clear. NotBuy F1 ≈ 0.65 here, consistent with the post-leak-fix value.

### R9 — Event-stream measurement (Sep 8 2026)

`scripts/measure_event_stream.py`, full cohort: 5,004 users, 3,017,180 rows
(1,916,832 real draws; 1,100,348 inserted, 36.5%). Written to
`results/event_stream_stats.json`.

- **Mark space is clean.** Every inserted row is decision 9; *zero* real rows
  are decision 9. Filtering `IsInserted` leaves exactly an 8-way mark space.
- **Sequences shrink 1.57×.** Discrete median 556 / max 1,512; continuous
  median 344 / max 1,021. The continuous representation covers 100% of users at
  S=1024, which fits (20.2 GB); the discrete one needs S=1536, which OOMs.
- **`IPT` is not the inter-purchase time.** It measures back to the previous
  *row*, and inserted rows are rows, so it is censored at the grid width
  (max 29 h). True real-to-real gaps reach 2,401 h. A point process must
  accumulate across intervening rows.
- **`IPT` is quantised at 0.01 h = 36 s** (100% of non-zero gaps on that grid;
  99 distinct values below 1 h), so 46.5% of true gaps round to exactly zero.
  Handle by interval-censoring each gap to [v, v+q) rather than regenerating.
- Clustering: P(next draw ≤ 1 min | previous gap ≤ 1 min) = 0.658 vs 0.488
  after a ≥ 24 h gap, a 1.35× risk ratio. Treat as a floor — the 36 s
  quantisation attenuates it.
- Holdout is well powered: 3,251 of 5,004 users (65%) draw in campaigns ≥ 28;
  260,953 real draws; 687,130 revenue units.

---

## Queued next

1. **Common scoring harness** on the existing discrete model — count / revenue
   CRPS and 8-way log-loss over campaigns 28–30. Prerequisite for everything
   below, and it also delivers the GRU/LSTM baseline comparison that is still
   missing.
2. **GRU / LSTM baselines** on the frozen split.
3. **Δ sweep** (24 / 48 / 96 h) — needs `scripts/regenerate_ipt.py` validated
   against the R output first.
4. **Continuous-time model** — interval-censored log-normal mixture over the
   gap plus a softmax over the 8 marks.
5. **Hyperparameter tuning** — deliberately last. Tuning before baselines exist
   optimises a number nobody can interpret.
