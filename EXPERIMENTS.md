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
| 40859–40882 | `b2_*` (24 runs) | 2026-09-12 | Seeded replication: 4 transformer variants, GRU and LSTM baselines, dropout sweep (0.10/0.25/0.40/0.55), 3 seeds each | Running |

All at S=1024, `VAL_MODE=late`, EPOCHS=40, default patience. Submitted by
`scripts/submit_batch2.sh`; read with `python3 scripts/summarize_batch.py`.
Batch 1 (40852–40855) is complete and produced R16 and R17.

> **Always pass `MAX_EVENTS` explicitly — the PBS default is 512, not 1024.**
> Jobs 40755–40758 omitted it and silently ran at 512 (recorded as R11).

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
| R5 | Sep 2026 | Split redesign | Four holdout conventions existed; none matched the benchmark | Adopted Lu & Kannan's 2×2 | `split_mode="both"` default |
| R6 | Sep 8 2026 | Regularisation sweep | Dropping the user embedding hurts; augmentation is a free win | **Both wrong.** Dropping the embedding *helped* (val NLL 1.013 → 0.951); aug+dropout was worse (0.957) | Keep `use_user_embedding=False`; re-run augmentation alone |
| R7 | Sep 8 2026 | Mixture-head port | Per-customer mixture beats a flat head | Better on the holdout period (NLL 1.121 vs 1.247, macro F1 0.484 vs 0.447), slightly worse on validation NLL and revenue MAE | Keep mix8; heterogeneity reading pending HP tuning |
| R8 | Sep 2026 | Laptop pilot | Smoke-test the full gen-5 path end to end | Runs; collapses to 2 classes | Baseline to beat |
| R9 | Sep 8 2026 | Event-stream measurement | Is a continuous-time formulation feasible? | Yes, with two data caveats | Proceed to a scoring harness |
| R10 | Sep 9 2026 | Cap sweep + memory arithmetic | S=1024 is a hardware ceiling | **Wrong** — it is a batching artefact. Memory ∝ B·S², so S=1536 at batch 1 needs ~11 GB, half of what S=1024 at batch 4 already uses | Build a token-budget sampler; do **not** request more GPU memory |
| R11 | Sep 9 2026 | Four arms at S=512 (submitted without `MAX_EVENTS` by mistake) | Re-run R6 under the R5 design | **R6 reverses.** The user embedding is the *best* holdout model; augmentation alone does nothing; the gain in the old aug+do25 arm was dropout | Re-run at S=1024 (40760–63) before concluding |
| R12 | Sep 9 2026 | Selection-metric study (7 runs, no GPU time) | Validation NLL mis-ranks models; another metric will do better | **Worse than expected.** ALL FIVE validation metrics are anti-correlated with holdout performance. Spearman(selected epoch, holdout NLL) = +0.83 | Diagnostic runs 40764/40765 submitted to separate training length from architecture |
| R13 | Sep 10 2026 | Per-epoch holdout tracking, both architectures | Separate training length from architecture | **Training length. Decisively.** Holdout peaks at epoch 3-4 in BOTH models while validation improves to epoch 10 and 34. Both architectures reach the SAME holdout optimum (1.0194 vs 1.0158) | Validation must be temporally shifted. Cross-run rankings at different epochs are void (see R14 for what survives) |
| R14 | Sep 11 2026 | v2 arms at S=1024 (40760-63), and which conclusions survive R13 | Separate matched-epoch comparisons from confounded ones | Augmentation no-op and dropout 0.25 gain both hold at matched epochs; S=512 ≈ S=1024 holds | Dropout sweep is justified; architecture comparisons wait for temporal validation |
| R15 | Sep 11 2026 | In/out-sample gap across epochs, both diag runs | Does knowing a customer help at the honest epoch? | **No.** At the holdout optimum the embedding's gap equals the no-embedding model's pure cohort gap. Late-epoch sign flip is an over-training artefact | Heterogeneity finding stands, now controlled |
| R16 | Sep 12 2026 | Gate: does late-calibration validation track the holdout? | Validation shifted in time can detect temporal drift | **Passes.** Selection cost 0.0000 / 0.0000 / 0.0006 nats (transformer no-emb / emb / GRU) against +0.211 for the old validation | `val_mode="late"` is now the default; batch 2 launched |
| R17 | Sep 12 2026 | GRU baseline vs transformer, matched pipeline | The transformer beats a recurrent baseline | **Wrong, and not close.** GRU holdout NLL 0.8818 vs 1.0062 for the best transformer, F1 0.605 vs 0.590, with 45% fewer parameters | Baselines become a headline result, not a formality |
| R18 | Sep 12 2026 | Component ablation + recency bias + features-only products (batch 3, 6 screening runs) | The transformer's early holdout peak is campaign memorisation via product identity in the offer stream; attention lacks recency; the cross-attention may or may not earn its cost | _pending_ | _pending_ |

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

### R11 — the S=512 arms (accidental, but decisive)

Submitted without `MAX_EVENTS`, so all four took the PBS default of 512 rather
than the intended 1024. They ran to completion in ~30 min each. Not comparable
to the S=1024 rows, but internally consistent, and they settle two questions
R6 could not.

| Run tag (all `gen5_hpcc_S512_b4_`) | user emb | aug | dropout | epochs (best) | **val NLL** | **holdout NLL** | **holdout F1** |
|---|---|---|---|---|---|---|---|
| `v2_emb` | **yes** | no | 0.10 | 14 (8) | 1.0326 | **1.0918** | **0.5605** |
| `v2_noemb` | no | no | 0.10 | 36 (30) | **0.9894** | 1.2037 | 0.4745 |
| `v2_noemb_aug` | no | **yes** | 0.10 | 36 (30) | 0.9893 | 1.2016 | 0.4740 |
| `v2_noemb_aug_do25` | no | yes | **0.25** | 41 (35) | 0.9932 | 1.1350 | 0.5284 |

Three findings.

1. **R6 reverses.** The user embedding gives the *best* holdout performance
   (NLL 1.0918, F1 0.5605) while having the *worst* validation NLL (1.0326).
   R6 concluded the opposite, but it ranked on validation NLL alone, before the
   R5 redesign. Selecting on validation NLL would discard the best model here.
2. **Augmentation alone does nothing.** `v2_noemb` vs `v2_noemb_aug` differ in
   the fourth decimal on every metric (1.2037 vs 1.2016 holdout NLL). This is
   the comparison the original sweep never got, because that arm produced no
   output.
3. **The gain in the old aug+do25 arm was dropout, not augmentation.** Holding
   augmentation on and raising dropout 0.10 → 0.25 moves holdout NLL 1.2016 →
   1.1350 and F1 0.4740 → 0.5284. Worth a dedicated dropout sweep.

**The heterogeneity pattern holds, and strengthens.** Out-of-sample customers
beat in-sample ones in all four arms — including `v2_emb`, which *has* a fitted
per-customer embedding (1.0590 out vs 1.0918 in). A model given real
per-customer parameters still does better on customers it has never seen,
where those parameters are replaced by the population mean ω̄. That is harder
to explain as cohort composition than the R7 result was. Still to be
re-checked after hyperparameter tuning.

**Memory model confirmed.** Peak GPU was 5.1 GB at S=512/batch 4. Scaling by
B·S² predicts 20.4 GB at S=1024/batch 4 against 20.23 GB measured — so the R10
projections, including S=1536 at batch 1 needing ~11 GB, rest on a validated
relationship rather than a single point.

### R12 — every validation metric points the wrong way

`scripts/selection_metric_study.py`, over the 7 runs carrying `test_cells`.
No GPU time: it reads `history.json` and `final.json`.

Runs sorted by holdout NLL, best first:

| run | val NLL | val F1 | val AUPRC | **HO NLL** | **HO F1** |
|---|---|---|---|---|---|
| `S512_v2_emb` | 1.0326 | 0.5720 | 0.5605 | **1.0590** | 0.5753 |
| `S1024_v2_emb` | 1.0351 | 0.5765 | 0.5618 | 1.0636 | 0.5763 |
| `S1024_jmr_mix8` | 0.9947 | 0.5843 | 0.5899 | 1.1109 | 0.4849 |
| `S512_v2_noemb_aug_do25` | 0.9932 | 0.5808 | 0.5877 | 1.1111 | 0.5325 |
| `S512_v2_noemb_aug` | 0.9893 | 0.5833 | 0.5877 | 1.1782 | 0.4859 |
| `S512_v2_noemb` | 0.9894 | 0.5839 | 0.5875 | 1.1810 | 0.4856 |
| `S1024_jmr_flat` | 0.9908 | 0.5851 | **0.5959** | **1.2330** | 0.4449 |

The ordering is close to exactly inverted. The best holdout model has the
*worst* validation NLL; the worst holdout model has the *best* validation F1
and AUPRC. Spearman, validation vs holdout:

| val metric | HO NLL | HO F1 |
|---|---|---|
| nll | −0.82 | +0.57 |
| f1_macro | +0.79 | **−0.96** |
| auprc_macro | +0.64 | −0.82 |
| hit | +0.75 | −0.46 |
| rev_mae | +0.64 | −0.64 |

Signs should be the opposite of this throughout. So the answer is not "select
on AUPRC instead of NLL" — no available metric selects correctly.

**Likely cause, and the confound.** Validation is held-out *customers* scored
on the *calibration period* — the same time regime as training. It therefore
cannot detect overfitting to that regime, and rewards it. Consistent with
that, Spearman(selected epoch, holdout NLL) = **+0.83**: the longer a run
trained, the worse it did on the holdout period.

But at n=7 training length is perfectly confounded with architecture — both
8-epoch runs are the ones with a user embedding. Jobs **40764** (`diag_noemb`)
and **40765** (`diag_emb`) run with `--track-holdout` and `PATIENCE=99`, which
scores the holdout every epoch as a diagnostic and lets the full curve appear.

- If holdout degrades *within* a run as epochs accumulate → validation is in
  the wrong time regime. The fix is a temporally shifted validation set, not
  a different metric.
- If holdout stays flat within a run → training length is not the cause and
  the user embedding is genuinely the better architecture.

Either way, **hyperparameter tuning stays blocked until this resolves.** Tuning
maximises whatever it is given; every metric currently available points the
wrong way.

### R13 — validation cannot see temporal overfitting

Jobs 40764 / 40765, `--track-holdout` with `PATIENCE=99` so the full curve is
visible. Same S=1024, same design, differing only in the user embedding.

| | val NLL best | **holdout NLL best** | HO NLL at the epoch validation picks | cost of selecting on validation |
|---|---|---|---|---|
| `diag_emb` | epoch 10 (1.0333) | **epoch 3 (1.0194)** | 1.0721 | **+0.053 nats** |
| `diag_noemb` | epoch 34 (0.9946) | **epoch 4 (1.0158)** | 1.1704 | **+0.155 nats** |

Three conclusions, in increasing order of how much they cost us.

**1. The confound is resolved: it is training length, not architecture.**
Within a single run, holding architecture fixed, holdout performance peaks at
epoch 3-4 and then degrades monotonically. In `diag_noemb`, validation NLL
*improves* for 34 consecutive epochs while holdout NLL rises 1.0158 → 1.1704.
Validation never signals to stop, because it sits in the calibration period
and the damage is to a later period it cannot observe.

**2. The two architectures are equivalent.** At their own optima they are
within noise of each other — 1.0194 (emb, ep 3) against 1.0158 (noemb, ep 4),
with the *no-embedding* model marginally ahead. The apparent superiority of
the embedding in R11 and R13's predecessors was entirely an artefact of it
early-stopping at epoch 8 rather than 23, which happened to land nearer the
holdout optimum. It was luck about when patience ran out, not a better model.

**3. Every ranking we have drawn is void.** Selected epochs across the runs in
R6, R7 and R11 range from 8 to 35, and holdout quality is monotone decreasing
in that range. Those comparisons measured which run stopped earliest, not
which configuration is better. That includes the R7 mixture-head result.

> **Correction (Sep 11).** This paragraph originally also voided the
> heterogeneity reading. That was wrong: the in-sample vs out-of-sample
> comparison is made *within* a single checkpoint, so stopping epoch is held
> fixed by construction. It survives, and R15 now controls it properly.

**Interpretation, with a caveat.** The leading explanation is genuine
distribution shift between the calibration and holdout periods: the model
learns calibration-period structure that does not transfer. An alternative is
that an epoch-3 model is simply under-trained and sits near the marginal class
frequencies, which happen to suit the holdout period better. These are
distinguishable — compare the epoch-3 and epoch-34 predicted class
distributions against each period's empirical distribution — and that check
should be run before the finding goes in a paper.

**The fix is not a different metric.** Validation must be temporally shifted
from training: drawn from the *late* calibration campaigns rather than spread
across the whole period, so early stopping can see drift. Then re-run the
architecture comparison at matched, honestly-selected epochs.

### R14 — the S=1024 arms, and what survives R13

| Run (S=1024, all `v2_`) | emb | aug | dropout | sel. epoch | val NLL | HO NLL | HO F1 |
|---|---|---|---|---|---|---|---|
| `emb` | Y | n | 0.10 | 8 | 1.0351 | 1.0636 | 0.576 |
| `noemb` | n | n | 0.10 | 23 | 0.9976 | 1.1575 | 0.497 |
| `noemb_aug` | n | **Y** | 0.10 | 23 | 0.9974 | 1.1602 | 0.496 |
| `noemb_aug_do25` | n | Y | **0.25** | 23 | 1.0070 | **1.0813** | **0.552** |

R13 showed holdout quality falls monotonically after epoch 3–4. So a
comparison between two runs is valid only if they stopped at the same epoch.
The three `noemb` arms all stopped at epoch 23, which makes them a clean
comparison by accident.

**Survives (matched epochs):**
- **Augmentation is a no-op.** 1.1575 vs 1.1602 at S=1024 (both ep 23);
  1.1810 vs 1.1782 at S=512 (both ep 30). Two matched pairs.
- **Dropout 0.25 helps.** 1.1602 → 1.0813 at matched epoch 23. Caveat: some of
  this may be dropout slowing the overfitting R13 exposed, i.e. the same
  mechanism as stopping earlier, rather than a separate benefit.
- **Context length barely matters.** `v2_emb` at S=512 and S=1024 both stopped
  at epoch 8: 1.0590 vs 1.0636.

**Void (different stopping epochs):**
- Embedding vs no embedding (8 vs 23). At honest optima they tie (R13).
- Mixture heads vs flat (29 vs 35).
- Anything pre-redesign vs anything after (different validation set).

### R15 — does knowing a customer help at the honest epoch?

Both diagnostic runs scored every holdout cell every epoch, so the in/out gap
can be traced through training. Gap = in-sample NLL − out-of-sample NLL;
negative would mean known customers are predicted better.

| epoch | `diag_noemb` gap | `diag_emb` gap |
|---|---|---|
| 3 (≈ holdout optimum) | +0.0144 | +0.0161 |
| 10 | +0.0084 | +0.0289 |
| 20 | +0.0103 | −0.0007 |
| 30 | +0.0123 | −0.1005 |
| 39 | +0.0121 | −0.1381 |

**`diag_noemb` is the control.** It has no customer-specific parameters, so it
treats both cohorts identically; its steady +0.01 to +0.02 gap is pure cohort
composition — the out-of-sample group is simply slightly easier to predict.

**At the holdout optimum the embedding adds nothing.** At epoch 3 the embedded
model's gap (+0.0161) matches the control's (+0.0144). Knowing a customer buys
no measurable advantage beyond the population mean ω̄.

**The late sign flip is not heterogeneity helping.** From epoch 17 the embedded
model's gap turns negative, but not because known customers improve — both
cells degrade, and the out-of-sample one degrades faster (1.0194 → 1.4590
against 1.0354 → 1.3209). Out-of-sample customers receive ω̄, the mean of the
trained embeddings; as those over-specialise, their mean becomes a poor stand-in
for anyone. That is an over-training artefact, and it is also why an
over-trained embedded model can *appear* to show heterogeneity when it does not.

### R16 — late validation tracks the holdout (gate passed)

Four runs, all S=1024, batch 4, dropout 0.10, no augmentation, no mixture
heads, holdout tracked every epoch.

| Run | Validation mode | Val picks | Holdout peaks | Cost |
|---|---|---|---|---|
| transformer, no embedding | late (campaign 27) | epoch 1 | epoch 1 | **0.0000 nats** |
| transformer, embedding | late | epoch 3 | epoch 3 | **0.0000 nats** |
| GRU baseline | late | epoch 17 | epoch 24 | **0.0006 nats** |
| transformer, no embedding | customers (old) | epoch 32 | epoch 2 | **+0.211 nats** |

The old design is worse than R13 measured (0.211 against 0.155), and the new
one is free on every architecture tried. `val_mode="late"` is now the config
default.

**Step 1 resolved: drift, not under-training.** The epoch-1 model beats the
best constant predictor by 0.63 nats, so it is not merely reproducing class
frequencies. Of the degradation that follows, 82% survives an oracle
prior-matching correction, i.e. it is a change in P(decision | history), not a
change in class frequencies (18%). The class mix does move between periods
(total variation 0.107), but that is not what costs the model its accuracy.

### R17 — the GRU baseline beats the transformer

Same feature pipeline (identical product-feature lookup, within-event pooling
and decision embedding); the only difference is how events combine over time.

| | GRU | Transformer (emb) | Transformer (no emb) |
|---|---|---|---|
| parameters | **669,077** | 1,905,430 | 1,215,382 |
| validation NLL | **0.9194** | 0.9933 | 1.0347 |
| in-sample × holdout NLL | **0.8968** | 1.0256 | 1.0320 |
| out-of-sample × holdout NLL | **0.8818** | 1.0062 | 1.0159 |
| holdout macro F1 | **0.605** | 0.590 | 0.585 |
| gain over constant predictor | **+0.765 nats** | +0.640 | +0.631 |

The recurrent baseline wins on every cell, by 0.12–0.15 nats, with 45% fewer
parameters than the smaller transformer. It also wins on the calibration-period
cell, so this is not a drift artefact.

Three caveats before this is quoted. It is a single seed. The GRU had **not
converged** — its holdout optimum is at the last epoch run (24 of 25), so its
number may improve further, while the transformers are at interior optima.
And neither architecture has been tuned; both use the same learning rate and
dropout, which need not suit both equally.

Batch 2 (24 runs, three seeds) tests whether it survives replication.

### R18 — why does the transformer peak at epoch 1–3? (hypothesis, pre-registered)

**Observation.** The transformer's holdout optimum is at epoch 1–3 and degrades
after; the GRU's is at 24+ and still improving; the GRU wins by 0.12–0.15 nats
on the same feature pipeline (R17). S=512 ≈ S=1024; dropout helps a lot;
the customer embedding adds nothing out of sample (R15).

**Hypothesis (H2).** Every product token is `id_embed(t) + γ·feat_proj(features)`.
The identity road is a lookup — fast to learn and a better fit to the
calibration data, because campaign-specific quirks live there. The LTO stream
is a campaign fingerprint (which products are on offer identifies the
campaign), so identity-conditioned patterns are campaign memorisation. Holdout
campaigns 28–30 have banners never seen in training; identity knowledge is
worthless there. Attention over the full history picks this shortcut up
easily; a GRU's 128-d recurrent bottleneck cannot carry campaign identity as
easily and is pushed toward the attribute road, which transfers.

**Predictions, stated before the runs.**
1. `tf_noid` (products by attributes only): the holdout optimum moves to a
   LATER epoch and improves. If it does not move, H2 is wrong.
2. `gru_noid`: a smaller gain than for the transformer, or none.
3. `tf_alibi` (recency bias): later optimum, some improvement — a remedy for
   the same problem by a different route.
4. `gru_cross` vs `gru_nocross` and `b2_tf_noemb` vs `tf_nocross`: whether the
   O(S²) offer-inventory cross-attention earns its cost on either encoder.

**Design.** Six screening runs, seed 1, S=1024, no customer embedding, late
validation, holdout tracked every epoch (`TRACK_HOLDOUT=1`) so the position of
the optimum is observed, not inferred. `scripts/submit_batch3.sh`. Winners get
three seeds. Together with batch 2's `tf_noemb` and `gru` arms these complete
the 2×2 {encoder: attention, GRU} × {cross-attention: on, off}.
