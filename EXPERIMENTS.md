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
| 41123–41194 | `b13_*` (72 runs) | 2026-09-20 | R29: vocabulary × decision × stock factorial | 45 done, 27 left (qstat 2026-09-21) |
| 41207–41254 | `b14_*` (48 runs) | 2026-09-21 | R30 stage 1: capacity frontier, level 7 | queued behind batch 13 (qstat 2026-09-21) |

Batch 7 (R22) is closed at two seeds per arm; batch 8 (R23) is complete.
Batches 10-12 are complete (R25, R27, R28). Batch 13 (R29) was submitted
2026-09-20: 72 runs, seed-major, read with
`python3 scripts/summarize_batch.py --prefix b13_`. Tags are
`b13_v{6,7}_<decision>[_<stock>]_s<seed>`.

Batches 1-4 (40852-40903) are complete and recorded as R16-R19. Batch 5
(40904-40909) is complete: 40904-40907 void (R20); 40908-40909 ran the lagged
clock under batch-5 tags and are recorded under R21 as an early read. Batch 6 (40922-40924) is complete and
recorded as R21. Batches 7–9 (R22–R24) submitted 2026-09-15, 36 jobs; read each with
`python3 scripts/summarize_batch.py --prefix b7_` (then `b8_`, `b9_`). Batch 9
is read against batch 8, its single-step control.

**Reading batch-5-tagged runs:** trust `cfg.time_bias_lag_ipt` in `final.json`,
not the tag. Absent or False = leaky (void); True = lagged (valid).

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
| R18 | Sep 13 2026 | Component ablation + recency bias + attributes-only products (batch 3) | Early holdout peak = campaign memorisation via product identity | **Prediction failed; the cause is recency.** Attributes-only did not move the optimum (still epoch 3). ALiBi moved it to epoch 19 and closed the gap: 0.888 vs GRU 0.880 +/- 0.008. Cross-attention worth 0.03-0.04 nats on both encoders | Batch 4: seed the winners and combine ALiBi with dropout 0.55 |
| R19 | Sep 13 2026 | Batch 4: tf_alibi x {plain, do25, do55, noid} and gru_cross, 3 seeds each | Attention + recency + cross-attention beats the GRU by more than seed noise | **No. A tie.** tf_alibi 0.886 +/- 0.006 vs GRU 0.880 +/- 0.008; paired seeds split 1-1 with one identical to 4 d.p. Dropout 0.55 now HURTS (+0.042); attributes-only +0.012; gru_cross 0.889 gains nothing over the plain GRU. Per-class profiles are indistinguishable | Pre-registered rule applies: on ordinal recency, attention buys interpretability, not accuracy |
| R20 | Sep 13 2026 | Recency measured in HOURS rather than events (batch 5, 6 runs) | Elapsed-time recency beats ordinal recency, because the event index mixes a minutes-scale burst clock with a weeks-scale silence clock | **VOID: label leakage.** 0.706 +/- 0.051, a 0.18-nat "gain" in every cell including calibration. Row t's clock included IPT_t, the gap ending at t, which carries 0.40 nats about y_t (gap 23.5-24.5h => NotBuy 99.3%; gap 0 => NotBuy 0.6%) | Lag the clock one row (now default); rerun as R21. Batch-5 numbers must never be cited |
| R21 | Sep 14 2026 | Recency in hours, clock lagged one row (batch 6, 3 seeds) | Honest elapsed-time recency beats ordinal recency (0.886) by more than seed noise | **No — it is worse.** 0.9005 +/- 0.008 vs ordinal 0.8860: +0.0145, worse on all 3 paired seeds (sd of the difference 0.002). Trails the GRU by 0.020. Leak confirmed gone (calibration cell 0.981) | Keep the ordinal ruler. Calendar time does not help as an attention kernel; timing goes to the continuous-time model, or a hybrid ruler is tested first |
| R22 | Sep 15 2026 | Capacity sweep: depth 2/4/8 and width 128/256, transformer + ALiBi and GRU (batch 7, 16 of 21 runs) | The behavioural mechanisms are deep, so more layers or width improve prediction | **No.** Every arm within 0.02 of its 4x128 reference and every deviation worse; 8x the parameters (885k to 7.42M) moves the transformer less than seed noise. Deeper transformers peak earlier (epoch 10.5 vs 18.7) | Capacity is not the constraint. Closed at 2 seeds per arm to free the queue for R23-R24 |
| R23 | Sep 15 2026 | Additive inventory slots on both backbones (batch 8) | The gen-5 inventory modules add nothing (−0.009 over a plain GRU) because inventory is mis-specified, not because satiation is unimportant | **Prediction failed.** Slots are WORSE on the transformer (0.9011 vs 0.8860, 0/3 seeds) and a wash on the GRU encoder (0.8916 vs 0.8889, 1/3). Neither beats the plain GRU (0.8802). The three data problems are real; fixing them buys nothing | The inventory path is not where the missing signal is. Batch 9 tests whether depth on slots changes that; if not, satiation is a small effect on this data |
| R24 | Sep 15 2026 | Deep satiation module on additive slots (batch 9) | Satiation needs several layers of offer–inventory computation | **Half held.** Depth beats the single step on slots (0.8908 vs 0.9011, 3/3 seeds) but does not recover the token baseline (0.8860) and never beats the plain GRU (0.8802, 0/3). 4 layers = 2 layers | Depth where the concept lives is real but small; the ceiling holds. Slots are not adopted |
| R25 | Sep 16 2026 | Recurrence + attention over past occasions (batch 10, 9 runs) | The two memories carry different information: recurrence supplies the decay prior, attention retrieves specific past occasions by content | No gain: best hybrid `hyb_stack` 0.8865 ± 0.0079, level with its better parent (0.8860), above the plain GRU (0.8802) | Not adopted. Combining the memories does not help; with R21–R24, the information-ceiling reading stands |
| R26 | Sep 17 2026 | Pre-build data checks for the additive attention stock; then verification of every product-id mapping (no GPU) | Copy tiers and substitution have variation to learn | **Checks void. The obtained-products stream is CONFIRMED corrupted in both gen-4 (`simple6`) and gen-5 (`_IPT`) data:** version-2 `NewProductIndex` codes are decoded with the version-6 table, which renumbered 89 of 122 products. 45 products get the wrong id; two 3-star weapons become Raiden Shogun and Xiao (92% of all "limited-time 5-star" acquisitions). Offers are correct except a campaign-30 A/B swap | Every result that reads the obtained stream is void until regenerated: inventory GRU, satiation attention, slots, deep satiation, Table 1 attribution, and gen-4 inventory results. First confirmed cause (key mismatch) was wrong; see correction below |
| R27 | Sep 17 2026 | Batch 11: core replication on the corrected obtained stream (15 runs) | The architecture ranking and the value of reading inventory survive the R26 correction | **Ranking survives, inventory does not.** GRU 0.8892 ± 0.0172 vs transformer 0.9083 ± 0.0055 (gap 0.0191, GRU wins 3/3 paired seeds); the offer-inventory attention now adds NOTHING on either encoder (−0.0041 GRU, +0.0005 transformer). Everything is worse than pre-fix, the transformer most (+0.0223) | Phase 2 skipped per rule 2. Safety rule 3 fired and is explained: correcting the stream cut its entropy 1.050 → 0.601 nats, because two common 3-star weapons lost their own ids. Next test is R28 (one id per product) |
| R28 | Sep 18 2026 | Batch 12: one token per product, 118 ids (9 runs) | Token resolution is what the pooled vocabulary was costing: per-product ids recover the ~0.02 nats R27 lost | **Neither prediction. A CROSSOVER.** Per-product ids HURT the GRU (0.8892 → 0.9096) and HELP the transformer (0.9083 → 0.8985). At level 7 the transformer beats the GRU by 0.0111, 3/3 paired seeds — the first time any transformer has led | Rule 3 fired: product identity is something attention exploits and recurrence cannot. The architecture question is reopened in the transformer's favour; batch 13 widens the level-7 comparison |
| R29 | Sep 20 2026 | Batch 13: full factorial, vocabulary × decision memory × stock memory (24 configs, 72 runs) | The R28 crossover is an interaction between vocabulary and decision-level memory, not a capacity artefact, and per-product counts finally make the stock path pay | _pending_ | _pending_ |
| R31 | Sep 21 2026 | Customer-level test of the vocabulary × architecture crossover, and where it lives (inference only, no training) | The crossover is product RETRIEVAL: real at the customer level, concentrated on offers of already-owned products, absent on the placebo | _pending_ | _pending_ |

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

**Default changed (14 Sep 2026).** `use_user_embedding` is now `False` in
`config5.py` and the model builders; `--user-embedding` (PBS `USER_EMB=1`) turns
it on. Every reported run since batch 2 already passed `USER_EMB=0`, so no
result changes. `submit_batch2.sh`'s `tf_emb` arm now passes `USER_EMB=1`
explicitly, so rerunning it reproduces the original arm.

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

### R18 — results: the mechanism was recency, not product identity

Batch 2 first, since it is the reference. 24 runs, three seeds each, late
validation, S=1024. Out-of-sample × holdout, mean ± sd at the validation-
selected epoch:

| config | sel. epoch | params | **NLL** | F1 |
|---|---|---|---|---|
| **gru** | 14.7 | 669,077 | **0.8802 ± 0.0078** | 0.608 |
| lstm | 19.7 | 801,173 | 0.9013 ± 0.0177 | 0.605 |
| tf_noemb_do55 | 12.7 | 1,215,382 | 0.9971 ± 0.0129 | 0.582 |
| tf_noemb_do25 | 3.3 | 1,215,382 | 1.0142 ± 0.0077 | 0.583 |
| tf_emb | 2.3 | 1,905,430 | 1.0151 ± 0.0058 | 0.581 |
| tf_noemb | 3.0 | 1,215,382 | 1.0203 ± 0.0243 | 0.578 |
| tf_noemb_do40 | 6.0 | 1,215,382 | 1.0225 ± 0.0108 | 0.580 |
| tf_mix8 | 3.7 | 1,271,676 | 1.0436 ± 0.0036 | 0.572 |

R17 replicates: the GRU beats every transformer variant in 3/3 paired seeds,
by ≥ 0.117 nats, against a typical seed sd of 0.009. Two smaller results:
dropout 0.55 is the best transformer setting and pushes its selected epoch
from 3 to 12.7; the mixture head is the *worst* transformer under honest
selection, and the customer embedding is a wash (1.0151 vs 1.0203), as R15
predicted.

Batch 3 screening runs, seed 1, holdout tracked every epoch. The seed-1
references are `b2_tf_noemb_s1` (1.0187, epoch 3) and the GRU mean above.

| arm | val picks | holdout peaks | **NLL at pick** | gain vs constant |
|---|---|---|---|---|
| tf_noid (attributes only) | 3 | 3 | 1.0038 | +0.643 |
| **tf_alibi (recency bias)** | 16 | 19 | **0.8881** | **+0.767** |
| tf_nocross | 4 | 3 | 1.0613 | +0.602 |
| gru_noid | 26 | 26 | 0.8726 | +0.774 |
| gru_cross (GRU encoder + cross-attn) | 20 | 21 | 0.8797 | +0.768 |
| gru_nocross | 10 | 14 | 0.9137 | +0.741 |

**Prediction 1 failed.** Removing product identity did *not* move the
transformer's optimum (still epoch 3) and improved it by only 0.015 nats,
about 1.5 seed-sd. The campaign-memorisation hypothesis is wrong, or at most
a minor contributor. It was pre-registered so that this could be said plainly.

**Prediction 3 succeeded, and it is the whole story.** With a recency bias
the optimum moved from epoch 3 to epoch 19, the gain over a constant
predictor rose from +0.63 to +0.77 nats — the GRU's level — and holdout NLL
fell from 1.019 to 0.888, within one seed-sd of the GRU's 0.880 ± 0.008.
The transformer was not memorising campaigns; it had **no way to represent
recency at all**. With only a causal mask the past is an unordered bag, so
what it learned in the first epochs was whatever a bag-of-history model can
learn, and further training fitted bag-level correlations specific to the
calibration era. The GRU never had that problem because recurrence is
recency. One additive bias with no parameters removes the gap.

**Prediction 4: the cross-attention earns its cost on both encoders.**
Removing it costs the transformer 0.043 nats (1.0187 → 1.0613) and the GRU
encoder 0.034 (0.8797 → 0.9137), 3–4 seed-sd each. Note the GRU encoder
*without* cross-attention (0.9137) is worse than the plain GRU baseline
(0.8802), so the inventory GRU on its own does not help; with the
cross-attention it recovers to parity. Single seed — batch 4 replicates.

**What this does and does not establish.** A recency-biased transformer
*ties* a GRU with 45% fewer parameters. That is not an advantage. Whether
one exists is the question batch 4 asks, by combining the recency bias with
the regularisation that helped (dropout 0.55) and replicating across seeds.

### R19 — pre-registered: can attention + recency + cross-attention beat the GRU?

Five configurations, three seeds each (`scripts/submit_batch4.sh`):
`tf_alibi` and `gru_cross` replicated; `tf_alibi` + dropout 0.25, + dropout
0.55, + attributes-only products.

Predictions. `tf_alibi` replicates within 0.01 of 0.888. Dropout 0.55 helps
it by roughly what it helped the plain transformer (~0.02), which would put it
at ~0.87, marginally ahead of the GRU; whether that clears seed noise
(~0.009) is the test. Attributes-only adds little (R18). `gru_cross`
replicates at ~0.88, i.e. the GRU gains nothing from the attention encoder
but the cross-attention is worth keeping on either.

If no transformer variant beats the GRU by more than ~0.02 across seeds, the
honest conclusion is that on this data attention buys interpretability (the
satiation cross-attention) and not accuracy, and the paper should say so.

### R19 — results: a tie, and the rule applies

Out-of-sample × holdout, validation-selected epoch (late validation, S=1024):

| config | seed 1 | seed 2 | seed 3 | mean ± sd | sel. epoch |
|---|---|---|---|---|---|
| gru (batch 2) | 0.8720 | 0.8812 | 0.8876 | **0.8802 ± 0.0078** | 14.7 |
| tf_alibi | 0.8924 | 0.8812 | 0.8844 | **0.8860 ± 0.0057** | 18.7 |
| gru_cross | 0.8892 | 0.8928 | 0.8847 | 0.8889 ± 0.0040 | 15.0 |
| tf_alibi_do25 | 0.8970 | 0.8827 | 0.8984 | 0.8927 ± 0.0087 | 24.0 |
| tf_alibi_noid | 0.8910 | 0.9056 | 0.8984 | 0.8983 ± 0.0073 | 20.7 |
| lstm (batch 2) | 0.8887 | 0.8935 | 0.9215 | 0.9013 ± 0.0177 | 19.7 |
| tf_alibi_do55 | 0.9263 | 0.9297 | 0.9281 | 0.9280 ± 0.0017 | 14.0 |

Against the predictions:

1. *tf_alibi replicates within 0.01 of 0.888* — **held** (0.886).
2. *Dropout 0.55 helps by ~0.02* — **failed, in the opposite direction**: it
   costs 0.042. Heavy dropout was compensating for the overfitting that the
   missing recency signal caused; once recency is present it only removes
   capacity. The batch-2 regularisation result does not transfer.
3. *Attributes-only adds little* — **held**, and slightly negative (+0.012).
4. *gru_cross replicates at ~0.88* — **held** (0.889). On a GRU backbone,
   gen 5's extras (inventory GRU + satiation cross-attention) add nothing.

**The decision rule fires.** No transformer variant beats the GRU; the best
one trails by 0.006, under one seed-sd, and the paired seeds split. Per-class
precision, recall, F1 and AUPRC of the two models agree to within 0.03 on
every class, so they are not two different models that happen to tie on
average — they extract the same signal. On this data, with ordinal recency,
attention buys interpretability (the satiation cross-attention) and not
accuracy.

### R20 — pre-registered: is recency better measured in hours than in events?

R18 established that the transformer's deficit was the absence of any recency
signal; a zero-parameter ordinal bias closed the whole gap to the GRU. But the
ordinal index is the wrong ruler for this data. R9 measured why: 46.5% of true
inter-purchase gaps round to zero at the field's 36-second resolution, while
the longest is 2,401 hours. Ten pulls four minutes apart and a three-week
silence are both "one event ago".

**Implementation.** `--time-bias time` replaces the ordinal distance with
elapsed hours, taken from `IPT.cumsum()` over the row sequence (correct in the
discrete representation, where inserted rows are rows — the censoring caveat
in R9 only applies if inserted rows are dropped). The penalty is
`s_h · log1p(t_i − t_j)` with a learnable per-head decay: log1p compresses the
0–2,401 h range to 0–7.8, so the fixed ALiBi schedule would be far too gentle,
and the rate is learned rather than guessed (initialised at 16× the ALiBi
values, softplus-constrained positive). Cost: **+n_heads parameters**.

Verified by unit test before submission — for the same event-index distance of
five steps, the head-0 penalty is −0.32 across a burst of one-minute gaps and
−26.94 across weekly gaps; causal masking, zero self-penalty, monotonicity and
gradient flow to the decay rates all hold. `--time-bias ordinal` reproduces
`--alibi` numerically, so the arms differ only in the ruler.

**Predictions.** Time-based recency beats ordinal by more than seed noise
(~0.009 nats), because it can distinguish the two clocks. If it merely ties,
the useful signal is "recency of any kind", which is a weaker but still
publishable claim and points at the continuous-time model rather than at
better position encoding. If it is *worse*, the likely cause is the learnable
decay being harder to optimise than a fixed schedule, and a fixed-scale
variant should be tried before abandoning it.

**Design.** `tb_time` and `tb_time_do55`, three seeds each
(`scripts/submit_batch5.sh`). The ordinal comparison is batch 4's `tf_alibi`
and `tf_alibi_do55` at the same three seeds, so only the ruler changes.

### R20 — results: void, the clock leaked the label

Three seeds of `tb_time` finished at **0.706 ± 0.051** out-of-sample × holdout
(0.668 / 0.685 / 0.763), hit rate 0.79 against 0.69 for every other model. The
gain was the same size in the out-of-sample × *calibration* cell (0.731 vs
0.977), where recency has no drift to fix, and the seed spread was six times
the usual. A mechanism that helps only across time should not do that.

**The channel.** IPT at row t is the gap that *ends* at row t. Query t's bias
toward key t−1 is −s·log1p(IPT_t), so the attention pattern at t encodes
IPT_t. But rows exist because of outcomes — a draw creates a row, a quiet day
creates an inserted NotBuy at the 24-hour mark — so the row's own timestamp
says what kind of row it is. `scripts/ipt_leak_check.py`, all 3,017,180
labelled rows, aggregates only:

| conditioning on | H(Y \| ·) nats | information | modal hit |
|---|---|---|---|
| nothing | 1.7062 | — | 0.365 |
| own gap IPT_t (leaky) | 1.3041 | **0.402** | 0.503 |
| previous gap IPT_t−1 (legitimate) | 1.5484 | 0.158 | 0.425 |

Two bins do most of the work: gaps that round to 0 h (29.7% of rows) are
NotBuy 0.6% of the time; gaps of 23.5–24.5 h (22.4% of rows) are NotBuy 99.3%.

Meeting 41's deck had flagged exactly this risk ("the gap may itself identify
[inserted rows] and re-create the leak") before R20 was built. It was not
checked. Lesson recorded: any field produced by the same process that creates
rows must be tested against the label *before* a model consumes it.

**The model uses it.** `scripts/ipt_leak_counterfactual.py` re-scored the three
batch-5 checkpoints locally (bf16), reproducing the HPCC numbers to within
0.001, then again with the clock stopped at row t−1:

| out-of-sample × holdout | seed 1 | seed 2 | seed 3 | mean | hit |
|---|---|---|---|---|---|
| as trained (leaky clock) | 0.669 | 0.686 | 0.763 | 0.706 | 0.793 |
| clock lagged at evaluation | 1.382 | 1.496 | 1.444 | **1.441** | 0.590 |

Losing the leak costs 0.73 nats, leaving the models worse than a transformer
with no recency at all (1.020): the weights were built around it. This bounds
reliance on the leak; it does not estimate a properly trained lagged model.

**Fix.** `time_bias_lag_ipt=True` (now the default; `--leaky-time-bias`
reproduces batch 5) rolls IPT forward one row, so row t's clock stops at row
t−1 — the same fix as R1's obtained stream. `scripts/test_time_bias_causality.py`
is the formal test: perturbing IPT_t leaves every logit at rows ≤ t unchanged
(exactly 0.0) and moves later rows; with the leaky clock row t moves.

**Found alongside: PyTorch's eval fast path mishandles the additive mask.**
Without autocast, eval-mode output of the biased encoder differed from
train-mode output (dropout 0) by ~2.4, for both ordinal and time biases, and a
changed gap altered only one row. With bf16 autocast the fast path is skipped
and the two agree exactly. All 18 batch-4/5 job logs show
`amp=torch.bfloat16` on CUDA, so **no reported number is affected**, but a CPU
or fp32 evaluation would have silently scored a different model. The encoder
now disables the fast path around its call; the same test checks eval = train.

### R21 — pre-registered: honest recency in hours (batch 6)

`tb_lag`, three seeds (`scripts/submit_batch6.sh`): identical to batch 5's
`tb_time` except the clock is lagged. Comparison is batch 4's `tf_alibi`
(0.886 ± 0.006) and the GRU (0.880 ± 0.008) at the same seeds.

Predictions, written before the runs:

1. The leak is gone: out-of-sample × calibration NLL returns to the
   0.97–0.99 band of every other model. If it stays near 0.73, a second
   channel exists and the result is void again.
2. Lagged time beats ordinal by **0.00–0.02 nats**. The legitimate signal
   (0.16 nats in IPT_t−1) is real, but it reaches the model only through how
   attention weights are spread, a weak channel.
3. If it beats the GRU by more than ~0.02, the fair follow-up is a GRU given
   lagged log1p(IPT) as an input feature, before crediting attention: the
   information, not the architecture, would be the source.

### R21 — early read: two accidental lagged runs (dropout 0.55)

Jobs 40908-40909 started after the fix was pulled, so they ran the lagged
clock with dropout 0.55 under `b5_tb_time_do55_s2/s3` (`final.json` confirms
`time_bias_lag_ipt=True`). Like-for-like comparator: batch 4's ordinal
`tf_alibi_do55` at the same seeds. Out-of-sample customers:

| seed | lagged hours: holdout | ordinal: holdout | Δ | lagged hours: calibration cell |
|---|---|---|---|---|
| 2 | 0.9527 | 0.9297 | +0.023 | 1.061 |
| 3 | 0.9345 | 0.9281 | +0.006 | 1.037 |

- **Prediction 1 holds: the leak is gone.** The out-of-sample × calibration
  cell is back at ~1.04-1.06, the band of every honest dropout-0.55 model
  (ordinal: 1.02-1.03), not 0.73.
- **Prediction 2 is not supported so far:** the lagged time ruler is *worse*
  than ordinal by 0.015 on average, in both seeds. Two seeds, and at dropout
  0.55, the setting R19 showed is harmful with recency — so this does not
  settle R21. Batch 6 (plain dropout, three seeds) does.
- For the record, the leaky arm at dropout 0.55 (40907, `_s1`, void) scored
  0.902: heavy dropout limited how far that model exploited the leak, compared
  with 0.668 for the same seed at dropout 0.10.

### R21 — results: calendar-time recency is worse than event order

Batch 6, `tb_lag`, three seeds, stopped early at epochs 28 / 21 / 18 with the
selected checkpoints at 22 / 15 / 12. Out-of-sample customers:

| seed | lagged hours: holdout | ordinal (b4): holdout | Δ vs ordinal | GRU (b2) | lagged hours: calibration cell |
|---|---|---|---|---|---|
| 1 | 0.9094 | 0.8924 | +0.0170 | 0.8720 | 0.9745 |
| 2 | 0.8943 | 0.8812 | +0.0131 | 0.8812 | 0.9855 |
| 3 | 0.8979 | 0.8844 | +0.0135 | 0.8876 | 0.9824 |
| **mean** | **0.9005 ± 0.0079** | **0.8860** | **+0.0145** | **0.8802** | **0.9808** |

Macro F1 0.605 and macro AUPRC 0.602 are below the ordinal arm's 0.607 and
0.610; hit rate is marginally higher (0.694 vs 0.693). The NLL loss is not
offset elsewhere.

Against the predictions:

1. *The leak is gone: the calibration cell returns to 0.97–0.99* — **held**
   (0.974–0.986, against 0.731 for the leaky batch 5).
2. *Lagged time beats ordinal by 0.00–0.02* — **failed.** It is worse by
   0.0145, and the difference is unusually stable: 0.013–0.017 on every seed,
   so this is not seed noise. The two accidental dropout-0.55 runs (R21 early
   read) pointed the same way.
3. *If it beats the GRU, test the information in a GRU* — does not arise.

**Interpretation (not yet tested).** On this grid, event distance already
encodes calendar time over quiet periods, because every silent day adds an
inserted row. What the hours ruler adds is a distinction *within bursts* — and
that is where it loses information: 46.5% of gaps round to zero, so
log(1 + hours) is near zero across a whole burst and the kernel treats every
pull in it as equally recent. The ordinal ruler still orders them. A hybrid
ruler (ordinal and time penalties added, each with its own slope) would test
this directly: if the explanation is right, it should at least match ordinal.
R20's pre-registration also named a second candidate — the learnable decay
being harder to optimise than a fixed schedule — which a fixed-slope time arm
would separate.

**Decision.** Keep the ordinal ruler as the recency kernel. Calendar time does
not earn its place as an attention bias on the discrete grid, which strengthens
the case for handling timing as an outcome in the continuous-time model rather
than as a feature.

### R22–R24 — pre-registered: capacity, additive inventory, deep satiation

**Why now.** R19 left the transformer tied with the GRU. Three
specification problems in the gen-5 inventory path were found on 15 Sep,
by reading the code and measuring the data (`scripts/inventory_dilution_check.py`,
aggregates only):

1. **Satiation gets one layer of computation.** The offer queries the inventory
   through a single attention step (no feed-forward, no residual, no stacking),
   and the 4-layer sequence model after it never re-reads the inventory.
2. **The inventory is not additive and decays on quiet days.** Acquisitions are
   pooled per row and passed through a GRU that gates and forgets; 37.9% of
   rows have an empty obtained block, and the GRU updates on every one.
3. **The inventory memory is diluted.** At the end of calibration the median
   customer's satiation attention spans 643 obtained tokens for 13 distinct
   products (p90: 1,322 for 19) — 50 tokens per product — and 72.7% of tokens
   are 3-star items. Items obtained before the 1,024-row window are invisible.

**R22 — capacity (batch 7, `scripts/submit_batch7.sh`).** Transformer + ALiBi
at 2 and 8 layers (width 128), width 256 (4 layers) and 8 × 256; plain GRU at
2 and 8 layers and width 256. Three seeds; references are `b4_tf_alibi`
(0.886) and `b2_gru` (0.880).
Prediction: **every arm within ±0.02 of its 4 × 128 reference.** The deficit
is in structure (point 1), and depth in the sequence stack cannot reach the
inventory. Wider or deeper transformers may also select earlier epochs.
Decision rule: capacity changes are adopted only if they beat the reference by
more than 0.02 on the three-seed mean.

**R23 — additive inventory slots (batch 8).** Replace the inventory GRU and the
token memory with one slot per product (at most 44): product embedding plus
log(1 + cumulative count) and log(1 + occasions since last acquired), counted
over the FULL history before truncation and lagged to occasion t−1; plus an
attribute stock (counts × the 34 product attributes). Keep the single-step
cross-attention, now over slots, so representation is isolated from depth.
Run on the transformer + ALiBi backbone and the GRU encoder backbone.
Predictions: (a) the GRU encoder with slots beats the plain GRU (0.880), turning
the −0.009 into a gain; (b) the transformer with slots beats `b4_tf_alibi`
(0.886); (c) the causality test passes — perturbing o_(t−1) never moves rows
before t. Memory falls roughly 230-fold in the cross-attention.

**R24 — deep satiation (batch 9).** On slots: 2 and 4 stacked blocks of
[offer self-attention → cross-attention to slots → feed-forward], residual and
norm, keeping one representation per offer slot. Prediction: a further gain
over R23's single step; if none, satiation is shallow and point 1 was not the
bottleneck.

**Step 4 (two-timescale campaign memory)** is conditional on R23–R24 leaving
cross-campaign effects under-captured.

### R22 — interim (seeds 1–2 of 3): capacity does not help

Out-of-sample × holdout, two seeds per arm unless noted; references at
4 layers × width 128 are `b2_gru` 0.8802 and `b4_tf_alibi` 0.8860. No job
failed and none ran out of memory (peak 20.5 GB of 44 GB at 8 × 256).

| arm | depth × width | params | holdout NLL | vs its 4 × 128 reference |
|---|---|---|---|---|
| gru_d256 | 4 × 256 | 2.62M | 0.8787 ± 0.0056 | −0.0015 |
| gru_N2 | 2 × 128 | 471k | 0.8803 ± 0.0053 | +0.0001 |
| tf_N8_d256 | 8 × 256 | 7.42M | 0.8883 ± 0.0034 | +0.0023 |
| gru_N8 | 8 × 128 | 1.07M | 0.8929 ± 0.0088 | +0.0127 |
| tf_N8 | 8 × 128 | 1.88M | 0.8974 ± 0.0123 | +0.0114 |
| tf_N2 (3 seeds) | 2 × 128 | 885k | 0.8978 ± 0.0110 | +0.0118 |
| tf_d256 | 4 × 256 | 4.79M | 0.9005 ± 0.0156 | +0.0145 |

**The prediction holds so far.** Every arm is within ±0.02 of its reference,
and every deviation is *worse*, not better. An 8× parameter increase (885k →
7.42M) moves the transformer by less than seed noise, and the best capacity
arm overall is a GRU whose 0.0015 edge is a fifth of a seed-sd. Deeper
transformers also select earlier epochs (10.5 at 8 × 256 against 18.7 for the
4 × 128 reference), the signature of capacity being spent on the calibration
period rather than on transferable structure.

**Seed 3 was cancelled for five of the seven arms** (40954–40958, 2026-09-16)
so that batches 8–9 could start: fixing the representation comes before tuning
capacity on top of it. R22 therefore rests on two seeds per arm, except
`tf_N2` and `tf_N8`, which have three. That is enough for the conclusion drawn
here — every arm within 0.02 and every deviation worse — but not enough to
rank arms against each other. The cancelled arms can be resubmitted with
`bash scripts/submit_batch7.sh` once a representation is chosen; the run
directories are stable per configuration. Read alongside R23–R24: if capacity is flat
while a *representation* change moves the number, the bottleneck is where the
computation is spent, not how much of it there is.

### R23 — results: the additive inventory does not help

Out-of-sample × holdout, three seeds, against matched controls from batch 4:

| arm | holdout NLL | control | Δ | seeds won | sel. epoch | calibration cell |
|---|---|---|---|---|---|---|
| transformer + ALiBi + slots | 0.9011 ± 0.0071 | `b4_tf_alibi` 0.8860 | **+0.0151** | 0/3 | 7.0 | 0.9936 |
| GRU encoder + slots | 0.8916 ± 0.0043 | `b4_gru_cross` 0.8889 | +0.0028 | 1/3 | 15.7 | 0.9810 |
| — | — | plain GRU 0.8802 | +0.011 / +0.021 | 0/3 | — | 0.9811 |

Against the predictions:

1. *The GRU encoder with slots beats the plain GRU, turning −0.009 into a gain*
   — **failed.** It is 0.0114 worse than the plain GRU, and only 0.003 from the
   token-based version it replaced: a wash, not a gain.
2. *The transformer with slots beats `b4_tf_alibi`* — **failed**, and by more
   than seed noise: +0.0151, worse on every seed.
3. *The causality test passes* — **held** (checked before submission).
   Memory fell from 5.1 GB to 0.33 GB at S=1024, as expected.

**What this rules out.** The three data problems behind R23 are real and
measured — the inventory GRU is not additive and decays on empty rows, the
token memory spans 643 tokens for 13 distinct products, and truncation hides
earlier acquisitions. Fixing all three at once does not improve prediction.
So they were not what limited the model.

**What it points to.** Two readings, not yet separated:

- *The compression loses something.* Slots keep counts and recency per product
  but discard the ORDER and co-occurrence of acquisitions that the token
  memory retained. The transformer's selected epoch fell from 18.7 to 7.0 and
  its calibration-period score worsened too (0.9769 → 0.9936), which is what a
  less informative input looks like rather than a drift effect.
- *Satiation is simply a small effect here.* Consistent with R19, where the
  gen-5 inventory modules added nothing to a GRU backbone, and with the
  per-class profiles of the leading models agreeing to within 0.03 everywhere.

Batch 9 separates them in part: if depth on slots recovers the gap, the
representation was adequate and the single step was the limit; if not, the
inventory path is a small effect on this data whichever way it is written.
A cheaper follow-up if batch 9 also fails: keep BOTH memories (slots for
counts, tokens for order) and test whether the combination beats either.

### R25 — pre-registered: recurrence + attention (batch 10)

**Why.** Across 44 completed configurations no transformer has beaten the best
recurrent model; the honest summary is a tie at a third to a half the
parameters. The pre-2017 architecture (Bahdanau 2015; Luong 2015) combined
recurrence with attention rather than replacing one with the other, and that
combination has never been tested here at the sequence level — `--encoder gru`
swaps the stack but keeps only the offer-inventory cross-attention.

In this project's terms the two are different memory systems: recurrence is a
learned exponential decay (Guadagni & Little's loyalty variable with a gate),
attention is distance-indifferent content retrieval. R18 showed attention
alone fails without a decay prior; R22 showed more of either alone does
nothing.

**Arms** (`scripts/submit_batch10.sh`, three seeds, inventory = tokens):
`hyb_gate` (both branches, learned gate, ALiBi on the attention half),
`hyb_stack` (interleaved layer by layer), `hyb_gate_norec` (gate, no recency
bias on the attention half). Controls at the same seeds: `b4_gru_cross`
(0.8889) and `b4_tf_alibi` (0.8860); `b2_gru` (0.8802) is the benchmark.

**Predictions.**

1. *If the ceiling is information*, all three land within 0.02 of
   `b4_gru_cross`, and the gate settles above 0.6 on the recurrent branch —
   the model mostly ignoring retrieval.
2. *If retrieval carries something recurrence misses*, `hyb_gate` beats both
   parents by more than 0.02, and the gate sits nearer 0.5.
3. `hyb_gate_norec` ≈ `hyb_gate` if recurrence already supplies recency,
   making the bias redundant alongside it. If it is much worse, the attention
   half needs its own decay prior even next to a GRU.
4. `hyb_stack` > `hyb_gate` would say composition matters more than blending,
   and argues for a deeper hybrid.

**Decision rule.** Adopt the hybrid only if it beats the better parent by more
than 0.02 nats on the three-seed mean. Otherwise record that combining the two
memories does not help on this data, which — with R22 (capacity), R23
(inventory) and R21 (calendar time) — would make the information-ceiling
reading hard to avoid.

**Cost note.** A hybrid trains at recurrent speed, so epochs are slower than
the pure transformer; the attention over 1,024 hidden states is cheap by
comparison (~67 MB).

### R24 — results: depth helps the slots, but not enough

Three seeds, out-of-sample × holdout:

| arm | holdout NLL | sel. epoch | params | vs its control |
|---|---|---|---|---|
| `b9_tf_sat2` (2 satiation layers) | 0.8908 ± 0.0121 | 14.0 | 1.67M | **−0.0103 vs `b8_tf_slots`, 3/3 seeds** |
| `b9_tf_sat4` (4 layers) | 0.8912 ± 0.0054 | 9.3 | 2.13M | +0.0004 vs `b9_tf_sat2` |
| `b9_gru_sat2` | 0.8925 ± 0.0037 | 10.0 | 1.40M | +0.0008 vs `b8_gru_slots` |
| reference `b4_tf_alibi` (tokens, 1 step) | 0.8860 ± 0.0057 | 18.7 | 1.22M | — |
| reference `b2_gru` | 0.8802 ± 0.0078 | 14.7 | 669k | — |

1. *A further gain over R23's single step* — **held on the transformer**:
   −0.0103 nats, better on all three seeds. Depth in the offer-inventory
   computation is real.
2. But it **does not recover the token baseline** (+0.0048 vs `b4_tf_alibi`,
   better on only 1 of 3 seeds), and **nothing beats the plain GRU**
   (+0.0106, 0/3). Two layers and four layers are identical (+0.0004), so the
   depth that helps is one extra layer, not a stack.
3. On the GRU encoder depth adds nothing at all (+0.0008).

**Reading.** Splitting R23 and R24 was worth it: the slots representation
costs about 0.015 and depth on top returns about 0.010 of it. So the two
diagnoses were both half right — satiation *is* computed too shallowly, and
the additive representation *does* lose something the token memory had — but
together they leave the model where it started. The inventory path is not
where the missing signal is.

**Correction to an interim read.** On 16 Sep the single finished seed of
`b9_tf_sat2` scored 0.8783, the best transformer number in the project, and
was reported as such with a one-seed caveat. With three seeds it is
0.8908 ± 0.0121: the seeds were 0.8783 / 0.9024 / 0.8917. The caveat was the
operative part; the number was noise.

### R25 — results: combining the memories does not help

**Interim (Sep 16 2026, seed 1 of 3 — not a result).** Holdout NLL, out-of-sample
customers: `hyb_stack` 0.8776, `hyb_gate` 0.8929, `hyb_gate_norec` 0.9046. Gate
weight on the recurrent branch climbs from ~0.52 and settles at 0.59 (`hyb_gate`)
and 0.60 (`hyb_gate_norec`). Only `hyb_stack` is below `b2_gru` (0.8802), but
`b9_tf_sat2` also read 0.878 on seed 1 and finished at 0.8908 ± 0.0121, so no
reading is drawn until seeds 2–3 finish. Pre-registered adoption bar: better
parent (`b4_tf_alibi`, 0.8860) − 0.02 = three-seed mean below 0.8660. Beating the
plain GRU benchmark by the same margin would need below 0.8602.

**Result (Sep 17 2026, 3 seeds, all 9 runs finished).** Out-of-sample customers ×
holdout period, at the validation-selected epoch:

| Arm | Params | NLL | F1 macro | AUPRC | Revenue MAE | Gate on GRU (final, per seed) |
|---|---|---|---|---|---|---|
| `hyb_stack` | 1.61M | **0.8865 ± 0.0079** | 0.608 | **0.613** | **1.107** | — |
| `hyb_gate` | 1.15M | 0.8909 ± 0.0030 | 0.606 | 0.610 | 1.147 | 0.591 / 0.572 / 0.588 |
| `hyb_gate_norec` | 1.15M | 0.8960 ± 0.0086 | 0.609 | 0.608 | 1.164 | 0.601 / 0.594 / 0.602 |
| *parent* `b4_tf_alibi` | 1.22M | 0.8860 | | | | |
| *parent* `b4_gru_cross` | 0.95M | 0.8889 | | | | |
| *benchmark* `b2_gru` | 0.67M | 0.8802 | | | | |

Against the predictions:

1. **Ceiling (prediction 1) — holds on fit.** All three arms land within 0.01 of
   both parents; none clears the pre-registered bar (three-seed mean below
   0.8660). The gate settled at 0.57–0.60, just under the 0.6 predicted for a
   model ignoring retrieval: attention is used, but using it buys no likelihood.
2. **Retrieval carries new information (prediction 2) — rejected.** `hyb_gate`
   beats neither parent.
3. **Recency redundant next to a GRU (prediction 3) — mostly.** Dropping the
   bias costs 0.0051, under one seed sd; the gate barely moves.
4. **Composition over blending (prediction 4) — not established.** `hyb_stack`
   leads `hyb_gate` by 0.0043 (2/3 paired seeds), well inside noise, at 40% more
   parameters. It does have the best revenue error of the batch (1.107).

**The seed-1 lesson repeated.** `hyb_stack` read 0.8776 on seed 1 and finished at
0.8865 — the same regression `b9_tf_sat2` showed (0.8783 → 0.8908). Single-seed
leads in this project have not survived twice; interim readings stay unreported.

**Decision.** Not adopted. With R18 (campaign memorisation), R21 (calendar
time), R22 (capacity), R23 (inventory representation), R24 (satiation depth)
and now R25 (recurrence + attention), six explanations for the transformer–GRU tie have been tested and rejected. The
plain GRU (0.8802) remains the best model; the architecture queue is empty.

### R26 — the obtained-products stream is mis-coded (confirmed)

**How it was found.** `scripts/copies_substitution_check.py`, the approved
pre-build check for the additive attention stock, returned impossible copy
counts: 92% of limited-time 5-star acquisitions duplicates, 81% beyond the copy
cap. Aggregates against the game's draw rates: 1.02 limited-time 5-star ids per
character-banner 10-pull (expected ≈ 0.1); only 13–18% of them equal to the
product on offer; 3-star weapons 72% of tokens (expected ≈ 85%); 4-stars 12.5%
(correct). Both checks are void and not reported.

**Correction to the first diagnosis (same day).** The first write-up blamed a key
mismatch (ProductIndex values looked up in a NewProductIndex-keyed map). That is
**refuted**: ProductIndex codes for 3-star weapons are 301–313 and would pass
through unmapped, yet the obtained stream contains no value above 59 in 4.57M
tokens. The symptom was right; the stated cause was not.

**Confirmed cause** (`scripts/verify_product_index.py`, lookup tables read as
game metadata, JSON as aggregates):

- `GenerateDecisionSequenceCode.R` encodes items with **`FigureWeaponIndex2.xlsx`
  NewProductIndex**. Version 3 (Mar 2025) inserted two standard 5-star characters
  (F5-014, F5-036) mid-table, renumbering **89 of 122** products; versions 3–6
  agree with each other. ProductIndex is identical across all versions.
- `GenerateJSON.R` and `InsertNotBuy_GenerateJSON_IPT.R` decode the obtained
  stream with `map_vec_lv6` built from **`FigureWeaponIndex6.xlsx`**. Reading
  version-2 codes through version 6 mis-codes **45 of 122 entries**: 17
  characters, 25 weapons and 3 special tokens shift to a neighbour's id; two
  3-star weapons (W3-001, W3-002) land on ids 36 and 37 (Raiden Shogun, Xiao).
- **The sharp prediction held:** ids 36 and 37 hold **91.9%** of all obtained
  tokens with limited-time ids in `_IPT` and **92.5%** in `simple6` (5.1% if
  spread evenly). No other explanation predicts those two specific ids.

**Offers are correct.** Offer slots match `CampaignWideIndex.xlsx` on 100% of
3,017,180 rows (ProductIndex path, stable across versions). Two offer-side
notes: (1) campaign 30, `Figure5AIndex`/`Figure5BIndex` are swapped relative to
their names (Kazuha/Klee) — a holdout-period campaign; (2) standard 5-star
weapons featured on 21 of 60 early weapon-banner slots are pooled into W5 (id 17)
by design.

**Scope.** Both gen-4 (`simple6`, used by the previous manuscript) and gen-5
(`_IPT`) data. Void until regenerated: the inventory GRU and satiation
cross-attention (R19), slots and deep satiation (R23, R24), the satiation
attribution in Table 1, and any gen-4 claim that depends on obtained products.
Decisions, offers, IPT, the validation design (R16), recency (R18), the clock
leak (R20) and the customer split stand, but headline architecture rankings
should be rerun on corrected data because every arm consumed the corrupted
stream.

**Fix.** Decode obtained items through ProductID, never through a
version-specific NewProductIndex: build `v2 NewProductIndex → ProductID →
v6 NewProductIndex6` in both JSON generators, fix the campaign-30 index swap,
regenerate `simple6` and `_IPT`, then require `verify_product_index.py` to print
"signature absent" before any model reads the data.

### R26 — fix applied, `_IPT` regenerated and verified (Sep 17 2026)

**Direct confirmation of the input encoding.** `FullDecisionSequenceInt` has no raw
item-id column, but carries cumulative per-product counts (`Cum<ProductID>`). Row
increments identify which product each `ItemsJustGotIndex` code stands for: in 80
sampled files, **111 of 111** products carry the FigureWeaponIndex2
`NewProductIndex` code (v6: 31/111; ProductIndex: 8/111).

**Fix.** Both generators (`GenerateJSON.R`, `InsertNotBuy_GenerateJSON_IPT.R`) now
decode obtained items as v2 code → ProductID → NewProductIndex6 (`map_obtained`,
with `stopifnot` on missing codes). Committed in the OneDrive `Code` repo, which
is now under git with a `.gitignore` for secrets, session state, data and knitted
outputs.

**Regeneration.** R 4.6.1 installed locally (per-user). Stage 2 of the IPT script
rerun from `FullDecisionSequence_IPT` through a local runner that reads OneDrive
and writes only to `PRODUCTGPT_DATA/regen_r26/`. One runner-only guard:
`PlayerSummary_IPT.xlsx` already contains the `player_index` columns, so the merge
is skipped when present (a second merge would suffix the columns and fail).

**Verification** (`scripts/compare_regenerated_json.py`, `verify_product_index.py`):

| Check | Result |
|---|---|
| Customers | 5,004 = 5,004, identical uid sets |
| Sequence lengths, offers, previous decisions, all other fields | identical |
| IPT | 4,351 customers differ only by −0.01 h on 0.49% of values (`sprintf` halfway rounding, macOS vs Windows) |
| Obtained tokens | 97.64% unchanged; 2.36% changed, **every one** exactly as the fix predicts; 0 unexpected |
| Bug signature (ids 36/37 share of limited-time ids) | 0.919 → 0.149: "signature absent" |
| Copy check | 43,086 limited-time 5-star acquisitions (was 644,393); beyond cap 0.12% (was 81%); no character above 7 copies |

The corrected file replaces `clean_list_int_wide4_simple6_IPT.json` in the laptop's
`PRODUCTGPT_DATA`; the original is kept as `..._pre_r26.json`. **HPCC still has the
old file**, and the OneDrive `Data` copy is untouched.

**Gen-4 `simple6` not fixed.** The current `GenerateJSON.R` no longer reproduces
the April 2025 file (5 fields dropped, 4,952 customers vs 5,291), and the buggy
mapping is many-to-one, so the old JSON cannot be repaired in place. The
regenerated file is quarantined as `UNFAITHFUL_do_not_use_*`.

**Substitution check on corrected data — provisional, NOT a finding.** Raw: owners
of a same-element character pull MORE on an unowned featured character (0.33 vs
0.25). Within customer × element with offered-product fixed effects: −0.107
(t = −30), −0.164 with controls. But the design has a mechanical confound that
must be removed before any reading: before a customer's first character of an
element is acquired, the "not owned" observations include the occasions spent
chasing that very character (high pull rate), which ends when it is acquired.
Next version: exclude, for each customer, the banners of characters they
eventually acquire, or use an event study on other-element banners as control.

**Consequence for the architecture ledger.** Every gen-5 run to date (R1–R25)
consumed the corrupted obtained stream. Rankings that do not depend on inventory
(recency, validation, the customer split) are unlikely to move, but the headline
arms and all inventory arms (R19, R23, R24) must be rerun on the corrected file
before any inventory or architecture claim is written.

### R27 — pre-registered: core replication on corrected data (batch 11)

**Data.** `clean_list_int_wide4_simple6_IPT.json` regenerated under R26 (sha256
`be6f52fe1355524e…`), identical on laptop and HPCC; the pre-fix file is kept as
`_pre_r26.json` in both places. Before submission: obtained-block leak check
"ok (block describes t−1)"; smoke test 6/6; time-bias causality PASS. Labels,
offers and timing are unchanged, so any movement is attributable to the
corrected inventory.

**Arms** (`scripts/submit_batch11.sh`; S=1024, late validation, 40 epochs, no
customer embedding, seeds 1–3; pre-fix references in brackets):

| Arm | Inventory read? | Pre-fix |
|---|---|---|
| `gru` — plain GRU baseline | outcome pooling only | 0.8802 ± 0.0078 |
| `gru_cross` — gen-5 GRU encoder | inventory GRU + offer–inventory attention | 0.8889 |
| `gru_nocross` | inventory GRU, no attention | 0.9137 (1 seed) |
| `tf_alibi` — transformer + ordinal recency | offer–inventory attention | 0.8860 ± 0.0057 |
| `tf_alibi_nocross` | none | new |

**Predictions and decision rules** (three-seed means, out-of-sample customers ×
holdout period; 0.02 nats is the adoption margin, seed sd ≈ 0.008):

1. *Ranking.* If `tf_alibi` and `gru` stay within 0.02, the tie (findings 5–7)
   carries over with corrected numbers. If either leads by more than 0.02, the
   architecture conclusion is reopened and the leader becomes the new baseline.
2. *Inventory value.* Define the attention gain as `*_nocross − *` on each
   encoder. If the attention gain exceeds 0.02 on either encoder, **or**
   `gru_cross` / `tf_alibi` beats the plain GRU by more than 0.02, inventory
   carries information the corrupted stream hid → run Phase 2 (slots, deep
   satiation, best hybrid, one capacity and one calendar-time arm).
   Otherwise R19/R23/R24's reading stands on corrected data and Phase 2 is
   skipped except where the paper reports a number.
3. *Direction.* Correct product identities should not make inventory-reading
   arms worse. If `gru_cross` or `tf_alibi` loses more than 0.02 against its
   pre-fix value while the plain GRU does not, stop and investigate the data
   before interpreting anything.

**Afterwards (not in this batch).** R28: one id per product (118 products,
attributes for all); substitution check with the chasing confound removed.

### R27 — results: the ranking survives, the inventory signal does not

All 15 runs finished. Job logs confirm the corrected file (278,720,977 bytes,
Sep 17) was the one read. Out-of-sample customers × holdout period:

| Arm | Pre-fix | Corrected | Δ |
|---|---|---|---|
| `gru` | 0.8802 ± 0.0078 | **0.8892 ± 0.0172** | +0.0090 |
| `gru_nocross` | 0.9137 (1 seed) | 0.8929 ± 0.0107 | −0.0208 |
| `gru_cross` | 0.8889 | 0.8970 ± 0.0075 | +0.0081 |
| `tf_alibi` | 0.8860 ± 0.0057 | 0.9083 ± 0.0055 | **+0.0223** |
| `tf_alibi_nocross` | — | 0.9088 ± 0.0037 | — |

**Rule 1 — ranking.** The gap between the plain GRU and the transformer is
0.0191, just inside the 0.02 margin, so the pre-registered "tie" verdict stands
by the letter of the rule. But it has grown from 0.006 to 0.019 and the GRU now
wins **3 of 3 paired seeds**. Read it as: the tie is weaker, and the transformer
is not ahead on corrected data.

**Rule 2 — inventory value.** The offer-inventory attention is worth nothing
once product ids are correct: −0.0041 on the GRU encoder (i.e. dropping it is
slightly BETTER) and +0.0005 on the transformer, both inside seed noise. Neither
inventory-reading arm beats the plain GRU. **Phase 2 is skipped**, as
pre-registered. R19/R23/R24's conclusion — the inventory path is not where the
missing signal is — survives the correction, and is now cleaner: on corrupted
data the attention looked worth 0.034; it is worth zero.

**Rule 3 — safety check fired, and here is why.** `tf_alibi` lost 0.0223 while
the plain GRU lost 0.0090. The mis-mapping was a deterministic relabelling, so
it did not add information — but it did change RESOLUTION. Two very common
3-star weapons (W3-001, W3-002) previously carried their own ids (36, 37: 13% of
obtained tokens); correcting them merges them into the pooled 3-star id, which
now holds 85% of tokens. Measured on the obtained stream: **entropy falls from
1.050 to 0.601 nats** with the same 44 distinct ids. The corrupted stream was
accidentally a finer-grained record of recent pull volume, which is predictive.
The transformer lost most because its satiation attention reads that memory
directly; the plain GRU only pools it.

So the degradation is a loss of token resolution, not a defect in the corrected
data, and it makes a testable prediction for **R28 (one id per product)**: giving
all 13 three-star weapons, 27 four-star weapons and 25 four-star characters their
own ids restores — and exceeds — the lost resolution. If the corrupted-data
advantage was resolution, R28 should recover roughly the 0.02 nats lost here.

**Calibration period is flat.** All five arms score 0.971–0.974 there, within
0.003 of each other. The architecture differences live entirely in the holdout
period, as before.

**Seed spread.** The plain GRU's seeds are 0.8715 / 0.8902 / 0.9057 (sd 0.0172),
twice the usual. Any future comparison against it needs three seeds; single-seed
reads of the GRU are unusable.

### R28 — pre-registered: one token per product (batch 12)

**Why.** R27 turned an accident into a measurement. The corrupted stream gave two
common 3-star weapons their own ids; correcting them merged those into the pooled
3-star token, entropy fell 1.050 → 0.601 nats, and every model lost 0.009–0.022
nats. That says token RESOLUTION, not product identity per se, was carrying
signal. Level 6 spends 44 tokens on 118 products: all 3-star weapons share one
id, as do the 4-stars and the standard 5-stars. Level 7 gives each product its
own token and embedding row.

**Data.** `perproduct/clean_list_int_wide4_simple7_IPT.json`, generated by the
same R script under `VOCAB_LEVEL=7` (`ProductVocab7.xlsx`: 118 products, ids
13–130, specials 131–133, vocab 134). Verified with
`scripts/verify_vocab7_data.py`: same 5,004 customers, same sequence lengths,
same decisions/campaigns/IPT, and **every token collapses exactly onto its
level-6 id** — so resolution is the only difference. Obtained-stream entropy
**0.664 → 3.236 nats**. Code: `scripts/test_vocab7.py` (14 checks) passes, and
level 6 is bit-for-bit unchanged.

**Arms** (`scripts/submit_batch12.sh`, 3 seeds, paired with their batch-11 twins):

| Arm | Level 6 (R27) | Level 7 |
|---|---|---|
| `gru` | 0.8892 ± 0.0172 | _pending_ |
| `tf_alibi` | 0.9083 ± 0.0055 | _pending_ |
| `gru_cross` | 0.8970 ± 0.0075 | _pending_ |

**Predictions and decision rules** (three-seed means; 0.02 adoption margin):

1. *Resolution was binding.* Each arm improves by more than 0.02 over its
   level-6 twin → adopt level 7 as the default vocabulary and rerun the
   architecture comparison on it.
2. *Resolution was not binding.* All arms land within 0.01 of their twins → the
   pooled vocabulary was not the constraint, R27's loss was specific to the two
   high-frequency ids, and the information-ceiling reading strengthens.
3. *Architecture interaction.* If `tf_alibi` gains more than `gru` by over 0.02,
   product identity is something attention exploits and recurrence cannot: the
   architecture conclusion (findings 5–7) is reopened in the transformer's
   favour. The reverse ordering would close it further.
4. *Inventory.* If `gru_cross` now beats `gru` by more than 0.02, the
   offer-inventory attention needed product identity to be worth anything, and
   R19/R23/R24 must be rerun at level 7 before the paper says inventory adds
   nothing.

**Follow-up analysis (not a rule).** Project the learned product embeddings to
two dimensions and check whether 3-stars, 4-stars and 5-stars separate, and
whether same-element characters cluster. That is the perception map; it is a
figure for the paper, not a selection criterion.

### R28 — results: the vocabulary decides which architecture wins

Nine runs, three seeds per arm, paired with their batch-11 twins (same seeds,
same customers, same labels; only token resolution differs).

| Arm | Level 6 (44 tokens) | Level 7 (118 tokens) | Δ level 7 − level 6 |
|---|---|---|---|
| `gru` | 0.8892 ± 0.0172 | 0.9096 ± 0.0022 | **+0.0204 (worse)** |
| `gru_cross` | 0.8970 ± 0.0075 | 0.9126 ± 0.0078 | +0.0156 (worse) |
| `tf_alibi` | 0.9083 ± 0.0055 | **0.8985 ± 0.0016** | **−0.0098 (better)** |

**Neither pre-registered prediction held.** Not "everything improves" (rule 1),
not "nothing changes" (rule 2). **Rule 3 fired**: the transformer gained
0.030 nats relative to the GRU, well past the 0.02 margin.

**The crossover.** At level 7 the transformer beats the plain GRU by 0.0111 with
3/3 paired seeds (transformer seeds 0.8980 / 0.9004 / 0.8972; GRU 0.9071 /
0.9113 / 0.9106). This is the first time in 60+ configurations that a
transformer has led, and the seed spread collapsed (sd 0.0016 vs the GRU's 0.017
at level 6), so the comparison is unusually clean.

**Reading.** Giving every product its own token adds 74 ids, most of them rare.
The GRU pools the offer and outcome tokens into a single vector per occasion, so
finer ids mostly dilute that pooled average — and its selected epoch fell to 10.7
from 22.0, i.e. it starts over-fitting sooner. The transformer reads products
through attention, which can select among ids instead of averaging them, so the
extra identity is usable. Product identity is exactly the kind of information
attention is supposed to be good at, and this is the first evidence in the
project that it is.

**Inventory, again.** `gru_cross` is still no better than the plain GRU at level 7
(+0.0030). Whatever the transformer gains from product identity, it is not coming
through the offer-inventory attention.

**Calibration period.** `gru` degrades there too (0.9896 vs 0.9737 at level 6)
while `gru_cross` and `tf_alibi` are flat, so the GRU's loss is not drift.

**What this does NOT yet establish.** One decision-level transformer against one
recurrent baseline at one width. Before any claim: batch 13 widens the level-7
comparison (LSTM, GRU encoder without cross-attention, transformer without the
recency prior, the stacked hybrid) and adds a GRU at matched capacity, since the
transformer has 1.8× the parameters.

### R29 — pre-registered: the full factorial (batch 13)

**Why.** R28's crossover rests on three arms at one vocabulary each. A reviewer
will ask three questions immediately: is it capacity (the transformer has 1.8×
the parameters), does it survive other stock-level choices, and does product
identity simply substitute for the recency prior? This batch crosses the design
so the interaction is measured rather than inferred.

**Design** (`scripts/submit_batch13.sh`; 3 seeds; 24 configurations; 72 runs;
seed-major, so a complete first pass lands in roughly a third of the time):

| Dimension | Levels |
|---|---|
| Vocabulary | 6 (118 products share 44 tokens) · 7 (one token per product) |
| Decision-level memory | `gru` (d=128) · `gruw` (d=176, **1.25M params, matched to the transformer's 1.22M**) · `gru_enc` · `tf` (+ALiBi) · `tf_norec` · `hyb` (GRU and attention interleaved) |
| Stock-level memory | `nostock` · `tokens` (inventory GRU + offer-inventory attention) · `slots2` (additive per-product counts + 2 satiation layers) |

Crossed: {`gru_enc`,`tf`,`hyb`} × {`nostock`,`tokens`,`slots2`} × {6,7} = 18;
plus {`gru`,`gruw`} × {6,7} = 4; plus `tf_norec` × `tokens` × {6,7} = 2.
Six cells repeat batch 11/12 configurations at the same seeds, which doubles as
a reproducibility check on those numbers.

**Hypotheses and decision rules** (three-seed means, out-of-sample × holdout;
0.02 nats is the adoption margin, seed sd ≈ 0.002–0.017):

1. **The crossover is an interaction, not an artefact.** Predicted: at level 6
   the recurrent arms lead; at level 7 `tf` leads; the difference-in-differences
   (tf − gru at level 7) − (tf − gru at level 6) exceeds 0.02 in at least 2 of
   the 3 stock variants. If it appears in 0 or 1, R28 was a single-cell result
   and must be reported as such.
2. **Capacity is not the explanation.** Predicted: `gruw` at level 7 closes less
   than half the transformer's 0.011 lead. If the width-matched GRU closes it,
   the finding is capacity, not attention, and the paper says so.
3. **Per-product counts finally make the stock path pay.** At level 7, `slots2`
   counts real products rather than pools. Predicted: `slots2` beats `tokens` by
   more than 0.02 at level 7 but not at level 6. If it fails at both, the stock
   path is not where the signal is, on any representation tried, and the
   additive-stock programme is justified on theory and interpretability only —
   not on fit.
4. **Identity does not substitute for recency.** Predicted: `tf_norec` stays far
   behind `tf` at both levels (pre-fix gap was ~0.13). If level 7 closes that
   gap substantially, product identity is doing part of what the recency prior
   did, which would be a finding in itself.
5. **The hybrid.** With attention now earning its place at level 7, predicted:
   `hyb` ≥ `tf` − 0.02 (i.e. no real gain). Adopt the hybrid only if it beats
   `tf` by more than 0.02 at level 7.

**Cost.** 72 runs at roughly 1.5 h each, two GPUs: about 2 days wall clock.

**Analysis plan, fixed now.** Report the 2 × 6 × 3 table of three-seed means; the
difference-in-differences for rule 1 with its paired-seed count; and the
per-class profile for the best cell at each vocabulary. The perception map of
the 118 learned product embeddings is a figure, not a selection criterion.

**R29 interim (Sep 21 2026, 45 of 72 runs, mostly 2 seeds — NOT a result).**
Holdout NLL, out-of-sample customers. Level 6 / level 7: plain GRU 0.8809 /
0.9092; width-matched GRU (d=176) 0.8911 / **0.8943**; GRU encoder + counts
0.8979 / 0.8949; transformer + tokens 0.9052 / 0.8992; transformer + counts
0.9016 / 0.8995; hybrid + counts 0.8911 / 0.8888 (1 seed).

- **Rule 1 (crossover is an interaction): failing.** The difference-in-differences
  (transformer − GRU encoder, level 7 − level 6) is −0.0301 with the token
  inventory, +0.0078 with no stock path and +0.0008 with per-product counts.
  It appears in 1 of 3 stock variants, not the 2 required — i.e. only in the
  configuration batch 12 happened to run.
- **Rule 2 (capacity is not the explanation): failing.** At level 7 the
  width-matched GRU beats the transformer (0.8943 vs 0.8992) and gains 0.0148
  over the d=128 GRU. R28's "first transformer lead" looks like a comparison
  against an undersized recurrent model.
- **Rule 3 (per-product counts pay at level 7): supported on the recurrent
  encoder.** Counts beat the token path by 0.0218 at level 7 and lose 0.0053 at
  level 6 — the predicted interaction, and the first time the inventory path has
  earned anything since the data fix.
- **Rule 4 (identity does not replace recency): holds.** Dropping the recency
  prior costs 0.128 (level 6) and 0.113 (level 7).

Nothing is adopted on two seeds; seed 3 lands tonight.

### R30 — the tuning and model-comparison programme (stage 1 submitted)

**Why now.** Nothing in this project has ever been tuned. Width, depth, heads,
dropout, learning rate, batch size and weight decay have been identical since
gen 4, inherited from a transformer sweep on different data. Tuning was blocked
in R12–R13 (validation ranked models backwards) and never resumed after R16
fixed it. R29's interim result shows the cost: one extra width on the GRU
(d=176) moved it 0.0148 at level 7 and overturned R28's headline. Every
architecture comparison we have made is a comparison at one arbitrary point on
each family's capacity curve.

**Protocol (fixed before any run).**

- **P1. Selection on campaign-27 validation only.** Search stages read
  `scripts/summarize_search.py`, which never opens a test cell. The holdout is
  opened once, at stage 4, with the configurations already frozen.
- **P2. Equal budget per family.** Same number of sampled configurations, same
  search space dimensions, same seeds.
- **P3. Seeds by stage.** Search 1 seed; confirmation 3 FRESH seeds (11, 12, 13)
  to blunt the winner's curse; final comparison 5 seeds (1–5).
- **P4. Search on the cheap stock path** (per-product counts, ~20× cheaper than
  the token inventory). The stock path is re-tested on the winner at stage 4.
- **P5. Frozen, not tuned:** the 2×2 evaluation design, campaign-27 validation,
  `split_seed=33`, `max_events=1024`, 40-epoch cap with patience 5, class
  weighting, and the recency prior (architectural — tested separately).
- **P6. Report the frontier, not a point:** every table carries parameters and
  minutes per epoch beside NLL.
- **P7. No single-seed result is ever reported.** Two have evaporated already.

**Stages.**

| Stage | Batch | Content | Runs |
|---|---|---|---|
| 0 ✅ | — | Expose `--lr`, `--weight-decay`, `--grad-accum`, `--warmup-frac`; validation-only summarizer | 0 |
| 1 ▶ | 14 | **Capacity frontier** (submitted 2026-09-21, jobs 41207–41254), level 7: 4 families × d ∈ {96,128,176,256} × N ∈ {2,4,6}, 1 seed | 48 |
| 2 | 15 | **Random search** per family around its stage-1 best: lr log-uniform [1e-4, 1.2e-3], dropout {0.05,0.1,0.2,0.3}, weight decay {0,0.01,0.05,0.1}, effective batch {8,16,32}, d_ff ratio {2,3,4}, warmup {0.02,0.05,0.1}; 24 configs × 4 families, 1 seed | 96 |
| 3 | 16 | **Confirmation**: top 3 per family × 3 fresh seeds; each family's config frozen by validation mean | 36 |
| 4 | 17 | **Final comparison**, holdout opened once: frozen config per family × 5 seeds × vocabulary {6,7}; plus stock-path ablation on the winner at level 7 | 55 |
| 5 | — | Diagnostics: per-class profiles, calibration cell, efficiency table, product-embedding perception map, substitution check | 0 |

Families: plain GRU · GRU encoder · transformer + recency · stacked hybrid.
Total ≈ 235 runs, ≈ 90 GPU-hours, ≈ 2 days wall on two GPUs.

**Decision rules for stage 4.**

1. **Family winner.** Best-of-family A beats B only if Δ > 0.02 nats on the
   five-seed holdout mean AND A wins at least 4 of 5 paired seeds.
2. **Vocabulary interaction.** The R28 crossover is real only if
   (A − B at level 7) − (A − B at level 6) > 0.02 with consistent sign in at
   least two families. R29 already shows it in 1 of 3 stock variants, so the
   prior is against it.
3. **Efficiency.** Parameters and runtime are reported whatever the NLL says. A
   tie at half the parameters is a result, and is the honest form of the
   "attention buys interpretability, not accuracy" claim.
4. **Ceiling.** If all four families land within 0.02 of each other at their own
   optima, the information-ceiling reading is reported WITH the tuning budget as
   evidence — that is a far stronger version of the claim than the untuned one.

**What this costs us if skipped.** Any reviewer can ask "did you tune the
baseline?" and today the answer is no. R29 shows the answer matters: the GRU
gained 0.015 from a single width change.

### R31 — pre-registered: is the crossover real, and where does it live? (prioritised over R30)

**Why.** R28's crossover rests on three seeds per cell, and R29 localises it to
the token-inventory configuration. Two questions can be answered from existing
checkpoints, without training, and are more urgent than tuning:

- **A1 — power.** Every model is scored on the same out-of-sample customers, so
  their per-occasion losses can be PAIRED. A cluster bootstrap over customers
  replaces n = 3 seeds with n ≈ 2,500 customers.
- **C — mechanism.** If the transformer's level-7 advantage is retrieval of
  specific past products, it must concentrate where that matters, and vanish on
  a placebo.

**Method** (`scripts/eval_per_occasion.py`, `scripts/r31_analysis.py`,
`scripts/r31_eval.pbs`). Each checkpoint's best.pt is rebuilt and scored on
every occasion of the out-of-sample × holdout cell. Seed-averaged per-occasion
NLL per model; statistic

    DiD = (A7 − B7) − (A6 − B6),   negative = crossover in A's favour

with a 2,000-draw cluster bootstrap over customers (one shared set of draws for
all subgroups, so subgroup contrasts are paired draw by draw). Subgroup tags are
computed in the level-6 id space for every model, so level-6 and level-7 models
put each occasion in the same subgroup. **Integrity gates:** every model must
score the identical occasions with identical labels, and each file's mean NLL
must match its run's recorded test-cell NLL to 5e-4 — otherwise that comparison
is not analysed.

**Comparisons.** Primary: transformer vs GRU encoder, token inventory (batches
11/12, 3 seeds each). Secondary: transformer vs plain GRU; the SAME contrast
with per-product counts (batch 13); transformer vs width-matched GRU.

**Subgroups (fixed now).** Purchase decisions (1–8) vs NotBuy (9). Offer
includes a limited 5-star the customer ALREADY OWNS vs limited offer, none owned
(**placebo**). Distinct limited 5-stars owned: ≤3, 4–8, ≥9. (Smoke test: every
held-out occasion offers at least one limited 5-star, so owned + placebo cover
all occasions.)

**Predictions and rules.**

1. **A1:** primary DiD over all occasions is negative with a 95% CI excluding 0.
   If the CI includes 0, R28's crossover is not established at the customer
   level and is reported as such.
2. **C, location:** DiD on owned-offer occasions is more negative than on the
   placebo; the owned-minus-placebo contrast has a CI excluding 0, and the
   placebo's own CI includes 0.
3. **C, decision type:** DiD is larger in magnitude on purchase decisions than on
   NotBuy.
4. **C, dose:** DiD grows in magnitude with the number of limited 5-stars owned
   (≤3 → 4–8 → ≥9).
5. **Contrast:** with per-product counts, the DiD CI includes 0 (R29 predicts no
   interaction there).
6. **Capacity:** against the width-matched GRU the DiD shrinks; if its CI
   includes 0, the crossover is capacity-dependent and the paper says so.

Predictions 2–4 all passing with 1 is the mechanism story. 1 passing with 2–4
failing means a real but unlocalised effect. 1 failing ends the crossover claim.

**Priority.** User decision (Sep 21): R31 runs before R30. Batch 13's and
batch 14's queued jobs are held while the evaluation jobs run, then released.
