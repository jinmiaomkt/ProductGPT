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

Fill this from an actual `qstat -u $USER` at the start of each HPCC session.
It is deliberately empty rather than stale — a wrong job id is worse than none.

| Job id | Run dir / tag | Submitted | Hypothesis | Status |
|---|---|---|---|---|
| _(fill from qstat)_ | | | | |

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
| R4 | Sep 2026 | Memory probe | Find the largest feasible `max_events` | S=1024 fits HPCC; S=1536 OOMs | Cap HPCC runs at S ≤ 1024 |
| R5 | Sep 2026 | Split redesign | Four holdout conventions existed; none matched the benchmark | Adopted Lu & Kannan's 2×2 | `split_mode="both"` default |
| R6 | Sep 2026 | Regularisation sweep | Augmentation is a free win | **Wrong** — slightly worse, and confounded with dropout | Re-run cleanly, one factor at a time |
| R7 | Sep 2026 | Mixture-head port | Per-customer mixture beats a plain user embedding | Submitted; **not recorded** | Pending |
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
each with its own `TAG` so run directories do not collide. Augmentation came
out slightly *worse*, not better; one variant confounded augmentation with a
dropout change, so that arm is uninterpretable. **Exact metrics not recorded.**
Re-run one factor at a time before drawing any conclusion.

### R7 — Mixture head (Lu & Kannan mechanism)

`UserMixtureOutputHead`: H output projections with per-customer weights
α_n = softmax(user_mix_logits[n]); unseen customers get ᾱ, the population mean.
H parameters per customer instead of a 128-dim embedding — roughly 32× less
memorisation capacity, and interpretable as soft segment membership.

Two arms were submitted, `jmr_flat` (no mixture) and `jmr_mix8`
(`MIX_HEADS=8`). **Outcomes not recorded here.** Read `final.json` in each run
directory and fill in the table above.

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
