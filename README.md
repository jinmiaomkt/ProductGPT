# ProductGPT

Research code for a sequence model of consumer purchase decisions, estimated on
Genshin Impact gacha logs. Each customer is a sequence of *decision occasions*.
At every occasion the model predicts which of nine decisions the customer takes,
given the limited-time offers on show, the products they already own, and what
they decided last time.

The project asks two questions of each model:

1. **Prediction:** how well does it forecast the next decision for customers and
   campaigns it never saw in training?
2. **Generation:** if it runs on its own decisions, does it produce behaviour that
   looks like real customers' behaviour?

This is unpublished, ongoing research. The customer data is confidential and is
not in this repository, so the training code will not run as is. Numbers can
change as experiments are re-run.

**Every run, its hypothesis and the decision it led to are in
[`EXPERIMENTS.md`](EXPERIMENTS.md).** Read the relevant entry before comparing two
numbers, because several early results were voided by fixes recorded there.

## Results so far

### Prediction

Negative log-likelihood on out-of-sample customers in the holdout campaigns
(217,303 occasions from the 5,004 customers with complete histories). Lower is
better.

| Model | NLL | Ledger entry |
|---|---|---|
| Generation 6, streaming hybrid (five seeds) | 0.878 | R44 |
| Generation 5, tuned hybrid (five seeds) | 0.882 | R44 |
| Plain GRU (five seeds) | 0.886 | R46, R54 |
| Muse Glimmer 30B, off the shelf, zero-shot | 0.928 | R54 |
| Centaur, off the shelf, zero-shot | 0.931 | R54 |
| Qwen3-32B, off the shelf, zero-shot | 0.948 | R49 |
| Latent-class parametric choice model (R6) | 0.983 | R46 |
| Gemma 4 31B, off the shelf, zero-shot | 1.153 | R54 |
| Constant at training-set shares | 1.664 | R48 |

Generations 5 and 6 tie: the gap is 0.0035 nats, and it favours generation 6 on
only 2 of 5 seeds. Adopting a change requires more than 0.02 nats on the
multi-seed mean and at least 4 of 5 paired seeds. Generation 6 was adopted for
what it can do (read any history length, roll out 3.9× faster), not for accuracy.

### Generation

Energy distance between generated and real holdout behaviour, on 300 customers ×
8 replicates, using free-running rollouts. Two halves of the real data differ by
up to 0.078 (the 95% band).

| Generator | Energy distance | Ledger entry |
|---|---|---|
| Latent-class parametric choice model (R6) | 0.318 | R46 |
| First-order Markov chain | 0.784 | R43 |
| Independent draws | 2.468 | R43 |
| Generation 5 | 2.563 | R43 |
| Generation 6 | 2.617 | R43 |

Nothing falls inside the band. The neural models are the most accurate one-step
predictors and among the weakest free-running generators, so predictive accuracy
alone does not justify using them as simulators. As a diagnostic, feeding the same
models the real past decisions (teacher forcing) brings them to about 0.35, so most
of the problem is in the closed loop. A known-truth simulation study (R51) is under
way, so these comparisons do not have to rest on energy distance alone.

### Other findings

- **Left-censoring costs little (R45).** A history missing up to its first 4
  campaigns loses at most about 0.004 nats. Adding 8,834 customers whose histories
  start before the data window changes no result cell.
- **Forgetting and campaign resets are not identified (R47).** The models cannot
  date how fast players forget their holdings. In the raw data, owners stay less
  likely than non-owners to pull on a rerun for weeks.
- **Cells differ in level because campaigns differ (R52).** The 0.097-nat gap
  between the calibration and holdout periods comes mainly from campaign
  difficulty. Compare models within a cell, not across cells.

## Repository layout

```
gen6_streaming/      Generation 6: the current model and trainer
gen5_multistream/    Generation 5: model, data pipeline, GRU/LSTM baselines, and
                     the training loop that generation 6 imports
shared/              Building blocks used by the model modules
simulation/          The game environment, rollout engine and sequence metrics
scripts/             Job scripts, one-off experiment scripts (rNN_*), and tests
  batches/           One submit script per experiment batch (2 to 42)
eval_inputs/         Metadata for the frozen customer split (uid lists excluded)
docs/                Run guide, notes and paper-revision material
archive/             Generations 0 to 4 and their AWS-era tooling
EXPERIMENTS.md       The experiment ledger
paths.py             The only place data and output paths are resolved
requirements.txt     Pinned packages (install PyTorch separately)
env.sh env.ps1 env.bat   Put the code folders on PYTHONPATH
```

Later experiments live on their own branches until they are merged:

| Branch | Contents |
|---|---|
| `main`, `gen6-streaming` | The code above, and the ledger for R1–R45, R47 and R52 |
| [`r45-censoring`](https://github.com/jinmiaomkt/ProductGPT/tree/r45-censoring) | Code for left-censoring experiments and the inclusive sample |
| [`r46-parametric`](https://github.com/jinmiaomkt/ProductGPT/tree/r46-parametric) | `parametric/`: the classical choice-model ladder and its ledger entries |
| [`r47-forgetting`](https://github.com/jinmiaomkt/ProductGPT/tree/r47-forgetting) | Code for the forgetting and campaign-reset tests |
| [`r48-zeroshot`](https://github.com/jinmiaomkt/ProductGPT/tree/r48-zeroshot) | `llm_zeroshot/`: scoring off-the-shelf LLMs, and ledger entries R48–R54 |

Those branches were cut before the 4 November 2026 reorganisation, so they still
have the old layout. [`archive/README.md`](archive/README.md) lists what moved.

## The task

One integer namespace is shared by inputs and labels. The labels are 1 to 9:

| ID | Decision | Banner | Draws |
|---|---|---|---|
| 1 / 2 | Buy1_Reg / Buy10_Reg | Regular | 1 / 10 |
| 3 / 4 | Buy1_FigA / Buy10_FigA | Featured A | 1 / 10 |
| 5 / 6 | Buy1_FigB / Buy10_FigB | Featured B | 1 / 10 |
| 7 / 8 | Buy1_Wep / Buy10_Wep | Weapon | 1 / 10 |
| 9 | NotBuy | none | 0 |

NotBuy is an ordinary class meaning no draw within a 24-hour interval. Sequences
end with PAD (0). The full token layout is in [`shared/vocab.py`](shared/vocab.py).

Each occasion is a 15-token event:

```
[ LTO offer x4 | products obtained since the last decision x10 | previous decision x1 ]
```

## Model generations

| Generation | Architecture | Status |
|---|---|---|
| 0, 12 | Encoder–decoder; generation 12 adds a limited-time-offer stream | Archived |
| 2 | Decoder-only Performer, offer + previous decision | Archived |
| 4 | Decoder-only Performer or FlashAttention, flattened 15-token stream | Archived |
| 5 | Offer, inventory and decision as separate tensors; recurrence + attention hybrid; reads the last 1,024 occasions | Reference and shared code |
| **6** | The same hybrid with sliding-window attention and a carried GRU state; reads every customer's full history | **Current** |

## Evaluation design

The design follows Lu & Kannan (JMR): customers are split in half, time is split at
campaign 28, and the cells are reported separately.

| | Calibration (campaigns ≤ 27) | Holdout (campaigns ≥ 28) |
|---|---|---|
| **In-sample customers** | Training; campaign 27 is validation | Reported |
| **Out-of-sample customers** | Reported | **Headline cell** |

- **Selection uses campaign 27 only.** Validation from the training period cannot
  detect temporal drift, and the holdout is opened once, at the end.
- **Experiments are pre-registered.** The hypothesis, prediction and decision rule
  go into the ledger before a run.
- **Families are tuned on equal budgets**, so a difference is not an artefact of
  tuning one side harder.
- **No headline comes from one seed.**

## Data

The data is confidential consumer data and must never be committed. `.gitignore`
blocks the data folder, `*.xlsx`, uid lists, checkpoints and run outputs.

The main sample has the 5,004 customers observed from their account's start. From
2 November 2026 the study moves to 13,838 customers: those 5,004 plus 8,834 whose
histories are left-censored, together about 85% of purchases. The results above
are on the 5,004 until the re-runs finish.

## Running the code

With access to the data, see the [run guide](docs/RUNNING.md) for setup, training
on a PBS cluster, the switches and the correctness checks.

Nine of the 13 test scripts use only synthetic data and run without the confidential
files (the run guide names the other four). For example:

```bash
source ./env.sh
CUDA_VISIBLE_DEVICES="" python scripts/test_gen6_streaming.py
```

## Licence

This repository has no licence file, so default copyright applies.
