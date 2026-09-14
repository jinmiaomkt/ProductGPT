# ProductGPT — Purchase-Sequence Transformer

A sequence model of consumer purchase decisions, estimated on Genshin Impact
gacha logs. Each customer is a sequence of *decision occasions*; at every
occasion the model predicts which of nine decisions the customer takes, given
the limited-time offers on show, the products they already own, and what they
decided last time.

> **Results do not live in this file.** Every run, the hypothesis it tested and
> the decision it produced are recorded in [`EXPERIMENTS.md`](EXPERIMENTS.md).
> Read the relevant ledger rows before comparing any two numbers — several
> early results were voided by fixes recorded there.

---

## Contents

1. [Where things stand](#1-where-things-stand)
2. [The model generations](#2-the-model-generations)
3. [Repository layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Running generation 5](#5-running-generation-5)
6. [Evaluation design](#6-evaluation-design)
7. [Correctness checks](#7-correctness-checks)
8. [Data](#8-data)
9. [The `shared/` package](#9-the-shared-package)
10. [Generation 4 and older](#10-generation-4-and-older)
11. [Working with git across laptop and HPCC](#11-working-with-git-across-laptop-and-hpcc)
12. [Known loose ends](#12-known-loose-ends)

---

## 1. Where things stand

- **The active line is `gen5_multistream/`.** One plain-PyTorch trainer runs the
  generation-5 transformer *and* matched GRU/LSTM baselines, on the Windows
  laptop and on the SMU HPCC cluster (OMEGA) without changes.
- **Every architectural choice is a command-line switch** (section 5.2), so each
  component can be credited or blamed against a matched control.
- **Evaluation follows Lu & Kannan's 2×2 design**, with checkpoints selected on
  the last calibration campaign so early stopping can see temporal drift
  (section 6).
- **Status, 14 Sep 2026:** with a recency bias the transformer ties a GRU on the
  same features; it does not beat it. See ledger rows R17–R21.

---

## 2. The model generations

Each generation is kept side by side so older results stay reproducible. **A
file's number tells you its generation** (`config4.py`, `model4_*.py` and
`train4_*.py` are all generation 4).

| Folder | Gen | Architecture | Input construction | Status |
|---|---|---|---|---|
| `gen0_encoder_decoder/` | 0 | Encoder–decoder, full softmax attention | `source_rate=10` | superseded |
| `gen12_encdec_lto/` | 12 | Encoder–decoder plus a third LTO stream | `source_rate=10`, `lto_rate=12` | superseded |
| `gen2_lp_duplet/` | 2 | Decoder-only Performer ("LP / duplet") | `ai_rate=5`: offer + previous decision | reference only |
| `gen4_full_productgpt/` | 4 | Decoder-only Performer or FlashAttention | `ai_rate=15`, flattened | previous main line; Linux + DeepSpeed |
| **`gen5_multistream/`** | **5** | **Multi-stream model + matched GRU/LSTM baselines** | **offer / inventory / previous decision as separate tensors** | **active** |

Generations 4 and 5 consume the same 15-token decision event; the earlier
generations used subsets of it (see the input column above):

```
[ LTO offer x4 | products obtained since the last decision x10 | previous decision x1 ]
```

Generation 4 flattens it into one long token stream (1,024 events = 15,360
tokens) and discards 14 of every 15 output positions. Generation 5 keeps the
three parts as separate tensors and routes each to its own module, which is its
single structural difference from generation 4.

### The nine decisions

One integer namespace is shared by inputs and labels. The labels are 1–9:

| ID | Decision | Banner | Draws |
|---|---|---|---|
| 1 / 2 | Buy1_Reg / Buy10_Reg | Regular | 1 / 10 |
| 3 / 4 | Buy1_FigA / Buy10_FigA | Featured A | 1 / 10 |
| 5 / 6 | Buy1_FigB / Buy10_FigB | Featured B | 1 / 10 |
| 7 / 8 | Buy1_Wep / Buy10_Wep | Weapon | 1 / 10 |
| 9 | **NotBuy** | — | 0 |

**NotBuy is an ordinary class, not end-of-sequence.** In the current (`_IPT`)
data it means "no draw within one 24-hour interval". Sequences end with PAD
(0). Products are tokens 13–56. The full layout is in `shared/vocab.py`.

---

## 3. Repository layout

Model stacks are grouped **by generation**; tooling is grouped **by function**.

```
gen5_multistream/              ACTIVE
  train5_multistream.py          the trainer: every switch, both architectures
  config5.py                     profiles (pilot | hpcc) and every default
  dataset_multistream.py         JSON records -> offer / inventory / decision tensors
  model_multistream_state_space.py   the gen-5 model
  model_recurrent_baseline.py    GRU / LSTM on the identical feature pipeline
  probe_memory.py                peak GPU memory for a given max_events
  train_multistream_patch.py     superseded patch guide, kept for history

gen4_full_productgpt/  gen2_lp_duplet/  gen12_encdec_lto/  gen0_encoder_decoder/
shared/                        building blocks imported by the model modules

scripts/
  gen5_train_hpcc.pbs            the HPCC training job (knobs via qsub -v)
  gen5_probe_hpcc.pbs            short memory-probe job
  submit_batch{2..6}.sh          one script per experiment batch
  summarize_batch.py             mean +/- sd per configuration across seeds
  drift_diagnostic.py            validation vs holdout curves, per epoch
  selection_metric_study.py      which validation metric predicts the holdout
  hpcc_status.sh  hpcc_cron.sh  notify.sh  telegram_setup.sh   monitoring
  backup_run.sh                  copy finished runs off HPCC (rclone)
  check_obtained_leak.py  ipt_leak_check.py  ipt_leak_counterfactual.py
  test_time_bias_causality.py  smoke_test_shared_refactor.py   correctness
  measure_event_stream.py        aggregate statistics of the event stream
  export_gen5_split.py  eval_gen5_per_class.py  regenerate_ipt.py
  (the rest: AWS/GCP-era setup, Ray Tune and gen-4 evaluation helpers)

baselines/   evaluation/   tuning/   crossval/   analysis/     gen 0-4 era tooling
vendor/transformer_xl/     upstream reference code, not this project
legacy/                    older snapshots
eval_inputs/               frozen uid split lists (the lists themselves are not in git)

EXPERIMENTS.md             the experiment ledger
paths.py                   the only place data and output paths are resolved
env.ps1  env.bat  env.sh   put the code folders on PYTHONPATH

checkpoints/  results/  runs/  logs/     outputs, ignored by git
```

---

## 4. Setup

### 4.1 Python

| Machine | Environment | Notes |
|---|---|---|
| Windows laptop | conda env `transformers`, Python 3.11, PyTorch 2.11 + cu128 | 6 GB GPU. cu128 is required for this Blackwell card |
| HPCC (OMEGA) | venv `~/ProductGPT/venvs/productgpt-eval`, modules `python/3.11.4 cuda/12.5` | activated by the PBS script |

On the laptop, open a Miniforge Prompt and run `conda activate transformers`
before anything below. In a shell where `python` is not found, call
`C:\Users\jinmiao\AppData\Local\miniforge3\envs\transformers\python.exe` by its
full path instead.

Generation 5 needs only `torch`, `numpy`, `pandas`, `scikit-learn` and
`openpyxl`. It does **not** need DeepSpeed, which is why it runs on Windows.

### 4.2 The data location

All code reads the data directory from one environment variable. Set it once
per machine:

```powershell
setx PRODUCTGPT_DATA "C:\Users\jinmiao\ResearchData\productgpt"
```

`setx` only affects terminals opened afterwards. On HPCC the PBS script exports
`PRODUCTGPT_DATA=/storage/home/jinmiao/ProductGPT/data` itself.

### 4.3 The import path — once per terminal

Scripts import each other by bare module name (`from config5 import ...`), so
the code folders must be on `PYTHONPATH`. From the repo root:

| Shell | Command |
|---|---|
| PowerShell (`PS C:\...>`) | `. .\env.ps1` — the leading dot matters |
| cmd.exe (`C:\...>`) | `.\env.bat` — the leading `.\` matters on this machine |
| Linux / HPCC | `source ./env.sh` |

`ModuleNotFoundError: No module named 'config5'` means this step was skipped.

---

## 5. Running generation 5

### 5.1 On the laptop

A one-epoch smoke run on 40 customers, which finishes in under 30 seconds:

```powershell
. .\env.ps1
python gen5_multistream\train5_multistream.py --profile pilot --epochs 1 --max-users 40 --max-events 128
```

Outputs go to `checkpoints/gen5_multistream/pilot/`. The `pilot` profile is
sized for the 6 GB card (`d_model=64`, `N=2`, batch 2 with accumulation 4).
Before raising `--max-events`, measure: the offer–inventory cross-attention
costs memory in proportion to the *square* of sequence length.

```powershell
python gen5_multistream\probe_memory.py
```

### 5.2 The switches

| Dimension | Flag | Values | Default |
|---|---|---|---|
| Sequence model | `--arch` | `transformer`, `gru`, `lstm` | `transformer` |
| Encoder inside gen 5 | `--encoder` | `transformer`, `gru` (keeps cross-attention and inventory GRU) | `transformer` |
| Satiation cross-attention | `--no-cross-attn` | on / off | on |
| Recency bias | `--time-bias` | `none`, `ordinal` (ALiBi, same as `--alibi`), `time` (elapsed hours) | `none` |
| Time clock | `--leaky-time-bias` | reproduces the batch-5 leak (section 7); never use for results | lagged |
| Product representation | `--no-product-id` | identity + attributes / attributes only | identity + attributes |
| Customer embedding | `--no-user-embedding` | on / off | on |
| Mixture head (Lu & Kannan) | `--mix-heads H` | 0 = single projection | 0 |
| Regularisation | `--dropout`, `--augment`, `--patience` | | 0.10, off, 5 |
| Validation design | `--val-mode`, `--val-from` | `customers`, `late`; first late campaign | `late`, 27 |
| Split | `--split-mode` | `user`, `temporal`, `both` | `both` |
| Seed | `--seed` | training seed; the customer partition is fixed separately | 33 |
| Diagnostics | `--track-holdout` | score the holdout every epoch; never used for selection | off |
| Frozen split | `--uids-dir` | directory written by `scripts/export_gen5_split.py` | derived |

Run `python gen5_multistream\train5_multistream.py --help` for the full text.

### 5.3 On HPCC

Log in with `ssh jinmiao@omega.smu.edu.sg`. The checkout lives in
`~/ProductGPT/work`. **Submit from that directory** — the job script runs the
code in the directory it was submitted from.

```bash
cd ~/ProductGPT/work
git pull                                     # pick up laptop changes first
qsub -v MAX_EVENTS=1024,USER_EMB=0,SEED=1,TAG=mytest scripts/gen5_train_hpcc.pbs
```

Every switch in 5.2 has a knob: `ARCH`, `ENCODER`, `CROSS_ATTN=0`, `ALIBI=1`,
`TIME_BIAS`, `PROD_ID=0`, `USER_EMB=0`, `MIX_HEADS`, `DROPOUT`, `AUGMENT=1`,
`PATIENCE`, `VAL_MODE`, `VAL_FROM`, `SPLIT`, `SEED`, `TRACK_HOLDOUT=1`, plus
`EPOCHS`, `BATCH_SIZE`, `MAX_EVENTS`, `RESUME=1` and `TAG`. The comments at the
top of `scripts/gen5_train_hpcc.pbs` document each one.

- **Always set `TAG`.** The run directory is
  `~/ProductGPT/runs/gen5_hpcc_S<max_events>_b<batch>_<TAG>`, stable per
  configuration so a resubmitted job resumes its own `last.pt`. Two jobs with
  the same tag overwrite each other.
- **At most two GPU jobs run per user;** a third queues.
- **Experiments are submitted as batches**, one script per batch:
  `bash scripts/submit_batch6.sh`, or `DRY_RUN=1 bash scripts/submit_batch6.sh`
  to print the `qsub` lines without submitting.

Monitoring:

```bash
qstat -u $USER                               # /opt/pbs/bin/qstat if not on PATH
tail -f ~/ProductGPT/logs/gen5_train_<jobid>.log
bash scripts/hpcc_status.sh --once --force   # queue plus each job's latest epoch
```

Notifications arrive three ways, all running on OMEGA: PBS mail on job end or
abort; a Telegram message when each job exits; and an hourly, change-only
Telegram feed from `scripts/hpcc_cron.sh` via `crontab`. Telegram credentials
live in `~/.telegram_env` (mode 600), populated by `scripts/telegram_setup.sh`,
and never in the repository.

### 5.4 Reading results

Each run directory holds `best.pt`, `last.pt`, `history.json` (one record per
epoch) and `final.json` (the selected checkpoint scored on every test cell,
with per-class metrics and the full configuration). For a seeded batch:

```bash
python3 scripts/summarize_batch.py --prefix b6_
```

`summarize_batch.py` reports mean ± sd per configuration on every cell, a
ranking with how many paired seeds the best configuration wins, and the typical
seed standard deviation. A difference much smaller than
that standard deviation is not a difference. For runs trained with
`TRACK_HOLDOUT=1`, `scripts/drift_diagnostic.py <run_dir>` compares validation
with the holdout epoch by epoch. Then write the conclusion — not the log — into
`EXPERIMENTS.md`.

---

## 6. Evaluation design

The design follows Lu & Kannan (JMR): customers are split in half, time is split
at campaign 28, and the cells are reported separately.

| | Calibration period (campaigns ≤ 27) | Holdout period (campaigns ≥ 28) |
|---|---|---|
| **In-sample customers (50%)** | training; campaign 27 is validation | `insample_users_holdout_period` |
| **Out-of-sample customers (50%)** | `outsample_users_calib_period` | **`outsample_users_holdout_period`** — the headline cell |

- **Validation is the last calibration campaign** (`--val-mode late`), not
  held-out customers. Validation drawn from the same period as training cannot
  detect temporal drift: in one diagnostic run it improved for 34 consecutive
  epochs while the holdout got worse (ledger R13, R16).
- **Out-of-sample customers get the population-mean customer parameter.**
- **The customer partition is fixed by `split_seed` (33)** and is independent of
  `--seed`, so runs that differ only in seed share one partition. It is derived
  from record order in the JSON; pass `--uids-dir` to pin it to explicit uid
  lists if the data file is ever regenerated.
- **The headline metric is NLL** on the out-of-sample × holdout cell. The best
  constant predictor scores about 1.65 there.

---

## 7. Correctness checks

Two label leaks have been found and fixed, and both came from the same cause:
a field produced by the same process that creates rows can reveal what kind of
row it is. **Any such field must be tested against the label before a model
consumes it.**

| Leak | What happened | Fix | Check |
|---|---|---|---|
| Obtained products (R1) | Row *t*'s inventory block recorded what was obtained *at* *t*; an all-zero block meant NotBuy with certainty | stream shifted one row (`shift_obtained=True`) | `scripts/check_obtained_leak.py` |
| Elapsed-time clock (R20) | Row *t*'s clock included the gap ending at *t*; a 24-hour gap means NotBuy 99.3% of the time | clock lagged one row (`time_bias_lag_ipt=True`) | `scripts/ipt_leak_check.py`, `scripts/test_time_bias_causality.py` |

After any edit to a model file, run both of these (a few seconds each):

```powershell
python scripts\smoke_test_shared_refactor.py
python scripts\test_time_bias_causality.py
```

The second also guards a PyTorch trap: without autocast, the inference fast
path mishandled the additive attention mask, so an evaluation in fp32 or on CPU
scored a different model from the one trained. The encoder now bypasses that
path, and the test checks that evaluation matches training.

---

## 8. Data

- **The data is confidential consumer data.** Never commit it, and never print
  or copy record contents; inspect only shapes, key names and aggregate
  statistics. `.gitignore` blocks `data/`, `*.xlsx`, `*.pt` and the output
  folders.
- **The master archive is on SMU OneDrive and is read-only.** Code reads a local
  working copy, found through `PRODUCTGPT_DATA`. The R preprocessing pipeline
  that produces the JSON lives in that archive, not in this repository.
- **Never write outputs into the data directory.** Checkpoints, results and logs
  go under the repository's ignored output folders, or `~/ProductGPT/runs` on
  HPCC.

| File | Used by | Notes |
|---|---|---|
| `clean_list_int_wide4_simple6_IPT.json` | gen 5 | Current revision. One inserted NotBuy per quiet 24-hour interval; adds `IPT` (hours since the previous row), `IsInserted`, `CampaignID`, holdout flags. 5,004 customers |
| `clean_list_int_wide4_simple6[_FeatureBasedTrain].json` | gen 2, gen 4 | Previous revision. NotBuy has a different meaning, so results are not comparable with `_IPT` |
| `SelectedFigureWeaponEmbeddingIndex.xlsx` | every feature model | The 34-attribute product table |

### `paths.py`

The single source of truth for locations. It resolves the data directory from
`PRODUCTGPT_DATA` and nothing else, and fails fast with `ProductGPTPathError`
naming the expected path when a file is missing.

```python
from paths import data_file, product_feature_xlsx, output_dir

train = data_file("clean_list_int_wide4_simple6_IPT.json")
feat = product_feature_xlsx()
```

`PRODUCTGPT_OUTPUT` optionally redirects checkpoints; it defaults to
`<repo>/checkpoints`. The PBS script points it at each run's directory.

---

## 9. The `shared/` package

Transformer building blocks that used to be copied byte-for-byte across eight
model modules now live in one place:

```
shared/vocab.py        token-ID layout
shared/features.py     FEATURE_COLS + load_feature_tensor(path)
shared/layers.py       LayerNormalization, FeedForwardBlock, InputEmbeddings, ...
shared/attention.py    CausalPerformer
shared/decoder.py      DecoderBlock, Decoder
shared/embeddings.py   SpecialPlusFeatureLookup (also used by gen 5)
```

Numerics are unchanged: each class is a verbatim copy of the canonical version,
and the mixture models' extra `gate` argument defaults to `None`. Importing a
model module no longer reads the spreadsheet; build the feature table
explicitly and pass it in:

```python
from shared.features import load_feature_tensor
feat = load_feature_tensor(path_to_xlsx)
model = build_transformer(..., feature_tensor=feat)
```

`model4_bigbird.py`, `model4_decoderonly_index_performer_original.py`,
`model4_decoderonly_feature_flash.py`, `model4_mixture_flash.py` and the
`model4_per*` family keep inline copies on purpose, because their components
genuinely differ.

---

## 10. Generation 4 and older

Generation 4 was the main line before gen 5. It has four independent variant
axes, which is why there are so many files:

- **`index` vs `feature`** — plain identity embeddings, or identity plus a
  projection of the 34 product attributes.
- **`performer` vs `flash`** — linear attention, or exact softmax through
  `F.scaled_dot_product_attention`. Flash won.
- **plain vs `mixture`** — per-customer mixture over H output projections.
- **`decision_only`** — trained on decision history alone.

Its best configuration was feature-based FlashAttention with `d_model=128`,
`d_ff=384`, `N=6`, `heads=8` (`evaluation/model_specs_hpcc.json`).

Things to know before running it:

- **Every `train4_*_aws.py` imports DeepSpeed at module level**, which has no
  Windows build. Generation 4 trains only on Linux.
- **Its reported numbers are not comparable with gen 5's.** Neither label leak
  in section 7 affects it — the `simple6` files it reads have no all-zero
  inventory blocks, and it has no time bias — but it predates the evaluation
  design in section 6 and reads data with a different meaning of NotBuy.
- **`evaluation/unified_model_eval_hpcc.py` hides a missing model.** It wraps its
  ProductGPT imports in `try/except`, and without DeepSpeed it silently reports
  only the baselines. Check `PRODUCTGPT_IMPORT_ERROR` before trusting a sparse
  table.
- Inside each generation folder, `configN.py` holds hyperparameters and paths,
  `datasetN*.py` builds tensors, `modelN*.py` defines the network and
  `trainN*.py` trains it.

---

## 11. Working with git across laptop and HPCC

There are three copies of the code: the laptop, GitHub
(`github.com/jinmiaomkt/ProductGPT`) and the HPCC checkout in
`~/ProductGPT/work`. Changes flow in one direction:

```
laptop  --git push-->  GitHub  --git pull-->  HPCC (~/ProductGPT/work)
```

- **Edit on the laptop, commit, push.** Then run `git pull` on HPCC before the
  next submission. A queued job reads the code when it *starts*, not when it
  was submitted.
- **Do not edit files directly on HPCC.** If you must, commit them there and
  pull on the laptop before editing the same file.
- **Commit messages say why, not what.** `Lag the time-bias clock one row`
  beats `change`; most of the history before September 2026 is named "change"
  and cannot be searched.
- **Never commit data, checkpoints or run outputs.** Ignore rules only prevent
  new additions; anything committed in the past stays in history.

```bash
git status                  # what has changed
git log --oneline -10       # the last ten commits
git pull                    # bring this copy up to date
```

---

## 12. Known loose ends

- **About 50 files in the gen 0–4 era tooling** (`baselines/`, `evaluation/`,
  `tuning/`, `crossval/`, `analysis/`) still mention `/home/ec2-user/...`,
  mostly as argparse defaults and usage examples. They work when paths are
  passed explicitly. `config0.py` and `config12.py` keep Colab-era
  `drive/MyDrive/...` paths.
- **`config0git` and `config12git` fail fast on the laptop**, because their data
  files were never copied into the working copy.
- **`scripts/regenerate_ipt.py`**, the Python port of the R generator, has not
  yet been validated against the R output. Validate it before using it to
  change the 24-hour interval.
- **The HPCC training runs use the derived customer split**, not frozen uid
  lists. It is stable while the data file is unchanged.
