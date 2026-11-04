# Run guide

How to set up, train and check the code. This needs the confidential data, except
where noted. For what the project is and what it has found, see the
[README](../README.md).

## Setup

### Python

The cluster environment is a venv with Python 3.11, built by
`scripts/setup_eval_env_hpcc.sh` from the repo root:

```bash
bash scripts/setup_eval_env_hpcc.sh
```

It installs PyTorch separately, then `requirements.txt`. Generations 5 and 6 do not
need DeepSpeed, so they also run on Windows. Some scripts under `simulation/` also
import `scipy`.

### Data location

All code reads the data directory from one environment variable. Set it once per
machine:

```bash
export PRODUCTGPT_DATA=/path/to/data                # Linux
```

```powershell
setx PRODUCTGPT_DATA "C:\path\to\data"              # Windows, affects new terminals only
```

[`paths.py`](../paths.py) resolves every location from that variable and fails with
`ProductGPTPathError`, naming the expected path, if a file is missing.
`PRODUCTGPT_OUTPUT` optionally redirects checkpoints, which otherwise go to
`<repo>/checkpoints`. Never write outputs into the data directory.

| File | Used by | Notes |
|---|---|---|
| `perproduct/clean_list_int_wide4_simple7_IPT.json` | Generation 6 by default; generation 5 with `--vocab-level 7` | Level-7 product coding, after the R26 correction |
| `clean_list_int_wide4_simple6_IPT.json` | Generation 5 by default | Level-6 product coding. One inserted NotBuy per quiet 24-hour interval; adds `IPT`, `IsInserted`, `CampaignID` and holdout flags |
| `clean_list_int_wide4_simple6[_FeatureBasedTrain].json` | Generations 2 and 4 | NotBuy has a different meaning, so results are not comparable with `_IPT` files |
| `SelectedFigureWeaponEmbeddingIndex.xlsx` | Every feature model | The 34-attribute product table |

### Import path, once per terminal

Scripts import each other by plain module name, so the code folders must be on
`PYTHONPATH`. From the repo root:

| Shell | Command |
|---|---|
| Linux, PBS jobs | `source ./env.sh` |
| PowerShell | `. .\env.ps1` (the leading dot matters) |
| cmd.exe | `.\env.bat` |

`ModuleNotFoundError: No module named 'config5'` means this step was skipped.

## Training

Train on the cluster. **Do not train on a laptop GPU:** on the project's 6 GB laptop,
three of three generation-6 pilot runs crashed the machine (fatal PCIe error). Unit
tests are fine on a laptop with the GPU hidden.

Submit from the repo root, because a job runs the code in the directory it was
submitted from. Each line of work has its own checkout on the cluster.

```bash
cd ~/ProductGPT/work_gen6
qsub -v SEED=1,TAG=g6_w384 scripts/gen6_train_hpcc.pbs          # generation 6
qsub -v MAX_EVENTS=1024,SEED=1,TAG=mytest scripts/gen5_train_hpcc.pbs   # generation 5
```

Directly, for generation 6:

```bash
python gen6_streaming/train6_streaming.py --profile hpcc --seed 1
```

- **Always set `TAG`.** The run directory is stable per configuration so a resubmitted
  job resumes its own `last.pt`, and two jobs with the same tag overwrite each other.
  Generation 5 uses `runs/gen5_hpcc_S<max_events>_b<batch>_<TAG>`, and generation 6 uses
  `runs/gen6_<profile>_W<window>_C<chunk>_b<batch>_<TAG>_s<SEED>`.
- **Always pass `MAX_EVENTS` to generation 5.** The PBS default is 512.
- **At most two GPU jobs run per user;** a third queues. CPU queues are free.
- **Experiments are submitted as batches**, one script each. From the repo root:
  `bash scripts/batches/submit_batch6.sh`, or
  `DRY_RUN=1 bash scripts/batches/submit_batch6.sh` to print the `qsub` lines.
- **A queued job reads the code when it starts, not when it was submitted.** Never
  `git pull` a checkout while its jobs are queued.

The comments at the top of each `.pbs` file document its knobs.

### Generation 5 switches

Every architectural choice is a command-line switch, so each component can be
credited or blamed against a matched control. Run
`python gen5_multistream/train5_multistream.py --help` for the full text.

| Dimension | Flag | Values | Default |
|---|---|---|---|
| Sequence model | `--arch` | `transformer`, `gru`, `lstm` | `transformer` |
| Encoder inside gen 5 | `--encoder` | `transformer`, `gru` (keeps cross-attention and inventory GRU) | `transformer` |
| Satiation cross-attention | `--no-cross-attn` | on / off | on |
| Inventory representation | `--inventory` | `tokens`, `slots` (additive per-product counts over the full history, lagged) | `tokens` |
| Satiation depth | `--sat-layers` | with `slots`: 0 = one attention step; L = L stacked blocks | 0 |
| Capacity | `--d-model`, `--n-layers`, `--n-heads`, `--d-ff` | width and depth | profile |
| Recency bias | `--time-bias` | `none`, `ordinal` (ALiBi, same as `--alibi`), `time` (elapsed hours) | `none` |
| Time clock | `--leaky-time-bias` | reproduces the batch-5 leak; never use for results | lagged |
| Product representation | `--no-product-id` | identity + attributes / attributes only | identity + attributes |
| Customer embedding | `--user-embedding` | on / off (R15: buys nothing measurable) | off |
| Mixture head | `--mix-heads H` | 0 = single projection | 0 |
| Regularisation | `--dropout`, `--augment`, `--patience` | | 0.10, off, 5 |
| Validation design | `--val-mode`, `--val-from` | `customers`, `late`; first late campaign | `late`, 27 |
| Split | `--split-mode` | `user`, `temporal`, `both` | `both` |
| Seed | `--seed` | training seed; the customer partition is fixed separately | 33 |
| Diagnostics | `--track-holdout` | score the holdout every epoch; never used for selection | off |
| Frozen split | `--uids-dir` | directory written by `scripts/export_gen5_split.py` | derived |

The customer partition is fixed by `split_seed` (33), independent of `--seed`. It is
derived from record order in the JSON, so pin it with `--uids-dir` if the data file is
ever regenerated.

### Monitoring and results

```bash
qstat -u $USER                      # /opt/pbs/bin/qstat if not on PATH
tail -f ~/ProductGPT/logs/gen6_train_<jobid>.log
bash scripts/hpcc_status.sh --once --force
```

Jobs can send a Telegram message on exit through `scripts/notify.sh`, and
`scripts/hpcc_cron.sh` can send an hourly change-only feed. Credentials live in
`~/.telegram_env` (mode 600), set up by `scripts/telegram_setup.sh`, and never in the
repository.

Each run directory holds `best.pt`, `last.pt`, `history.json` (one record per epoch)
and `final.json` (the selected checkpoint scored on every test cell, with per-class
metrics and the full configuration). For a seeded batch:

```bash
python3 scripts/summarize_batch.py --prefix b6_
```

It reports mean ± sd per configuration on every cell, a ranking with how many paired
seeds the best configuration wins, and the typical seed standard deviation. Then write
the conclusion, not the log, into `EXPERIMENTS.md`.

## Correctness checks

Run the tests on a CPU, with the GPU hidden. From the repo root, after setting the
import path:

```bash
for t in scripts/test_*.py scripts/smoke_test_shared_refactor.py; do
  CUDA_VISIBLE_DEVICES="" python -W ignore "$t" || echo "FAILED: $t"
done
```

All 13 take about 2.5 minutes. `test_pity_stream.py`, `test_rollout.py`,
`test_seq_metrics.py` and `test_vocab7.py` need the data; the other nine run on
synthetic data alone. After any edit to a model file, run at least
`smoke_test_shared_refactor.py`, `test_time_bias_causality.py` and
`test_inventory_slots.py`.

Two label leaks have been found and fixed, both from one cause: a field produced by
the same process that creates rows can reveal what kind of row it is. **Any such field
must be tested against the label before a model consumes it.**

| Leak | What happened | Fix | Check |
|---|---|---|---|
| Obtained products (R1) | Row *t*'s inventory block recorded what was obtained at *t*; an all-zero block meant NotBuy with certainty | Stream shifted one row (`shift_obtained=True`) | `scripts/check_obtained_leak.py` |
| Elapsed-time clock (R20) | Row *t*'s clock included the gap ending at *t*; a 24-hour gap means NotBuy 99.3% of the time | Clock lagged one row (`time_bias_lag_ipt=True`) | `scripts/ipt_leak_check.py`, `scripts/test_time_bias_causality.py` |

`test_time_bias_causality.py` also guards a PyTorch trap: without autocast, the
inference fast path mishandled the additive attention mask, so an evaluation in fp32
or on CPU scored a different model from the one trained. The encoder now bypasses that
path.

## The `shared/` package

Building blocks that used to be copied across eight model modules:

```
shared/vocab.py        token-ID layout
shared/features.py     FEATURE_COLS + load_feature_tensor(path)
shared/layers.py       LayerNormalization, FeedForwardBlock, InputEmbeddings, ...
shared/attention.py    CausalPerformer
shared/decoder.py      DecoderBlock, Decoder
shared/embeddings.py   SpecialPlusFeatureLookup
```

Importing a model module does not read the spreadsheet. Build the feature table and
pass it in:

```python
from shared.features import load_feature_tensor
feat = load_feature_tensor(path_to_xlsx)
model = build_transformer(..., feature_tensor=feat)
```

## Working with git

Code flows one way: laptop, then GitHub, then the cluster checkout.

- **Edit on the laptop, commit, push,** then pull on the cluster before the next
  submission, once that checkout has no queued jobs.
- **Each experiment line gets a branch and a worktree,** with a matching cluster
  checkout such as `~/ProductGPT/work_r46`.
- **Do not edit files directly on the cluster.**
- **Commit messages say why, not what.**
- **Never commit data, checkpoints or run outputs.** Ignore rules only prevent new
  additions, and anything committed stays in history. This repository is public.

## Known loose ends

- `scripts/regenerate_ipt.py`, the Python port of the R generator, has not been
  validated against the R output. Validate it before using it to change the 24-hour
  interval.
- The cluster training runs use the derived customer split, not frozen uid lists. It
  is stable while the data file is unchanged.
- The test scripts stay in `scripts/` rather than a subfolder because open branches add
  to them and their job scripts call them by path. Group them after those branches
  merge.
