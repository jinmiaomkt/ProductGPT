# ProductGPT — Purchase-Sequence Transformer

A Transformer for consumer sequential purchase behavior. Each user is a sequence of
*decision events*; the model predicts which of 9 decisions the user takes at each event,
given the offers shown, the products they already own, and their previous decision.

---

## 1. The generations

The model has been rebuilt several times. Rather than delete old versions, each
generation is kept side by side so results stay reproducible. **A file's number tells
you its generation** (`config4.py`, `model4_*.py`, `train4_*.py` are all generation 4).

| Folder | Gen | Architecture | Input layout | Status |
|---|---|---|---|---|
| `gen0_encoder_decoder/` | 0 | Encoder–decoder, full softmax attention | 2 streams, `source_rate=10` | superseded |
| `gen12_encdec_lto/` | 12 | Encoder–decoder + third LTO stream, two cross-attentions | `source_rate=10`, `lto_rate=12` | superseded |
| `gen2_lp_duplet/` | 2 | Decoder-only Performer ("LP / Duplet") | `ai_rate=5` — offer + previous decision | comparison baseline |
| `gen4_full_productgpt/` | 4 | Decoder-only Performer **or** FlashAttention | `ai_rate=15` — offer + inventory + previous decision | **current main line** |
| `gen5_multistream/` | 5 | Multi-stream state-space hybrid | separate offer / inventory / decision tensors | **experimental, not yet wired up** |

### Generation 4 in detail (the one in use)

Each decision event is 15 tokens:

```
[ LTO offer x4 | products obtained since last decision x10 | previous decision x1 ]
```

with `seq_len_tgt = 1024` events, so `seq_len_ai = 15 x 1024 = 15360` tokens per user.

Generation 4 has four independent variants, which is why there are so many files:

- **`index` vs `feature`** — `index` uses plain ID embeddings; `feature` adds a projection
  of 34 product attributes (rarity, weapon type, attack stats...). This is the core ablation.
- **`performer` vs `flash`** — Performer is linear-attention (random features, O(T));
  Flash is exact softmax attention via `F.scaled_dot_product_attention`. Flash currently wins.
- **plain vs `mixture`** — the mixture head gives each user their own blend over H output
  projections, to model consumer heterogeneity.
- **`decision_only`** — ablation trained on decision history alone (`train1_decision_only_performer_aws.py`).

**Current champion** (see `evaluation/model_specs_hpcc.json`):
feature-based + FlashAttention, `d_model=128, d_ff=384, N=6, heads=8`.

---

## 2. Repository layout

Model stacks are grouped **by generation**. Tooling is grouped **by function**, because
most tools span generations (and their filenames already carry the generation number).

```
gen0_encoder_decoder/     config0*, dataset0*, model0*, train0*
gen12_encdec_lto/         config12*, dataset12*, model12*, train12*
gen2_lp_duplet/           config2, dataset2*, model2*, train2*
gen4_full_productgpt/     config4, dataset4*, model4*, train4*   <-- start here
gen5_multistream/         multistream model + dataset + patch guide

shared/                   model building blocks used by the generations above

baselines/                GRU / LSTM comparison models and their predictors
evaluation/               unified_model_eval*, predict_*_and_eval, calibrators, model_specs_*.json
tuning/                   Ray Tune drivers, hP_tuning*, Phase A/B ranking, sweep helpers
crossval/                 fold/UID split generation, cross-validation runners
analysis/                 metric tables, ROC curves, confusion matrices, embedding plots
scripts/                  shell scripts and the SMU HPCC PBS job script
vendor/transformer_xl/    upstream Transformer-XL reference code (NOT this project)
legacy/                   older snapshots kept for reference
checkpoints/              local model weights (NOT tracked by git)
```

## 2a. The shared/ package

The basic Transformer blocks were copy-pasted, byte-for-byte, across eight model
modules. They now live in one place:

```
shared/vocab.py       token-ID layout (PAD, decisions 1-9, products 13-56, ...)
shared/features.py    FEATURE_COLS + load_feature_tensor(path)
shared/layers.py      gelu_approx, LayerNormalization, FeedForwardBlock,
                      InputEmbeddings, PositionalEncoding, ResidualConnection,
                      ProjectionLayer
shared/attention.py   CausalPerformer
shared/decoder.py     DecoderBlock, Decoder
shared/embeddings.py  SpecialPlusFeatureLookup
```

Use them like this:

```python
from shared.attention import CausalPerformer
from shared.features import load_feature_tensor
```

Two things to know:

1. **Numerics are unchanged.** Every class is a verbatim copy of the canonical
   implementation. Where the mixture models had threaded an extra `gate`
   argument through Decoder/DecoderBlock/CausalPerformer, the shared version
   takes `gate=None` by default, which reproduces the non-mixture behaviour
   exactly. `model4_mixture2_*` keeps its own `ProjectionLayer`, the one
   component that genuinely differs.

2. **Importing a model file no longer reads the spreadsheet.** The model modules
   used to run `pd.read_excel("/home/ec2-user/data/...")` at import time, which
   made them unimportable anywhere but the original EC2 box. Build the table
   explicitly instead:

   ```python
   from shared.features import load_feature_tensor
   feat = load_feature_tensor(path_to_xlsx)
   model = build_transformer(..., feature_tensor=feat)
   ```

   The trainers already did exactly this, so nothing changes at runtime.

Still on the old inline copies (left alone on purpose, both have genuinely
divergent components): `model4_bigbird.py`, `model4_decoderonly_index_performer_original.py`,
`model4_decoderonly_feature_flash.py`, `model4_mixture_flash.py`, and the `model4_per*` family.

Inside each generation folder the four files always mean the same thing:

- `configN.py` — all hyperparameters and paths in one dict
- `datasetN*.py` — turns raw JSON records into padded tensors
- `modelN*.py` — the network definition
- `trainN*.py` — the training loop, evaluation, and checkpointing

---

## 3. Running anything (important)

Scripts import each other by plain module name (`from config4 import get_config`).
That used to work because every file was in one folder. Now you must put the code
folders on `PYTHONPATH` first — **once per terminal session**, from the repo root:

Pick the one matching your shell. The prompt tells you which you're in:
`C:\...>` is cmd.exe, `PS C:\...>` is PowerShell.

**Windows / cmd.exe:**
```
env.bat
```

**Windows / PowerShell:**
```powershell
. .\env.ps1
```
(the leading dot matters — it runs the script in your current shell)

**Linux / SMU HPCC:**
```bash
source ./env.sh
```

Then run scripts by their new path:

```powershell
python gen4_full_productgpt\train4_decoderonly_flash_feature_aws.py
python evaluation\unified_model_eval_hpcc.py --config evaluation\model_specs_hpcc.json ...
```

If you get `ModuleNotFoundError: No module named 'config4'`, you forgot this step.

The folder move itself edited no Python file: the import namespace is identical to
the old flat layout, just relocated. The later shared/ extraction did edit six model
modules, but only to delete duplicated blocks and import them instead - see section 2a.

---

## 4. Data

- Master data lives on SMU OneDrive (institution-managed, read-only).
- The working copy path comes from the `PRODUCTGPT_DATA` environment variable.
- Never write checkpoints, results, or logs into the data folder or OneDrive.

### paths.py — where data paths come from

`paths.py` at the repo root is the single source of truth. It resolves the data
directory from `PRODUCTGPT_DATA` and nothing else — no path is hard-coded, so the
same code runs on the Windows laptop and on HPCC with only the variable differing.

```python
from paths import data_file, product_feature_xlsx, output_dir

train = data_file("clean_list_int_wide4_simple6_FeatureBasedTrain.json")
feat  = product_feature_xlsx()
```

Set it once per machine:

```
setx PRODUCTGPT_DATA "C:\Users\jinmiao\ResearchData\productgpt"     (Windows, then open a NEW terminal)
export PRODUCTGPT_DATA=/storage/home/jinmiao/ProductGPT/data        (HPCC — already in the .pbs)
```

`PRODUCTGPT_OUTPUT` is optional and controls where checkpoints go; it defaults to
`<repo>/checkpoints`, which git ignores. The PBS script points it at each job's own
output directory so concurrent runs don't collide.

**Fail-fast by design:** `get_config()` now resolves and *verifies* its data files,
so a missing file raises `ProductGPTPathError` naming the exact expected path
instead of a confusing traceback later. A consequence worth knowing: `config0git`
and `config12git` raise immediately on this laptop, because their data files were
never copied to the working copy (`clean_list_int_wide4_simple4_IndexBasedTrain.json`
is on OneDrive; `clean_list_int_wide12.json` isn't anywhere). That is correct
behaviour — those generations genuinely can't run here until the files are copied.

**Migrated so far:** the 4 configs that hard-coded `/home/ec2-user`
(`config4`, `config2`, `config0git`, `config12git`) and the 4 `FEAT_FILE`
constants in the `train4_*_aws.py` trainers.

**Still hard-coded (known, lower priority):** roughly 20 eval/predict scripts have
`--feat-xlsx` argparse *defaults* pointing at `/home/ec2-user/data/...`. They still
work because you pass `--data`/`--labels` explicitly anyway; only the default is
stale. `config0.py` and `config12.py` keep Colab-era `drive/MyDrive/...` paths —
a different environment, deliberately left alone.

---

## 5. Git quick reference

You are on a branch called `reorganize-structure`. Branches are cheap, isolated copies
of the project — nothing you do on one affects `main` until you merge it.

```bash
git status                  # what has changed right now
git branch                  # which branches exist; * marks the current one
git log --oneline -10       # last 10 commits
git switch main             # go back to the old flat layout
git switch reorganize-structure   # come back to this one
```

To keep this reorganization permanently:

```bash
git switch main
git merge reorganize-structure
git push
```

To throw it away entirely and forget it happened:

```bash
git switch main
git branch -D reorganize-structure
```

Two habits worth building:

1. **Commit small and often**, with a message saying *why*, not *what*.
   `git commit -m "Group model files by generation"` beats `git commit -m "change"`.
   (The history currently has 20+ commits all named "change" — those are unsearchable.)
2. **Never commit data or checkpoints.** `.gitignore` now blocks `data/`, `checkpoints/`,
   `results/`, and `*.pt`. Files already committed in the past stay in history; the
   ignore rules only prevent new ones.
