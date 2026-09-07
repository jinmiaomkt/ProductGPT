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

No Python file was edited during the reorganization — the import namespace is
identical to the old flat layout, just relocated.

---

## 4. Data

- Master data lives on SMU OneDrive (institution-managed, read-only).
- The working copy path comes from the `PRODUCTGPT_DATA` environment variable.
- Never write checkpoints, results, or logs into the data folder or OneDrive.

**Known gap:** no script reads `PRODUCTGPT_DATA` yet — 62 files still hard-code
AWS paths like `/home/ec2-user/data/...`. Centralizing this is the top follow-up item.

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
