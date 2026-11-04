# Archive

Generations 0 to 4 and the tooling of that era. Nothing here is used by the current
code, and it is kept so that older results stay traceable. Files are unchanged
except for the paths in `scripts/productgpt_eval_hpcc.pbs`.

`env.sh`, `env.ps1` and `env.bat` still put these folders on `PYTHONPATH`, so
imports by plain module name work as before.

## What moved on 4 November 2026

Ledger entries and comments written before this date use the old paths. Each
ledger entry names the commit it ran from, and that commit has the old layout.

| Old path | New path |
|---|---|
| `gen0_encoder_decoder/` | `archive/gen0_encoder_decoder/` |
| `gen12_encdec_lto/` | `archive/gen12_encdec_lto/` |
| `gen2_lp_duplet/` | `archive/gen2_lp_duplet/` |
| `gen4_full_productgpt/` | `archive/gen4_full_productgpt/` |
| `baselines/` | `archive/baselines/` |
| `evaluation/` | `archive/evaluation/` |
| `tuning/` | `archive/tuning/` |
| `crossval/` | `archive/crossval/` |
| `analysis/` | `archive/analysis/` |
| `legacy/` | `archive/legacy/` |
| `vendor/` | `archive/vendor/` |
| `evaluation/requirements_eval_hpcc.txt` | `requirements.txt` |
| `scripts/submit_batch*.sh` | `scripts/batches/submit_batch*.sh` |
| `scripts/productgpt_eval_hpcc.pbs` | `archive/scripts/productgpt_eval_hpcc.pbs` |
| `scripts/CloudShell.sh`, `commmand_collection.sh`, `setup_dependencies.sh`, `setup_dependencies_git.sh`, `setup_dependency_gcp.sh`, `run_embed_sweeps.sh`, `run_ray_tune_mixture2.sh`, `run_unified_eval_example.sh` | `archive/scripts/` |
| `PAPER_REVISION_PLAN.md` | `docs/PAPER_REVISION_PLAN.md` |

The submit scripts still run from the repo root, for example
`bash scripts/batches/submit_batch6.sh`. Their header comments show the old path.

## Folders

| Folder | Contents |
|---|---|
| `gen0_encoder_decoder/` | Generation 0: encoder–decoder with full softmax attention |
| `gen12_encdec_lto/` | Encoder–decoder with a third limited-time-offer stream |
| `gen2_lp_duplet/` | Generation 2: decoder-only Performer on offer + previous decision |
| `gen4_full_productgpt/` | Generation 4: decoder-only Performer or FlashAttention |
| `baselines/` | GRU and LSTM baselines of that era |
| `evaluation/` | Prediction and evaluation scripts |
| `tuning/` | Ray Tune searches and campaign sweeps |
| `crossval/` | Cross-validation runners |
| `analysis/` | Tables and result analysis |
| `legacy/` | Older training snapshots |
| `vendor/` | Upstream Transformer-XL reference code, not this project's |
| `scripts/` | AWS and GCP setup, and old command collections |

Inside each generation folder, `configN.py` holds hyperparameters and paths,
`datasetN*.py` builds tensors, `modelN*.py` defines the network and `trainN*.py`
trains it.

## Generation 4

Generation 4 has four independent variant axes, which is why it has so many files:

- **`index` or `feature`:** plain identity embeddings, or identity plus a projection
  of the 34 product attributes.
- **`performer` or `flash`:** linear attention, or exact softmax through
  `F.scaled_dot_product_attention`. Flash won.
- **plain or `mixture`:** a per-customer mixture over H output projections.
- **`decision_only`:** trained on decision history alone.

Its best configuration was feature-based FlashAttention with `d_model=128`,
`d_ff=384`, `N=6` and `heads=8` (`evaluation/model_specs_hpcc.json`).

Before running it:

- **Linux only.** Every `train4_*_aws.py` imports DeepSpeed at module level, which
  has no Windows build.
- **Its numbers are not comparable with generations 5 and 6.** It predates the
  current evaluation design and reads data in which NotBuy has a different meaning.
  Its results also predate `EXPERIMENTS.md`, and R26 voids any claim that depends on
  *what* was obtained.
- **A missing model can be hidden.** `evaluation/unified_model_eval_hpcc.py` wraps its
  ProductGPT imports in `try/except`, and without DeepSpeed it silently reports only
  the baselines. Check `PRODUCTGPT_IMPORT_ERROR` before trusting a sparse table.

## Known issues

- `analysis/Feature-based Embedding.py` contains pseudocode and does not parse.
- About 50 files still mention `/home/ec2-user/...`, mostly as argparse defaults and
  usage examples. They work when paths are passed explicitly.
- `config0.py` and `config12.py` keep Colab-era `drive/MyDrive/...` paths.
