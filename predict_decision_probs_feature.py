#!/usr/bin/env python
"""
predict_decision_probs_decisiononly.py

Inference for “Decision-Only” Performer model.
Outputs one JSON line per user with the 9-way decision distribution.
FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2
"""

import argparse, json, torch, deepspeed
from pathlib import Path
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from config4 import get_config
from model4_decoderonly_feature_performer import build_transformer
from train4_decision_only_performer_aws import _ensure_jsonl, JsonLineDataset, _build_tok

# ───────────── CLI ─────────────
cli = argparse.ArgumentParser()
cli.add_argument("--data", required=True, help="ND-JSON events file")
cli.add_argument("--ckpt", required=True, help="*.pt checkpoint")
cli.add_argument("--out",  required=True, help="output *.jsonl path")
args = cli.parse_args()

# ─────────── Config ────────────
cfg              = get_config()
cfg["ai_rate"]   = 15
cfg["batch_size"] = 512

# ──────── Tokenizer ────────────
tok_path = Path(cfg["model_folder"]) / "tokenizer_tgt.json"
tok_tgt  = (Tokenizer.from_file(str(tok_path)) if tok_path.exists()
            else _build_tok())
pad_id   = tok_tgt.token_to_id("[PAD]")

# ══════════════════════════════ 1. Constants ═══════════════════════════
PAD_ID = 0
DECISION_IDS = list(range(1, 10))  # 1‑9
SOS_DEC_ID, EOS_DEC_ID, UNK_DEC_ID = 10, 11, 12
FIRST_PROD_ID, LAST_PROD_ID = 13, 56
EOS_PROD_ID, SOS_PROD_ID, UNK_PROD_ID = 57, 58, 59
SPECIAL_IDS = [
    PAD_ID,
    SOS_DEC_ID,
    EOS_DEC_ID,
    UNK_DEC_ID,
    EOS_PROD_ID,
    SOS_PROD_ID,
]
MAX_TOKEN_ID = UNK_PROD_ID  # 59

# ══════════════════════════════ 2. Data helpers ════════════════════════
FEAT_FILE = Path("/home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx")
FEATURE_COLS: List[str] = [
    # stats
    "Rarity",
    "MaxLife",
    "MaxOffense",
    "MaxDefense",
    # categorical one‑hots
    "WeaponTypeOneHandSword",
    "WeaponTypeTwoHandSword",
    "WeaponTypeArrow",
    "WeaponTypeMagic",
    "WeaponTypePolearm",
    "EthnicityIce",
    "EthnicityRock",
    "EthnicityWater",
    "EthnicityFire",
    "EthnicityThunder",
    "EthnicityWind",
    "GenderFemale",
    "GenderMale",
    "CountryRuiYue",
    "CountryDaoQi",
    "CountryZhiDong",
    "CountryMengDe",
    # misc
    "type_figure",
    "MinimumAttack",
    "MaximumAttack",
    "MinSpecialEffect",
    "MaxSpecialEffect",
    "SpecialEffectEfficiency",
    "SpecialEffectExpertise",
    "SpecialEffectAttack",
    "SpecialEffectSuper",
    "SpecialEffectRatio",
    "SpecialEffectPhysical",
    "SpecialEffectLife",
    "LTO",
]


def load_feature_tensor(xls_path: Path) -> torch.Tensor:
    """Load product‑level feature embeddings – (V, D) FloatTensor."""
    df = pd.read_excel(xls_path, sheet_name=0)
    feat_dim = len(FEATURE_COLS)
    arr = np.zeros((MAX_TOKEN_ID + 1, feat_dim), dtype=np.float32)
    for _, row in df.iterrows():
        token_id = int(row["NewProductIndex6"])
        if FIRST_PROD_ID <= token_id <= LAST_PROD_ID:
            arr[token_id] = row[FEATURE_COLS].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)


# ══════════════════════════════ 3. Tokenisers ═════════════════════════=

def _base_tokeniser(extra_vocab: Dict[str, int] | None = None) -> Tokenizer:
    """Word‑level tokeniser with a fixed numeric vocabulary."""
    vocab: Dict[str, int] = {
        "[PAD]": PAD_ID,
        **{str(i): i for i in range(1, 10)},  # decisions
        "[SOS]": SOS_DEC_ID,
        "[EOS]": EOS_DEC_ID,
        "[UNK]": UNK_DEC_ID,
    }
    if extra_vocab:
        vocab.update(extra_vocab)
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    return tok


def build_tokenizer_src() -> Tokenizer:  # with product IDs
    prod_vocab = {str(i): i for i in range(FIRST_PROD_ID, UNK_PROD_ID + 1)}
    return _base_tokeniser(prod_vocab)


def build_tokenizer_tgt() -> Tokenizer:  # decisions only
    return _base_tokeniser()

# ─────── Dataset helpers ───────
def to_int_or_pad(tok: str) -> int:
    try:             return int(tok)
    except ValueError:return pad_id

class PredictDataset(JsonLineDataset):
    def __getitem__(self, idx):
        row     = super().__getitem__(idx)
        seq_raw = row["AggregateInput"]

        if isinstance(seq_raw, list):
            seq_str = (" ".join(map(str, seq_raw))
                       if not (len(seq_raw)==1 and isinstance(seq_raw[0], str))
                       else seq_raw[0])
        else:
            seq_str = seq_raw

        toks  = [to_int_or_pad(t) for t in seq_str.strip().split()]
        uid   = row["uid"][0] if isinstance(row["uid"], list) else row["uid"]
        return {"uid": uid, "x": torch.tensor(toks, dtype=torch.long)}

def collate(batch):
    uids = [b["uid"] for b in batch]
    lens = [len(b["x"]) for b in batch]
    Lmax = max(lens)
    X    = torch.full((len(batch), Lmax), pad_id, dtype=torch.long)
    for i,(item,L) in enumerate(zip(batch,lens)):
        X[i,:L] = item["x"]
    return {"uid": uids, "x": X}

loader = DataLoader(
            PredictDataset(_ensure_jsonl(args.data)),
            batch_size=cfg["batch_size"],
            collate_fn=collate,
            shuffle=False)

# ───────── Build model ──────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2
model  = build_transformer(
            vocab_size_tgt=cfg["vocab_size_tgt"],
            vocab_size_src=cfg["vocab_size_src"],
            d_model     = 32,
            n_layers    = 6,
            n_heads     = 4,
            d_ff        = 32,
            dropout     = 0.0,
            nb_features = 16,
            max_seq_len = 15360,
            kernel_type = cfg["kernel_type"],
            feature_tensor=load_feature_tensor(FEAT_FILE),
            special_token_ids=SPECIAL_IDS,).to(device).eval()

# ─────── LOAD CHECKPOINT ───────
def clean_state_dict(raw):
    def strip_prefix(k): return k[7:] if k.startswith("module.") else k
    ignore = ("weight_fake_quant", "activation_post_process")
    return {strip_prefix(k): v for k, v in raw.items()
            if not any(tok in k for tok in ignore)}

state = torch.load(args.ckpt, map_location=device)

if "model_state_dict" in state:
    raw_sd = state["model_state_dict"]
elif "module" in state:
    raw_sd = state["module"]
else:
    raw_sd = state

state_dict = clean_state_dict(raw_sd)
model.load_state_dict(state_dict, strict=True)

# ───────── Inference loop ───────
focus_ids = torch.arange(1, 10, device=device)  # decision classes 1-9

out_path = Path(args.out).expanduser()
with out_path.open("w") as fout, torch.no_grad():
    for batch in loader:
        x    = batch["x"].to(device)
        uids = batch["uid"]

        probs = torch.softmax(model(x), dim=-1)
        pos   = torch.arange(cfg["ai_rate"]-1, x.size(1), cfg["ai_rate"],
                             device=device)

        prob_dec_focus = probs[:, pos, :][..., focus_ids]  # (B, N, 9)

        for i, uid in enumerate(uids):
            fout.write(json.dumps({
                "uid": uid,
                "probs": prob_dec_focus[i].cpu().tolist()
            }) + "\n")

print(f"[✓] predictions written → {out_path}")
