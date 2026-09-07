"""
Token-ID layout for ProductGPT.

The vocabulary is a fixed integer partition, not learned. These constants were
previously re-declared in most model and trainer files; import them from here.

    0        PAD
    1..9     decisions
    10,11,12 SOS / EOS / UNK for decisions
    13..56   products
    57,58,59 EOS / SOS / UNK for products
"""
from __future__ import annotations

PAD_ID = 0

# ── decisions ────────────────────────────────────────────────────────────
FIRST_DECISION_ID = 1
LAST_DECISION_ID = 9
NUM_DECISION_CLASSES = 9
DECISION_IDS = list(range(FIRST_DECISION_ID, LAST_DECISION_ID + 1))  # 1..9

SOS_DEC_ID = 10
EOS_DEC_ID = 11
UNK_DEC_ID = 12

# ── products ─────────────────────────────────────────────────────────────
FIRST_PROD_ID = 13
LAST_PROD_ID = 56
PRODUCT_IDS = list(range(FIRST_PROD_ID, LAST_PROD_ID + 1))  # 13..56

EOS_PROD_ID = 57
SOS_PROD_ID = 58
UNK_PROD_ID = 59

MAX_TOKEN_ID = UNK_PROD_ID  # 59

SPECIAL_IDS = [
    PAD_ID,
    SOS_DEC_ID,
    EOS_DEC_ID,
    UNK_DEC_ID,
    EOS_PROD_ID,
    SOS_PROD_ID,
]

# Product tokens that carry an attribute row in the feature table.
# UNK_PROD_ID is included because the embedding treats it as a product whose
# feature row is all zeros (this matches the original models' behaviour).
FEATURE_BEARING_IDS = PRODUCT_IDS + [UNK_PROD_ID]

# ── event layout (generation 4) ──────────────────────────────────────────
# Each decision event is AI_RATE tokens:
#     [ LTO offer x4 | products obtained since last decision x10 | prev decision x1 ]
LTO_LEN = 4
OBTAINED_LEN = 10
PREV_DEC_LEN = 1
AI_RATE = LTO_LEN + OBTAINED_LEN + PREV_DEC_LEN  # 15

# Revenue attached to each decision 1..9 (decision 9 = no purchase).
REVENUE_PER_DECISION = [1, 10, 1, 10, 1, 10, 1, 10, 0]
