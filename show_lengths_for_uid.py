#!/usr/bin/env python3
"""
show_lengths_for_uid.py
-----------------------
Print how many decision slots we have *in the labels* and *in the predictions*
for a single user ID (uid).

Edit the three paths below if yours are different, then:

    python show_lengths_for_uid.py
"""

import json, gzip, sys
from pathlib import Path

UID        = "3bee09aa08a35465"          # <--  target uid
LABEL_PATH = Path("/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/Data/clean_list_int_wide4_simple6.json")
PRED_PATH  = Path('/Users/jxm190071/Dropbox/Mac/Desktop/E2 Genshim Impact/TuningResult/FeatureBasedLP/LP_feature_predictions.jsonl')

# ---------------- 1) load labels ---------------------------------
label_data = json.loads(LABEL_PATH.read_text())

if isinstance(label_data, list):                          # row‑oriented
    records = label_data
else:                                                     # columnar
    records = [{k: label_data[k][i] for k in label_data}
               for i in range(len(label_data["uid"]))]

def flat_uid(u):                                          # 'abc' or ['abc'] → 'abc'
    return str(u[0] if isinstance(u, list) else u)

label_rec = next((rec for rec in records
                  if flat_uid(rec["uid"]) == UID), None)

if label_rec is None:
    print(f"UID {UID} not found in label file.", file=sys.stderr)
    sys.exit(1)

n_labels = len(label_rec["Decision"])
print(f"Labels: {n_labels} decision slots")

# ---------------- 2) load predictions ----------------------------
open_pred = gzip.open if PRED_PATH.suffix == ".gz" else open
with open_pred(PRED_PATH, "rt", errors="replace") as f:
    for line in f:
        if not line.lstrip().startswith("{"):
            continue
        rec = json.loads(line)
        if flat_uid(rec.get("uid", "")) != UID:
            continue

        probs_field = rec["probs"]
        # old format: list‑of‑lists, new format: single list
        if probs_field and isinstance(probs_field[0], list):
            n_preds = len(probs_field)
        else:
            n_preds = 1    # one vector for the whole sequence

        print(f"Predictions: {n_preds} decision slots")
        break