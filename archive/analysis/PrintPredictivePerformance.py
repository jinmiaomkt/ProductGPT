import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, average_precision_score
from sklearn.preprocessing import label_binarize

MODELS  = ["featurebased", "indexbased", "gru", "lstm"]
CLASSES = np.arange(1, 10)                     # decision classes 1..9

def load_metrics(tag):
    y_true  = np.load(f"{tag}_val_labels.npy")   # (N,)
    y_score = np.load(f"{tag}_val_scores.npy")   # (N, 9)

    # macro-AUPRC
    y_bin   = label_binarize(y_true, classes=CLASSES)
    auprc   = average_precision_score(y_bin, y_score, average="macro")

    # macro-F1  (argmax → predicted class ID 1-9)
    y_pred  = y_score.argmax(axis=1) + 1
    f1      = f1_score(y_true, y_pred, average="macro")
    return len(y_true), f1, auprc

print(f"{'Model':<14} {'N_dec':>7}   {'macro-F1':>8}   {'macro-AUPRC':>11}")
print("-" * 45)
for m in MODELS:
    try:
        n, f1, pr = load_metrics(m)
        print(f"{m:<14} {n:7d}   {f1:8.4f}   {pr:11.4f}")
    except FileNotFoundError:
        print(f"{m:<14}   (files not found)")
