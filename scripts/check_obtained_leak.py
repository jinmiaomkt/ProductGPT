"""
Test whether the `obtained` stream leaks the current label.

Hypothesis: AggregateInput's ObtainedProducts block at event t describes what
the user obtained AT t (not at t-1). If so, an all-zero block is a perfect
tell for y_t == 9 (NotBuy), because InsertNotBuy_GenerateJSON_IPT.R writes
"0 0 0 0 0 0 0 0 0 0" into inserted no-buy rows.

Prints ONLY aggregate contingency statistics -- never token values or records.
"""
import sys
from collections import Counter

sys.path.insert(0, r"C:\Users\jinmiao\ProductGPT\ProductGPT")
sys.path.insert(0, r"C:\Users\jinmiao\ProductGPT\ProductGPT\gen5_multistream")

import torch
from dataset_multistream import TransformerDataset, load_json_dataset
import paths

PATH = paths.data_file("clean_list_int_wide4_simple6_IPT.json")
N_USERS = int(sys.argv[1]) if len(sys.argv) > 1 else 300

print(f"loading (first {N_USERS} users) ...")
raw = load_json_dataset(str(PATH))[:N_USERS]
SHIFT = (len(sys.argv) > 2 and sys.argv[2] == "shift")
print(f"shift_obtained = {SHIFT}")
ds = TransformerDataset(raw, ai_rate=15, lto_len=4, obtained_len=10, shift_obtained=SHIFT,
                        prev_dec_len=1, max_events=1024)

# contingency: is the obtained block all-zero, vs is the label NotBuy(9)
tab = Counter()
prev_tab = Counter()
n_events = 0

for i in range(len(ds)):
    it = ds[i]
    obt, lab = it["obtained"], it["label"]
    prev = it["prev_decision"]
    S = min(obt.size(0), lab.size(0))
    if S == 0:
        continue
    obt, lab, prev = obt[:S], lab[:S], prev[:S]

    valid = (lab >= 1) & (lab <= 9)
    allzero = (obt == 0).all(dim=1)

    is9 = (lab == 9) & valid
    not9 = (lab != 9) & valid
    n_events += int(valid.sum())

    tab["zero_and_9"] += int((allzero & is9).sum())
    tab["zero_and_not9"] += int((allzero & not9).sum())
    tab["nonzero_and_9"] += int((~allzero & is9).sum())
    tab["nonzero_and_not9"] += int((~allzero & not9).sum())

    # same test against the PREVIOUS label, which is what the docstring claims
    # the obtained block describes
    prev9 = (prev == 9)
    prev_tab["zero_and_prev9"] += int((allzero & prev9 & valid).sum())
    prev_tab["zero_and_prevnot9"] += int((allzero & ~prev9 & valid).sum())
    prev_tab["nonzero_and_prev9"] += int((~allzero & prev9 & valid).sum())
    prev_tab["nonzero_and_prevnot9"] += int((~allzero & ~prev9 & valid).sum())

print(f"\nscored events: {n_events:,}\n")


def report(name, a, b, c, d, pos_label):
    # a = zero & pos, b = zero & neg, c = nonzero & pos, d = nonzero & neg
    print(f"=== obtained-block all zero  vs  {name} ===")
    print(f"{'':>22}{pos_label:>14}{'other':>14}")
    print(f"{'obtained all zero':>22}{a:>14,}{b:>14,}")
    print(f"{'obtained has products':>22}{c:>14,}{d:>14,}")
    tot = a + b + c + d
    if a + b:
        print(f"  P({pos_label} | all zero)      = {a/(a+b):.4f}")
    if c + d:
        print(f"  P({pos_label} | has products) = {c/(c+d):.4f}")
    if a + c:
        print(f"  P(all zero | {pos_label})      = {a/(a+c):.4f}")
    # A perfect 2x2 separation means the feature determines the label.
    leak = (b == 0 and c == 0) or (a == 0 and d == 0)
    print(f"  perfectly separating: {leak}")
    print()


report("label y_t == 9", tab["zero_and_9"], tab["zero_and_not9"],
       tab["nonzero_and_9"], tab["nonzero_and_not9"], "y_t=9")

report("prev decision y_(t-1) == 9", prev_tab["zero_and_prev9"],
       prev_tab["zero_and_prevnot9"], prev_tab["nonzero_and_prev9"],
       prev_tab["nonzero_and_prevnot9"], "y_(t-1)=9")

print("Interpretation:")
print("  If the FIRST table separates perfectly, the obtained block encodes the")
print("  CURRENT decision -> the model can read y_t off its own input, and the")
print("  NotBuy metrics are leakage, not prediction.")
print("  If only the SECOND separates, the block describes t-1 as documented")
print("  and there is no leak from this source.")
