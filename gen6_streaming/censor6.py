"""
R45: controlled left-censoring on complete histories.

Every customer in the sample is observed from the account's start (the
beginner-banner filter). Chopping the start of those histories reproduces what
a left-censored customer looks like, with the truth known. A chop removes two
things, and the variants below separate them:

    state="zero"    the chopped rows are gone: no context, and the computed
                    state (inventory, stock, pity) restarts from zero -- a real
                    left-censored customer
    state="true"    the chopped rows are gone from the model's context but are
                    handed to the computed state, so inventory, stock and pity
                    enter the remaining rows exactly as in the full history
    state="labels"  nothing is removed; only the chopped rows' LABELS are masked.
                    The volume control: the same lost training signal, with
                    context and state intact

Chop points (per customer, deterministic, never into the validation or holdout
period -- the cut is capped at the first row of campaign `val_from`):
    mode="calendar" drop the customer's first `depth` active campaigns, the
                    mechanism of real censoring (accounts that began before
                    the data window)
    mode="count"    drop the first `depth` occasions

`ChopView` wraps any per-customer dataset (a TemporalRoleView subset) and
`collate_chopped` batches it, left-padding the prefixes so that every prefix
ends at row -1 (see StreamingHybrid.stream_forward).
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from dataset_multistream import PAD_ID

ROW_KEYS = ("lto", "obtained", "prev_decision", "label", "ipt", "is_inserted", "campaign")
MODES = ("none", "calendar", "count")
STATES = ("zero", "true", "labels")


def chop_index(campaign: torch.Tensor, mode: str, depth: int, val_from: int) -> int:
    """Rows [0, s) are chopped. Capped at the first row of campaign val_from."""
    if mode == "none" or depth <= 0:
        return 0
    c = campaign.tolist()
    cap = next((i for i, x in enumerate(c) if x >= val_from), len(c))
    if mode == "count":
        return min(int(depth), cap)
    seen: List[int] = []
    for i, x in enumerate(c[:cap]):
        if x > 0 and x not in seen:
            seen.append(x)
            if len(seen) == depth + 1:
                return i
    return cap


class ChopView(Dataset):
    """A per-customer dataset with each history's start chopped."""

    def __init__(self, base: Dataset, mode: str, depth: int, state: str, val_from: int):
        if mode not in MODES:
            raise ValueError(f"chop mode must be one of {MODES}, got {mode!r}")
        if state not in STATES:
            raise ValueError(f"chop state must be one of {STATES}, got {state!r}")
        self.base, self.mode, self.depth = base, mode, int(depth)
        self.state, self.val_from = state, int(val_from)

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        it = dict(self.base[i])
        s = chop_index(it["campaign"], self.mode, self.depth, self.val_from)
        it["chopped"] = torch.tensor(s, dtype=torch.long)
        it["censored"] = torch.tensor(int(s > 0), dtype=torch.long)
        if s == 0:
            return it
        if self.state == "labels":
            lab = it["label"].clone()
            lab[:s] = PAD_ID
            it["label"] = lab
            it["censored"] = torch.tensor(0, dtype=torch.long)   # nothing is hidden
            return it
        if self.state == "true":
            it["prefix_obtained"] = it["obtained"][:s]
            it["prefix_prev"] = it["prev_decision"][:s]
        for k in ROW_KEYS:
            if k in it:
                it[k] = it[k][s:]
        return it


def collate_prefix(items: List[Dict[str, Any]], out: Dict[str, Any]) -> Dict[str, Any]:
    """Add the chop fields to a collated batch; prefixes are LEFT-padded so every
    prefix ends at row -1 and PAD rows (no acquisitions, decision 0) change
    nothing in the computed state."""
    if "chopped" in items[0]:
        out["chopped"] = torch.stack([it["chopped"] for it in items])
        out["censored"] = torch.stack([it["censored"] for it in items])
    if any("prefix_obtained" in it for it in items):
        L = max(int(it["prefix_obtained"].size(0)) if "prefix_obtained" in it else 0 for it in items)
        width = items[0]["obtained"].size(1)
        po = torch.full((len(items), L, width), PAD_ID, dtype=torch.long)
        pp = torch.full((len(items), L), PAD_ID, dtype=torch.long)
        for b, it in enumerate(items):
            if "prefix_obtained" in it:
                n = int(it["prefix_obtained"].size(0))
                if n:
                    po[b, L - n:] = it["prefix_obtained"]
                    pp[b, L - n:] = it["prefix_prev"]
        out["prefix_obtained"], out["prefix_prev"] = po, pp
    return out
