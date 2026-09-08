"""
Multi-stream dataset for the generation-5 state-space Transformer.

WHY THIS WAS REWRITTEN (Sep 2026)
---------------------------------
The previous version required two JSON fields, LTO_ObtainedProducts and
LTO_PreviousDecision. The new IPT data generation dropped both. Rather than
regenerate the JSON, this version derives all three streams from
AggregateInput, which every generation of the data still carries.

That is safe because the fields were only ever rearrangements of the same
15-token event block (verified: 50/50 records reconstructed exactly on
clean_list_int_wide4_simple6.json, which still has all three fields):

    AggregateInput        = [ LTO x4 | Obtained x10 | PrevDecision x1 ]   (15)
    LTO_ObtainedProducts  = [ LTO x4 | Obtained x10 ]                     (14)
    LTO_PreviousDecision  = [ LTO x4 | PrevDecision x1 ]                  ( 5)

So this loader works unchanged on:
  - clean_list_int_wide4_simple6.json          (old, has the LTO_* fields)
  - clean_list_int_wide4_simple6_IPT.json      (new, does not)

It also carries the two new IPT-only fields through when present (IPT and
IsInserted), so a future model can consume them; they are simply absent
from the batch for older files.

__getitem__ returns:
    lto:             (S, 4)   int64   offer tokens x_t
    obtained:        (S, 10)  int64   products obtained since last decision, o_{t-1}
    prev_decision:   (S,)     int64   y_{t-1}
    label:           (S,)     int64   y_t, values in 1..9 (0 = PAD)
    aggregate_input: (S*15,)  int64   kept for backward compatibility
    uid:             str
    user_id:         scalar int64     index into the model's user embedding
    ipt:             (S,) float32     hours since previous decision  [IPT files only]
    is_inserted:     (S,) int64       1 = synthetic NotBuy row       [IPT files only]
"""
from __future__ import annotations

import gzip
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import torch
from torch.utils.data import Dataset

PAD_ID = 0
UNK_ID = 12  # what the WordLevel tokenizer maps an unrecognised token to


# ─────────────────────────────── loading ───────────────────────────────
def load_json_dataset(
    path: str,
    keep_uids: Optional[Iterable[str]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Load a JSON array or JSONL file, optionally gzipped, filtered by uid."""
    keep: Optional[Set[str]] = set(map(str, keep_uids)) if keep_uids is not None else None

    def _match(rec_uid) -> bool:
        if keep is None:
            return True
        if rec_uid is None:
            return False
        if isinstance(rec_uid, (list, tuple, set)):
            return any(x is not None and str(x) in keep for x in rec_uid)
        return str(rec_uid) in keep

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
            return [r for r in data if isinstance(r, dict) and _match(r.get("uid"))]
        out: List[Dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict) and _match(rec.get("uid")):
                out.append(rec)
        return out


def parse_token_ids(x: Any) -> List[int]:
    """
    Flatten a field into a list of ints.

    The R export writes literal 'NA' for missing values (rare: 4 in ~684k
    tokens sampled). The WordLevel tokenizer used by every other generation
    silently maps unknown strings to [UNK] = 12, so this does the same thing
    rather than crashing or inventing a different sentinel.
    """
    def one(t) -> int:
        try:
            return int(t)
        except (TypeError, ValueError):
            return UNK_ID

    if x is None:
        return []
    if isinstance(x, str):
        return [one(t) for t in x.split()]
    if isinstance(x, (int, float)):
        return [one(x)]
    if isinstance(x, (list, tuple)):
        out: List[int] = []
        for v in x:
            if isinstance(v, str):
                out.extend(one(t) for t in v.split())
            else:
                out.append(one(v))
        return out
    raise TypeError(f"Cannot parse token field of type {type(x)}")


def parse_floats(x: Any) -> List[float]:
    def one(t) -> float:
        try:
            return float(t)
        except (TypeError, ValueError):
            return 0.0

    if x is None:
        return []
    if isinstance(x, str):
        return [one(t) for t in x.split()]
    if isinstance(x, (int, float)):
        return [one(x)]
    if isinstance(x, (list, tuple)):
        out: List[float] = []
        for v in x:
            if isinstance(v, str):
                out.extend(one(t) for t in v.split())
            else:
                out.append(one(v))
        return out
    raise TypeError(f"Cannot parse float field of type {type(x)}")


# ─────────────────────────────── dataset ───────────────────────────────
class TransformerDataset(Dataset):
    def __init__(
        self,
        data: Sequence[Dict[str, Any]],
        tok_src=None,
        tok_tgt=None,
        seq_len_ai: int = 0,
        seq_len_tgt: int = 0,
        num_heads: int = 0,
        ai_rate: int = 15,
        pad_token: int = PAD_ID,
        augment_permute_obtained: bool = False,
        lto_len: int = 4,
        obtained_len: int = 10,
        prev_dec_len: int = 1,
        base_seed: int = 12345,
        permute_mode: str = "event_obtained",
        only_if_no_zero: bool = True,
        keep_zeros_tail: bool = True,
        max_events: Optional[int] = None,
        uid_to_index: Optional[Dict[str, int]] = None,
        shift_obtained: bool = True,
        truncate: str = "head",
        **kwargs,
    ):
        """
        shift_obtained
            LABEL LEAKAGE FIX (Sep 2026). Leave this True unless you are
            deliberately reproducing a pre-fix run.

            AggregateInput's ObtainedProducts block at event t records what the
            user obtained AT t, not at t-1 -- despite this class's original
            docstring claiming otherwise. Because InsertNotBuy_GenerateJSON_IPT.R
            writes "0 0 0 0 0 0 0 0 0 0" into inserted no-buy rows, an all-zero
            block is a perfect tell for y_t == 9 (NotBuy).

            Measured on 239,716 events of clean_list_int_wide4_simple6_IPT.json:
                P(y_t = 9 | obtained block all zero) = 1.0000  (83,664 events,
                                                                no exceptions)
                P(obtained block all zero | y_t = 9) = 0.9955
            i.e. the model could read the label off its own input. The first
            gen-5 run scored precision = recall = F1 = AUPRC = 1.000 on NotBuy
            for exactly this reason.

            With this True the stream is rolled forward one event, so row t
            carries o_{t-1} and the first event sees padding. That is what the
            architecture was designed for: the model should infer the decision
            from the offer, the inventory it held BEFORE deciding, and its
            previous decision.

            The other two streams need no shift: lto is x_t, the offer shown
            before the decision, and prev_decision is already y_{t-1} by
            construction in the R generator.

        max_events
            Truncate every user to their first N decision events. This is the
            memory knob for gen 5: the offer-inventory cross-attention builds a
            (B, H, S, 4, S*10) score tensor, so cost grows with S**2. At S=2048
            that single tensor is ~2.7 GB in fp32 before autograd overhead.
            None keeps the full sequence.

        uid_to_index
            Pass the TRAIN split's mapping into val/test so a user embedding
            means the same thing across splits. Unseen uids map to 0.
        """
        self.data = list(data)
        self.ai_rate = int(ai_rate)
        self.lto_len = int(lto_len)
        self.obtained_len = int(obtained_len)
        self.prev_dec_len = int(prev_dec_len)
        self.pad_id = int(pad_token)
        self.max_events = int(max_events) if max_events else None
        self.shift_obtained = bool(shift_obtained)
        if truncate not in ("head", "tail"):
            raise ValueError(f"truncate must be 'head' or 'tail', got {truncate!r}")
        # "head" keeps a user's FIRST max_events, "tail" their LAST.
        # Under a temporal split, head truncation deletes exactly the late
        # campaigns that form validation and test -- measured on 200 users at
        # max_events=256 it left train with 49,195 scored events against 948
        # for val and 488 for test. Temporal runs must use "tail".
        self.truncate = truncate

        self.augment_permute_obtained = bool(augment_permute_obtained)
        self.base_seed = int(base_seed)
        self.permute_mode = str(permute_mode)
        self.only_if_no_zero = bool(only_if_no_zero)
        self.keep_zeros_tail = bool(keep_zeros_tail)
        self.epoch = 0

        if self.ai_rate != self.lto_len + self.obtained_len + self.prev_dec_len:
            raise ValueError(
                f"ai_rate must equal lto_len+obtained_len+prev_dec_len, got "
                f"{self.ai_rate} vs {self.lto_len}+{self.obtained_len}+{self.prev_dec_len}"
            )

        # user index -------------------------------------------------------
        if uid_to_index is not None:
            self.uid_to_index = dict(uid_to_index)
        else:
            uniq = sorted({self._uid(r) for r in self.data})
            self.uid_to_index = {u: i + 1 for i, u in enumerate(uniq)}
        self.num_users = max(self.uid_to_index.values(), default=0) + 1

        # caches -----------------------------------------------------------
        self._uid_cache: List[str] = []
        self._lto: List[torch.Tensor] = []
        self._obt: List[torch.Tensor] = []
        self._prev: List[torch.Tensor] = []
        self._label: List[torch.Tensor] = []
        self._ipt: List[Optional[torch.Tensor]] = []
        self._ins: List[Optional[torch.Tensor]] = []
        self._hf: List[Optional[torch.Tensor]] = []   # FeatureBasedHoldout
        self._hi: List[Optional[torch.Tensor]] = []   # IndexBasedHoldout

        self.has_ipt = False
        self.has_is_inserted = False
        self.has_holdout = False

        for rec in self.data:
            uid = self._uid(rec)

            ai = parse_token_ids(rec["AggregateInput"])
            dec = parse_token_ids(rec.get("Decision", []))

            n_full = len(ai) // self.ai_rate
            if dec:
                n_full = min(n_full, len(dec))
            S = n_full
            if self.max_events is not None:
                S = min(S, self.max_events)
            if S <= 0:
                # keep the record but make it entirely padding, so indices stay aligned
                S = n_full = 1
                ai = [self.pad_id] * self.ai_rate
                dec = [self.pad_id]

            # Which S of the user's n_full events to keep. "tail" is required
            # for a temporal split, where the held-out campaigns are at the end.
            start = n_full - S if self.truncate == "tail" else 0
            self._event_offset = start          # used by the holdout slices below

            blocks = torch.tensor(
                ai[start * self.ai_rate: (start + S) * self.ai_rate],
                dtype=torch.long,
            ).view(S, self.ai_rate)

            a = self.lto_len
            b = self.lto_len + self.obtained_len
            obtained = blocks[:, a:b].contiguous()

            if self.shift_obtained:
                # Row t must describe o_{t-1}, not o_t -- see the shift_obtained
                # note in __init__. Roll forward one event; the first event has
                # no predecessor, so it gets padding.
                obtained = torch.cat(
                    [torch.full((1, obtained.size(1)), self.pad_id,
                                dtype=obtained.dtype),
                     obtained[:-1]],
                    dim=0,
                )

            self._lto.append(blocks[:, :a].contiguous())
            self._obt.append(obtained)
            self._prev.append(blocks[:, b].contiguous())
            self._label.append(torch.tensor(dec[start:start + S], dtype=torch.long))
            self._uid_cache.append(uid)

            if "IPT" in rec:
                v = parse_floats(rec["IPT"])[start:start + S]
                v = v + [0.0] * (S - len(v))
                self._ipt.append(torch.tensor(v, dtype=torch.float32))
                self.has_ipt = True
            else:
                self._ipt.append(None)

            if "IsInserted" in rec:
                v = parse_token_ids(rec["IsInserted"])[start:start + S]
                v = v + [0] * (S - len(v))
                self._ins.append(torch.tensor(v, dtype=torch.long))
                self.has_is_inserted = True
            else:
                self._ins.append(None)

            # Per-event temporal holdout flags, written by the R generator as
            #   FeatureBasedHoldout = CampaignID >= 28
            #   IndexBasedHoldout   = CampaignID >= 29
            # Together they define a three-way TEMPORAL split, which is the
            # holdout the dataset was actually designed for.
            if "FeatureBasedHoldout" in rec and "IndexBasedHoldout" in rec:
                hf = parse_token_ids(rec["FeatureBasedHoldout"])[start:start + S]
                hi = parse_token_ids(rec["IndexBasedHoldout"])[start:start + S]
                hf = hf + [0] * (S - len(hf))
                hi = hi + [0] * (S - len(hi))
                self._hf.append(torch.tensor(hf, dtype=torch.long))
                self._hi.append(torch.tensor(hi, dtype=torch.long))
                self.has_holdout = True
            else:
                self._hf.append(None)
                self._hi.append(None)

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _uid(rec: Dict[str, Any]) -> str:
        u = rec.get("uid", "")
        if isinstance(u, (list, tuple)):
            u = u[0] if u else ""
        return str(u)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._lto)

    def _permute_obtained(self, obtained: torch.Tensor, idx: int,
                          sample_index: Optional[int]) -> torch.Tensor:
        """
        Shuffle the obtained-products slots within each event. Encodes the prior
        that inventory is a set, not a sequence. Deterministic in
        (base_seed, epoch, idx, sample_index) so runs stay reproducible.
        """
        if not self.augment_permute_obtained:
            return obtained
        si = int(sample_index or 0)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.base_seed + 1_000_003 * self.epoch + 9_917 * idx + 104_729 * si)

        out = obtained.clone()
        for t in range(out.size(0)):
            row = out[t]
            if self.only_if_no_zero:
                if torch.any(row == self.pad_id):
                    continue
                out[t] = row[torch.randperm(row.numel(), generator=g)]
            else:
                nz = (row != self.pad_id).nonzero(as_tuple=False).view(-1)
                if nz.numel() <= 1:
                    continue
                last = int(nz[-1].item())
                prefix = row[: last + 1]
                if torch.any(prefix == self.pad_id):
                    continue
                out[t, : last + 1] = prefix[torch.randperm(prefix.numel(), generator=g)]
        return out

    def __getitem__(self, idx: int, sample_index: Optional[int] = None):
        obtained = self._permute_obtained(self._obt[idx], idx, sample_index)
        uid = self._uid_cache[idx]

        item: Dict[str, Any] = {
            "lto": self._lto[idx],
            "obtained": obtained,
            "prev_decision": self._prev[idx],
            "label": self._label[idx],
            "uid": uid,
            "user_id": torch.tensor(self.uid_to_index.get(uid, 0), dtype=torch.long),
        }
        if self._ipt[idx] is not None:
            item["ipt"] = self._ipt[idx]
        if self._ins[idx] is not None:
            item["is_inserted"] = self._ins[idx]
        if self._hf[idx] is not None:
            item["holdout_feature"] = self._hf[idx]
            item["holdout_index"] = self._hi[idx]
        return item


class TemporalRoleView(Dataset):
    """
    A temporal-split view over one TransformerDataset.

    The R generator marks each EVENT, not each user:
        FeatureBasedHoldout = CampaignID >= 28
        IndexBasedHoldout   = CampaignID >= 29
    which gives a three-way split in time:
        train : FeatureBasedHoldout == 0                    (campaigns <= 27)
        val   : FeatureBasedHoldout == 1 and Index... == 0  (campaign 28)
        test  : IndexBasedHoldout == 1                      (campaigns >= 29)

    Every user appears in every role, at different points in their own
    history. That is the intended design: predict a user's FUTURE behaviour,
    not a stranger's. It also means the per-user embedding is meaningful
    here, unlike under a user-disjoint split.

    Implemented by masking labels to PAD outside the role rather than by
    slicing sequences. The model still sees the full history leading up to
    each scored event, which is what makes it a forecast rather than a
    truncation, and the loss and metrics already ignore PAD. Wrapping one
    cached dataset also avoids holding three copies of ~600 MB of tensors.
    """

    # Three-way roles use both flags. The two-way roles follow Lu & Kannan
    # (JMR 2025), who split periods once into a calibration block and a
    # holdout block and take validation from held-out CUSTOMERS inside
    # calibration rather than from a slice of time.
    ROLES = ("train", "val", "test", "calibration", "holdout")

    def __init__(self, base: TransformerDataset, role: str,
                 holdout_flag: str = "feature"):
        if role not in self.ROLES:
            raise ValueError(f"role must be one of {self.ROLES}, got {role!r}")
        if holdout_flag not in ("feature", "index"):
            raise ValueError("holdout_flag must be 'feature' or 'index'")
        # Which boundary the two-way roles use. The R generator defines a
        # different one per model family:
        #   FeatureBasedHoldout = CampaignID >= 28   (feature models)
        #   IndexBasedHoldout   = CampaignID >= 29   (index models)
        # Gen 5 consumes the product feature table, so "feature" is correct
        # for it and campaign 28 belongs to the holdout, not to validation.
        self.holdout_flag = holdout_flag
        if not base.has_holdout:
            raise ValueError(
                "This data file has no FeatureBasedHoldout / IndexBasedHoldout "
                "fields, so a temporal split is not possible. Use the "
                "user-disjoint split instead."
            )
        self.base = base
        self.role = role

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __getitem__(self, idx: int, sample_index: Optional[int] = None):
        item = dict(self.base.__getitem__(idx, sample_index=sample_index))
        hf = item.pop("holdout_feature")
        hi = item.pop("holdout_index")

        keep = self._keep_mask(hf, hi)

        lab = item["label"].clone()
        lab[~keep] = self.base.pad_id
        item["label"] = lab
        return item

    def _keep_mask(self, hf: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """Which events this role scores."""
        if self.role == "train":
            return hf == 0
        if self.role == "val":
            return (hf == 1) & (hi == 0)
        if self.role == "test":
            return hi == 1
        flag = hf if self.holdout_flag == "feature" else hi
        return flag == 0 if self.role == "calibration" else flag == 1

    def subset(self, indices: List[int]) -> "IndexSubset":
        """Restrict this temporal role to a subset of users."""
        return IndexSubset(self, indices)

    def scored_events(self) -> int:
        """How many events this role actually scores (for a sanity check)."""
        n = 0
        for i in range(len(self.base)):
            it = self.base[i]
            hf, hi = it["holdout_feature"], it["holdout_index"]
            lab = it["label"]
            keep = self._keep_mask(hf, hi)
            n += int((keep & (lab >= 1) & (lab <= 9)).sum())
        return n


class IndexSubset(Dataset):
    """
    torch's Subset, but forwarding set_epoch and the sample_index kwarg that
    RepeatWithPermutation-style augmentation relies on.

    Used to cross a temporal role with a set of users, giving the 2x2:

                        users seen in training | users never seen
        campaigns <= 27   TRAIN                | cold-start test
        campaigns >= 29   forecasting test     | strict test

    Each cell isolates a different kind of generalisation. The forecasting
    cell asks whether we can predict a KNOWN customer's future; the
    cold-start cell whether we can predict a stranger during a period we
    trained on; the strict cell both at once, which is the hardest and the
    closest to deploying on a newly acquired customer.
    """

    def __init__(self, base: Dataset, indices: List[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __getitem__(self, i: int, sample_index: Optional[int] = None):
        j = self.indices[i]
        try:
            return self.base.__getitem__(j, sample_index=sample_index)
        except TypeError:
            return self.base[j]

    def scored_events(self) -> int:
        n = 0
        for i in range(len(self)):
            lab = self[i]["label"]
            n += int(((lab >= 1) & (lab <= 9)).sum())
        return n


# ─────────────────────────────── collate ───────────────────────────────
def collate_multistream(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad a batch to the longest sequence in it.

    Padding to the batch max rather than to a global maximum matters here:
    the cross-attention is O(S**2), so letting one long user set the length
    for the whole batch is what makes short batches cheap.
    """
    S = max(int(b["lto"].size(0)) for b in batch)
    lto_len = batch[0]["lto"].size(1)
    obt_len = batch[0]["obtained"].size(1)

    def pad2(t: torch.Tensor, width: int) -> torch.Tensor:
        n = S - t.size(0)
        if n <= 0:
            return t[:S]
        return torch.cat([t, torch.full((n, width), PAD_ID, dtype=t.dtype)], dim=0)

    def pad1(t: torch.Tensor, fill=PAD_ID) -> torch.Tensor:
        n = S - t.size(0)
        if n <= 0:
            return t[:S]
        return torch.cat([t, torch.full((n,), fill, dtype=t.dtype)], dim=0)

    out: Dict[str, Any] = {
        "lto": torch.stack([pad2(b["lto"], lto_len) for b in batch]),
        "obtained": torch.stack([pad2(b["obtained"], obt_len) for b in batch]),
        "prev_decision": torch.stack([pad1(b["prev_decision"]) for b in batch]),
        "label": torch.stack([pad1(b["label"]) for b in batch]),
        "user_id": torch.stack([b["user_id"] for b in batch]),
        "uid": [b["uid"] for b in batch],
    }
    if "ipt" in batch[0]:
        out["ipt"] = torch.stack(
            [pad1(b["ipt"].to(torch.float32), 0.0).to(torch.float32) for b in batch]
        )
    if "is_inserted" in batch[0]:
        out["is_inserted"] = torch.stack([pad1(b["is_inserted"]) for b in batch])
    return out
