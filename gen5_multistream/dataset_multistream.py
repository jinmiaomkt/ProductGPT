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
        **kwargs,
    ):
        """
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

        self.has_ipt = False
        self.has_is_inserted = False

        for rec in self.data:
            uid = self._uid(rec)

            ai = parse_token_ids(rec["AggregateInput"])
            dec = parse_token_ids(rec.get("Decision", []))

            S = len(ai) // self.ai_rate
            if self.max_events is not None:
                S = min(S, self.max_events)
            S = min(S, len(dec)) if dec else S
            if S <= 0:
                # keep the record but make it entirely padding, so indices stay aligned
                S = 1
                ai = [self.pad_id] * self.ai_rate
                dec = [self.pad_id]

            blocks = torch.tensor(
                ai[: S * self.ai_rate], dtype=torch.long
            ).view(S, self.ai_rate)

            a = self.lto_len
            b = self.lto_len + self.obtained_len
            self._lto.append(blocks[:, :a].contiguous())
            self._obt.append(blocks[:, a:b].contiguous())
            self._prev.append(blocks[:, b].contiguous())
            self._label.append(torch.tensor(dec[:S], dtype=torch.long))
            self._uid_cache.append(uid)

            if "IPT" in rec:
                v = parse_floats(rec["IPT"])[:S]
                v = v + [0.0] * (S - len(v))
                self._ipt.append(torch.tensor(v, dtype=torch.float32))
                self.has_ipt = True
            else:
                self._ipt.append(None)

            if "IsInserted" in rec:
                v = parse_token_ids(rec["IsInserted"])[:S]
                v = v + [0] * (S - len(v))
                self._ins.append(torch.tensor(v, dtype=torch.long))
                self.has_is_inserted = True
            else:
                self._ins.append(None)

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
        return item


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
