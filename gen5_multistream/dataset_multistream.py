from __future__ import annotations

import gzip
import json
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
from torch.utils.data import Dataset


def load_json_dataset(
    path: str,
    keep_uids: Optional[Iterable[str]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Loads:
      - JSON array:  [ {...}, {...}, ... ]
      - JSONL:       one JSON object per line
      - optionally gzipped (.gz) versions of either

    Optional:
      keep_uids: if provided, only keep records whose rec.get("uid") is in keep_uids.
    """
    keep: Optional[Set[str]] = set(keep_uids) if keep_uids is not None else None

    def _uid_matches(rec_uid, keep_set: Optional[Set[str]]) -> bool:
        if keep_set is None:
            return True
        if rec_uid is None:
            return False
        if isinstance(rec_uid, (list, tuple, set)):
            return any((x is not None and str(x) in keep_set) for x in rec_uid)
        return str(rec_uid) in keep_set

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
            return [
                rec for rec in data
                if isinstance(rec, dict) and _uid_matches(rec.get("uid"), keep)
            ]

        out: List[Dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            if _uid_matches(rec.get("uid"), keep):
                out.append(rec)
        return out


def parse_token_ids(x) -> List[int]:
    """
    Handles:
      - string: "19 0 42 40 ..."
      - list/tuple: [19, 0, 42, 40, ...]
    """
    if isinstance(x, str):
        x = x.strip()
        return [int(v) for v in x.split()] if x else []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    raise TypeError(f"Unsupported sequence type: {type(x)}")


class TransformerDataset(Dataset):
    """
    Multi-stream dataset for the final state-space Transformer.

    Required JSON fields:
        uid
        Decision
        LTO_ObtainedProducts
        LTO_PreviousDecision

    Optional legacy field:
        AggregateInput

    The prepared streams are interpreted as:

        LTO_ObtainedProducts:
            repeated blocks of length 14:
                [x_t(4), o_{t-1}(10)]

        LTO_PreviousDecision:
            repeated blocks of length 5:
                [x_t(4), y_{t-1}(1)]

        Decision:
            current labels y_t, length S

    __getitem__ returns:
        lto:             (S,4)
        obtained:        (S,10)
        prev_decision:   (S,)
        aggregate_input: (S*15,) legacy compatibility
        label:           (S,)
        uid:             str
        user_id:         scalar long tensor
    """

    def __init__(
        self,
        data,
        tok_src=None,
        tok_tgt=None,
        seq_len_ai: int = 0,
        seq_len_tgt: int = 0,
        num_heads: int = 0,
        ai_rate: int = 15,
        pad_token: int = 0,
        augment_permute_obtained: bool = False,
        lto_len: int = 4,
        obtained_len: int = 10,
        prev_dec_len: int = 1,
        base_seed: int = 12345,
        permute_mode: str = "event_obtained",
        only_if_no_zero: bool = True,
        keep_zeros_tail: bool = True,
        **kwargs,
    ):
        self.data = list(data)
        self.tok_src = tok_src
        self.tok_tgt = tok_tgt
        self.seq_len_ai = int(seq_len_ai)
        self.seq_len_tgt = int(seq_len_tgt)
        self.ai_rate = int(ai_rate)
        self.pad_id = int(pad_token)

        self.augment_permute_obtained = bool(augment_permute_obtained)
        self.lto_len = int(lto_len)
        self.obtained_len = int(obtained_len)
        self.prev_dec_len = int(prev_dec_len)
        self.base_seed = int(base_seed)
        self.permute_mode = str(permute_mode)
        self.only_if_no_zero = bool(only_if_no_zero)
        self.keep_zeros_tail = bool(keep_zeros_tail)

        if self.ai_rate != self.lto_len + self.obtained_len + self.prev_dec_len:
            raise ValueError(
                f"ai_rate must equal lto_len+obtained_len+prev_dec_len, got "
                f"{self.ai_rate} vs {self.lto_len}+{self.obtained_len}+{self.prev_dec_len}"
            )

        self.epoch = 0

        unique_uids = sorted({self._normalize_uid(rec.get("uid", "")) for rec in self.data})
        self.uid_to_index = {u: i + 1 for i, u in enumerate(unique_uids)}
        self.num_users = len(self.uid_to_index) + 1
        self.index_to_uid = {0: "[UNK]"}
        self.index_to_uid.update({idx: uid for uid, idx in self.uid_to_index.items()})

        self._uid_cache: List[str] = []
        self._lto_cache: List[torch.Tensor] = []
        self._obtained_cache: List[torch.Tensor] = []
        self._prev_dec_cache: List[torch.Tensor] = []
        self._label_cache: List[torch.Tensor] = []
        self._enc_cache: List[torch.Tensor] = []

        for rec in self.data:
            uid = self._normalize_uid(rec.get("uid", ""))

            # Decide number of decision occasions.
            dec_ids_raw = parse_token_ids(rec["Decision"])
            S = self.seq_len_tgt if self.seq_len_tgt > 0 else len(dec_ids_raw)

            # 1. [LTO, obtained] stream.
            xo_ids = parse_token_ids(rec["LTO_ObtainedProducts"])
            xo_ids = self._pad(xo_ids, S * (self.lto_len + self.obtained_len))
            xo = torch.tensor(xo_ids, dtype=torch.long).view(
                S, self.lto_len + self.obtained_len
            )
            lto_from_xo = xo[:, : self.lto_len].contiguous()
            obtained = xo[:, self.lto_len :].contiguous()

            # 2. [LTO, previous decision] stream.
            xy_ids = parse_token_ids(rec["LTO_PreviousDecision"])
            xy_ids = self._pad(xy_ids, S * (self.lto_len + self.prev_dec_len))
            xy = torch.tensor(xy_ids, dtype=torch.long).view(
                S, self.lto_len + self.prev_dec_len
            )
            prev_dec = xy[:, self.lto_len].contiguous()

            # Canonical LTO stream comes from LTO_ObtainedProducts.
            lto = lto_from_xo

            # 3. Current decision labels y_t.
            label = torch.tensor(self._pad(dec_ids_raw, S), dtype=torch.long)

            # 4. Legacy aggregate_input for old utilities/evaluation.
            if "AggregateInput" in rec:
                agg_ids = parse_token_ids(rec["AggregateInput"])
                L = self.seq_len_ai if self.seq_len_ai > 0 else S * self.ai_rate
                agg_ids = self._pad(agg_ids, L)
                aggregate_input = torch.tensor(agg_ids, dtype=torch.long)
            else:
                aggregate_input = torch.cat(
                    [lto, obtained, prev_dec.unsqueeze(-1)],
                    dim=-1,
                ).reshape(-1)
                L = self.seq_len_ai if self.seq_len_ai > 0 else S * self.ai_rate
                aggregate_input = torch.tensor(
                    self._pad(aggregate_input.tolist(), L),
                    dtype=torch.long,
                )

            self._uid_cache.append(uid)
            self._lto_cache.append(lto)
            self._obtained_cache.append(obtained)
            self._prev_dec_cache.append(prev_dec)
            self._label_cache.append(label)
            self._enc_cache.append(aggregate_input)

    def __len__(self):
        return len(self.data)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _normalize_uid(self, uid) -> str:
        if uid is None:
            return ""
        if isinstance(uid, (list, tuple, set)):
            vals = [str(x) for x in uid if x is not None]
            return "|".join(sorted(vals)) if vals else ""
        return str(uid)

    def _pad(self, ids: List[int], L: int) -> List[int]:
        ids = ids[:L]
        if len(ids) < L:
            ids = ids + [self.pad_id] * (L - len(ids))
        return ids

    def _permute_obtained_by_event(
        self,
        obtained: torch.Tensor,
        *,
        idx: int,
        sample_index: Optional[int],
    ) -> torch.Tensor:
        if not self.augment_permute_obtained:
            return obtained

        out = obtained.clone()
        si = int(sample_index) if sample_index is not None else 0

        seed = (
            self.base_seed
            + 1_000_003 * self.epoch
            + 9_917 * int(idx)
            + 104_729 * si
        )
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        for s in range(out.size(0)):
            row = out[s]
            nonpad = row.ne(self.pad_id)
            n = int(nonpad.sum().item())
            if n <= 1:
                continue

            # If PADs appear in the middle, skip to avoid corrupting the outcome.
            nonpad_idx = nonpad.nonzero(as_tuple=False).view(-1)
            if nonpad_idx.numel() > 0:
                last_nonpad = int(nonpad_idx[-1].item())
                if torch.any(row[: last_nonpad + 1].eq(self.pad_id)):
                    continue

            perm = torch.randperm(n, generator=g)
            vals = row[nonpad][perm]
            row[nonpad] = vals
            out[s] = row

        return out

    def __getitem__(self, idx: int, sample_index: Optional[int] = None):
        lto = self._lto_cache[idx].clone()
        obtained = self._obtained_cache[idx].clone()
        prev_dec = self._prev_dec_cache[idx].clone()
        label = self._label_cache[idx].clone()
        aggregate_input = self._enc_cache[idx].clone()

        obtained = self._permute_obtained_by_event(
            obtained,
            idx=idx,
            sample_index=sample_index,
        )

        uid = self._uid_cache[idx]
        user_id = self.uid_to_index.get(uid, 0)

        return {
            "lto": lto,
            "obtained": obtained,
            "prev_decision": prev_dec,
            "aggregate_input": aggregate_input,
            "label": label,
            "uid": uid,
            "user_id": torch.tensor(user_id, dtype=torch.long),
        }
