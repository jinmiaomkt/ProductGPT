import json
import gzip
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Iterable, Set
# from __future__ import annotations

def load_json_dataset(
    path: str,
    keep_uids: Optional[Iterable[str]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Loads:
      - JSON array:  [ {...}, {...}, ... ]
      - JSONL:       one JSON object per line
      - optionally gzipped (.gz) versions of either

    Optional:
      keep_uids: if provided, only keep records whose rec.get("uid") is in keep_uids.
      **kwargs: ignored (forward-compat for callers that pass extra options).
    """
    keep: Optional[Set[str]] = set(keep_uids) if keep_uids is not None else None

    # def _uid_matches(rec_uid, keep_set: Set[str] | None) -> bool:
    def _uid_matches(rec_uid, keep_set: Optional[Set[str]]) -> bool:
        if keep_set is None:
            return True
        if rec_uid is None:
            return False

        # uid sometimes is a list (unhashable) -> keep if any element is in keep
        if isinstance(rec_uid, (list, tuple, set)):
            return any((x is not None and str(x) in keep_set) for x in rec_uid)

        return str(rec_uid) in keep_set

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        # JSON array
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {path}, got {type(data)}")
            if keep is None:
                return data
            # return [rec for rec in data if isinstance(rec, dict) and rec.get("uid") in keep]
            return [rec for rec in data if isinstance(rec, dict) and _uid_matches(rec.get("uid"), keep)]

        # JSONL
        out: List[Dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            if keep is not None and not _uid_matches(rec.get("uid"), keep):
                continue
            out.append(rec)
        return out

class TransformerDataset(Dataset):
    def __init__(
        self,
        data,
        tok_src,
        tok_tgt,
        seq_len_ai,
        seq_len_tgt,
        num_heads,
        ai_rate,
        pad_token=0,
        augment_permute_obtained: bool = False,
        lto_len: int = 4,
        obtained_len: int = 10,
        prev_dec_len: int = 1,
        base_seed: int = 12345,
        permute_mode: str = "last_block",
        only_if_no_zero: bool = True,
        keep_zeros_tail: bool = True,
        **kwargs,
    ):
        self.keep_zeros_tail = keep_zeros_tail
        self.data = data
        self.tok_src = tok_src
        self.tok_tgt = tok_tgt
        self.seq_len_ai = seq_len_ai
        self.seq_len_tgt = seq_len_tgt
        self.ai_rate = ai_rate
        self.pad_id = int(pad_token)

        self.augment_permute_obtained = augment_permute_obtained
        self.lto_len = int(lto_len)
        self.obtained_len = int(obtained_len)
        self.prev_dec_len = int(prev_dec_len)
        self.base_seed = int(base_seed)
        self.permute_mode = str(permute_mode)
        self.only_if_no_zero = bool(only_if_no_zero)

        self.epoch = 0

        self._enc_cache: List[torch.Tensor] = []
        self._lab_cache: List[torch.Tensor] = []
        self._uid_cache: List[str] = []

        # self.uid_to_index = {str(u): i for i, u in enumerate(sorted({rec.get("uid","") for rec in self.data}))}
        # self.num_users = len(self.uid_to_index)

        # Collect normalized user ids (handle scalar or list)
        uid_values = []

        for rec in self.data:
            uid = rec.get("uid", "")
            if isinstance(uid, (list, tuple, set)):
                for u in uid:
                    if u is not None:
                        uid_values.append(str(u))
            else:
                if uid is not None:
                    uid_values.append(str(uid))

        unique_uids = sorted(set(uid_values))
        self.uid_to_index = {u: i for i, u in enumerate(unique_uids)}
        self.num_users = len(self.uid_to_index)

        for rec in self.data:
            uid = rec.get("uid", "")

            agg = rec["AggregateInput"]
            src_txt = " ".join(map(str, agg)) if isinstance(agg, (list, tuple)) else str(agg)
            ai_ids = self._pad(self.tok_src.encode(src_txt).ids, self.seq_len_ai)

            dec = rec["Decision"]
            tgt_txt = " ".join(map(str, dec)) if isinstance(dec, (list, tuple)) else str(dec)
            tgt_ids = self._pad(self.tok_tgt.encode(tgt_txt).ids, self.seq_len_tgt)

            self._uid_cache.append(uid)
            self._enc_cache.append(torch.tensor(ai_ids, dtype=torch.long))
            self._lab_cache.append(torch.tensor(tgt_ids, dtype=torch.long))

        if self.lto_len + self.obtained_len + self.prev_dec_len > self.ai_rate:
            raise ValueError(
                f"Invalid offsets: lto_len({self.lto_len}) + obtained_len({self.obtained_len}) + "
                f"prev_dec_len({self.prev_dec_len}) > ai_rate({self.ai_rate})."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.data)

    def _pad(self, ids, L):
        ids = ids[:L]
        if len(ids) < L:
            ids = ids + [self.pad_id] * (L - len(ids))
        return ids

    def _maybe_permute_slice_(self, ids: torch.Tensor, a: int, z: int, g: torch.Generator) -> None:
        """
        Permute ids[a:z] in-place subject to the configured PAD/zero policy.
        """
        if a < 0 or z > ids.numel() or a >= z:
            return

        slice_ = ids[a:z]
        if slice_.numel() <= 1:
            return

        pad = self.pad_id

        # Policy 1: only permute if there are NO PADs in the slice.
        if self.only_if_no_zero:
            if torch.any(slice_ == pad):
                return
            perm = torch.randperm(slice_.numel(), generator=g)
            ids[a:z] = slice_[perm]
            return

        # Policy 2: keep PADs as a trailing tail if present (skip if PADs appear in the middle).
        if self.keep_zeros_tail:
            if not torch.any(slice_ == pad):
                perm = torch.randperm(slice_.numel(), generator=g)
                ids[a:z] = slice_[perm]
                return

            # Find last non-PAD; permute only the prefix [0:last_nonpad+1] if it contains no PAD.
            nonpad_idx = (slice_ != pad).nonzero(as_tuple=False).view(-1)
            if nonpad_idx.numel() == 0:
                return  # all PADs
            last_nonpad = int(nonpad_idx[-1].item())

            prefix = slice_[: last_nonpad + 1]
            if torch.any(prefix == pad):
                return  # PADs in the middle -> skip

            perm = torch.randperm(prefix.numel(), generator=g)
            ids[a : a + prefix.numel()] = prefix[perm]
            # tail PADs remain untouched
            return

        # Policy 3: permute including PADs (rarely desired, but supported)
        perm = torch.randperm(slice_.numel(), generator=g)
        ids[a:z] = slice_[perm]

    def _permute_obtained_inplace(self, ids: torch.Tensor, *, idx: int, sample_index: Optional[int]) -> None:
        """
        ids: (seq_len_ai,) int64 tensor
        Deterministically permute obtained-products slice based on (epoch, idx, sample_index).
        """
        if not self.augment_permute_obtained:
            return

        si = int(sample_index) if sample_index is not None else 0

        # deterministic per (epoch, idx, sample_index)
        # primes reduce collisions; keep under 64-bit comfortably.
        seed = (
            self.base_seed
            + 1_000_003 * self.epoch
            + 9_917 * int(idx)
            + 104_729 * si
        )

        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        start_offset = self.lto_len
        end_offset = self.lto_len + self.obtained_len  # exclusive
        L = ids.numel()

        if self.permute_mode == "last_block":
            b0 = max(0, L - self.ai_rate)
            a = b0 + start_offset
            z = b0 + end_offset
            if z <= L:
                self._maybe_permute_slice_(ids, a, z, g)
            return

        if self.permute_mode == "all_blocks":
            for b0 in range(0, L, self.ai_rate):
                a = b0 + start_offset
                z = b0 + end_offset
                if z > L:
                    break
                self._maybe_permute_slice_(ids, a, z, g)
            return

        raise ValueError(f"Unknown permute_mode={self.permute_mode!r}. Use 'last_block' or 'all_blocks'.")

    def __getitem__(self, idx: int, *, sample_index: Optional[int] = None):
        enc_input = self._enc_cache[idx].clone()   # clone so permutation doesn't corrupt cache
        label_tensor = self._lab_cache[idx]
        uid = self._uid_cache[idx]
        user_id = self.uid_to_index.get(uid, 0)   # or reserve an UNK user bucket

        # IMPORTANT: use (idx, sample_index) so RepeatWithPermutation creates distinct permutations
        self._permute_obtained_inplace(enc_input, idx=idx, sample_index=sample_index)

        return {"uid": uid, "aggregate_input": enc_input, "label": label_tensor,     "user_id": torch.tensor(user_id, dtype=torch.long)}