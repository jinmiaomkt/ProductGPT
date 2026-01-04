import json
import gzip
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Iterable, Set

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
            return [rec for rec in data if isinstance(rec, dict) and rec.get("uid") in keep]

        # JSONL
        out: List[Dict[str, Any]] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            if keep is not None and rec.get("uid") not in keep:
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
        self.pad_id = pad_token

        self.augment_permute_obtained = augment_permute_obtained
        self.lto_len = lto_len
        self.obtained_len = obtained_len
        self.prev_dec_len = prev_dec_len
        self.base_seed = base_seed
        self.permute_mode = permute_mode
        self.only_if_no_zero = only_if_no_zero

        self.epoch = 0

        self._enc_cache = []
        self._lab_cache = []
        self._uid_cache = []

        for rec in self.data:
            uid = rec.get("uid","")
            agg = rec["AggregateInput"]
            src_txt = " ".join(map(str, agg)) if isinstance(agg, (list, tuple)) else str(agg)
            ai_ids = self._pad(self.tok_src.encode(src_txt).ids, self.seq_len_ai)

            dec = rec["Decision"]
            tgt_txt = " ".join(map(str, dec)) if isinstance(dec, (list, tuple)) else str(dec)
            tgt_ids = self._pad(self.tok_tgt.encode(tgt_txt).ids, self.seq_len_tgt)

            self._uid_cache.append(uid)
            self._enc_cache.append(torch.tensor(ai_ids, dtype=torch.long))
            self._lab_cache.append(torch.tensor(tgt_ids, dtype=torch.long))

        # Basic sanity: expected block length
        # Some setups have ai_rate == lto_len + obtained_len + prev_dec_len (+1 pad/special)
        # We do not hard-fail here, but you should confirm offsets are correct.
        if self.lto_len + self.obtained_len + self.prev_dec_len > self.ai_rate:
            raise ValueError(
                f"Invalid offsets: lto_len({lto_len}) + obtained_len({obtained_len}) + "
                f"prev_dec_len({prev_dec_len}) > ai_rate({ai_rate})."
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
        Permute ids[a:z] in-place if it satisfies the 'no zero' caveat.
        """
        slice_ = ids[a:z]
        if slice_.numel() <= 1:
            return

        # Caveat: only permute if obtained slice contains NO zeros
        if self.only_if_no_zero and torch.any(slice_ == 0):
            return

        perm = torch.randperm(slice_.numel(), generator=g)
        ids[a:z] = slice_[perm]

    def _permute_obtained_inplace(self, ids: torch.Tensor, sample_index: int) -> None:
        """
        ids: (seq_len_ai,) int64 tensor
        Permute the obtained-products slice according to permute_mode.
        """
        if not self.augment_permute_obtained:
            return

        # deterministic per (epoch, sample_index)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.base_seed + 1000003 * self.epoch + sample_index)

        start_offset = self.lto_len
        end_offset = self.lto_len + self.obtained_len  # exclusive

        L = ids.numel()

        if self.permute_mode == "last_block":
            # Permute obtained slice in the LAST ai_rate tokens (most consistent with decision-only samples)
            b0 = max(0, L - self.ai_rate)
            a = b0 + start_offset
            z = b0 + end_offset
            if z <= L:
                self._maybe_permute_slice_(ids, a, z, g)
            return

        if self.permute_mode == "all_blocks":
            # Permute obtained slice in EVERY full block
            for b0 in range(0, L, self.ai_rate):
                a = b0 + start_offset
                z = b0 + end_offset
                if z > L:
                    break
                self._maybe_permute_slice_(ids, a, z, g)
            return

        raise ValueError(f"Unknown permute_mode={self.permute_mode!r}. Use 'last_block' or 'all_blocks'.")

    # def __getitem__(self, idx: int):
    #     rec = self.data[idx]

    #     # ----- UID -----
    #     uid = rec.get("uid", "")

    #     # ----- INPUT: AggregateInput -----
    #     # In your explode_record, AggregateInput is a string already.
    #     # But keep compatibility with list/tuple just in case.
    #     agg = rec["AggregateInput"]
    #     if isinstance(agg, (list, tuple)):
    #         src_txt = " ".join(map(str, agg))
    #     else:
    #         src_txt = str(agg)

    #     ai_ids = self.tok_src.encode(src_txt).ids
    #     ai_ids = self._pad(ai_ids, self.seq_len_ai)
    #     enc_input = torch.tensor(ai_ids, dtype=torch.long)

    #     # ----- TARGET: Decision -----
    #     # In explode_record you set Decision = lab (likely int)
    #     dec = rec["Decision"]
    #     if isinstance(dec, (list, tuple)):
    #         tgt_txt = " ".join(map(str, dec))
    #     else:
    #         tgt_txt = str(dec)

    #     tgt_ids = self.tok_tgt.encode(tgt_txt).ids
    #     tgt_ids = self._pad(tgt_ids, self.seq_len_tgt)
    #     label_tensor = torch.tensor(tgt_ids, dtype=torch.long)

    #     # ----- Augmentation (training only) -----
    #     self._permute_obtained_inplace(enc_input, idx)

    #     return {
    #         "uid": uid,
    #         "aggregate_input": enc_input,
    #         "label": label_tensor,
    #     }
    def __getitem__(self, idx: int):
        enc_input = self._enc_cache[idx].clone()   # clone so permutation doesn't corrupt cache
        label_tensor = self._lab_cache[idx]
        uid = self._uid_cache[idx]

        self._permute_obtained_inplace(enc_input, idx)
        return {"uid": uid, "aggregate_input": enc_input, "label": label_tensor}
