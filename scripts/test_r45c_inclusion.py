"""
Tests for R45 part C, the inclusion of the real left-censored players
(build_loaders' extra file, censor6.CensorMark, the trainer's wiring). Real
data, CPU only; aggregates only. Reads the cfg of R44's frozen gen-6 run, so
the loaders are built exactly as arm F's were.

  P1  the main sample is untouched: with the extra file, validation and the
      three main test cells hold the same customers in the same order with
      identical tensors, and so does training when the extra half does not train.
      When it trains, training is the main training set, unchanged, plus it.
  P2  the headline cell scores the same occasions in every build.
  P3  the extra partition: disjoint halves covering every censored player,
      identical in two builds and for extra_train 0 and 1, never mixed with the
      main sample's cells.
  P4  CensorMark flags every censored customer and no main one.
  P5  the flag reaches the model through the trainer's own loader and
      evaluation path: with non-zero flag parameters, a censored customer's
      logits move and a main customer's do not, in one batch.

    python scripts/test_r45c_inclusion.py --ckpt <R44 gen-6 run>/gen6_streaming/hpcc/best.pt
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "gen5_multistream", REPO / "gen6_streaming", REPO / "simulation",
          REPO / "shared"):
    sys.path.insert(0, str(p))

import train5_multistream as t5  # noqa: E402
import train6_streaming as t6  # noqa: E402
from censor6 import CensorMark  # noqa: E402
from model6_streaming import build_from_cfg  # noqa: E402
from rollout import load_checkpoint  # noqa: E402

EXTRA = "perproduct/censored_list_int_wide4_simple7_IPT.json"
HEAD = "outsample_users_holdout_period"
MAIN = ("insample_users_holdout_period", "outsample_users_calib_period", HEAD)
CENS = ("censored_insample_holdout_period", "censored_outsample_holdout_period")
FAIL = []
torch.set_num_threads(1)


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    if not ok:
        FAIL.append(name)


def fp(it):
    """Fingerprint of one item: every tensor's shape and bytes."""
    h = hashlib.sha1()
    for k in sorted(it):
        v = it[k]
        if torch.is_tensor(v):
            h.update(k.encode())
            h.update(str(tuple(v.shape)).encode())
            h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


def summary(cfg):
    """uid order and fingerprints of every cell, then the datasets are dropped."""
    tr, va, te, _ = t5.build_loaders(cfg)
    cells = {"train": tr.dataset, "val": va.dataset, **{k: v.dataset for k, v in te.items()}}
    out = {k: [(ds[i]["uid"], fp(ds[i])) for i in range(len(ds))] for k, ds in cells.items()}
    events = {k: ds.scored_events() for k, ds in te.items() if hasattr(ds, "scored_events")}
    return out, events, cfg.pop("_extra_uids", None), cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    _, cfg0 = load_checkpoint(Path(a.ckpt), device="cpu")

    print("building: no extra file (arm F) ...", flush=True)
    s0, e0, x0, _ = summary(dict(cfg0))
    print("building: extra file, not trained ...", flush=True)
    s1, e1, x1, _ = summary(dict(cfg0, extra_data_file=EXTRA, extra_train=False))
    print("building: extra file, trained ...", flush=True)
    s2, e2, x2, cells2 = summary(dict(cfg0, extra_data_file=EXTRA, extra_train=True))

    print("P1 main sample untouched")
    for k in ("val",) + MAIN:
        check(f"{k}: {len(s0[k]):,} customers, identical in both extra builds",
              s0[k] == s1[k] == s2[k])
    check(f"train, extra not trained: identical ({len(s0['train']):,} customers)",
          s0["train"] == s1["train"])
    n0 = len(s0["train"])
    check(f"train, extra trained: the main {n0:,} unchanged, then "
          f"{len(s2['train']) - n0:,} censored", s2["train"][:n0] == s0["train"]
          and {u for u, _ in s2["train"][n0:]} <= set(x2))

    print("P2 headline occasions")
    check(f"{HEAD} scores {e0.get(HEAD, 0):,} occasions in every build",
          e0.get(HEAD) == e1.get(HEAD) == e2.get(HEAD) and e0.get(HEAD, 0) > 0)

    print("P3 extra partition")
    xin = [u for u, _ in s2[CENS[0]]]
    xout = [u for u, _ in s2[CENS[1]]]
    check(f"halves {len(xin):,} / {len(xout):,}: disjoint, cover all {len(x2):,}",
          not set(xin) & set(xout) and set(xin) | set(xout) == set(x2)
          and len(xin) + len(xout) == len(x2))
    check("identical for extra_train 0 and 1", s1[CENS[0]] == s2[CENS[0]]
          and s1[CENS[1]] == s2[CENS[1]] and x1 == x2)
    main_u = {u for k in ("train", "val") + MAIN for u, _ in s0[k]}
    check("no censored player in a main cell, no main customer in a censored cell",
          not main_u & set(x2) and x0 is None)
    check(f"extra_train 1 trains exactly the in-sample half ({len(xin):,})",
          [u for u, _ in s2["train"][n0:]] == xin)

    print("P4 CensorMark")
    marks = {k: [int(CensorMark(cells2[k], x2)[i]["censored"]) for i in range(len(cells2[k]))]
             for k in ("val",) + MAIN + CENS}
    check("every censored-cell customer flagged",
          all(all(v == 1 for v in marks[k]) for k in CENS))
    check("no main-cell customer flagged",
          all(not any(marks[k]) for k in ("val",) + MAIN))

    print("P5 the flag reaches the model through the trainer's path")
    cfg = dict(cfg0, censor_flag=True)
    feat = t6.load_feature_tensor(t6.config5.feature_file_path(cfg),
                                  id_column=cfg.get("feature_id_column"),
                                  first_prod_id=cfg["first_prod_id"],
                                  last_prod_id=cfg["last_prod_id"],
                                  max_token_id=cfg["vocab_size_src"] - 1)
    torch.manual_seed(0)
    m = build_from_cfg(cfg, feat.float(), 20_000).eval()
    m.chunk = 256
    mixed = t5.Subset(torch.utils.data.ConcatDataset(
        [CensorMark(cells2[HEAD], x2), CensorMark(cells2[CENS[1]], x2)]),
        [0, len(cells2[HEAD])])          # one main, one censored customer
    batch = t6.collate_scored([mixed[0], mixed[1]])
    check(f"the batch carries censored = {batch['censored'].tolist()}",
          batch["censored"].tolist() == [0, 1])
    kw = t6._inventory_kwargs6(m, batch, torch.device("cpu"))
    check("evaluation passes it to the model", "censored" in kw)
    with torch.no_grad():
        args = (batch["lto"], batch["obtained"], batch["prev_decision"], batch["user_id"])
        lo = m(*args, **kw)
        m.censor_vec.fill_(0.5)
        m.censor_h0.fill_(0.3)
        hi = m(*args, **kw)
    d_main = float((hi[0] - lo[0]).abs().max())
    d_cens = float((hi[1] - lo[1]).abs().max())
    check(f"main customer unchanged ({d_main:.1e}), censored customer moves ({d_cens:.1e})",
          d_main < 1e-6 and d_cens > 1e-4)

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED:")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all R45 part C checks passed")


if __name__ == "__main__":
    main()
