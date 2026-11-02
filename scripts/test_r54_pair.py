"""R54: r49_pair_llms.py --restrict-to-ref on synthetic shards (no data, no model).

The reference scored 3 of 6 customers.  An arm that scored all 6 must be reduced to
those 3 before the gate, recover a planted gap exactly on them, and fail the gate if
one of their rows is missing.

    python scripts/test_r54_pair.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
HEADLINE = 4


def shard(path, uids, nll, drop_last=False):
    rows = 6
    uid = np.repeat(np.array(uids, np.int64), rows)
    pos = np.tile(np.arange(rows, dtype=np.int32), len(uids))
    if drop_last:
        uid, pos = uid[:-1], pos[:-1]
    n = len(uid)
    np.savez_compressed(path, uid_hash=uid, pos=pos, label=np.full(n, 3, np.int8),
                        cell=np.full(n, HEADLINE, np.int8), nll=np.full(n, nll, np.float32),
                        logp9=np.zeros((n, 9), np.float32), mass=np.zeros(n, np.float32))


def run(tmp, arm):
    out = tmp / "out.json"
    cmd = [sys.executable, str(REPO / "scripts" / "r49_pair_llms.py"), "--ref", f"ref:{tmp}:1",
           "--arm", f"{arm}:{tmp}:1", "--out", str(out), "--draws", "50", "--restrict-to-ref"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, json.loads(out.read_text())


def main():
    fails = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        shard(tmp / "ref_shard0.npz", [11, 12, 13], 1.0)
        shard(tmp / "full_shard0.npz", [11, 12, 13, 14, 15, 16], 0.75)
        shard(tmp / "gap_shard0.npz", [11, 12, 13], 0.75, drop_last=True)
        code, res = run(tmp, "full")
        d0 = res["headline"]["d"]["full_minus_ref"]["est"] if code == 0 else None
        ok = code == 0 and res["customers"] == 3 and res["headline"]["n"] == 18 and abs(d0 + 0.25) < 1e-6
        print(f"[{'PASS' if ok else 'FAIL'}] the arm is reduced to the reference's customers, d = {d0}")
        if not ok:
            fails.append("restrict")
        code, res = run(tmp, "gap")
        ok = code != 0 and res["gate_pass"] is False
        print(f"[{'PASS' if ok else 'FAIL'}] a missing row of a sampled customer fails the gate")
        if not ok:
            fails.append("gate")
    print("ALL PASS" if not fails else "FAILED: " + ", ".join(fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
