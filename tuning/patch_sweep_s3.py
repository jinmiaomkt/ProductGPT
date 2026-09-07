#!/usr/bin/env python3
"""
patch_sweep_s3.py

Patches run_campaign28_sweep.py in-place to write all outputs to S3
instead of local disk. Run once:

    python3 patch_sweep_s3.py --sweep /home/ec2-user/ProductGPT/run_campaign28_sweep.py

Then run the sweep with two new flags:
    --s3_bucket  productgptbucket
    --s3_prefix  outputs/c28_v1          (no leading/trailing slash)
"""

import argparse
import re
from pathlib import Path

PATCHES = []

# ── 1. Add boto3 import after torch import ────────────────────────────────────
PATCHES.append((
    "import torch",
    "import torch\nimport boto3\nimport io",
    "add boto3 import",
))

# ── 2. Replace Manifest class entirely ───────────────────────────────────────
OLD_MANIFEST = '''class Manifest:
    """Incrementally-written JSON manifest tracking every (uid, lto28) pair."""

    def __init__(self, path: Path):
        self.path = path
        self._records: Dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                self._records = json.load(f)

    def _key(self, uid: str, lto28_name: str) -> str:
        return f"{uid}|{lto28_name}"

    def mark_started(self, uid: str, lto28_name: str):
        key = self._key(uid, lto28_name)
        self._records[key] = {
            "uid": uid, "lto28_name": lto28_name,
            "status": "running", "started_at": datetime.utcnow().isoformat(),
            "finished_at": None, "n_runs": 0, "error": None,
        }
        self._flush()

    def mark_done(self, uid: str, lto28_name: str, n_runs: int):
        key = self._key(uid, lto28_name)
        self._records[key]["status"] = "done"
        self._records[key]["finished_at"] = datetime.utcnow().isoformat()
        self._records[key]["n_runs"] = n_runs
        self._flush()

    def mark_error(self, uid: str, lto28_name: str, error: str):
        key = self._key(uid, lto28_name)
        self._records[key]["status"] = "error"
        self._records[key]["finished_at"] = datetime.utcnow().isoformat()
        self._records[key]["error"] = error
        self._flush()

    def is_done(self, uid: str, lto28_name: str) -> bool:
        return self._records.get(self._key(uid, lto28_name), {}).get("status") == "done"

    def summary(self) -> Dict[str, int]:
        counts: Counter = Counter(v["status"] for v in self._records.values())
        return dict(counts)

    def _flush(self):
        with open(self.path, "w") as f:
            json.dump(self._records, f, indent=2)'''

NEW_MANIFEST = '''def _s3_put(s3_client, bucket: str, key: str, body: str) -> None:
    """Upload a string as a UTF-8 S3 object."""
    s3_client.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"))


def _s3_get(s3_client, bucket: str, key: str) -> Optional[str]:
    """Download an S3 object as a string, or None if it doesn\'t exist."""
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read().decode("utf-8")
    except s3_client.exceptions.NoSuchKey:
        return None
    except Exception:
        return None


class Manifest:
    """S3-backed manifest tracking every (uid, lto28) pair."""

    def __init__(self, s3_client, bucket: str, key: str):
        self.s3  = s3_client
        self.bucket = bucket
        self.key = key
        self._records: Dict[str, Any] = {}
        existing = _s3_get(s3_client, bucket, key)
        if existing:
            self._records = json.loads(existing)

    def _key(self, uid: str, lto28_name: str) -> str:
        return f"{uid}|{lto28_name}"

    def mark_started(self, uid: str, lto28_name: str):
        k = self._key(uid, lto28_name)
        self._records[k] = {
            "uid": uid, "lto28_name": lto28_name,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None, "n_runs": 0, "error": None,
        }
        self._flush()

    def mark_done(self, uid: str, lto28_name: str, n_runs: int):
        k = self._key(uid, lto28_name)
        self._records[k]["status"] = "done"
        self._records[k]["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._records[k]["n_runs"] = n_runs
        self._flush()

    def mark_error(self, uid: str, lto28_name: str, error: str):
        k = self._key(uid, lto28_name)
        self._records[k]["status"] = "error"
        self._records[k]["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._records[k]["error"] = error
        self._flush()

    def is_done(self, uid: str, lto28_name: str) -> bool:
        return self._records.get(self._key(uid, lto28_name), {}).get("status") == "done"

    def summary(self) -> Dict[str, int]:
        counts: Counter = Counter(v["status"] for v in self._records.values())
        return dict(counts)

    def _flush(self):
        _s3_put(self.s3, self.bucket, self.key, json.dumps(self._records, indent=2))'''

PATCHES.append((OLD_MANIFEST, NEW_MANIFEST, "replace Manifest with S3-backed version"))

# ── 3. Fix datetime imports to include timezone ───────────────────────────────
PATCHES.append((
    "from datetime import datetime",
    "from datetime import datetime, timezone",
    "add timezone import",
))

# ── 4. Replace run_user_lto28 local file write with in-memory buffer ──────────
OLD_RUN = '''    results = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "w") as fout:
        for r in range(n_seeds):'''

NEW_RUN = '''    results = []
    lines_buffer = []

    for r in range(n_seeds):'''

PATCHES.append((OLD_RUN, NEW_RUN, "remove local file open in run_user_lto28"))

OLD_WRITE_LINE = '''                line = json.dumps(payload)
                fout.write(line + "\\n")
                results.append(payload)
                if not quiet:
                    print(line, flush=True)

            except Exception as exc:
                err_payload = {
                    "uid": uid, "lto28_name": lto28_cfg["name"],
                    "run": r, "seed": seed, "error": str(exc),
                }
                fout.write(json.dumps(err_payload) + "\\n")
                print(f"  [WARN] uid={uid} lto28={lto28_cfg['name']} run={r} error: {exc}")

    return results'''

NEW_WRITE_LINE = '''                line = json.dumps(payload)
                lines_buffer.append(line)
                results.append(payload)
                if not quiet:
                    print(line, flush=True)

            except Exception as exc:
                err_payload = {
                    "uid": uid, "lto28_name": lto28_cfg["name"],
                    "run": r, "seed": seed, "error": str(exc),
                }
                lines_buffer.append(json.dumps(err_payload))
                print(f"  [WARN] uid={uid} lto28={lto28_cfg['name']} run={r} error: {exc}")

    # Upload buffered lines to S3 in one shot
    if lines_buffer and s3_client is not None:
        body = "\\n".join(lines_buffer) + "\\n"
        s3_key = f"{s3_prefix}/raw/{lto28_cfg['name']}/{uid}.jsonl"
        _s3_put(s3_client, s3_bucket, s3_key, body)
        if not quiet:
            print(f"  [S3] Uploaded {len(lines_buffer)} lines -> s3://{s3_bucket}/{s3_key}")

    return results'''

PATCHES.append((OLD_WRITE_LINE, NEW_WRITE_LINE, "buffer lines and upload to S3"))

# ── 5. Add s3_client/bucket/prefix params to run_user_lto28 signature ─────────
PATCHES.append((
    "def run_user_lto28(\n    *,\n    model,\n    history_tokens: List[int],\n    uid: str,\n    lto28_cfg: Dict,\n    n_seeds: int,\n    seed_base: int,\n    device: torch.device,\n    calibrator,\n    args,\n    max_seq_len: int,\n    raw_path: Path,\n    quiet: bool,\n) -> List[Dict]:",
    "def run_user_lto28(\n    *,\n    model,\n    history_tokens: List[int],\n    uid: str,\n    lto28_cfg: Dict,\n    n_seeds: int,\n    seed_base: int,\n    device: torch.device,\n    calibrator,\n    args,\n    max_seq_len: int,\n    raw_path: Path,\n    quiet: bool,\n    s3_client=None,\n    s3_bucket: str = \"\",\n    s3_prefix: str = \"\",\n) -> List[Dict]:",
    "add s3 params to run_user_lto28",
))

# ── 6. Add --s3_bucket and --s3_prefix CLI args ───────────────────────────────
PATCHES.append((
    '    parser.add_argument("--skip_done", action="store_true",',
    '    parser.add_argument("--s3_bucket", default="",\n                        help="S3 bucket for all outputs (skips local disk writes)")\n    parser.add_argument("--s3_prefix", default="outputs",\n                        help="S3 key prefix, e.g. outputs/c28_v1")\n    parser.add_argument("--skip_done", action="store_true",',
    "add s3 CLI args",
))

# ── 7. Replace local sweep_dir creation with S3 setup ────────────────────────
OLD_DIR_SETUP = '''    # ── output directory setup ────────────────────────────────────────────────
    ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(args.out_root) / f"{args.sweep_name}_{ts_str}"
    raw_dir     = sweep_dir / "raw"
    summary_dir = sweep_dir / "summary"
    for d in [sweep_dir, raw_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)'''

NEW_DIR_SETUP = '''    # ── output directory setup ────────────────────────────────────────────────
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_name_ts = f"{args.sweep_name}_{ts_str}"

    use_s3 = bool(args.s3_bucket)
    if use_s3:
        s3_client = boto3.client("s3")
        s3_prefix = f"{args.s3_prefix.strip('/')}/{sweep_name_ts}"
        print(f"[SWEEP] S3 output mode: s3://{args.s3_bucket}/{s3_prefix}/")
        # Small local tmp dir only for compatibility (not written to for outputs)
        sweep_dir = Path("/tmp") / sweep_name_ts
        sweep_dir.mkdir(parents=True, exist_ok=True)
    else:
        s3_client = None
        s3_prefix = ""
        sweep_dir = Path(args.out_root) / sweep_name_ts
        raw_dir     = sweep_dir / "raw"
        summary_dir = sweep_dir / "summary"
        for d in [sweep_dir, raw_dir, summary_dir]:
            d.mkdir(parents=True, exist_ok=True)'''

PATCHES.append((OLD_DIR_SETUP, NEW_DIR_SETUP, "replace local dir setup with S3 setup"))

# ── 8. Replace config snapshot write with S3-aware version ───────────────────
OLD_CONFIG_WRITE = '''    with open(sweep_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)
    print(f"[SWEEP] Output directory: {sweep_dir}")'''

NEW_CONFIG_WRITE = '''    config_body = json.dumps(config_snapshot, indent=2)
    if use_s3:
        _s3_put(s3_client, args.s3_bucket, f"{s3_prefix}/config.json", config_body)
        print(f"[SWEEP] Config -> s3://{args.s3_bucket}/{s3_prefix}/config.json")
    else:
        with open(sweep_dir / "config.json", "w") as f:
            f.write(config_body)
        print(f"[SWEEP] Output directory: {sweep_dir}")'''

PATCHES.append((OLD_CONFIG_WRITE, NEW_CONFIG_WRITE, "S3-aware config write"))

# ── 9. Replace Manifest instantiation ────────────────────────────────────────
OLD_MANIFEST_INIT = '    manifest = Manifest(sweep_dir / "manifest.json")'
NEW_MANIFEST_INIT = '''    if use_s3:
        manifest = Manifest(s3_client, args.s3_bucket, f"{s3_prefix}/manifest.json")
    else:
        manifest = Manifest(s3_client=None, bucket="", key=str(sweep_dir / "manifest.json"))'''

# The local Manifest needs a fallback for non-S3 mode — handle differently.
# Instead, we add a simpler approach: make Manifest work in both modes.
# Skip this patch and instead handle it via the unified Manifest class.

# ── 10. Replace CSV open_csv with S3-buffered version ────────────────────────
OLD_OPEN_CSV = '''def open_csv(path: Path, cols: List[str]):
    """Open (or append to) a CSV file, writing header only if new."""
    is_new = not path.exists()
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return fh, writer'''

NEW_OPEN_CSV = '''def open_csv(path: Path, cols: List[str]):
    """Open (or append to) a CSV file, writing header only if new."""
    is_new = not path.exists()
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return fh, writer


def make_memory_csv(cols: List[str]):
    """Return an in-memory (StringIO, DictWriter) pair for S3 upload."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    return buf, writer'''

PATCHES.append((OLD_OPEN_CSV, NEW_OPEN_CSV, "add make_memory_csv helper"))

# ── 11. Replace CSV initialization in main ───────────────────────────────────
OLD_CSV_INIT = '''    # ── open summary CSVs ─────────────────────────────────────────────────────
    all_runs_fh,   all_runs_writer   = open_csv(summary_dir / "all_runs.csv",   ALL_RUNS_COLS)
    stop_stats_fh, stop_stats_writer = open_csv(summary_dir / "stop_stats.csv", STOP_STATS_COLS)'''

NEW_CSV_INIT = '''    # ── open summary CSVs ─────────────────────────────────────────────────────
    if use_s3:
        all_runs_fh,   all_runs_writer   = make_memory_csv(ALL_RUNS_COLS)
        stop_stats_fh, stop_stats_writer = make_memory_csv(STOP_STATS_COLS)
    else:
        all_runs_fh,   all_runs_writer   = open_csv(summary_dir / "all_runs.csv",   ALL_RUNS_COLS)
        stop_stats_fh, stop_stats_writer = open_csv(summary_dir / "stop_stats.csv", STOP_STATS_COLS)'''

PATCHES.append((OLD_CSV_INIT, NEW_CSV_INIT, "S3-aware CSV init"))

# ── 12. Replace CSV flush call in the run loop ───────────────────────────────
PATCHES.append((
    "                all_runs_fh.flush()\n\n                # ── write stop_stats.csv row ───────────────────────────────\n                stats = compute_stop_stats(uid, lto28_cfg, results)\n                stop_stats_writer.writerow(stats)\n                stop_stats_fh.flush()",
    "                if not use_s3:\n                    all_runs_fh.flush()\n\n                # ── write stop_stats.csv row ───────────────────────────────\n                stats = compute_stop_stats(uid, lto28_cfg, results)\n                stop_stats_writer.writerow(stats)\n                if not use_s3:\n                    stop_stats_fh.flush()",
    "conditional flush for S3 mode",
))

# ── 13. Update run_user_lto28 call site to pass s3 params ────────────────────
OLD_CALL = '''                results = run_user_lto28(
                    model=model,
                    history_tokens=history_tokens,
                    uid=uid,
                    lto28_cfg=lto28_cfg,
                    n_seeds=args.n_seeds,
                    seed_base=args.seed_base,
                    device=device,
                    calibrator=calibrator,
                    args=args,
                    max_seq_len=max_seq_len,
                    raw_path=raw_path,
                    quiet=args.quiet,
                )'''

NEW_CALL = '''                results = run_user_lto28(
                    model=model,
                    history_tokens=history_tokens,
                    uid=uid,
                    lto28_cfg=lto28_cfg,
                    n_seeds=args.n_seeds,
                    seed_base=args.seed_base,
                    device=device,
                    calibrator=calibrator,
                    args=args,
                    max_seq_len=max_seq_len,
                    raw_path=raw_path,
                    quiet=args.quiet,
                    s3_client=s3_client,
                    s3_bucket=args.s3_bucket,
                    s3_prefix=s3_prefix,
                )'''

PATCHES.append((OLD_CALL, NEW_CALL, "pass s3 params to run_user_lto28"))

# ── 14. Replace final CSV close + final summary with S3-aware version ─────────
OLD_CLOSE = '''    # ── close CSVs ────────────────────────────────────────────────────────────
    all_runs_fh.close()
    stop_stats_fh.close()'''

NEW_CLOSE = '''    # ── close / upload CSVs ───────────────────────────────────────────────────
    if use_s3:
        _s3_put(s3_client, args.s3_bucket,
                f"{s3_prefix}/summary/all_runs.csv",   all_runs_fh.getvalue())
        _s3_put(s3_client, args.s3_bucket,
                f"{s3_prefix}/summary/stop_stats.csv", stop_stats_fh.getvalue())
        print(f"[S3] Uploaded summary CSVs to s3://{args.s3_bucket}/{s3_prefix}/summary/")
    else:
        all_runs_fh.close()
        stop_stats_fh.close()'''

PATCHES.append((OLD_CLOSE, NEW_CLOSE, "S3-aware CSV close/upload"))

# ── 15. Fix Manifest init for local (non-S3) mode ────────────────────────────
OLD_MANIFEST_INIT2 = '    manifest = Manifest(sweep_dir / "manifest.json")'
NEW_MANIFEST_INIT2 = '''    if use_s3:
        manifest = Manifest(s3_client, args.s3_bucket, f"{s3_prefix}/manifest.json")
    else:
        # Local fallback: monkey-patch _flush to write to a local file
        local_manifest_path = sweep_dir / "manifest.json"
        manifest = Manifest.__new__(Manifest)
        manifest.s3 = None
        manifest.bucket = ""
        manifest.key = str(local_manifest_path)
        manifest._records = {}

        def _local_flush(self=manifest):
            with open(self.key, "w") as _f:
                json.dump(self._records, _f, indent=2)
        manifest._flush = _local_flush'''

PATCHES.append((OLD_MANIFEST_INIT2, NEW_MANIFEST_INIT2, "dual-mode manifest init"))


# ── Apply all patches ─────────────────────────────────────────────────────────

def apply_patches(filepath: str):
    text = Path(filepath).read_text()
    applied = 0
    skipped = 0
    for old, new, desc in PATCHES:
        if old in text:
            text = text.replace(old, new, 1)
            print(f"  [OK]  {desc}")
            applied += 1
        else:
            print(f"  [SKIP] already applied or not found: {desc}")
            skipped += 1
    Path(filepath).write_text(text)
    print(f"\nApplied {applied} patches, skipped {skipped}.")
    print(f"Patched file: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="/home/ec2-user/ProductGPT/run_campaign28_sweep.py")
    args = parser.parse_args()
    print(f"Patching {args.sweep} ...")
    apply_patches(args.sweep)