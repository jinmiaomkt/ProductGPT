#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import ClientError


def should_skip(rel_path: str, exclude_patterns: list[str]) -> bool:
    # Normalize to POSIX style for matching
    rel_path = rel_path.replace("\\", "/")
    for pat in exclude_patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
    return False


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def upload_tree(
    *,
    s3_client,
    bucket: str,
    local_root: Path,
    s3_prefix: str,
    exclude_patterns: list[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Upload local_root recursively to s3://bucket/s3_prefix/
    preserving relative paths.
    Returns (num_files, total_bytes).
    """
    if not local_root.exists():
        print(f"[WARN] local path not found, skipping: {local_root}")
        return 0, 0

    local_root = local_root.resolve()
    s3_prefix = s3_prefix.strip("/")

    file_count = 0
    byte_count = 0

    print(f"\n[INFO] Uploading tree")
    print(f"  local: {local_root}")
    print(f"  s3   : s3://{bucket}/{s3_prefix}/")

    for fp in iter_files(local_root):
        rel = fp.relative_to(local_root).as_posix()
        if should_skip(rel, exclude_patterns):
            continue

        s3_key = f"{s3_prefix}/{rel}"
        size = fp.stat().st_size

        if dry_run:
            print(f"[DRYRUN] {fp}  ->  s3://{bucket}/{s3_key}  ({size} bytes)")
        else:
            try:
                s3_client.upload_file(str(fp), bucket, s3_key)
                print(f"[UP] {rel} ({size} bytes)")
            except ClientError as e:
                print(f"[ERROR] Failed upload: {fp}")
                print(f"        {e}")
                continue

        file_count += 1
        byte_count += size

    print(f"[DONE] Uploaded {file_count} files, {byte_count:,} bytes")
    return file_count, byte_count


def main():
    ap = argparse.ArgumentParser(description="Upload mixture-model results to S3 with unambiguous prefixes.")
    ap.add_argument("--bucket", default="productgptbucket", help="S3 bucket name")
    ap.add_argument(
        "--ray-dir",
        default="/home/ec2-user/ProductGPT/ray_results/ProductGPT_RayTune",
        help="Local Ray Tune experiment directory",
    )
    ap.add_argument(
        "--ray-s3-prefix",
        default="ray_results/mixture_head/ProductGPT_RayTune",
        help="Destination S3 prefix for Ray Tune results",
    )

    # Optional artifact dirs (Phase B outputs, etc.)
    ap.add_argument(
        "--metrics-dir",
        default="/home/ec2-user/output/metrics",
        help="Local metrics directory (optional)",
    )
    ap.add_argument(
        "--checkpoints-dir",
        default="/home/ec2-user/output/checkpoints",
        help="Local checkpoints directory (optional)",
    )
    ap.add_argument(
        "--predictions-dir",
        default="/home/ec2-user/output/predictions",
        help="Local predictions directory (optional)",
    )

    ap.add_argument(
        "--upload-artifacts",
        action="store_true",
        help="Also upload local metrics/checkpoints/predictions to mixture_head prefixes",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads only, do not upload",
    )

    # Default exclusions for Ray/log folders (adjust if needed)
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern (relative path) to exclude. Can be passed multiple times.",
    )

    args = ap.parse_args()

    # Reasonable defaults for exclusions (mainly for Ray results upload)
    # You can override by passing additional --exclude
    default_excludes = [
        "*.tmp",
        "*.lock",
        "**/*.tmp",
        "**/*.lock",
        # If trial-local checkpoints exist and you only want logs/metadata, uncomment these:
        # "**/*.pt",
        # "**/*.bin",
    ]
    exclude_patterns = default_excludes + args.exclude

    s3 = boto3.client("s3")

    total_files = 0
    total_bytes = 0

    # 1) Upload Ray Tune results/logs
    n, b = upload_tree(
        s3_client=s3,
        bucket=args.bucket,
        local_root=Path(args.ray_dir),
        s3_prefix=args.ray_s3_prefix,
        exclude_patterns=exclude_patterns,
        dry_run=args.dry_run,
    )
    total_files += n
    total_bytes += b

    # 2) Optional: upload Phase-B artifacts to disambiguated prefixes
    if args.upload_artifacts:
        artifact_jobs = [
            # local_dir, s3_prefix
            (Path(args.metrics_dir), "FullProductGPT/mixture_head/performer/FeatureBased/metrics"),
            (Path(args.checkpoints_dir), "FullProductGPT/mixture_head/performer/FeatureBased/checkpoints"),
            (Path(args.predictions_dir), "CV/mixture_head/predictions"),
        ]

        for local_dir, s3_prefix in artifact_jobs:
            n, b = upload_tree(
                s3_client=s3,
                bucket=args.bucket,
                local_root=local_dir,
                s3_prefix=s3_prefix,
                exclude_patterns=[],
                dry_run=args.dry_run,
            )
            total_files += n
            total_bytes += b

    print("\n========== SUMMARY ==========")
    print(f"Bucket       : {args.bucket}")
    print(f"Dry run      : {args.dry_run}")
    print(f"Total files  : {total_files}")
    print(f"Total bytes  : {total_bytes:,}")


if __name__ == "__main__":
    main()