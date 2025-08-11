# run_cv_gru.py  — replace the helper block with this
from __future__ import annotations
import boto3, pandas as pd, io, json, numpy as np
from botocore.exceptions import ClientError
from train_fold_gru import run_single_fold, BUCKET, PREFIX

S3_MERGED    = f"{PREFIX}/tables/cv_gru_metrics.csv"
S3_SUM_CSV   = f"{PREFIX}/tables/cv_gru_summary.csv"
S3_SUM_JSON  = f"{PREFIX}/tables/cv_gru_summary.json"

_s3 = boto3.client("s3")

def _download_csv(bucket: str, key: str) -> pd.DataFrame | None:
    try:
        obj = _s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None
        raise
    except Exception:
        return None

def _upload_csv(df: pd.DataFrame, bucket: str, key: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    _s3.upload_fileobj(buf, bucket, key)

def _upsert_row(row: dict) -> pd.DataFrame:
    """Insert/replace this fold’s row, sort, and upload merged CSV."""
    df_existing = _download_csv(BUCKET, S3_MERGED)
    add = pd.DataFrame([row])

    if df_existing is None:
        df = add
    else:
        if "fold" in df_existing.columns:
            df_existing = df_existing[df_existing["fold"] != row.get("fold")]
        df = pd.concat([df_existing, add], ignore_index=True)

    if "fold" in df.columns:
        df = df.sort_values("fold").reset_index(drop=True)

    _upload_csv(df, BUCKET, S3_MERGED)
    return df

def _write_summary(df: pd.DataFrame):
    """Write mean/std/SE for numeric columns; robust to partial folds."""
    numeric = df.select_dtypes(include=[np.number]).copy()
    # drop columns with all NaN to keep summary clean
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.empty:
        summary_all = pd.DataFrame()  # nothing to write yet
    else:
        n = len(numeric)
        mean = numeric.mean().to_frame().T
        mean.index = ["mean"]
        std  = numeric.std(ddof=1).to_frame().T if n > 1 else pd.DataFrame([np.zeros(len(numeric.columns))], columns=numeric.columns, index=["std"])
        std.index = ["std"]
        se = (numeric.std(ddof=1) / np.sqrt(max(n, 1))).to_frame().T if n > 1 else pd.DataFrame([np.zeros(len(numeric.columns))], columns=numeric.columns, index=["se"])
        se.index = ["se"]

        summary_all = pd.concat([mean.round(4), std.round(4), se.round(4)], axis=0)

    _upload_csv(summary_all, BUCKET, S3_SUM_CSV)
    _s3.put_object(
        Bucket=BUCKET, Key=S3_SUM_JSON,
        Body=json.dumps(summary_all.to_dict(orient="index"), indent=2).encode("utf-8")
    )

def main():
    for k in range(10):  # folds 0..9
        print(f"\n[GRU CV] fold {k} starting …")
        row = run_single_fold(k)
        # compact preview of key metrics if present
        preview_keys = [x for x in ["fold","val_f1","val_auprc","val_auc","val_loss"] if x in row]
        print(f"[GRU CV] fold {k} metrics: " + ", ".join(f"{kk}={row[kk]}" for kk in preview_keys))
        df = _upsert_row(row)   # incremental CSV
        _write_summary(df)      # incremental summary
        print(f"[GRU CV] fold {k} merged → s3://{BUCKET}/{S3_MERGED}")

if __name__ == "__main__":
    main()
