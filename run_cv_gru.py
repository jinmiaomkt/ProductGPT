# run_cv_gru.py
from __future__ import annotations
import boto3, pandas as pd, io, json
from train_fold_gru import run_single_fold, BUCKET, PREFIX

S3_MERGED = f"{PREFIX}/tables/cv_gru_metrics.csv"
S3_SUM_CSV= f"{PREFIX}/tables/cv_gru_summary.csv"
S3_SUM_JSON=f"{PREFIX}/tables/cv_gru_summary.json"

_s3 = boto3.client("s3")

def _download_csv(bucket: str, key: str):
    try:
        obj = _s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    except _s3.exceptions.NoSuchKey:
        return None
    except Exception:
        return None

def _upload_csv(df: pd.DataFrame, bucket: str, key: str):
    buf = io.BytesIO(); df.to_csv(buf, index=False); buf.seek(0)
    _s3.upload_fileobj(buf, bucket, key)

def _upsert_row(row: dict):
    df = _download_csv(BUCKET, S3_MERGED)
    add = pd.DataFrame([row])
    if df is None:
        df = add
    else:
        if "fold" in df.columns:
            df = df[df["fold"] != row.get("fold")]
            df = pd.concat([df, add], ignore_index=True)
        else:
            df = pd.concat([df, add], ignore_index=True)
    if "fold" in df.columns:
        df = df.sort_values("fold")
    _upload_csv(df, BUCKET, S3_MERGED)
    return df

def _write_summary(df: pd.DataFrame):
    # Pick numeric cols commonly present; adjust if your trainer uses other keys.
    num = df.select_dtypes("number")
    summary = num.agg(["mean", "std"]).round(4)
    # also include standard error
    se = (num.std(ddof=1) / (len(df) ** 0.5)).rename(lambda c: f"{c}_se")
    summary_se = pd.DataFrame(se).T
    summary_all = pd.concat([summary, summary_se], axis=0)

    _upload_csv(summary_all, BUCKET, S3_SUM_CSV)
    _s3.put_object(Bucket=BUCKET, Key=S3_SUM_JSON,
                   Body=json.dumps(summary_all.to_dict(), indent=2).encode())

def main():
    for k in range(10):             # folds 0..9
        print(f"\n[GRU CV] fold {k} starting …")
        row = run_single_fold(k)
        print(f"[GRU CV] fold {k} metrics: { {k: row.get(k) for k in ['fold','val_f1','val_auprc','val_auc','val_loss'] if k in row} }")
        df = _upsert_row(row)       # incremental CSV update
        _write_summary(df)          # incremental summary update
        print(f"[GRU CV] fold {k} done & merged to s3://{BUCKET}/{S3_MERGED}")

if __name__ == "__main__":
    main()
