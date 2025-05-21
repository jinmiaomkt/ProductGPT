"""
Parallel sweep for Decision-Only model
– Each worker spawns its own TRAIN script (see above)
– Only a one-liner appears per finished combo
"""
import multiprocessing as mp, itertools, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch, boto3, botocore
from config4_decision_only_git import get_config
from train4_decision_only_aws import train_model, json_safe   # ← ADD import

mp.set_start_method("spawn", force=True)
GPU = torch.cuda.device_count()
WORKERS = GPU or max(1, mp.cpu_count()-1)

grid = itertools.product(
    [32,64,128,256],            # seq_len_ai
    [256,512],                  # d_model
    [128,256,512],              # d_ff
    [6,8,10],                   # N
    [8,16,32],                  # heads
    [1e-4],                     # lr
    [1,2,4]                     # weight
)

def s3():                                # singleton
    if not hasattr(s3,"c"):
        try: s3.c=boto3.client("s3")
        except botocore.exceptions.BotoCoreError: s3.c=None
    return s3.c

def run(hp):
    L,dm,df,N,H,lr,w = hp
    cfg=get_config(); cfg.update(
        seq_len_ai=L, seq_len_tgt=L//cfg["ai_rate"],
        d_model=dm, d_ff=df, N=N, num_heads=H,
        lr=lr, weight=w)
    uid=f"ctx{L}_dmodel{dm}_ff{df}_N{N}_heads{H}_lr{lr}_weight{w}"
    cfg["model_basename"]=f"DecisionOnly_{uid}"
    if GPU: os.environ["CUDA_VISIBLE_DEVICES"]=str(hash(uid)%GPU)
    stats=train_model(cfg)

    # small local JSON for quick check
    p=Path(f"{uid}.json"); p.write_text(json.dumps(json_safe(stats),indent=2))
    bkt=cfg["s3_bucket"]; key=f"DecisionOnly/metrics/{p.name}"
    if (c:=s3()): 
        try: c.upload_file(str(p),bkt,key); print(f"[S3] {p.name} → s3://{bkt}/{key}")
        except botocore.exceptions.BotoCoreError: pass
    p.unlink(missing_ok=True)
    return uid

def main():
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        fut={ex.submit(run,h):h for h in grid}
        for f in as_completed(fut):
            h=fut[f]
            try: print("[Done]",f.result())
            except Exception as e: print(f"[Error] params={h} -> {e}")

if __name__=="__main__":
    main()
