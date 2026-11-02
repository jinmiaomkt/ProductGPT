#!/bin/bash -l
# ============================================================================
# R54 read-out, login node (numpy only), once every shard has exited 0:
# shard metadata and self-checks, the row gate per full arm, the pairing with the
# five GRU seeds (H1), each arm against Qwen3-32B (descriptive), and Centaur
# against its base on R50's sample (H4).
#
#     cd ~/ProductGPT/work_r54 && bash -l scripts/r54_readout.sh
# ============================================================================
set -e
cd ~/ProductGPT/work_r54
echo "work_r54 at $(git log --oneline -1)"
source /storage/home/jinmiao/ProductGPT/venvs/productgpt-llm/bin/activate
export OMP_NUM_THREADS=2 PRODUCTGPT_DATA=/storage/home/jinmiao/ProductGPT/data
R=/storage/home/jinmiao/ProductGPT/runs
FULL="muse_glimmer_30b_out gemma4_31b_out centaur_70b_out"
echo "== shard metadata"
python -c "
import json,glob
for tag in '$FULL llama31_70b_out'.split():
    tot=0; mx=0
    for f in sorted(glob.glob('$R/r54/'+tag+'_shard[0-9].json')):
        d=json.load(open(f)); t=d.get('tokens') or {}; s=d.get('selfcheck') or {}
        tot+=t.get('total',0); mx=max(mx,t.get('max',0))
        print(f.split('/')[-1], 'commit', d.get('commit'), 'customers', d.get('customers'),
              'tokens', t.get('total'), 'sec', d.get('seconds'), 'peak', d.get('gpu_peak_gb'), 'mem', d.get('memcheck'),
              '| t3_true', s.get('t3_true_mean'), 'repeat', s.get('repeat_max'), 't4c_before', s.get('t4c_before_max'),
              't4_after', s.get('t4_after_max'), 'adapter', s.get('adapter_effect_max'),
              'one-prefix', None if not s else (s.get('nll_onepass',0)-s.get('nll_prefix',0)))
    print(tag, 'TOTAL tokens', tot, 'max', mx)
"
for TAG in $FULL; do
  echo "== summarise $TAG"
  python scripts/r48_summarise.py --table $R/r46/occasions_v6_S1024.npz \
      --run-dir $R/r54 --tag $TAG --n-shards 4 --who out
  echo "== pair_gru $TAG"
  python scripts/r48_pair_gru.py --table $R/r46/occasions_v6_S1024.npz \
      --llm-dir $R/r54 --llm-tag $TAG --n-shards 4 \
      --gru-dir $R/r48/gru_b17 --runs-root $R \
      --gru-tags b17_gru_v6_s1 b17_gru_v6_s2 b17_gru_v6_s3 b17_gru_v6_s4 b17_gru_v6_s5
done
echo "== pair_llms, each full arm against Qwen3-32B (descriptive)"
python scripts/r49_pair_llms.py --ref qwen3_32b_out:$R/r49:4 \
    --arm muse_glimmer_30b_out:$R/r54:4 --arm gemma4_31b_out:$R/r54:4 --arm centaur_70b_out:$R/r54:4 \
    --out $R/r54/r54_vs_qwen3_32b.json
echo "== pair_llms, Centaur against its base on the sample (H4)"
python scripts/r49_pair_llms.py --ref llama31_70b_out:$R/r54:1 \
    --arm centaur_70b_out:$R/r54:4 --restrict-to-ref \
    --out $R/r54/r54_centaur_vs_base.json
echo "== done $(date)"
