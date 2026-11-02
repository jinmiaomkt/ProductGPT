#!/bin/bash -l
# ============================================================================
# R53 read-out, login node (numpy only), once all four shards have exited 0:
# shard metadata and self-checks, the row gate, the pairing with the five GRU
# seeds (H1) and the pairing with Qwen3.6-27B and Qwen3-32B (H2, descriptive).
#
#     cd ~/ProductGPT/work_r48 && bash -l scripts/r53_readout.sh
# ============================================================================
set -e
cd ~/ProductGPT/work_r48
echo "work_r48 at $(git log --oneline -1)"
source /storage/home/jinmiao/ProductGPT/venvs/productgpt-llm/bin/activate
export OMP_NUM_THREADS=2 PRODUCTGPT_DATA=/storage/home/jinmiao/ProductGPT/data
R=/storage/home/jinmiao/ProductGPT/runs
TAG=qwen38_27b_out
echo "== shard metadata"
python -c "
import json,glob
tot=0; mx=0
for f in sorted(glob.glob('$R/r53/$TAG'+'_shard[0-9].json')):
    d=json.load(open(f)); t=d.get('tokens') or {}; s=d.get('selfcheck') or {}
    tot+=t.get('total',0); mx=max(mx,t.get('max',0))
    print(f.split('/')[-1], 'commit', d.get('commit'), 'customers', d.get('customers', d.get('n_customers')),
          'tokens', t.get('total'), 'sec', d.get('seconds'),
          '| t3_true', s.get('t3_true_mean'), 'repeat', s.get('repeat_max'), 't4c_before', s.get('t4c_before_max'),
          't4_after', s.get('t4_after_max'), 'one-prefix', None if not s else (s.get('nll_onepass',0)-s.get('nll_prefix',0)))
print('TOTAL tokens', tot, 'max', mx, '(Qwen3.6-27B: 46,451,481, max 37,092)')
"
echo "== summarise $TAG"
python scripts/r48_summarise.py --table $R/r46/occasions_v6_S1024.npz \
    --run-dir $R/r53 --tag $TAG --n-shards 4 --who out
echo "== pair_gru $TAG"
python scripts/r48_pair_gru.py --table $R/r46/occasions_v6_S1024.npz \
    --llm-dir $R/r53 --llm-tag $TAG --n-shards 4 \
    --gru-dir $R/r48/gru_b17 --runs-root $R \
    --gru-tags b17_gru_v6_s1 b17_gru_v6_s2 b17_gru_v6_s3 b17_gru_v6_s4 b17_gru_v6_s5
echo "== pair_llms, against Qwen3.6-27B (H2)"
python scripts/r49_pair_llms.py --ref qwen36_27b_out:$R/r49:4 \
    --arm $TAG:$R/r53:4 --out $R/r53/r53_vs_qwen36.json
echo "== pair_llms, against Qwen3-32B (descriptive)"
python scripts/r49_pair_llms.py --ref qwen3_32b_out:$R/r49:4 \
    --arm $TAG:$R/r53:4 --out $R/r53/r53_vs_qwen3_32b.json
echo "== done $(date)"
