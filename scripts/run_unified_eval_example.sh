#!/usr/bin/env bash
set -euo pipefail

python3 unified_model_eval.py \
  --config /mnt/data/model_specs_example.json \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --uids-val s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt \
  --uids-test s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt \
  --fold-id 0 \
  --compare-on test \
  --output-dir /home/ec2-user/eval_unified_fold0 \
  --s3 s3://productgptbucket/evals/unified_compare_$(date +%F_%H%M%S)/ \
  --save-preds
