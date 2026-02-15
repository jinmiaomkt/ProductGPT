python infer_new_campaign.py \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_step1.jsonl \
  --lto28 28 37 55 17 \
  --fixed_outcomes 0 0 0 0 0 0 0 0 0 0 \
  --max_steps28 200

python infer_new_campaign.py \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_step1_test1.jsonl \
  --lto28 28 37 55 17 \
  --fixed_outcomes 0 0 0 0 0 0 0 0 0 0 \
  --max_steps28 200 \
  --first \
  --seed 123

python infer_new_campaign.py \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_step1_test1.jsonl \
  --lto28 28 37 55 17 \
  --fixed_outcomes 0 0 0 0 0 0 0 0 0 0 \
  --max_steps28 1024 \
  --n_users 3 --seed 333

python infer_new_campaign.py \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_step1_30runs.jsonl \
  --lto28 28 37 55 17 \
  --fixed_outcomes 0 0 0 0 0 0 0 0 0 0 \
  --max_steps28 200 \
  --uid 000b0fc7320f6070 \
  --repeat 30 \
  --seed_base 1000

python infer_new_campaign_advanced.py \
  --step 2 \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_adv.jsonl \
  --lto28 28 37 55 17 \
  --max_steps28 200 \
  --first \
  --repeat 30 --seed_base 1000

python infer_new_campaign_advanced.py \
  --step 2 \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_adv.jsonl \
  --lto28 28 37 55 17 \
  --max_steps28 200 \
  --first \
  --repeat 1 --seed_base 1000 \
  --trace --trace_max_steps 50

  python infer_new_campaign_advanced.py \
  --step 2 \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_adv.jsonl \
  --lto28 28 37 55 17 \
  --max_steps28 200 \
  --first \
  --repeat 1 --seed_base 1000 \
  --trace --trace_max_steps 50

  python infer_new_campaign_advanced.py \
  --step 2 \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_adv.jsonl \
  --lto28 28 37 55 17 \
  --max_steps28 200 \
  --first \
  --repeat 1 --seed_base 1000 \
  --validate --validate_users 20 --validate_last_k 10
  

  python infer_new_campaign_advanced.py \
  --step 2 \
  --trace --trace_max_steps 50 \
  --trace_triplets \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_adv.jsonl \
  --lto28 30 0 54 51 \
  --seed_base 1000 \
  --quiet

  python infer_new_campaign_advanced.py \
  --step 2 \
  --trace --trace_max_steps 50 \
  --trace_triplets \
  --trace_out /home/ec2-user/output/campaign28_adv.trace.jsonl \
  --validate_trace_states \
  --ckpt /home/ec2-user/output/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N8_heads4_lr0.0001_w4.pt \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
  --out  /home/ec2-user/output/campaign28_adv.jsonl \
  --lto28 30 0 54 51 \
  --seed_base 1000 \
  --quiet
