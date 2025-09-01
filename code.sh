# FullProductGPT_featurebased_performerfeatures16_dmodel128_ff128_N6_heads4_lr0.0001_w2_fold0.pt .
# FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0.pt

python3 predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt FullProductGPT_featurebased_performerfeatures16_dmodel128_ff128_N6_heads4_lr0.0001_w2_fold0.pt \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/eval/FullProductGPT/ \
  --pred-out /tmp/fullproductgpt_preds.jsonl.gz \

python3 predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0.pt \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/eval/FullProductGPT/ \
  --pred-out /tmp/fullproductgpt_preds.jsonl.gz \


MASTER_ADDR=127.0.0.1 MASTER_PORT=29617 \
python3 predict_duplet_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt LP_ProductGPT_featurebased_performerfeatures32_dmodel128_ff128_N8_heads4_lr0.001_w2.pt \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/eval/LPProductGPT/ \
  --pred-out /tmp/LP_feature_preds.jsonl.gz


python3 run_cv_productgpt_eval.py \
  --num-folds 10 \
  --seed 33 \
  --predict-eval-script /home/ec2-user/bin/predict_productgpt_and_eval.py \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --data   /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3-bucket productgptbucket \
  --s3-prefix ProductGPT/CV/exp_001 \
  --eval-batch-size 32 \
  --thresh 0.5


python3 run_cv_duplet_eval.py \
  --num-folds 10 \
  --seed 33 \
  --predict-eval-script /home/ec2-user/project/predict_duplet_and_eval.py \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --data   /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --lp-rate 5 \
  --eval-batch-size 32 \
  --thresh 0.5 \
  --s3-bucket productgptbucket \
  --s3-prefix DupletCV/exp_001

python3 predict_gru_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /home/ec2-user/ProductGPT/gru_h128_lr0.001_bs4.pt \
  --hidden-size 128 \
  --input-dim 15 \
  --batch-size 128 \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --s3 s3://productgptbucket/GRU/eval/h128_lr0.001_bs4/
