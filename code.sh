# FullProductGPT_featurebased_performerfeatures16_dmodel128_ff128_N6_heads4_lr0.0001_w2_fold0.pt .
# FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0.pt

python3 predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt LP_ProductGPT_featurebased_performerfeatures32_dmodel128_ff128_N8_heads4_lr0.001_w2.pt \
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


  python3 run_cv_gru_eval.py \
    --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
    --data   /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json \
    --predict-eval-script /home/ec2-user/ProductGPT/predict_gru_and_eval.py \
    --s3-bucket productgptbucket \
    --s3-prefix GRU/CV/h128_lr0.001_bs4 \
    --hidden-size 128 --lr 0.001 --train-batch-size 4 --eval-batch-size 128 \
    --epochs 80 --class9-weight 5.0 --input-dim 15

python3 predict_lstm_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /home/ec2-user/ProductGPT/lstm_h128_lr0.0001_bs4.pt \
  --hidden-size 128 \
  --input-dim 15 \
  --batch-size 128 \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --s3 s3://productgptbucket/LSTM/eval/h128_lr0.0001_bs4/ \
  --pred-out /tmp/lstm_raw_preds.jsonl.gz

python3 run_cv_lstm_eval.py \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --data   /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --predict-eval-script /home/ec2-user/ProductGPT/predict_lstm_and_eval.py \
  --s3-bucket productgptbucket \
  --s3-prefix LSTM/CV/h128_lr0.0001_bs4 \
  --hidden-size 128 --lr 0.0001 --epochs 80 \
  --train-batch-size 4 --eval-batch-size 128 \
  --class9-weight 5.0 --input-dim 15 \
  --num-folds 10 --seed 33 \
  --upload-ckpt --pred-out

python3 run_cv_lstm_eval.py \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --data   /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --predict-eval-script /home/ec2-user/ProductGPT/predict_lstm_and_eval.py \
  --s3-bucket productgptbucket \
  --s3-prefix LSTM/CV/h128_lr0.0001_bs4 \
  --hidden-size 128 --lr 0.0001 --epochs 80 \
  --train-batch-size 4 --eval-batch-size 8 \
  --class9-weight 5.0 --input-dim 15 \
  --num-folds 10 --seed 33 \
  --upload-ckpt --pred-out

python3 /home/ec2-user/ProductGPT/predict_duplet_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /tmp/fold0.pt \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/DupletCV/exp_001/eval/fold0 \
  --pred-out /tmp/fold0_preds.jsonl.gz \
  --uids-val  s3://productgptbucket/DupletCV/exp_001/train/fold0/uids_val.txt \
  --uids-test s3://productgptbucket/DupletCV/exp_001/train/fold0/uids_test.txt \
  --fold-id 0 \
  --batch-size 32 \
  --thresh 0.5 \
  --lp-rate 5

python3 run_cv_duplet_eval.py \
  --num-folds 10 \
  --seed 33 \
  --predict-eval-script /home/ec2-user/ProductGPT/predict_duplet_and_eval.py \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --data   /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --lp-rate 5 \
  --eval-batch-size 32 \
  --thresh 0.5 \
  --s3-bucket productgptbucket \
  --s3-prefix DupletCV/exp_001

python3 /home/ec2-user/ProductGPT/predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /tmp/fold0.pt \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/FullProductGPT/CV/exp_001/eval/fold0 \
  --pred-out /tmp/fold0_preds.jsonl.gz \
  --uids-val  s3://productgptbucket/FullProductGPT/CV/exp_001/train/fold0/uids_val.txt \
  --uids-test s3://productgptbucket/FullProductGPT/CV/exp_001/train/fold0/uids_test.txt \
  --fold-id 0 \
  --batch-size 32 \
  --thresh 0.5 \
  --ai-rate 15

python3 run_cv_productgpt_eval.py \
  --num-folds 10 \
  --seed 33 \
  --predict-eval-script /home/ec2-user/ProductGPT/predict_productgpt_and_eval.py \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --data   /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --ai-rate 15 \
  --eval-batch-size 32 \
  --thresh 0.5 \
  --s3-bucket productgptbucket \
  --s3-prefix FullProductGPT/CV/exp_001


python3 /home/ec2-user/ProductGPT/predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /home/ec2-user/ProductGPT/checkpoints/FullProductGPT_featurebased_performerfeatures16_dmodel128_ff128_N6_heads4_lr0.0001_w2_fold0.pt \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/FullProductGPT/CV/exp_001/eval/fold0 \
  --pred-out /tmp/fold0_preds.jsonl.gz \
  --uids-val  s3://productgptbucket/FullProductGPT/CV/exp_001/train/fold0/uids_val.txt \
  --uids-test s3://productgptbucket/FullProductGPT/CV/exp_001/train/fold0/uids_test.txt \
  --fold-id 0 \
  --batch-size 32 \
  --thresh 0.5 \
  --ai-rate 15 \
  --nb-features 16 \
  --d-model 128 \
  --d-ff 128 \
  --N 6 \
  --num-heads 4
