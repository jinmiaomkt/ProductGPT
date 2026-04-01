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

python3 predict_productgpt_and_eval.py   --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json   --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json   --ckpt /tmp/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt   --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx   --s3 s3://productgptbucket/evals/PhaseB_paperstyle_$(date +%F_%H%M%S)/   --pred-out /tmp/phaseB_paperstyle_preds.jsonl.gz   --fold-id 0   --calibration none   --split-data-frac 1.0   --split-seed 33   --split-subsample-seed 33


python3 predict_duplet_and_eval.py   --data /home/ec2-user/data/clean_list_int_wide4_simple6_FeatureBasedTrain.json   --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json   --ckpt /tmp/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt   --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx   --s3 s3://productgptbucket/evals/PhaseB_paperstyle_$(date +%F_%H%M%S)/   --pred-out /tmp/phaseB_paperstyle_preds.jsonl.gz   --fold-id 0   --calibration none   --split-data-frac 1.0   --split-seed 33   --split-subsample-seed 33


mkdir -p /tmp/recovered_ckpts
mkdir -p /home/ec2-user/output/evals/phaseB_latest_fold0

aws s3 cp \
"s3://productgptbucket/FullProductGPT/performer/FeatureBased/checkpoints/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt" \
"/tmp/recovered_ckpts/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt"

python3 /home/ec2-user/ProductGPT/predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /tmp/recovered_ckpts/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/evals/phaseB_latest_fold0_$(date +%F_%H%M%S)/ \
  --pred-out /home/ec2-user/output/evals/phaseB_latest_fold0/preds.jsonl.gz \
  --calibration none


  python3 predict_productgpt_and_eval.py  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json   --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json  --ckpt "/tmp/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0.pt"   --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx   --s3 "s3://productgptbucket/evals/PrevBest_restore_$(date +%F_%H%M%S)/"   --pred-out /tmp/prevbest_preds.jsonl.gz   --ai-rate 15   --batch-size 4 --uids-val s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt --uids-test s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt 

[2026-03-16 14:27:53,748] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1773671277.042768  613232 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1773671277.048977  613232 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
[INFO] S3 upload prefix: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_142750/
[INFO] Using EXACT training split via fold-spec: fold=0, val=476, test=477, seed=33, data_frac=1.0
[INFO] No calibration applied (raw model probabilities)
[INFO] Calibration method: none
[INFO] parsed: 5291 users accepted, 0 users missing labels.
[INFO] length mismatches: {'pred>lbl': 3966}

=============  BUCKET SIZES & PREVALENCE  =======================
      Task       Group Split      N    Pos    Neg   Prev
 BuyFigure Calibration  test 162298  87239  75059 0.5375
   BuyNone Calibration  test 162298   3855 158443 0.0238
    BuyOne Calibration  test 162298 133668  28630 0.8236
BuyRegular Calibration  test 162298  53951 108347 0.3324
    BuyTen Calibration  test 162298  24775 137523 0.1527
 BuyWeapon Calibration  test 162298  17253 145045 0.1063
 BuyFigure    HoldoutA  test  12744   7274   5470 0.5708
   BuyNone    HoldoutA  test  12744    287  12457 0.0225
    BuyOne    HoldoutA  test  12744  10152   2592 0.7966
BuyRegular    HoldoutA  test  12744   3131   9613 0.2457
    BuyTen    HoldoutA  test  12744   2305  10439 0.1809
 BuyWeapon    HoldoutA  test  12744   2052  10692 0.1610
 BuyFigure    HoldoutB  test  11800   7316   4484 0.6200
   BuyNone    HoldoutB  test  11800    455  11345 0.0386
    BuyOne    HoldoutB  test  11800   9184   2616 0.7783
BuyRegular    HoldoutB  test  11800   3025   8775 0.2564
    BuyTen    HoldoutB  test  11800   2161   9639 0.1831
 BuyWeapon    HoldoutB  test  11800   1004  10796 0.0851
 BuyFigure Calibration   val 162024  89185  72839 0.5504
   BuyNone Calibration   val 162024   3803 158221 0.0235
    BuyOne Calibration   val 162024 131739  30285 0.8131
BuyRegular Calibration   val 162024  52299 109725 0.3228
    BuyTen Calibration   val 162024  26482 135542 0.1634
 BuyWeapon Calibration   val 162024  16737 145287 0.1033
 BuyFigure    HoldoutA   val  13732   8466   5266 0.6165
   BuyNone    HoldoutA   val  13732    309  13423 0.0225
    BuyOne    HoldoutA   val  13732  10551   3181 0.7684
BuyRegular    HoldoutA   val  13732   3523  10209 0.2566
    BuyTen    HoldoutA   val  13732   2872  10860 0.2091
 BuyWeapon    HoldoutA   val  13732   1434  12298 0.1044
 BuyFigure    HoldoutB   val  12809   7860   4949 0.6136
   BuyNone    HoldoutB   val  12809    498  12311 0.0389
    BuyOne    HoldoutB   val  12809  10024   2785 0.7826
BuyRegular    HoldoutB   val  12809   3623   9186 0.2828
    BuyTen    HoldoutB   val  12809   2287  10522 0.1785
 BuyWeapon    HoldoutB   val  12809    828  11981 0.0646
============================================================

=============  MULTICLASS TOP-1 HIT / MACRO-F1 TABLE  =======================
                Hit         MacroF1        
                val    test     val    test
Group                                      
Calibration  0.4837  0.4838  0.2395  0.2328
HoldoutA     0.4482  0.4387  0.2060  0.2114
HoldoutB     0.4786  0.4908  0.1857  0.1879
============================================================

=============  SELECTED BINARY AUC TABLE - CALIBRATION  =======================
Split                    val    test
TaskPretty                          
Buy One               0.7534  0.7436
Buy Ten               0.7792  0.7692
Character Event Wish  0.6629  0.6632
Weapon Event Wish     0.7380  0.7236
Regular Wish          0.7081  0.7011
============================================================

=============  SELECTED BINARY AUC TABLE - HOLDOUT A  =======================
Split                    val    test
TaskPretty                          
Buy One               0.7757  0.7616
Buy Ten               0.8021  0.7887
Character Event Wish  0.6377  0.6490
Weapon Event Wish     0.7047  0.7248
Regular Wish          0.6746  0.6762
============================================================

=============  SELECTED BINARY AUC TABLE - HOLDOUT B  =======================
Split                    val    test
TaskPretty                          
Buy One               0.7287  0.7407
Buy Ten               0.7671  0.7714
Character Event Wish  0.6627  0.6795
Weapon Event Wish     0.7025  0.7268
Regular Wish          0.6842  0.7076
============================================================
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_142750/paper_multiclass_table.csv
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_142750/paper_auc_calibration.csv
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_142750/paper_auc_holdoutA.csv
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_142750/paper_auc_holdoutB.csv
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_142750/prevbest_preds.jsonl.gz
[ec2-user@ip-10-231-4-180 ProductGPT]$ cd ../data/
[ec2-user@ip-10-231-4-180 data]$ ls -all
total 4364956
drwxr-xr-x.  2 ec2-user ec2-user      16384 Jun  8  2025 .
drwx------. 17 ec2-user ec2-user      16384 Mar 16 07:26 ..
-rw-r--r--.  1 ec2-user ec2-user      15492 Apr 19  2025 SelectedFigureWeaponEmbeddingIndex.xlsx
-rw-r--r--.  1 ec2-user ec2-user  326245706 Apr 18  2025 clean_list_int_wide4_simple6.json
-rw-r--r--.  1 ec2-user ec2-user  325827712 Jun  8  2025 clean_list_int_wide4_simple6.jsonl
-rw-r--r--.  1 ec2-user ec2-user  281395046 Apr 18  2025 clean_list_int_wide4_simple6_FeatureBasedTrain.json
-rw-r--r--.  1 ec2-user ec2-user  280977052 May 26  2025 clean_list_int_wide4_simple6_FeatureBasedTrain.jsonl
-rw-r--r--.  1 ec2-user ec2-user  305394922 Apr 18  2025 clean_list_int_wide4_simple6_IndexBasedTrain.json
-rw-r--r--.  1 ec2-user ec2-user 1036787055 Apr 18  2025 output_list_int_wide4_simple6.json
-rw-r--r--.  1 ec2-user ec2-user  924904335 Apr 18  2025 output_list_int_wide4_simple6_FeatureBasedTrain.json
-rw-r--r--.  1 ec2-user ec2-user  988126175 Apr 18  2025 output_list_int_wide4_simple6_IndexBasedTrain.json


python3 predict_productgpt_and_eval.py   
--data /home/ec2-user/data/clean_list_int_wide4_simple6.json   
--labels /home/ec2-user/data/clean_list_int_wide4_simple6.json   
--ckpt "/tmp/FullProductGPT_featurebased_performerfeatures16_dmodel32_ff32_N6_heads4_lr0.0001_w2_fold0.pt"   
--feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx   
--s3 "s3://productgptbucket/evals/PrevBest_restore_$(date +%F_%H%M%S)/"   
--pred-out /tmp/prevbest_preds.jsonl.gz   
--ai-rate 15   
--batch-size 4
--uids-val s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt 
--uids-test s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt 


[2026-03-16 05:03:54,005] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to cuda (auto detect)
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1773637437.744552  588533 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1773637437.750894  588533 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
[INFO] S3 upload prefix: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_050350/
[INFO] Using EXACT training split via fold-spec: fold=0, val=476, test=477, seed=33, data_frac=1.0
[INFO] No calibration applied (raw model probabilities)
[INFO] Calibration method: none
[INFO] parsed: 5291 users accepted, 0 users missing labels.
[INFO] length mismatches: {'pred>lbl': 3966}

=============  BUCKET SIZES & PREVALENCE  =======================
      Task       Group Split      N    Pos    Neg   Prev
 BuyFigure Calibration  test 162298  87239  75059 0.5375
   BuyNone Calibration  test 162298   3855 158443 0.0238
    BuyOne Calibration  test 162298 133668  28630 0.8236
BuyRegular Calibration  test 162298  53951 108347 0.3324
    BuyTen Calibration  test 162298  24775 137523 0.1527
 BuyWeapon Calibration  test 162298  17253 145045 0.1063
 BuyFigure    HoldoutA  test  12744   7274   5470 0.5708
   BuyNone    HoldoutA  test  12744    287  12457 0.0225
    BuyOne    HoldoutA  test  12744  10152   2592 0.7966
BuyRegular    HoldoutA  test  12744   3131   9613 0.2457
    BuyTen    HoldoutA  test  12744   2305  10439 0.1809
 BuyWeapon    HoldoutA  test  12744   2052  10692 0.1610
 BuyFigure    HoldoutB  test  11800   7316   4484 0.6200
   BuyNone    HoldoutB  test  11800    455  11345 0.0386
    BuyOne    HoldoutB  test  11800   9184   2616 0.7783
BuyRegular    HoldoutB  test  11800   3025   8775 0.2564
    BuyTen    HoldoutB  test  11800   2161   9639 0.1831
 BuyWeapon    HoldoutB  test  11800   1004  10796 0.0851
 BuyFigure Calibration   val 162024  89185  72839 0.5504
   BuyNone Calibration   val 162024   3803 158221 0.0235
    BuyOne Calibration   val 162024 131739  30285 0.8131
BuyRegular Calibration   val 162024  52299 109725 0.3228
    BuyTen Calibration   val 162024  26482 135542 0.1634
 BuyWeapon Calibration   val 162024  16737 145287 0.1033
 BuyFigure    HoldoutA   val  13732   8466   5266 0.6165
   BuyNone    HoldoutA   val  13732    309  13423 0.0225
    BuyOne    HoldoutA   val  13732  10551   3181 0.7684
BuyRegular    HoldoutA   val  13732   3523  10209 0.2566
    BuyTen    HoldoutA   val  13732   2872  10860 0.2091
 BuyWeapon    HoldoutA   val  13732   1434  12298 0.1044
 BuyFigure    HoldoutB   val  12809   7860   4949 0.6136
   BuyNone    HoldoutB   val  12809    498  12311 0.0389
    BuyOne    HoldoutB   val  12809  10024   2785 0.7826
BuyRegular    HoldoutB   val  12809   3623   9186 0.2828
    BuyTen    HoldoutB   val  12809   2287  10522 0.1785
 BuyWeapon    HoldoutB   val  12809    828  11981 0.0646
============================================================

=============  MULTICLASS TOP-1 HIT / MACRO-F1 TABLE  =======================
                Hit         MacroF1        
                val    test     val    test
Group                                      
Calibration  0.4837  0.4838  0.2395  0.2328
HoldoutA     0.4482  0.4387  0.2060  0.2114
HoldoutB     0.4786  0.4908  0.1857  0.1879
============================================================

=============  SELECTED BINARY AUC TABLE - CALIBRATION  =======================
Split                    val    test
TaskPretty                          
Buy One               0.7534  0.7436
Buy Ten               0.7792  0.7692
Character Event Wish  0.6629  0.6632
Weapon Event Wish     0.7380  0.7236
Regular Wish          0.7081  0.7011
============================================================

=============  SELECTED BINARY AUC TABLE - HOLDOUT A  =======================
Split                    val    test
TaskPretty                          
Buy One               0.7757  0.7616
Buy Ten               0.8021  0.7887
Character Event Wish  0.6377  0.6490
Weapon Event Wish     0.7047  0.7248
Regular Wish          0.6746  0.6762
============================================================

=============  SELECTED BINARY AUC TABLE - HOLDOUT B  =======================
Split                    val    test
TaskPretty                          
Buy One               0.7287  0.7407
Buy Ten               0.7671  0.7714
Character Event Wish  0.6627  0.6795
Weapon Event Wish     0.7025  0.7268
Regular Wish          0.6842  0.7076
============================================================
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_050350/paper_multiclass_table.csv
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_050350/paper_auc_calibration.csv
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_050350/paper_auc_holdoutA.csv
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_050350/paper_auc_holdoutB.csv
[S3] uploaded: s3://productgptbucket/evals/PrevBest_restore_2026-03-16_050350/prevbest_preds.jsonl.gz
[ec2-user@ip-10-231-4-180 ProductGPT]$ Read from remote host ec2-44-204-135-11.compute-1.amazonaws.com: Operation timed out
Connection to ec2-44-204-135-11.compute-1.amazonaws.com closed.



Yes — from the logs you posted, the Phase B no-shuffle checkpoint for fold 0 is:

s3://productgptbucket/FullProductGPT/performer/FeatureBased/checkpoints/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt


mkdir -p /tmp/recovered_ckpts
mkdir -p /home/ec2-user/output/evals/phaseB_latest_fold0

aws s3 cp \
"s3://productgptbucket/FullProductGPT/performer/FeatureBased/checkpoints/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt" \
"/tmp/recovered_ckpts/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt"

python3 /home/ec2-user/ProductGPT/predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /tmp/recovered_ckpts/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/evals/phaseB_latest_fold0_$(date +%F_%H%M%S)/ \
  --pred-out /home/ec2-user/output/evals/phaseB_latest_fold0/preds.jsonl.gz \
  --calibration none



mkdir -p /tmp/recovered_ckpts
mkdir -p /home/ec2-user/output/evals/phaseB_latest_fold0

aws s3 cp \
"s3://productgptbucket/FullProductGPT/performer/FeatureBased/checkpoints/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt" \
"/tmp/recovered_ckpts/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt"

python3 /home/ec2-user/ProductGPT/predict_productgpt_and_eval.py \
  --data /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --labels /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt /tmp/recovered_ckpts/FullProductGPT_featurebased_performerfeatures32_dmodel96_ff192_N3_heads4_lr0.0001992193018167218_w1_fold0.pt \
  --feat-xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3 s3://productgptbucket/evals/phaseB_latest_fold0_$(date +%F_%H%M%S)/ \
  --pred-out /home/ec2-user/output/evals/phaseB_latest_fold0/preds.jsonl.gz \
  --calibration none

# ── S3 paths ────────────────────────────────────────────────────────────────
S3_BASE="s3://productgptbucket/FullProductGPT/performer/FeatureBased/checkpoints"
MODEL_STEM="featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0"

S3_CKPT="${S3_BASE}/FullProductGPT_${MODEL_STEM}.pt"
S3_CAL="${S3_BASE}/calibrator_${MODEL_STEM}.pt"

# ── Local paths ──────────────────────────────────────────────────────────────
LOCAL_CKPT="/tmp/FullProductGPT_${MODEL_STEM}.pt"
LOCAL_CAL="/tmp/calibrator_${MODEL_STEM}.pt"

# ── Step 1: Download checkpoints ─────────────────────────────────────────────
echo "[INFO] Downloading model checkpoint..."
aws s3 cp "${S3_CKPT}" "${LOCAL_CKPT}"

echo "[INFO] Downloading calibrator checkpoint..."
aws s3 cp "${S3_CAL}" "${LOCAL_CAL}"

echo "[INFO] Downloads complete."
ls -lh /tmp/FullProductGPT_*.pt /tmp/calibrator_*.pt

python3 predict_productgpt_and_eval.py \
  --data        /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --labels      /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt        "${LOCAL_CKPT}" \
  --feat-xlsx   /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3          "s3://productgptbucket/evals/phaseB_fold0_${TIMESTAMP}/" \
  --pred-out    /tmp/preds_phaseB_fold0.jsonl.gz \
  --uids-val    s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt \
  --uids-test   s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt \
  --fold-id     0 \
  --calibration calibrator \
  --ai-rate     15 \
  --batch-size  2 \
  --split-data-frac 1.0


python3 predict_productgpt_and_eval_9class.py \
  --data        /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --labels      /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt        /tmp/FullProductGPT_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0.pt \
  --feat-xlsx   /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3          "s3://productgptbucket/evals/phaseB_fold0_$(date +%F_%H%M%S)/" \
  --pred-out    /tmp/preds_phaseB_fold0.jsonl.gz \
  --uids-val    s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt \
  --uids-test   s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt \
  --fold-id     0 \
  --calibration calibrator \
  --ai-rate     15 \
  --batch-size  2 \
  --split-data-frac 1.0

  
python3 infer_new_campaign_calibrated.py --step 2 --data /home/ec2-user/data/clean_list_int_wide4_simple6.json --ckpt /tmp/FullProductGPT_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0.pt --calibrator_ckpt /tmp/calibrator_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0.pt  --calibrator_type auto --feat_xlsx /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx --out /home/ec2-user/outputs/campaign28_calibrated.jsonl --lto28 30 0 54 51  --temperature 1.0 --seed_base 42 --repeat 5 --first


python3 predict_productgpt_and_eval_both.py \
  --data        /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --labels      /home/ec2-user/data/clean_list_int_wide4_simple6.json \
  --ckpt        /tmp/FullProductGPT_featurebased_performerfeatures64_dmodel64_ff192_N3_heads2_lr0.000510707329019641_w1_fold0.pt \
  --feat-xlsx   /home/ec2-user/data/SelectedFigureWeaponEmbeddingIndex.xlsx \
  --s3          "s3://productgptbucket/evals/phaseB_fold0_$(date +%F_%H%M%S)/" \
  --pred-out    /tmp/preds_phaseB_fold0.jsonl.gz \
  --uids-val    s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_val.txt \
  --uids-test   s3://productgptbucket/ProductGPT/CV/exp_001/train/fold0/uids_test.txt \
  --fold-id     0 \
  --calibration calibrator \
  --ai-rate     15 \
  --batch-size  2 \
  --split-data-frac 1.0