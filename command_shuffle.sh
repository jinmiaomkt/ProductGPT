watch -n 20 "ls -lh /home/ec2-user/ProductGPT/<your_model_folder>/metrics/*curve*; tail -n 5 /home/ec2-user/ProductGPT/<your_model_folder>/metrics/*_curve.csv"

watch -n 20 "ls -lh /*curve*; tail -n 5 /*_curve.csv"

python3 ShufTrain_postHT.py --rank 0 --epochs 200 --augment-train true --do-infer true --plot-every 1
python3 ShufTrain_mixture_postHT.py --rank 0 --epochs 200 --augment-train true --do-infer true --plot-every 1

