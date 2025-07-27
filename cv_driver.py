import ray, argparse, pandas as pd
from train_fold import run_single_fold

ray.init(address="auto")          # or ray.init() on a workstation

@ray.remote(num_gpus=1)
def _one(fold, spec):             # each task consumes one GPU
    return run_single_fold(fold, spec)

def main(spec_uri):
    tasks = [_one.remote(f, spec_uri) for f in range(10)]
    results = ray.get(tasks)
    pd.DataFrame(results).to_csv("cv_metrics.csv", index=False)
    print("Results → cv_metrics.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    main(parser.parse_args().spec)
