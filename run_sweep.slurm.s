#!/bin/bash
#SBATCH --job-name=hyperparam_sweep
#SBATCH --output=hyperparam_sweep_%j.log
#SBATCH --error=hyperparam_sweep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1        # If you need 1 GPU
#SBATCH --time=24:00:00     # Max run time (HH:MM:SS)
#SBATCH --partition=YOUR_GPU_PARTITION  # depends on cluster (e.g., 'normal', 'rtx', 'gpu', etc.)

# 1) Load any required modules (depends on your cluster)
module load anaconda3

# 2) Activate your conda environment or venv
source activate my_pytorch_env

# Or if you have a .venv in the same folder:
# source .venv/bin/activate

# 3) Move to your project folder if needed
cd /path/to/your/code

# 4) Run your python script
python hyperparam_sweep.py

echo "Done with job $SLURM_JOBID"
