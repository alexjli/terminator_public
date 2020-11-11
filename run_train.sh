#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --partition=sched_system_all
#SBATCH --time=12:00:00
#SBATCH -o logfiles/test-output_reg.out
#SBATCH -e logfiles/test-error_reg.out

HOME2=/nobackup/users/vsundar
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate pynew
ulimit -s unlimited
python train.py --dataset=features_singlechain --train=fold_splits/train_fold0.in --val=fold_splits/val_fold0.in --test=fold_splits/test_fold0.in --run_name=test_run_fold0_reg
