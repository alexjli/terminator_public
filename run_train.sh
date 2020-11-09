#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --partition=sched_any
#SBATCH --constraint=centos7
#SBATCH --time=12:00:00
#SBATCH -o test-output_checkpoint.out
#SBATCH -e test-error_checkpoint.out

. /etc/profile.d/modules.sh
module load python/3.8.3
module load anaconda3/2019.10
source activate myenv
python train.py --dataset=features_singlechain --train=fold_splits/train_fold0.in --val=fold_splits/val_fold0.in --test=fold_splits/test_fold0.in --run_name=test_run_fold0
