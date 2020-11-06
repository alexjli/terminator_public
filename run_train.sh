#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --partition=sched_any
#SBATCH --constraint=centos7
#SBATCH --time=12:00:00
#SBATCH -o test-output.out
#SBATCH -e test-error_2.out

. /etc/profile.d/modules.sh
module load python/3.8.3
module load anaconda3/2019.10
source activate myenv
python train.py --dataset=features_singlechain
