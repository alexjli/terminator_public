#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --partition=sched_system_all
#SBATCH --time=12:00:00
#SBATCH -o logfiles/test-output_RUNNAME_foldFOLD_runRUNNO.out
#SBATCH -e logfiles/test-error_RUNNAME_foldFOLD_runRUNNO.out

HOME2=/nobackup/users/alexjli
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate pytorch1.5
ulimit -s unlimited

python train_multi.py --dataset=features_singlechain --train=fold_splits/train_foldFOLD.in --val=fold_splits/val_foldFOLD.in --test=fold_splits/test_foldFOLD.in --run_name=test_run_foldFOLD_RUNNAME --hparams=hparams/HPARAMS.json
