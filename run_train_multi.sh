#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:volta:2
#SBATCH --time=96:00:00
#SBATCH --exclusive
#SBATCH -o logfiles/test-output_RUNNAME_foldFOLD_runRUNNO.out
#SBATCH -e logfiles/test-error_RUNNAME_foldFOLD_runRUNNO.out

HOME2=/nobackup/users/alexjli
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate pytorch1.7
ulimit -s unlimited

python train_multi.py --dataset=features_multichain --train=fold_splits/train_foldFOLD.in --val=fold_splits/val_foldFOLD.in --test=fold_splits/test_foldFOLD.in --run_name=test_run_foldFOLD_RUNNAME --hparams=hparams/HPARAMS.json
