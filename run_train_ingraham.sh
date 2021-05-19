#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:volta:2
#SBATCH --time=96:00:00
#SBATCH --exclusive
#SBATCH -o logfiles/test-output_RUNNAME_ingraham_runRUNNO.out
#SBATCH -e logfiles/test-error_RUNNAME_ingraham_runRUNNO.out

HOME2=/nobackup/users/alexjli
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate pytorch1.7
ulimit -s unlimited

python train_multi.py --dataset=features_ingraham --run_name=test_run_ingraham_RUNNAME --hparams=hparams/HPARAMS.json
