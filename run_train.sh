#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=256GB
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH -o logfiles/test-output_RUNNAME_foldFOLD_runRUNNO.out
#SBATCH -e logfiles/test-error_RUNNAME_foldFOLD_runRUNNO.out

conda activate pytorch1.7
ulimit -s unlimited

python train.py --dataset=features_singlechain --train=fold_splits/train_foldFOLD.in --val=fold_splits/val_foldFOLD.in --test=fold_splits/test_foldFOLD.in --run_name=test_run_foldFOLD_RUNNAME --hparams=hparams/HPARAMS.json
