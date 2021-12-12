#!/bin/bash
#SBATCH -N 1
#SBATCH --mincpu=40
#SBATCH --gres=gpu:volta:2
#SBATCH --time=144:00:00
#SBATCH --exclusive
#SBATCH -o OUTPUTDIR/train-output_runRUNNO.out
#SBATCH -e OUTPUTDIR/train-error_runRUNNO.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2019b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator
ulimit -s unlimited

python train.py --dataset=DATASET --run_dir=OUTPUTDIR --hparams=HPARAMS
