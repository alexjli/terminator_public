#!/bin/bash
#SBATCH -N 1
#SBATCH --mincpu=40
#SBATCH --gres=gpu:volta:2
#SBATCH --time=144:00:00
#SBATCH --exclusive
#SBATCH -o logfiles/eval-output_DATANAME_RUNNAME.out
#SBATCH -e logfiles/eval-error_DATANAME_RUNNAME.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2019b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator
ulimit -s unlimited

python eval.py --dataset=DATASET --model_dir=MODELDIR --output_dir=OUTPUTDIR
