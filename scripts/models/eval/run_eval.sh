#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:volta:1
#SBATCH --time=0:10:00
#SBATCH --mem=30G
#SBATCH -o OUTPUTDIR/eval-output.out
#SBATCH -e OUTPUTDIR/eval-error.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2019b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator-nightly
ulimit -s unlimited

python eval.py \
    --dataset=DATASET \
    --model_dir=MODELDIR \
    --output_dir=OUTPUTDIR \
    --subset=SUBSET
