#!/bin/bash
#SBATCH -N 1
#SBATCH --mincpu=16
#SBATCH --time=2:00:00
#SBATCH --mem=50G
#SBATCH -o OUTPUTDIR/eval-output.out
#SBATCH -e OUTPUTDIR/eval-error.out
#SBATCH -p normal

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2019b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator-nightly
ulimit -s unlimited

python eval.py \
    --dataset=DATASET \
    --model_dir=MODELDIR \
    --output_dir=OUTPUTDIR \
    --dev=cpu \
    --subset=SUBSET
