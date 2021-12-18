#!/bin/bash
#SBATCH -p xeon-p8
#SBATCH -o /dev/null

# activate conda
CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2019b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator

python summarize_results.py --output_dir=$1 --dtermen_data=$2
