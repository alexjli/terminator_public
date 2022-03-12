#!/bin/bash
#SBATCH -N 1
#SBATCH --mincpu=32
#SBATCH --gres=gpu:volta:1
#SBATCH --time=HOURS:00:00
#SBATCH --mem=50G
#SBATCH -o RUNDIR/train-output_runRUNNO.out
#SBATCH -e RUNDIR/train-error_runRUNNO.out

CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2019b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate terminator
ulimit -s unlimited
ulimit -n 10000

python train.py \
  --dataset=DATASET \
  --hparams=HPARAMS \
  --run_dir=RUNDIR \
  --out_dir=OUTPUTDIR \
  --lazy
