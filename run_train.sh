#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --partition=sched_system_all
#SBATCH --time=12:00:00
#SBATCH -o logfiles/test-output_RUNNAME_runRUNNO.out
#SBATCH -e logfiles/test-error_RUNNAME_runRUNNO.out

HOME2=/nobackup/users/vsundar
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate pynew
ulimit -s unlimited

export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python train.py --dataset=features_singlechain --train=fold_splits/train_foldFOLD.in --val=fold_splits/val_foldFOLD.in --test=fold_splits/test_foldFOLD.in --run_name=test_run_foldFOLD_RUNNAME

echo "Run completed at:- "
date
