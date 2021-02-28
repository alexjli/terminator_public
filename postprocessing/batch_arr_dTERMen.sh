#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=1000
#SBATCH --partition=defq
#SBATCH --time=0:15:00
#SBATCH -o /home/alexjli/TERMinator/postprocessing/logfiles/batch_arr-output.out
#SBATCH -e /home/alexjli/TERMinator/postprocessing/logfiles/batch_arr-error.err

. /etc/profile.d/modules.sh
ids_file=$2
readarray -t RUNLIST < $ids_file
ID=${RUNLIST[$SLURM_ARRAY_TASK_ID]}
echo $ID
runfile="run_${ID}.sh"
output_dir=$1
cd $1
bash $runfile > "${output_dir}/${ID}-output.out" 2> "${output_dir}/${ID}-error.err"

