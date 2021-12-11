#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=1000
#SBATCH --partition=defq
#SBATCH --time=0:15:00
#SBATCH -o /dev/null

. /etc/profile.d/modules.sh
ids_file=$2
batch_size=$3
readarray -t RUNLIST < $ids_file
RUNLIST_LEN=${#RUNLIST[@]}

let start=$batch_size*$SLURM_ARRAY_TASK_ID
# compute the end of the batch
let next=$batch_size*($SLURM_ARRAY_TASK_ID+1)
if [[ $next -gt $RUNLIST_LEN ]]
then
  let end=$RUNLIST_LEN-1
else
  let end=$next-1
fi

output_dir=$1
cd $1

# run the batch
for i in {$start..$end}
do
  ID=${RUNLIST[$i]}
  echo $ID
  runfile="run_${ID}.sh"
  bash $runfile > "${ID}-output.out" 2> "${ID}-error.err"
end
