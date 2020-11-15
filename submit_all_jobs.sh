#!/bin/bash

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/0/g' -e 's/RUNNO/0/g' <run_train.sh >bash_files/run_$1_fold0_run0.sh
jid1=$(sbatch --parsable bash_files/run_$1_fold0_run0.sh)
echo $jid1

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/0/g' -e 's/RUNNO/1/g' <run_train.sh >bash_files/run_$1_fold0_run1.sh
jid2=$(sbatch --parsable --dependency=afterany:$jid1 bash_files/run_$1_fold0_run1.sh)
echo $jid2
