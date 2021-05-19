#!/bin/bash

mkdir logfiles

sed -e "s/RUNNAME/$1/g" -e 's/RUNNO/0/g' -e "s/HPARAMS/$2/g" <run_train_ingraham.sh >bash_files/run_$1_ingraham_run0.sh
jid0=$(sbatch --parsable bash_files/run_$1_ingraham_run0.sh)
echo $jid0

