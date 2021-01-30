#!/bin/bash

mkdir logfiles

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/0/g' -e 's/RUNNO/0/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold0_run0.sh
jid0=$(sbatch --parsable bash_files/run_$1_fold0_run0.sh)
echo $jid0

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/0/g' -e 's/RUNNO/1/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold0_run1.sh
jid1=$(sbatch --parsable --dependency=afterany:$jid0 bash_files/run_$1_fold0_run1.sh)
echo $jid1

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/2/g' -e 's/RUNNO/0/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold2_run0.sh
jid2=$(sbatch --parsable --dependency=afterany:$jid1 bash_files/run_$1_fold2_run0.sh)
echo $jid2

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/2/g' -e 's/RUNNO/1/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold2_run1.sh
jid3=$(sbatch --parsable --dependency=afterany:$jid2 bash_files/run_$1_fold2_run1.sh)
echo $jid3

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/4/g' -e 's/RUNNO/0/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold4_run0.sh
jid4=$(sbatch --parsable --dependency=afterany:$jid3 bash_files/run_$1_fold4_run0.sh)
echo $jid4

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/4/g' -e 's/RUNNO/1/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold4_run1.sh
jid5=$(sbatch --parsable --dependency=afterany:$jid4 bash_files/run_$1_fold4_run1.sh)
echo $jid5

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/6/g' -e 's/RUNNO/0/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold6_run0.sh
jid6=$(sbatch --parsable --dependency=afterany:$jid5 bash_files/run_$1_fold6_run0.sh)
echo $jid6

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/6/g' -e 's/RUNNO/1/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold6_run1.sh
jid7=$(sbatch --parsable --dependency=afterany:$jid6 bash_files/run_$1_fold6_run1.sh)
echo $jid7

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/8/g' -e 's/RUNNO/0/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold8_run0.sh
jid8=$(sbatch --parsable --dependency=afterany:$jid7 bash_files/run_$1_fold8_run0.sh)
echo $jid8

sed -e "s/RUNNAME/$1/g" -e 's/FOLD/8/g' -e 's/RUNNO/1/g' -e "s/HPARAMS/$2/g" <run_train_multi.sh >bash_files/run_$1_fold8_run1.sh
jid9=$(sbatch --parsable --dependency=afterany:$jid8 bash_files/run_$1_fold8_run1.sh)
echo $jid9
