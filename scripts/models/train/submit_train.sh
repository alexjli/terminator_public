#!/bin/bash

if [[ ! -d logfiles ]];
then
  mkdir logfiles
fi
if [[ ! -d bash_files ]];
then
  mkdir bash_files
fi

DATASET=$1
DATANAME="basename $1"
HPARAMS=$2
OUTPUTDIR=$3
RUNNAME="basename $3"
sed \
  -e "s|DATASET|${DATASET}|g" \
  -e "s|DATANAME|${DATANAME}|g" \
  -e 's|RUNNO|0|g' \
  -e "s|HPARAMS|${HPARAMS}|g" \
  -e "s|RUNNAME|${RUNNAME}|g" \
  <run_train.sh \
  >bash_files/run_${DATANAME}_${RUNNAME}_run0.sh
  /
jid0=$(sbatch --parsable bash_files/run_${DATANAME}_${RUNNAME}_run0.sh)
echo $jid0
