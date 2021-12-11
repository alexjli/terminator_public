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
MODELDIR=$2
OUTPUTDIR=$3
RUNNAME="basename $2"
sed \
  -e "s|DATASET|${DATASET}|g" \
  -e "s|DATANAME|${DATANAME}|g" \
  -e "s|MODELDIR|${MODELDIR}|g" \
  -e "s|RUNNAME|${RUNNAME}|g" \
  <run_eval.sh \
  >bash_files/eval_${DATANAME}_${RUNNAME}.sh
  /
jid0=$(sbatch --parsable bash_files/eval_${DATANAME}_${RUNNAME}.sh)
echo $jid0
