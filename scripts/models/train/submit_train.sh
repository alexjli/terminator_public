#!/bin/bash

# compute what directory this file is in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
# DIR is the directory this file is in, e.g. postprocessing
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

cd $DIR

if [[ ! -d bash_files ]];
then
  mkdir bash_files
fi

DATASET=$1
DATANAME=${1##*/}
HPARAMS=$2
RUNDIR=$3
RUNNAME=${3##*/}
OUTPUTDIR=$4
HOURS=$5
echo "$DATANAME $RUNNAME $OUTPUTDIR"

if [[ ! -d $RUNDIR ]];
then
  mkdir $RUNDIR
fi

if [[ ! -d $OUTPUTDIR ]];
then
  mkdir $OUTPUTDIR
fi

sed \
  -e "s|DATASET|${DATASET}|g" \
  -e "s|DATANAME|${DATANAME}|g" \
  -e 's|RUNNO|0|g' \
  -e "s|HPARAMS|${HPARAMS}|g" \
  -e "s|RUNDIR|${RUNDIR}|g" \
  -e "s|OUTPUTDIR|${OUTPUTDIR}|g" \
  -e "s|RUNNAME|${RUNNAME}|g" \
  -e "s|HOURS|${HOURS}|g" \
  <run_train.sh \
  >bash_files/run_${DATANAME}_${RUNNAME}_run0.sh
jid0=$(sbatch --parsable bash_files/run_${DATANAME}_${RUNNAME}_run0.sh)
echo $jid0
