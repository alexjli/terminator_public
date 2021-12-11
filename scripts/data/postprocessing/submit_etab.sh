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

# check if shell_scripts exists and if not create it
if [[ ! -d $DIR/shell_scripts ]];
then
  mkdir $DIR/shell_scripts
fi

RUNNAME="basename $2"
sed -e "s|DTERMENDATA|$1|g" -e "s|OUTPUTDIR|$2|g" -e "s|RUNNAME|${RUNNAME}|g" <${DIR}/to_etab.sh >${DIR}/shell_scripts/to_etab_${RUNNAME}.sh
sbatch ${DIR}/shell_scripts/to_etab_${RUNNAME}.sh
