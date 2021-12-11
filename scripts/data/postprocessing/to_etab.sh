#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=sched_mem1TB
#SBATCH --time=06:00:00
#SBATCH -n 16
#SBATCH --mem=0
#SBATCH -o logfiles/etab-output-RUNNAME.out
#SBATCH -e logfiles/etab-error-RUNNAME.out

# activate conda
CONDA_ROOT=/state/partition1/llgrid/pkg/anaconda/anaconda3-2019b/
source ${CONDA_ROOT}/etc/profile.d/conda.sh

# compute what directory this file is in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
# DIR is the directory this file is in, e.g. postprocessing
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

python $DIR/to_etab.py --output_dir=OUTPUTDIR --dtermen_root=DTERMENDATA --num_cores=16 -u
python $DIR/batch_arr_dTERMen.py --output_dir=OUTPUTDIR
