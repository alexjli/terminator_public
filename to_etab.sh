#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=sched_mem1TB_centos7
#SBATCH --time=06:00:00
#SBATCH -n 16
#SBATCH --mem=0
#SBATCH -o logfiles/etab-output-OUTPUTDIR.out
#SBATCH -e logfiles/etab-error-OUTPUTDIR.out

. /etc/profile.d/modules.sh
module add gcc
module add slurm
module add c3ddb/miniconda
python to_etab.py --output_dir=OUTPUTDIR
