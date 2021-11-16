#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=sched_mem1TB_centos7
#SBATCH --time=06:00:00
#SBATCH -n 16
#SBATCH --mem=0
#SBATCH -o logfiles/compress-files-output-OUTPUTDIR.out
#SBATCH -e logfiles/compress-files-error-OUTPUTDIR.out

tar -cvzf /scratch/users/alexjli/TERMinator_runs/OUTPUTDIR/etabs.tar.gz /scratch/users/alexjli/TERMinator_runs/OUTPUTDIR/etabs/
rm -rf /scratch/users/alexjli/TERMinator_runs/OUTPUTDIR/etabs/