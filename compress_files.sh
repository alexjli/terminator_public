#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=sched_mem1TB_centos7
#SBATCH --time=06:00:00
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH -o logfiles/compress-files-output-OUTPUTDIR.out
#SBATCH -e logfiles/compress-files-error-OUTPUTDIR.out

tar -cvzf /scratch/users/vsundar/TERMinator/outputs/OUTPUTDIR/etabs.tar.gz /scratch/users/vsundar/TERMinator/outputs/OUTPUTDIR/etabs/
rm -rf /scratch/users/TERMinator/outputs/OUTPUTDIR/etabs/
