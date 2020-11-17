#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=8000
#SBATCH --partition=defq
#SBATCH --time=1-00:00:00
#SBATCH -o /scratch/users/vsundar/TERMinator/outputs/test_run_fold0_reg_par_fixed/etabs/ID-output.out
#SBATCH -e /scratch/users/vsundar/TERMinator/outputs/test_run_fold0_reg_par_fixed/etabs/ID-error.err

. /etc/profile.d/modules.sh
/scratch/users/swans/MST_workspace/MST/bin/design --p ID.red.pdb --o ID --c /scratch/users/vsundar/TERMinator/default_configfile
