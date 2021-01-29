#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=8000
#SBATCH --partition=defq
#SBATCH --time=1-00:00:00
#SBATCH -o /scratch/users/alexjli/ablate_s2s_runs/OUTPUTDIR/etabs/ID-output.out
#SBATCH -e /scratch/users/alexjli/ablate_s2s_runs/OUTPUTDIR/etabs/ID-error.err

. /etc/profile.d/modules.sh
/scratch/users/swans/MST_workspace/MST/bin/design --p ID.pdb --o ID --c /scratch/users/vsundar/TERMinator/default_configfile
