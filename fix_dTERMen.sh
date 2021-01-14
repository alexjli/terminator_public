#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=8000
#SBATCH --partition=defq
#SBATCH --time=1-00:00:00
#SBATCH -o /scratch/users/alexjli/TERMinator/fixed_dTERMen/ID/design.oFIXED
#SBATCH -e /scratch/users/alexjli/TERMinator/fixed_dTERMen/ID/design.eFIXED

. /etc/profile.d/modules.sh
module load c3ddb/glibc/2.14
module load gcc
/scratch/users/swans/MST_workspace/MST/bin/design --p ID.pdb --o ID --c /scratch/users/vsundar/TERMinator/default_configfile
