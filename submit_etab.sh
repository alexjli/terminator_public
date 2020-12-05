#!/bin/bash
sed -e 's/OUTPUTDIR/$1/g' </home/vsundar/TERMinator_code/to_etab.sh >/home/vsundar/TERMinator_code/shell_scripts/to_etab_$1.sh
sbatch /home/vsundar/TERMinator_code/shell_scripts/to_etab_$1.sh
