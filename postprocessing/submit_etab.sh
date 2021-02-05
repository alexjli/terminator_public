#!/bin/bash

sed -e "s/OUTPUTDIR/$1/g" </home/alexjli/TERMinator/postprocessing/to_etab.sh >/home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_$1.sh
sbatch /home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_$1.sh
