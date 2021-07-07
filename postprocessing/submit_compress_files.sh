#!/bin/bash
sed -e "s/OUTPUTDIR/$1/g" </home/alexjli/TERMinator/postprocessing/compress_files.sh >/home/alexjli/TERMinator/postprocessing/shell_scripts/compress_files_$1.sh
sbatch /home/alexjli/TERMinator/postprocessing/shell_scripts/compress_files_$1.sh
