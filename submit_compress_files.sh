#!/bin/bash
sed -e "s/OUTPUTDIR/$1/g" </home/vsundar/TERMinator_code/compress_files.sh >/home/vsundar/TERMinator_code/shell_scripts/compress_files_$1.sh
sbatch /home/vsundar/TERMinator_code/shell_scripts/compress_files_$1.sh
