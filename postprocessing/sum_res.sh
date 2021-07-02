#!/bin/bash
#SBATCH -p defq
#SBATCH -o /dev/null

source ~/.bashrc
conda activate analysis
python summarize_results.py --output_dir=$1
