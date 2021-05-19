#!/bin/bash
#SBATCH -p defq

source ~/.bashrc
conda activate analysis
python summarize_results.py --output_dir=$1
