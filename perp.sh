#!/bin/bash
#SBATCH --gres=gpu:volta:1
#SBATCH --time=00:45:00

for i in {0..8..2}; do
	python eval_perplexity.py --dataset=features_multichain --run_name=test_run_fold${i}_$1 --test=fold_splits/test_fold$i.in
done
