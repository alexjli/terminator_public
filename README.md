# TERMinator
This repo contains code for the TERMinator neural net, a neuronal implementation of dTERMen.

## Documentation
The living documentation for this package can be found [here](https://docs.google.com/document/d/1xiaKvsUgBG5gzdJVc7iZQBsFyWzoPZx4k-vBip66Q20/edit?usp=sharing).

## Requirements
* python3
* pytorch = 1.6.0
* pandas
* numpy
* scipy
* tqdm
* matplotlib
* seaborn

Above should be all that's needed, but an `env.yaml` file is included just in case (though it's not at all the minimal set needed). 

## Instructions for Running Pipeline

0. Ensure your path references are correct for both c3 and Supercloud.

0. Define a set of hyperparameters and store them in `hparams/<hparams_filename>.json`

0. Run `submit_all_jobs.sh <output_directory_suffix> <hparams_filename>` on Supercloud. This submits jobs to run the pipeline on folds 0, 2, 4, 6, and 8 in sequence automatically. The jobs can take up to 4 days to run, depending on the given hyperparameters. Use `submit_ingraham.sh <output_directory_suffix> <hparams_filename>` instead for running on the Ingraham dataset.

0. Move the output directories from Supercloud to c3ddb.

0. Enter the `postprocessing` folder, all subsequent commands are run from there.

0. Run `submit_etab.sh <output_directory_name>` for each output directory for an individual fold (e.g. Ingraham set) or `submit_pipeline.sh <output_directory_suffix>` to automatically submit folds 0, 2, 4, 6, 8. Each job should take about an 1 hr per 1000 structures and generates the energy tables from the output. 
The next two steps are run automatically, but below are the commands if something goes wrong in-between:

    a. Run dTERMen on the resulting energy tables via `python batch_arr_dTERMen.py --output_dir=<output_directory_name>` for each output directory (one for each fold). This step should relatively quickly submit all the individual jobs.

    b. Once all the dTERMen jobs are done running, run `touch to_run.out` inside the TERMinator directory to create a `to_run.out` file (to be used in step 7). Then run `python summarize_results.py --output_dir=<output_directory_name>` for each output directory (one for each fold). This step generates a summary .csv file in the output directory with all the relevant results.

    Although these two steps are run automatically, oftentimes certain dTERMen jobs will have not finished (sometimes jobs stall if they're placed on a busy node, so we set the time quota to be small and opt to rerun them). Run 4b again if you see no `summary_results.csv` in the output directory, and it will resubmit all dTERMen jobs that didn't complete.

0. Some dTERMen jobs may not have computed the native sequence recovery for the baseline dTERMen model. If this is an issue, step 6 will list the relevant files in `to_run.out`. Fix this by running `python fix_dTERMen.py`. I believe all such issues should have been fixed already, but you can use this to check.

0. The energy tables take up quite a lot of space on c3ddb's scratch. To conserve memory, run `submit_compress_files.sh <output_directory_name>` for each output directory (one for each fold). This will tarball the `etabs/` folder and then delete the original folder, conserving space. 
