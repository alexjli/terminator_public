# TERMinator
This repo contains code for the TERMinator neural net, a neuronal implementation of dTERMen.

## Documentation
The living documentation for this package can be found [here](https://docs.google.com/document/d/1xiaKvsUgBG5gzdJVc7iZQBsFyWzoPZx4k-vBip66Q20/edit?usp=sharing).

## Requirements
* python3
* pytorch = 1.1.0
    * Later versions break the codebase for some reason.
* numpy
* scipy

## Instructions for Running Pipeline

1. Define a set of hyperparameters and store them in `hparams/<hparams_filename>.json`

2. Run `submit_all_jobs.sh <output_directory_name> <hparams_filename>` on Satori. This submits jobs to run the pipeline on folds 0, 2, 4, 6, and 8 in sequence automatically. The jobs should take around 2 days to run, depending on the given hyperparameters.

3. Move the output directories from Satori to c3ddb.

4. Run `submit_etab.sh <output_directory_name>` for each output directory (one for each fold). This step takes a couple hours to run typically and generates the energy tables from the output.

5. Run dTERMen on the resulting energy tables by `python test_dTERMen_mass.py --output_dir=<output_directory_name>` for each output directory (one for each fold). This step should relatively quickly submit all the individual jobs.

6. Once all the dTERMen jobs are done running, run `touch to_run.out` inside the TERMinator directory to create a `to_run.out` file (to be used in step 7). Then run `python summarize_results.py --output_dir=<output_directory_name>` for each output directory (one for each fold). This step generates a summary .csv file in the output directory with all the relevant results.

7. Some dTERMen jobs may not have computed the native sequence recovery for the baseline dTERMen model. If this is an issue, step 6 will list the relevant files in `to_run.out`. Fix this by running `python fix_dTERMen.py`. I believe all such issues should have been fixed already, but you can use this to check.

8. The energy tables take up quite a lot of space on c3ddb's scratch. To conserve memory, run `submit_compress_files.sh <output_directory_name>` for each output directory (one for each fold). This will tarball the `etabs/` folder and then delete the original folder, conserving space. 
