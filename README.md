# TERMinator
This repo contains code for the TERMinator neural net, a neuronal implementation of dTERMen.

## (outline for how getting set up with TERMinator should work)

(maybe make a high-level "main.py" script that you can run other scripts from?)
- `./main.py featurize <in> <out> *args` to featurize raw data
- `./main.py train <hparams> <data> <run_dir>` to train models
- `./main.py eval <run_dir> <data> <eval_dir>` to eval models on a dataset
- `./main.py post <eval_dir>` to postprocess an eval dir
    have a flag for running with sequence complexity filter?
- `./main.py sum_res <eval_dir>` to summarize results from an eval dir
- `./main.py npy_etabs <eval_dir>` to convert dTERMen etabs to numpy etabs in an eval dir
- `./main.py compress <eval_dir>` to compress dTERMen etabs

0. setup environment using env.yaml file. this will create a conda env called `terminator`
0. modify `scripts/config.sh` at the top level, specifying the path to your MST installation (e.g. `~/MST_workspace/MST`)
0. run `conda activate terminator` to activate the environment
0. run `python setup.py install`, which will install the TERMinator software suite as module `terminator` in the environment
0. it's recommended to place data folders in `data/`, though not strictly required
0. it's also recommended to place model run folders in `models/`, if using pretrained models
0. to generate feature files from your raw data, use `python scripts/data/preprocessing/generateDataset.py <input_data_folder> <output_features_folder> -n <num_cores> <-u if you want to force update feature files>`.
0. to train on supercloud, run `./scripts/models/train/submit_<dataset>.sh <dataset_dir> <hparams_path> <output_dir>`. this will train on the given dataset using TERMinator with the given hyperparameters, and place the trained model and results in the output directory
0. to eval on supercloud, run `./terminator/models/eval/submit_eval.sh <output_dir_suffix> <input_features_folder>`. this will load the model from `models/runs/test_run_<output_dir_suffix>`, evaluate the features from `<input_features_folder>` using that model, and place them in `data/outputs/<output_dir_suffix>`
0. to perform postprocessing, run `./terminator/data/postprocesing/submit_etab.sh <output_directory_name>`. this will read the output dump file in `data/outputs/<output_directory_name>` and submit an etab job to the cluster, dumping the etabs in `data/outputs/<output_Directory_name>/etabs`.
    (perhaps it'd be nice to specify which net.out to use? since you might do multiple runs with the same model)

    a. when it completes, it should submit a batch job to run dTERMen on each of the etabs. in case you want to run this manually, this command is `python terminator/data/postprocessing/batch_arr_dTERMen.py --output_dir=<output_directory_name>`. these files are also dumped in `data/outputs/dTERMen/<output_directory_name>`.

    b. after the previous step completes, a summarization script should also automatically run. this command is `python summarize_results --output_dir=<output_directory_name>`. this will be located at `data/outputs/<output_directory_name>/summary_results.csv`.

    Although these two steps are run automatically, oftentimes certain dTERMen jobs will have not finished (sometimes jobs stall if they're placed on a busy node, so we set the time quota to be small and opt     to rerun them). Run the above step again if you see no `summary_results.csv` in the output directory, and it will resubmit all dTERMen jobs that didn't complete.

0. to convert dTERMen etabs to numpy etabs, run `python terminator/utils/<TODO_SCRIPT> --dir=<directory_name>`. this will search the folder `data/outputs/<directory_name>/etabs` for `*.etab` files and convert them into numpy files, which will be dumped in `data/outputs/<directory_name>/npy_etabs`

0. to compress files, `./terminator/data/postprocessing/submit_compress_files.sh <output_directory_name>`

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
