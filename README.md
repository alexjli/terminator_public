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

0. setup conda environment using env.yaml file. this will create a conda env called `terminator`
0. modify `scripts/config.sh` specifying the path to your MST installation (e.g. `~/MST_workspace/MST`)
0. run `conda activate terminator` to activate the environment
0. run `python setup.py install`, which will install the TERMinator software suite as module `terminator` in the environment
0. to generate feature files from your raw data, use `python scripts/data/preprocessing/generateDataset.py <input_data_folder> <output_features_folder> -n <num_cores> <-u if you want to force update feature files>`
0. to train on supercloud, run `./scripts/models/train/submit_train.sh <dataset_dir> <hparams_path> <output_dir>`. this will train on the given dataset using TERMinator with the given hyperparameters, and place the trained model and results in the output directory
    a. note that the train script assumes you place the train, val, and test splits in "<dataset_dir>/train.in", "<dataset_dir>/validation.in", and "<dataset_dir>/test.in", respectively.
0. to eval on supercloud, run `./terminator/models/eval/submit_eval.sh <model_directory> <dataset_dir> <output_dir> [subset_file]`. this will load the model, evaluate the features from using that model, and place them in the output dir. `subset_file` is optional: if provided, only that subset will be evaluated from `dataset_dir`, otherwise the whole dataset will be evaluated
0. to perform postprocessing, run `./terminator/data/postprocesing/submit_etab.sh <dtermen_data_root> <pdb_root> <output_dir>`. `dtermen_data_root` should be the parent directory to all the dTERMen runs you've run (e.g. for every dtermen dataset DATA, the directory should be structured `DATA/<pdb_id>/(dTERMen run files for pdb_id)`). `pdb_root` is similar but should be the parent directory to all databases in databaseCreator format (e.g. for database DATA, the directory should be structured `DATA/PDB/<pdb_id_mid>/<pdb_id>.pdb`).

    a. when it completes, it should submit a batch job array to run dTERMen on each of the etabs. in case you want to run this manually, this command is `python scripts/data/postprocessing/batch_arr_dTERMen.py --output_dir=<output_directory_name> --pdb_root=<pdb_root> --dtermen_data=<dtermen_data_root> --batch_size=10`. `batch_size` specifies how many dTERMen runs each job in the job array will run. the resultant files are also dumped in `<output_dir>/etabs/`.

    b. after the previous step completes, a summarization script should also automatically run. this command is `python scripts/data/postprocessing/summarize_results.py --output_dir=<output_dir> --dtermen_data=<dtermen_data_root>`. this will be located at `<output_dir>/summary_results.csv`.

    Although these two steps are run automatically, oftentimes certain dTERMen jobs will have not finished (sometimes jobs stall if they're placed on a busy node, so we set the time quota to be small and opt to rerun them). Run the above step again if you see no `summary_results.csv` in the output directory, and it will resubmit all dTERMen jobs that didn't complete.

0. to convert dTERMen etabs to numpy etabs, run `python scripts/analysis/dtermen2npEtabs.py --out_folder=<np_etab_folder> --in_list=(file containing list of paths to .etab files) --num_cores=N`. this will read the etab files in `in_list`, convert them into numpy files, and dump them in `np_etab_folder`

0. to compress etab files, `scripts/data/postprocessing/submit_compress_files.sh <output_dir>`

## Documentation
The living documentation for this package can be found [here](https://docs.google.com/document/d/1xiaKvsUgBG5gzdJVc7iZQBsFyWzoPZx4k-vBip66Q20/edit?usp=sharing).

## Requirements
* python3
* pytorch = 1.10.1
* pandas
* numpy
* scipy
* tqdm
* matplotlib
* seaborn
* pylint
* pytest-pylint
* yapf

Above should be all that's needed, but an `env.yaml` file is included just in case (though it's not at all the minimal set needed).
