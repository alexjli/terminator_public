# TERMinator
This repo contains code for the TERMinator neural net, a neuronal implementation of dTERMen.

The following instructions assume you are running on SuperCloud, one of MIT's HPCC.
These SLURM scripts are included in the repo, and can be adapted to other HPCC systems if necessary.

## Documentation
As of now, we don't have the docs hosted anywhere, but they're pretty nice! You can build the docs and view them locally following the instructions in the `docs` folder. The "Getting Started" guide is also given below.

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

Above should be all that's needed, and an `env.yaml` is included that specifies these.

## Setup
The following instructions assume you are running on SuperCloud, one of MIT's HPCC.
These SLURM scripts are included in the repo, and can be adapted to other HPCC systems if necessary.

Setup the proper conda environment using the `env.yaml` file (e.g. `conda env create -f env.yaml`).
This will create a conda env called `terminator`, which can be activated using `conda activate terminator`.

Next, run `python setup.py install`, which will install the TERMinator software suite as an importable module `terminator` in the environment.

Additionally, you'll need to modify `scripts/config.sh` specifying the path to your MST installation (e.g. `~/MST_workspace/MST`).
This is necessary for using `MST/bin/design`.

## Feature Generation
First, you'll need a folder that of dTERMen runs e.g. a folder of structure `<dataset>/<pdb_id>/<pdb_id>.<ext>`,
where `<ext>` must include `.dat` and `.red.pdb` outputted from running `MST/bin/design`.

To generate feature files from this folder, use

```
python scripts/data/preprocessing/generateDataset.py \
    --in_folder <input_data_folder> \
    --out_folder <output_features_folder> \
    -n <num_cores> \
    <-u if you want to overwrite existing feature files>
```


which will create a dataset `<output_features_folder>` that you can feed into TERMinator.

### TERMless TERMinator
You can run TERMinator without mining TERMs! This requires a bit of extra preprocessing.
To generate feature files from your raw data, use

```
python cleanStructs.py \
    --in_list_path <pdb_paths_file> \
    --out_folder <output_folder> \
    -n <num_processes>
```

which will clean the PDB files listed in `<pdb_paths_file>`. Be sure that <pdb_paths_file> is a file containing a list of PDB paths,
with one path per line. The outputted `<output_folder>` can then be fed into `generateDataset.py` above
with the additional flag `--coords_only` to featurize these structures for TERMinator.

## Training and evaluation TERMinator
To train a new model, run

```
./scripts/models/train/submit_train.sh \
    <dataset_dir> \
    <hparams_path> \
    <output_dir>
```

This will submit a job to train on the given dataset using TERMinator with the given hyperparameters, and place the trained model and results in the output directory.
Note that the train script assumes you place the train, val, and test splits in `<dataset_dir>/train.in`, `<dataset_dir>/validation.in`, and `<dataset_dir>/test.in`, respectively.
The model will automatically evaluate on the test set and dump that into `net.out` in `<output_dir>`, which can be used in postprocessing.

If you instead want to evaluate on a pretrained model, run

```
  ./scripts/models/eval/submit_eval.sh \
      <model_directory> \
      <dataset_dir> \
      <output_dir> \
      [subset_file]
```

This will load the model, evaluate the features from using that model, and place them in the output dir.
`subset_file` is optional: if provided, only that subset will be evaluated from `dataset_dir`, otherwise the whole dataset will be evaluated.

## Postprocessing
To perform postprocessing, run

```
./scripts/data/postprocesing/submit_etab.sh \
    <dtermen_data_root> \
    <pdb_root> \
    <output_dir>
```

`dtermen_data_root` should be the parent directory to all the dTERMen runs you've run
(e.g. for every dtermen dataset DATA, the directory should be structured `DATA/<pdb_id>/(dTERMen run files for pdb_id)`).

`pdb_root` is similar but should be the parent directory to all databases in databaseCreator format
(e.g. for database DATA, the directory should be structured `DATA/PDB/<pdb_id_mid>/<pdb_id>.pdb`).

### Automatic steps afterwards
`submit_etab.sh` automatically calls the following two scripts to automate postprocessing.

First, it should submit a batch job array to run dTERMen on each of the etabs.
In case you want to run this manually, this command is

```
python scripts/data/postprocessing/batch_arr_dTERMen.py \
    --output_dir=<output_directory_name> \
    --pdb_root=<pdb_root> \
    --dtermen_data=<dtermen_data_root> \
    --batch_size=10
```

`batch_size` specifies how many dTERMen runs each job in the job array will run.
For a batch size of 10, each job in the job array should take roughly 1-2 hours.
The resultant files are also dumped in `<output_dir>/etabs/`.

After the previous step completes, a summarization script should also automatically run.
This command is
```
  python scripts/data/postprocessing/summarize_results.py \
      --output_dir=<output_dir> \
      --dtermen_data=<dtermen_data_root>
```

This will be located at `<output_dir>/summary_results.csv`.

Although these two steps are run automatically, oftentimes certain dTERMen jobs will have not finished
(e.g. sometimes jobs stall if they're placed on a busy node, causing jobs to hit the wall time).
Run the above step again if you see no `summary_results.csv` in the output directory or it's empty,
and it will resubmit all dTERMen jobs that didn't complete.

## Other Potentially Useful Scripts
To convert dTERMen etabs to numpy etabs, run

```
python scripts/analysis/dtermen2npEtabs.py \
  --out_folder=<np_etab_folder> \
  --in_list=<file containing list of paths to .etab files> \
  --num_cores=N
```

This will read the etab files in `in_list`, convert them into numpy files, and dump them in `np_etab_folder`

To compress etab files,

```
./scripts/data/postprocessing/submit_compress_files.sh <output_dir>
```

## TODO

(maybe make a high-level "main.py" script that you can run other scripts from?)
- `./main.py featurize <in> <out> *args` to featurize raw data
- `./main.py train <hparams> <data> <run_dir>` to train models
- `./main.py eval <run_dir> <data> <eval_dir>` to eval models on a dataset
- `./main.py post <eval_dir>` to postprocess an eval dir
    have a flag for running with sequence complexity filter?
- `./main.py sum_res <eval_dir>` to summarize results from an eval dir
- `./main.py npy_etabs <eval_dir>` to convert dTERMen etabs to numpy etabs in an eval dir
- `./main.py compress <eval_dir>` to compress dTERMen etabs
