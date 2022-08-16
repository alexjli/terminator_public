***************
Getting Started
***************

This page details how to get started with TERMinator.

The following instructions assume you are running on SuperCloud, one of MIT's HPCC.
These SLURM scripts are included in the repo, and can be adapted to other HPCC systems if necessary.

Setup
=====
Setup the proper conda environment using the :code:`env.yaml` file (e.g. :code:`conda env create -f env.yaml`).
This will create a conda env called :code:`terminator`, which can be activated using :code:`conda activate terminator`.

Next, run :code:`pip install -e .`, which will install the TERMinator software suite as an importable module :code:`terminator` in the environment.

Additionally, you'll need to modify :code:`scripts/config.sh` specifying the path to your MST installation (e.g. :code:`~/MST_workspace/MST`).
This is necessary for using :code:`MST/bin/design`.

TERMinator Feature Generation
==============================
First, you'll need a folder that of dTERMen runs e.g. a folder of structure :code:`<dataset>/<pdb_id>/<pdb_id>.<ext>`,
where :code:`<ext>` must include :code:`.dat` and :code:`.red.pdb` outputted from running :code:`MST/bin/design`.

To generate feature files from this folder, use

.. code-block::

  python scripts/data/preprocessing/generateDataset.py \
      --in_folder <input_data_folder> \
      --out_folder <output_features_folder> \
      -n <num_cores> \
      <-u if you want to overwrite existing feature files>

which will create a dataset :code:`<output_features_folder>` that you can feed into TERMinator.

COORDinator Feature Generation
==============================
COORDinator preprocessing requires a bit of extra preprocessing. To generate feature files from your raw data, use


.. code-block::

  python scripts/data/preprocessing/cleanStructs.py \
      --in_list_path <pdb_paths_file> \
      --out_folder <output_folder> \
      -n <num_processes>

which will clean the PDB files listed in `<pdb_paths_file>`. Be sure that `<pdb_paths_file>` is a file containing a list of PDB paths,
with one path per line. The outputted `<output_folder>` can then be fed into `generateDataset.py` above
with the additional flag `--coords_only` to featurize these structures for COORDinator.

Training and evaluation
=======================
To train a new model, run

.. code-block::

  ./scripts/models/train/submit_train.sh \
      <dataset_dir> \
      <model_hparams_path> \
      <run_hparams_path> \
      <run_dir> \
      <output_dir> \
      <run_wall_time>

This will submit a job to train on the given dataset with the given hyperparameters, place the trained model and related files in the run directory, and results in the output directory. For TERMinator, use `model_hparams=hparams/model/terminator.json` and `run_hparams=hparams/run/default.json`. For COORDinator, use `model_hparams=hparams/model/coordinator.json` and `run_hparams/run/seq_batching.json`; you may also want to remove the `--lazy` option in `scripts/models/train/run_train_gpu.sh`, which can speed up training by loading the whole dataset into memory first.
Note that the train script will assume you placed the train, val, and test splits in `<dataset_dir>/train.in`, `<dataset_dir>/validation.in`, and `<dataset_dir>/test.in`, respectively. If you need different behavior, you can directly call the `scripts/models/train/train.py` script.
The model will automatically evaluate on the test set and dump the results into `net.out` in `<output_dir>`, which can be used in postprocessing.


If you instead want to evaluate on a pretrained model, run

.. code-block::

  ./scripts/models/eval/submit_eval.sh \
      <model_directory> \
      <dataset_dir> \
      <output_dir> \
      [subset_file]

This will load the model, evaluate the features from using that model, and place them in the output dir.
:code:`subset_file` is optional: if provided, only that subset will be evaluated from :code:`dataset_dir`, otherwise the whole dataset will be evaluated.

Postprocessing
==============
To perform postprocessing, run

.. code-block::

  ./scripts/data/postprocesing/submit_etab.sh \
      <dtermen_data_root> \
      <pdb_root> \
      <output_dir>

:code:`dtermen_data_root` should be the parent directory to all the dTERMen runs you've run
(e.g. for every dtermen dataset DATA, the directory should be structured :code:`DATA/<pdb_id>/(dTERMen run files for pdb_id)`). For COORDinator, you can use the outputs of `scripts/data/preprocessing/cleanStructs.py`.

:code:`pdb_root` is similar but should be the parent directory to all databases in databaseCreator format
(e.g. for database DATA, the directory should be structured :code:`DATA/PDB/<pdb_id_mid>/<pdb_id>.pdb`).

Automatic steps afterwards
##########################
:code:`submit_etab.sh` automatically calls the following two scripts to automate postprocessing.

First, it should submit a batch job array to run dTERMen on each of the etabs.
In case you want to run this manually, this command is
.. code-block::

  python scripts/data/postprocessing/batch_arr_dTERMen.py \
      --output_dir=<output_directory_name> \
      --pdb_root=<pdb_root> \
      --dtermen_data=<dtermen_data_root> \
      --batch_size=48

`batch_size` specifies how many dTERMen runs each job in the job array will run in parallel. Each job in the job array should take 5-10 minutes, but could take longer depending on protein size.
The resultant files are also dumped in :code:`<output_dir>/etabs/`.

After the previous step completes, a summarization script should also automatically run.
This command is
.. code-block::

  python scripts/data/postprocessing/summarize_results.py \
      --output_dir=<output_dir> \
      --dtermen_data=<dtermen_data_root>


This will be located at :code:`<output_dir>/summary_results.csv`.

Although these two steps are run automatically, oftentimes certain dTERMen jobs will have not finished
(e.g. sometimes jobs stall if they're placed on a busy node, causing jobs to hit the wall time).
Run the above step again if you see no :code:`summary_results.csv` in the output directory or it's empty,
and it will resubmit all dTERMen jobs that didn't complete.

Other Potentially Useful Scripts
================================
To convert dTERMen etabs to numpy etabs, run

.. code-block::

  python scripts/analysis/dtermen2npEtabs.py \
  --out_folder=<np_etab_folder> \
  --in_list=(file containing list of paths to .etab files) \
  --num_cores=N

This will read the etab files in :code:`in_list`, convert them into numpy files, and dump them in :code:`np_etab_folder`

To compress etab files,

.. code-block::

  ./scripts/data/postprocessing/submit_compress_files.sh <output_dir>
