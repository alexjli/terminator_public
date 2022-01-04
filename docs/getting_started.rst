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

Next, run :code:`python setup.py install`, which will install the TERMinator software suite as an importable module :code:`terminator` in the environment.

Additionally, you'll need to modify :code:`scripts/config.sh` specifying the path to your MST installation (e.g. :code:`~/MST_workspace/MST`).
This is necessary for using :code:`MST/bin/design`.

Feature Generation
==================
First, you'll need a folder that of dTERMen runs e.g. a folder of structure :code:`<dataset>/<pdb_id>/<pdb_id>.<ext>`,
where :code:`<ext>` must include :code:`.dat` and :code:`.red.pdb` outputted from running :code:`MST/bin/design`.

To generate feature files from this folder, use

.. code-block::

  python scripts/data/preprocessing/generateDataset.py \
      <input_data_folder> \
      <output_features_folder> \
      -n <num_cores> \
      <-u if you want to overwrite existing feature files>

which will create a dataset :code:`<output_features_folder>` that you can feed into TERMinator.

TERMless TERMinator
###################
You can run TERMinator without mining TERMs! This requires a bit of extra preprocessing.
To generate feature files from your raw data, use

.. code-block::

  python cleanStructs.py \
      <pdb_paths_file> \
      <output_folder> \
      -n <num_processes>

which will clean the PDB files listed in :code:`<pdb_paths_file>`. Be sure that <pdb_paths_file> is a file containing a list of PDB paths,
with one path per line. The outputted :code:`<output_folder>` can then be fed into :code:`generateDataset.py` above
with the additional flag :code:`--coords_only` to featurize these structures for TERMinator.

Training and evaluation TERMinator
==================================
To train a new model, run

.. code-block::

  ./scripts/models/train/submit_train.sh \
      <dataset_dir> \
      <hparams_path> \
      <output_dir>

This will submit a job to train on the given dataset using TERMinator with the given hyperparameters, and place the trained model and results in the output directory.
Note that the train script assumes you place the train, val, and test splits in :code:`<dataset_dir>/train.in`, :code:`<dataset_dir>/validation.in`, and :code:`<dataset_dir>/test.in`, respectively.
The model will automatically evaluate on the test set and dump that into :code:`net.out` in :code:`<output_dir>`, which can be used in postprocessing.

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
(e.g. for every dtermen dataset DATA, the directory should be structured :code:`DATA/<pdb_id>/(dTERMen run files for pdb_id)`).

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
      --batch_size=10

:code:`batch_size` specifies how many dTERMen runs each job in the job array will run.
For a batch size of 10, each job in the job array should take roughly 1-2 hours.
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
