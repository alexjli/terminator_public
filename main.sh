#!/bin/bash

HELP="
Helper script for running common TERMinator jobs
Usage: $0 {featurize|train|eval|post|sum_res|npy_etabs|compress|help} [ARGS ...]

  featurize:
    Generate feature files from raw data
    Usage: $0 featurize <in_data_folder> <out_feature_folder> [-u] [-n NUM_CORES]

  train:

  eval:

  post:

  sum_res:

  npy_etabs:

  compress:

  help, -h, --help:
    Display this message.
"

case $1 in

  "featurize")
    python terminator/data/preprocessing/generateDataset.py ${@:2}
  ;;

  "train")

  ;;

  "-h" | "--help" | "help")
    echo "$HELP"
  ;;

  *)
    echo "Invalid keyword command."
    echo "Usage: $0 {featurize|train|eval|post|sum_res|npy_etabs|compress|help} <args>"
    echo "See --help for more information."
  ;;
esac
