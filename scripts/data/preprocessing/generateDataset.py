import argparse
import functools
import glob
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback

import numpy as np
from packageTensors import dumpCoordsTensors, dumpTrainingTensors


# when subprocesses fail you usually don't get an error...
def generateDatasetParallel(in_folder,
                            out_folder,
                            cutoff=50,
                            num_cores=1,
                            update=True,
                            coords_only=False,
                            dummy_terms=None):
    print('num cores', num_cores)
    print(('warning! it seems that if subprocesses fail right now you don\'t get an error message. '
           'be wary of this if the number of files you\'re getting seems off'))
    # make folder where the dataset files are gonna be placed
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # generate absolute paths so i dont have to think about relative references
    in_folder = os.path.abspath(in_folder)
    out_folder = os.path.abspath(out_folder)

    os.chdir(in_folder)

    process_func = functools.partial(dataGen,
                                     cutoff=cutoff,
                                     coords_only=coords_only,
                                     dummy_terms=dummy_terms)

    pool = mp.Pool(num_cores, maxtasksperchild=10)
    # process folder by folder
    for folder in glob.glob("*"):
        # folders that aren't directories aren't folders!
        if not os.path.isdir(folder):
            continue

        full_folder_path = os.path.join(out_folder, folder)
        if not os.path.exists(full_folder_path):
            os.mkdir(full_folder_path)

        for idx, file in enumerate(glob.glob(folder + '/*.red.pdb')):
            name = file[:-len(".red.pdb")]
            if not update:
                out_file = os.path.join(out_folder, name)
                if os.path.exists(out_file + '.features'):
                    continue

            res = pool.apply_async(process_func,
                                   args=(file, folder, out_folder),
                                   error_callback=raise_error)

    pool.close()
    pool.join()


def raise_error(error):
    traceback.print_exception(Exception, error, None)


# inner loop we wanna parallize
def dataGen(file, folder, out_folder, cutoff, coords_only, dummy_terms):
    name = file[:-len(".red.pdb")]
    out_file = os.path.join(out_folder, name)
    print('out file', out_file)
    try:
        if coords_only:
            dumpCoordsTensors(name, out_path=out_file)
        else:
            dumpTrainingTensors(name,
                                out_path=out_file,
                                cutoff=cutoff,
                                coords_only=coords_only,
                                dummy_terms=dummy_terms)
    except Exception as e:
        print(out_file, file=sys.stderr)
        raise e


if __name__ == '__main__':
    # idek how to do real parallelism but this should fix the bug of stalling when processes crash
    mp.set_start_method("spawn")  # i should use context managers but low priority change
    parser = argparse.ArgumentParser('Generate features data files from dTERMen .dat files')
    parser.add_argument(
        'in_folder',
        help='input folder containing .dat files in proper directory structure',
    )
    parser.add_argument('out_folder',
                        help='folder where features will be placed')
    parser.add_argument('--cutoff',
                        dest='cutoff',
                        help='max number of match entries per TERM',
                        default=50,
                        type=int)
    parser.add_argument('-n',
                        dest='num_cores',
                        help='number of processes to use',
                        default=1,
                        type=int)
    parser.add_argument('-u',
                        dest='update',
                        help='if added, update existing files. else, files that already exist will not be overwritten',
                        default=False,
                        action='store_true')
    parser.add_argument('--weight_fn',
                        help='weighting function for rmsd to use when generating statistics',
                        default='neg')
    parser.add_argument('-s',
                        dest='stats',
                        help='if added, compute singleton and pair stats as features',
                        default=False,
                        action='store_true')
    parser.add_argument('--coords_only',
                        dest='coords_only',
                        help='if added, only include coordinates-relevant data in the feature files',
                        default=False,
                        action='store_true')
    parser.add_argument('--dummy_terms',
                        help='option for how to use dummy TERMs in the feature files',
                        default=None)
    args = parser.parse_args()
    generateDatasetParallel(args.in_folder,
                            args.out_folder,
                            cutoff=args.cutoff,
                            num_cores=args.num_cores,
                            update=args.update,
                            stats=args.stats,
                            weight_fn=args.weight_fn,
                            coords_only=args.coords_only,
                            dummy_terms=args.dummy_terms)
