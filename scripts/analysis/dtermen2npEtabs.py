import sys
import argparse
import multiprocessing as mp
import numpy as np
import os
import glob

# Code from preprocessing folder
sys.path.insert(0, '..')
from preprocessing.parseEtab import parseEtab

def parseEtabs(out_folder, in_list, num_cores=1):
    print('num cores', num_cores)
    print('warning! it seems that if subprocesses fail right now you don\'t get an error message. be wary of this if the number of files you\'re getting seems off')
    out_folder = os.path.abspath(out_folder)

    pool = mp.Pool(num_cores, maxtasksperchild = 10)
    
    for filepath in in_list:
        res = pool.apply_async(dataGen, args=(filepath, out_folder), error_callback=raise_error)

    pool.close()
    pool.join()

def raise_error(error):
    traceback.print_exception(Exception, error, None)

# inner loop we wanna parallize
def dataGen(filepath, out_folder):
    name = filepath.split('/')[-1]
    out_file = os.path.join(out_folder, name)
    print('out file', out_file)
    try:
        _, _, etab = parseEtab(filepath, save=False)
        np.save(out_file, etab)
    except Exception as e:
        print(out_file, file=sys.stderr)
        raise e
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse dTERMen etabs and dump numpy versions')
    parser.add_argument('out_folder', help='folder where numpy etabs will be placed')
    parser.add_argument('in_list', help='file containing etab paths', default=None)
    parser.add_argument('num_cores', help='number of proceses to use', type=int, default=1)
    args = parser.parse_args()
    
    with open(args.in_list) as fp:
        in_list = [l.strip() for l in fp]

    parseEtabs(args.out_folder, in_list, num_cores=args.num_cores)
    
