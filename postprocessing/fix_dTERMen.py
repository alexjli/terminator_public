import os
import argparse
import glob

from filepaths import *

if __name__ == '__main__':

    with open('to_run.out', 'r') as f:
        for line in f:
            filepath = line[:-1]
            pdb_id = filepath[-4:]
            print(pdb_id)
            os.system(f'cp -r {filepath} /scratch/users/alexjli/TERMinator/fixed_dTERMen/')
            os.system(f'cp {PDB_PATH}{pdb_id.lower()[1:3]}/{pdb_id}.pdb /scratch/users/alexjli/TERMinator/fixed_dTERMen/{pdb_id}/{pdb_id}.pdb')
            os.system(f"sed -e \"s/ID/{pdb_id}/g\" </home/alexjli/TERMinator/fix_dTERMen.sh >/scratch/users/alexjli/TERMinator/fixed_dTERMen/{pdb_id}/fix_{pdb_id}.sh")
            os.system(f"cd /scratch/users/alexjli/TERMinator/fixed_dTERMen/{pdb_id}/ && sbatch fix_{pdb_id}.sh")
