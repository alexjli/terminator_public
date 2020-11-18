import os
import argparse
import glob

OUTPUT_DIR = '/scratch/users/vsundar/TERMinator/outputs/'


if __name__ == '__main__':

    with open('to_run.out', 'r') as f:
        for line in f:
            filepath = line[:-1]
            pdb_id = filepath[-4:]
            print(pdb_id)
            os.system(f'cp -r {filepath} /scratch/users/vsundar/TERMinator/fixed_dTERMen/{pdb_id}')
            os.system(f"sed -e \"s/ID/{pdb_id}/g\" </home/vsundar/TERMinator_code/fix_dTERMen.sh >/scratch/users/vsundar/TERMinator/fixed_dTERMen/{pdb_id}/fix_{pdb_id}.sh")
            os.system(f"cd /scratch/users/vsundar/TERMinator/fixed_dTERMen/{pdb_id}/ && sbatch fix_{pdb_id}.sh")
