import os
import argparse
import glob

OUTPUT_DIR = '/scratch/users/vsundar/TERMinator/outputs/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run dTERMen for testing.')
    parser.add_argument('--output_dir', help = 'Output directory', default = 'test_run')
    args = parser.parse_args()

    output_path = os.path.join(OUTPUT_DIR, args.output_dir, 'etabs')
    for filename in glob.glob(os.path.join(output_path, '*.etab')):
        pdb_id = filename[-9:-5]
        print(pdb_id)
        os.system(f"sed -e \"s/ID/{pdb_id}/g\" </home/vsundar/TERMinator_code/run_dTERMen.sh >{output_path}/run_{pdb_id}.sh")
        os.system(f"cd {output_path} && sbatch run_{pdb_id}.sh")
