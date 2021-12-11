import os
import argparse
import glob

from .search_utils import find_pdb_folder

DIR = os.path.dirname(os.path.abspath(__file__))
assert DIR[0] == "/", "DIR should be an abspath"

if __name__ == '__main__':
    # TODO: create argument for parser to take in folder w proper dir structure for pdbs
    parser = argparse.ArgumentParser('Run dTERMen for testing.')
    parser.add_argument(
        '--output_dir',
        help='Output directory',
        default='test_run'
    )
    parser.add_argument(
        "--pdb_root",
        help="The root for all raw data PDB databases"
    )
    args = parser.parse_args()
    os.chdir(DIR)

    pdbs = []
    output_path = os.path.join(args.output_dir, 'etabs')
    basename = os.path.basename(args.output_dir)

    for filename in glob.glob(os.path.join(output_path, '*.etab')):
        pdb_id = os.path.basename(filename)[:-5]
        print(pdb_id)

        pdb_folder = find_pdb_folder(pdb_id)
        os.system(f"cp {pdb_folder}/{pdb_id}.pdb {output_path}/{pdb_id}.pdb")
        os.system(
            (
                f"sed -e \"s|ID|{pdb_id}|g\" "
                f"-e \"s|OUTPUTDIR|{output_path}|g\" "
                f"< run_dTERMen.sh "
                f" >{output_path}/run_{pdb_id}.sh"
            )
        )
        pdbs.append(pdb_id)

    with open(f"logfiles/{basename}.list", 'w') as fp:
        for pdb in pdbs:
            fp.write(pdb + "\n")

    batch_size = 20
    num_batches = len(pdbs)//batch_size + 1

    bid = os.popen(
        (
            f"sbatch --parsable --array=0-{num_batches} "
            f"{os.path.join(DIR, 'batch_arr_dTERMen.sh')} {output_path} logfiles/{basename}.list {batch_size}"
        )
    ).read()
    bid = int(bid.strip())
    os.system(f"sbatch --dependency=afterany:{bid} sum_res.sh {args.output_dir}")
