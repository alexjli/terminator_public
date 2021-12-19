import os
import argparse
import glob

from search_utils import find_pdb_path

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
    parser.add_argument('--dtermen_data', help="Root directory for dTERMen runs")
    parser.add_argument(
        '--batch_size', 
        help='number of dTERMen runs to run per node',
        default=5,
        type=int
    )
    args = parser.parse_args()
    os.chdir(DIR)

    pdbs = []
    output_path = os.path.join(args.output_dir, 'etabs')
    basename = os.path.basename(args.output_dir)

    for filename in glob.glob(os.path.join(output_path, '*.etab')):
        pdb_id = os.path.basename(filename)[:-5]
        print(pdb_id)

        pdb_path = find_pdb_path(pdb_id, args.pdb_root)
        os.system(f"cp {pdb_path} {output_path}/{pdb_id}.pdb")
        os.system(
            (
                f"sed -e \"s|ID|{pdb_id}|g\" "
                f"-e \"s|OUTPUTDIR|{output_path}|g\" "
                f"-e \"s|POSTDIR|{DIR}|g\" "
                f"< run_dTERMen.sh "
                f" >{output_path}/run_{pdb_id}.sh"
            )
        )
        pdbs.append(pdb_id)

    batch_arr_list = os.path.join(args.output_dir, f"{basename}_batch_arr.list")

    with open(batch_arr_list, 'w') as fp:
        for pdb in pdbs:
            fp.write(pdb + "\n")

    num_batches = len(pdbs)//args.batch_size + 1

    bid = os.popen(
        (
            f"sbatch --parsable --array=0-{num_batches} "
            f"{os.path.join(DIR, 'batch_arr_dTERMen.sh')} {output_path} {batch_arr_list} {args.batch_size}"
        )
    ).read()
    bid = int(bid.strip())
    os.system(f"sbatch --dependency=afterany:{bid} sum_res.sh {args.output_dir} {args.dtermen_data}")
