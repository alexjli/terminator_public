import os
import argparse
import glob

from filepaths import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run dTERMen for testing.')
    parser.add_argument('--output_dir', help = 'Output directory', default = 'test_run')
    args = parser.parse_args()

    pdbs = []
    if 'ingraham' in args.output_dir or 'h128':
        PDB_PATH = INGRAHAM_PDB_PATH

    output_path = os.path.join(OUTPUT_DIR, args.output_dir, 'etabs')

    for filename in glob.glob(os.path.join(output_path, '*.etab')):
        pdb_id = os.path.basename(filename)[:-5] #filename[-9:-5] #filename[-11:-5]
        print(pdb_id)
        #os.system(f"cp {PDB_PATH}{pdb_id.lower()[1:3]}/{pdb_id}.pdb {output_path}/{pdb_id}.pdb")
        os.system(f"cp {PDB_PATH}{pdb_id[1:3].lower()}/{pdb_id}.pdb {output_path}/{pdb_id}.pdb")
        os.system(f"sed -e \"s/ID/{pdb_id}/g\" -e 's/OUTPUTDIR/{args.output_dir}/g' </home/alexjli/TERMinator/postprocessing/run_dTERMen.sh >{output_path}/run_{pdb_id}.sh")
        pdbs.append(pdb_id)
    
    with open(f"logfiles/{args.output_dir}.list", 'w') as fp:
        for pdb in pdbs:
            fp.write(pdb + "\n")

    bid = os.popen(f"sbatch --parsable --array=0-{len(pdbs)-1} batch_arr_dTERMen.sh {output_path} logfiles/{args.output_dir}.list").read()
    bid = int(bid.strip())
    os.system(f"sbatch --dependency=afterany:{bid} sum_res.sh {args.output_dir}")

