import glob
import os


def find_pdb_path(pdb, root):
    pattern = os.path.join(root, "*", "PDB", pdb[1:3].lower(), f"{pdb}.pdb")
    for pdb_path in glob.glob(pattern):
        if os.path.exists(pdb_path):
            return pdb_path
    raise ValueError(f"{pdb} not found in $RAW_DATA")


def find_dtermen_folder(pdb, root):
    for dataDir in glob.glob(os.path.join(root, "*")):
        pdb_folder = os.path.join(dataDir, pdb)
        if os.path.isdir(pdb_folder):
            return pdb_folder
    raise ValueError(f"{pdb} not found in $INPUT_DATA")
