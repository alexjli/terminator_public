import os
import glob


def find_pdb_folder(pdb, root):
    pattern = os.path.join(
        root,
        "*",
        "PDB",
        pdb[1:3].lower(),
        pdb
    )
    for dataDir in glob.glob(pattern):
        pdb_folder = os.path.join(dataDir, pdb)
        if os.path.isdir(pdb_folder):
            return pdb_folder
    raise InvalidArgumentException(f"{pdb} not found in $RAW_DATA")


def find_dtermen_folder(pdb, root):
    for dataDir in glob.glob(os.path.join(root, "*")):
        pdb_folder = os.path.join(dataDir, pdb)
        if os.path.isdir(pdb_folder):
            return pdb_folder
    raise InvalidArgumentException(f"{pdb} not found in $INPUT_DATA")
