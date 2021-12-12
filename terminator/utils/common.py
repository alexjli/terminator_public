# zero is used as padding
AA_to_int = {
'A' : 1, 'ALA' : 1,
'C' : 2, 'CYS' : 2,
'D' : 3, 'ASP' : 3,
'E' : 4, 'GLU' : 4,
'F' : 5, 'PHE' : 5,
'G' : 6, 'GLY' : 6,
'H' : 7, 'HIS' : 7,
'I' : 8, 'ILE' : 8,
'K' : 9, 'LYS' : 9,
'L' : 10, 'LEU' : 10,
'M' : 11, 'MET' : 11,
'N' : 12, 'ASN' : 12,
'P' : 13, 'PRO' : 13,
'Q' : 14, 'GLN' : 14,
'R' : 15, 'ARG' : 15,
'S' : 16, 'SER' : 16,
'T' : 17, 'THR' : 17,
'V' : 18, 'VAL' : 18,
'W' : 19, 'TRP' : 19,
'Y' : 20, 'TYR' : 20,
'X' : 21
}


AA_to_int = {key: val-1 for key, val in AA_to_int.items()}

int_to_AA = {y:x for x,y in AA_to_int.items() if len(x) == 1}

"""
Given a string of one-letter encoded AAs, return its corresponding integer encoding
"""
def seq_to_ints(sequence):
    return [AA_to_int[residue] for residue in sequence]

# wrapper for AA_to_int
def aa_to_int(residue):
    return AA_to_int[residue]

# wrapper for AA_to_int
def int_to_aa(i):
    return int_to_AA[i]


