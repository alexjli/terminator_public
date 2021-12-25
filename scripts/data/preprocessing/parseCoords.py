import pickle
import numpy as np
from terminator.utils.common import aa_three_to_one


def parseCoords(filename, save = True):
    chain_dict = {}
    with open(filename, 'r') as fp:
        for line in fp:
            data = line.strip()
            if data == 'TER' or data == 'END':
                continue 
            try:
                element = data[13:16].strip()
                residue = data[17:20].strip()
                residx = data[22:27].strip()
                chain = data[21]
                x = data[30:38].strip()
                y = data[38:46].strip()
                z = data[46:54].strip()
                coords = [float(coord) for coord in [x,y,z]]
                #print(element, chain, coords)
            except Exception as e:
                print(data)
                raise e

            if chain not in chain_dict.keys():
                chain_dict[chain] = {
                    element: [] for element in ['N', 'CA', 'C', 'O']
                }
                chain_dict[chain]["seq_dict"] = {}

            # naively model terminal carboxylate as a single O atom 
            # (i cant find the two oxygens so im just gonna use OXT)
            if element == 'OXT':
                element = 'O'
            chain_dict[chain][element].append(coords)

            seq_dict = chain_dict[chain]["seq_dict"]
            if residx not in seq_dict.keys():
                if residue == 'MSE':  # convert MSE (seleno-met) to MET
                    residue = 'MET'
                elif residue == 'SEP':  # convert SEP (phospho-ser) to SER
                    residue = 'SER'
                elif residue == 'TPO':  # convert TPO (phospho-thr) to THR
                    residue = 'THR'
                elif residue == 'PTR':  # convert PTR (phospho-tyr) to TYR
                    residue = 'TYR' 
                elif residue == 'CSO':  # convert CSO (hydroxy-cys) to CYS
                    residue = 'CYS'
                elif residue == 'SEC':  # convert SEC (seleno-cys) to CYS
                    residue = 'CYS'
                seq_dict[residx] = aa_three_to_one(residue)

    chain_tensors = {}
    seq = ""
    for chain in chain_dict.keys():
        coords = [chain_dict[chain][element] for element in ['N', 'CA', 'C', 'O']]
        chain_tensors[chain] = np.stack(coords, 1)
        seq_dict = chain_dict[chain]["seq_dict"]
        s = "".join([seq_dict[i] for i in seq_dict.keys()])
        assert len([seq_dict[i] for i in seq_dict.keys()]) == chain_tensors[chain].shape[0], (s, chain_tensors[chain].shape, filename)
        seq += "".join([seq_dict[i] for i in seq_dict.keys()])

    if save:
        with open(filename[:-8] + '.coords', 'wb') as fp:
            pickle.dump(chain_tensors, fp)
        with open(filename[:-8] + '.seq', 'w') as fp:
            fp.write(seq)

    return chain_tensors, seq


def extractBackbone(filename, outpath, valid_elements = ['N', 'CA', 'C', 'O']):
    """
    Given a PDB structure, extract the protein backbone atoms and dump it in a redesigned PDB file
    """
    struct_dict = {}
    valid_entry_lines = []
    with open(filename, 'r') as fp:
        entry_lines = [l for l in fp]

    for line_num, line in enumerate(entry_lines):
        data = line.strip()
        if data == 'TER' or data == 'END':
            valid_entry_lines.append(line_num)
            continue
        try:
            element = data[13:16].strip()
            residue = data[17:20].strip()
            residx = data[22:27].strip()
            chain = data[21]
            x = data[30:38].strip()
            y = data[38:46].strip()
            z = data[46:54].strip()
            coords = [float(coord) for coord in [x,y,z]]
        except Exception as e:
            print(data)
            raise e

        if (chain, residx) not in struct_dict.keys():
            struct_dict[(chain, residx)] = {
                "elements": np.array([False for _ in range(5)]),
                "line_numbers": []
            }

        if element in valid_elements:
            struct_dict[(chain, residx)]["elements"][valid_elements.index(element)] = True
            struct_dict[(chain, residx)]["line_numbers"].append(line_num)
        elif element == 'OXT':
            struct_dict[(chain, residx)]["elements"][-1] = True
            struct_dict[(chain, residx)]["oxt_num"] = line_num


    for struct_vals in struct_dict.values():
        elem_arr = struct_vals["elements"]
        if elem_arr[:4].all():
            # if we have N, CA, C, O, we take those lines 
            # and ignore OXT even if present
            valid_entry_lines += struct_vals["line_numbers"]
        elif elem_arr[[0,1,2,4]].all() and not elem_arr[3]:
            # if we have N, CA, C, OXT, but no O
            # we take OXT as O
            assert len(struct_vals["line_numbers"]) == 3, struct_vals["line_numbers"]
            valid_entry_lines += struct_vals["line_numbers"]
            valid_entry_lines.append(struct_vals["oxt_num"])

    valid_entry_lines.sort()

    with open(outpath, 'w') as fp:
        for idx in range(len(valid_entry_lines)):
            cur_line_num = valid_entry_lines[idx]
            prev_line_num = valid_entry_lines[idx-1] if idx>0 else valid_entry_lines[0]
            cur_line = entry_lines[cur_line_num]
            prev_line = entry_lines[prev_line_num]
            if prev_line.strip() == 'TER' and cur_line.strip() == 'TER':
                # prevent redundant TER if we filter out a whole section
                continue
            fp.write(cur_line)

