import pickle
import numpy as np

def parseCoords(filename, save = True):
    chain_dict = {}
    with open(filename, 'r') as fp:
        for line in fp:
            data = line.strip()
            if data == 'TER' or data == 'END':
                continue 
            try:
                element = data[13:16].strip()
                chain = data[21]
                x = data[31:39].strip()
                y = data[39:47].strip()
                z = data[47:55].strip()
                coords = [float(coord) for coord in [x,y,z]]
                #print(element, chain, coords)
            except Exception as e:
                print(data)
                raise e

            if chain not in chain_dict.keys():
                chain_dict[chain] = {element: [] for element in ['N', 'CA', 'C', 'O']}

            chain_dict[chain][element].append(coords)

    chain_tensors = {}
    
    for chain in chain_dict.keys():
        coords = [chain_dict[chain][element] for element in ['N', 'CA', 'C', 'O']]
        chain_tensors[chain] = np.stack(coords, 1)

    if save:
        with open(filename[:-8] + '.coords', 'wb') as fp:
            pickle.dump(chain_tensors, fp)

    return chain_tensors
            


        
