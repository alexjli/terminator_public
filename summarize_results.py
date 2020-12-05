import os
import argparse
import glob
import pandas as pd

INPUT_DATA = '/scratch/users/alexjli/TERMinator'
OUTPUT_DIR = '/scratch/users/vsundar/TERMinator/outputs/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse all results.')
    parser.add_argument('--output_dir', help = 'Output directory', default = 'test_run')
    args = parser.parse_args()

    output_path = os.path.join(OUTPUT_DIR, args.output_dir, 'etabs')
    # p0 = '/scratch/users/vsundar/TERMinator/fixed_dTERMen/'
    p1 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1/')
    p2 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1_p2/')
    p3 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1_p3/')
    p4 = os.path.join(INPUT_DATA, 'monomer_DB_1/')
    p5 = os.path.join(INPUT_DATA, 'monomer_DB_2/')
    p6 = os.path.join(INPUT_DATA, 'monomer_DB_3/')
    # p = [p0, p1, p2, p3, p4, p5, p6]
    p = [p1, p2, p3, p4, p5, p6]
    ids = []
    pred_sequences = []
    real_sequences = []
    dtermen_pred_sequences = []
    recovery = []
    dtermen_recovery = []
    for filename in glob.glob(os.path.join(output_path, '*-output.out')):
        pdb_id = filename[-len('-output.out')-4:-len('-output.out')]
        ids += [pdb_id]
        with open(filename, 'r') as f:
            f.readline()
            f.readline()
            pred_sequences += [f.readline().split('|')[0]]
            real_sequences += [f.readline().split('|')[0]]
            recovery += [float(f.readline().split('|')[0][:-2])]

        for testfolder in p:
            if os.path.isdir(os.path.join(testfolder, pdb_id)):
                already_found = False
                for dtermen_filename in glob.glob(os.path.join(testfolder, pdb_id, 'design.o*')):
                    with open(dtermen_filename, 'r') as f:
                        for line in f:
                            linesplit = line.split('|')
                            if len(linesplit) >= 3:
                                if linesplit[2] == ' lowest-energy sequence\n' and not already_found:
                                    dtermen_pred_sequences += [linesplit[0]]
                                    already_found = True
                                elif len(linesplit) >= 4 and linesplit[3] == ' recovery\n':
                                    dtermen_recovery += [float(linesplit[0][:-2])]
                                    break
                break
        
        if len(dtermen_pred_sequences) < len(pred_sequences):
            dtermen_pred_sequences += ['UNKNOWN']
        if len(dtermen_recovery) < len(pred_sequences):
            dtermen_recovery += [-1]
            with open('to_run.out', 'a') as f:
                f.write(f'{testfolder}{pdb_id}\n')

    results_dict = {'ids': ids, 'pred_sequences': pred_sequences, 'real_sequences': real_sequences, 'dtermen_pred_sequences': dtermen_pred_sequences, 'recovery': recovery, 'dtermen_recovery': dtermen_recovery}
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(OUTPUT_DIR, args.output_dir, 'summary_results.csv'))
