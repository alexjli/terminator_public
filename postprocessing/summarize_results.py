import os
import argparse
import glob
import pandas as pd
import traceback

from filepaths import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse all results.')
    parser.add_argument('--output_dir', help = 'Output directory', default = 'test_run')
    args = parser.parse_args()

    output_path = os.path.join(OUTPUT_DIR, args.output_dir, 'etabs')
    p0 = '/scratch/users/alexjli/TERMinator/fixed_dTERMen/'
    p1 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1/')
    p2 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1_p2/')
    p3 = os.path.join(INPUT_DATA, 'dTERMen_speedtest200_clique1_p3/')
    p4 = os.path.join(INPUT_DATA, 'monomer_DB_1/')
    p5 = os.path.join(INPUT_DATA, 'monomer_DB_2/')
    p6 = os.path.join(INPUT_DATA, 'monomer_DB_3/')
    p7 = os.path.join(INPUT_DATA, 'seq_id_50_resid_500/')
    p8 = os.path.join(INPUT_DATA, 'ingraham_data/')
    p = [p0, p1, p2, p3, p4, p5, p6, p7, p8]
    # p = [p1, p2, p3, p4, p5, p6]

    errors = []

    ids = []
    pred_sequences = []
    real_sequences = []
    dtermen_pred_sequences = []
    recovery = []
    dtermen_recovery = []
    for filename in glob.glob(os.path.join(output_path, '*-output.out')):
        pdb_id = os.path.basename(filename)[:-len('-output.out')]#filename[-len('-output.out')-4:-len('-output.out')]
        #pdb_id = filename[-len('-output.out')-6:-len('-output.out')]
        ids += [pdb_id]
        print(pdb_id)
        try:
            with open(filename, 'r') as f:
                f.readline()
                f.readline()
                pred_sequences += [f.readline().split('|')[0]]
                real_sequences += [f.readline().split('|')[0]]
                recovery += [float(f.readline().split('|')[0][:-2])]
        except Exception as e:
            traceback.print_exc()
            errors.append(pdb_id)
            continue

        last_testfolder = None
        for testfolder in p:
            already_found = False
            already_found_recov = False
            if os.path.isdir(os.path.join(testfolder, pdb_id)):
                last_testfolder = testfolder
                if os.path.isfile(os.path.join(testfolder, pdb_id, 'design.oFIXED')):
                    fixed_file = os.path.join(testfolder, pdb_id, 'design.oFIXED')
                    with open(fixed_file, 'r') as f:
                        for line in f:
                            linesplit = line.split('|')
                            if len(linesplit) >= 3:
                                if linesplit[2] == ' lowest-energy sequence\n' and not already_found:
                                    dtermen_pred_sequences += [linesplit[0]]
                                    already_found = True
                                elif len(linesplit) >= 4 and linesplit[3] == ' recovery\n' and not already_found_recov:
                                    dtermen_recovery += [float(linesplit[0][:-2])]
                                    already_found_recov = True
                                    break
 
                for dtermen_filename in glob.glob(os.path.join(testfolder, pdb_id, 'design.o*')):
                    with open(dtermen_filename, 'r') as f:
                        for line in f:
                            linesplit = line.split('|')
                            if len(linesplit) >= 3:
                                #if linesplit[2] == ' lowest-energy sequence\n' and not already_found:
                                #    dtermen_pred_sequences += [linesplit[0]]
                                #    already_found = True
                                if len(linesplit) >= 4 and linesplit[3] == ' recovery\n' and not already_found_recov:
                                    dtermen_recovery += [float(linesplit[0][:-2])]
                                    already_found_recov = True
                                    break
            if already_found_recov:
                break
        
        if len(dtermen_pred_sequences) < len(pred_sequences):
            dtermen_pred_sequences += ['UNKNOWN']
        if len(dtermen_recovery) < len(pred_sequences):
            dtermen_recovery += [-1]
            with open('to_run.out', 'a') as f:
                f.write(f'{last_testfolder}{pdb_id}\n')

    print("Errors:", errors)
    os.chdir(output_path)
    for e in errors:
        sbatch_file = os.path.join(output_path, f"run_{e}.sh")
        os.system(f"sbatch {sbatch_file}")

    results_dict = {'ids': ids, 'pred_sequences': pred_sequences, 'real_sequences': real_sequences, 'dtermen_pred_sequences': dtermen_pred_sequences, 'recovery': recovery, 'dtermen_recovery': dtermen_recovery}
    print([(key, len(val)) for key, val in results_dict.items()])
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(OUTPUT_DIR, args.output_dir, 'summary_results.csv'))
