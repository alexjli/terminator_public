#!/bin/bash

sed -e "s/OUTPUTDIR/test_run_fold0_$1/g" </home/alexjli/TERMinator/postprocessing/to_etab.sh >/home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold0_$1.sh
sbatch /home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold0_$1.sh

sed -e "s/OUTPUTDIR/test_run_fold2_$1/g" </home/alexjli/TERMinator/postprocessing/to_etab.sh >/home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold2_$1.sh
sbatch /home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold2_$1.sh

sed -e "s/OUTPUTDIR/test_run_fold4_$1/g" </home/alexjli/TERMinator/postprocessing/to_etab.sh >/home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold4_$1.sh
sbatch /home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold4_$1.sh

sed -e "s/OUTPUTDIR/test_run_fold6_$1/g" </home/alexjli/TERMinator/postprocessing/to_etab.sh >/home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold6_$1.sh
sbatch /home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold6_$1.sh

sed -e "s/OUTPUTDIR/test_run_fold8_$1/g" </home/alexjli/TERMinator/postprocessing/to_etab.sh >/home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold8_$1.sh
sbatch /home/alexjli/TERMinator/postprocessing/shell_scripts/to_etab_test_run_fold8_$1.sh
