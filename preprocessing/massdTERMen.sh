#!/bin/bash
MST_WORKSPACE=~/Documents/keating_lab/MST_workspace
MST_PATH=$MST_WORKSPACE/MST
DESIGN_BIN=$MST_PATH/bin/design
CONFIG=$MST_WORKSPACE/design.configfile
OUTPATH=$MST_WORKSPACE/dTERMen_data
PDB=$MST_WORKSPACE/mini_data/PDB

# create directory for outpath if it doesn't exist
if [ ! -d $OUTPATH ]; then
    mkdir $OUTPATH
fi
cd $PDB

# for all subdirectories in the PDB folder
for dir in $(ls -d */)
do
    # create subdirectory for outpath
    if [ ! -d $OUTPATH/$dir ]; then
        mkdir $OUTPATH/$dir
    fi

    cd $dir
    # run dTERMen on each pdb in folder
    for pdb in $(ls *.pdb)
    do
        name=$(basename $pdb .pdb)
        echo $name
        OUT=$OUTPATH/$dir/$name
        SECONDS=0
        $DESIGN_BIN --p $pdb --c $CONFIG --o $OUT --s 'chain A' --w > $OUT.log
        echo "$(($SECONDS / 60)) minutes elapsed" >> $OUT.log
        echo "$(($SECONDS / 60)) minutes elapsed"
    done
    cd ..
done
