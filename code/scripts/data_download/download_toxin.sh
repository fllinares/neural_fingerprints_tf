#!/bin/bash

########################################################################################################################

# Path where data will be stored
TOXIN_PATH=${NF_HOMEDIR}/data/smiles/toxin

# Create output folder (if they do not exist)
mkdir -p ${TOXIN_PATH}

# Download preprocessed toxin dataset (link working as of 01.06.2017)
wget https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-05-22-tox/sr-mmp.smiles-processed.csv

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
awk -F"," 'NR!=1{ print $9 }' sr-mmp.smiles-processed.csv > ${TOXIN_PATH}/smiles.dat
awk -F"," 'NR!=1{ print (1 - $10) "," $10 }' sr-mmp.smiles-processed.csv > ${TOXIN_PATH}/targets.dat

# Delete original file
rm sr-mmp.smiles-processed.csv