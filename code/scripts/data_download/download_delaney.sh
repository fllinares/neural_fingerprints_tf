#!/bin/bash

########################################################################################################################

# Path where data will be stored
DELANEY_PATH=${NF_HOMEDIR}/data/smiles/delaney

# Create output folder (if they do not exist)
mkdir -p ${DELANEY_PATH}

# Download preprocessed delaney dataset (link working as of 01.06.2017)
wget https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-05-24-delaney/delaney-processed.csv

# Remove commas inside quotes
awk -F'"' -v OFS='' '{ for (i=2; i<=NF; i+=2) gsub(",", "", $i) } 1' delaney-processed.csv | cat > delaney-processed2.csv
# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
awk -F"," 'NR!=1{ print $10 }' delaney-processed2.csv > ${DELANEY_PATH}/smiles.dat
awk -F"," 'NR!=1{ print $9 }' delaney-processed2.csv > ${DELANEY_PATH}/targets.dat

# Delete original file
rm delaney-processed.csv
rm delaney-processed2.csv