#!/bin/bash

########################################################################################################################

# Path where data will be stored
REDUCED_MALARIA_PATH=${NF_HOMEDIR}/data/smiles/reduced_malaria

# Create output folder (if they do not exist)
mkdir -p ${REDUCED_MALARIA_PATH}

# Download preprocessed malaria dataset (link working as of 01.06.2017)
wget https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
awk -F"," 'NR!=1{ print $1 }' malaria-processed.csv > ${REDUCED_MALARIA_PATH}/smiles.dat
awk -F"," 'NR!=1{ print $2 }' malaria-processed.csv > ${REDUCED_MALARIA_PATH}/targets.dat

# Delete original file
rm malaria-processed.csv

########################################################################################################################

# Path where data will be stored
MALARIA_PATH=${NF_HOMEDIR}/data/smiles/malaria

# Create output folder (if they do not exist)
mkdir -p ${MALARIA_PATH}

# Download raw malaria dataset (link working as of 01.06.2017)
wget https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/raw_csv.csv

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
awk -F"," 'NR!=1{ print $5 }' raw_csv.csv > ${MALARIA_PATH}/smiles.dat
awk -F"," 'NR!=1{ print log($4) }' raw_csv.csv > ${MALARIA_PATH}/targets.dat

# Delete original file
rm raw_csv.csv