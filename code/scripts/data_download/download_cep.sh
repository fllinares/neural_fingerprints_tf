#!/bin/bash

########################################################################################################################

# Path where data will be stored
CEP_PATH=${NF_HOMEDIR}/data/smiles/cep
CEP_NO_OUTLIERS_PATH=${NF_HOMEDIR}/data/smiles/cep_no_outliers

# Create output folders (if they do not exist)
mkdir -p ${CEP_PATH}
mkdir -p ${CEP_NO_OUTLIERS_PATH}

# Download complete CEP dataset (link working as of 01.06.2017)
wget https://github.com/HIPS/neural-fingerprint/raw/master/data/2015-06-02-cep-pce/data_cep.tar.gz

# Uncompress and delete original compressed file
tar -xvzf data_cep.tar.gz
rm data_cep.tar.gz

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
awk -F"," 'NR!=1{ print substr($1, 2, length($1)-2) }' data_tmp_moldata.csv > ${CEP_PATH}/smiles.dat
awk -F"," 'NR!=1{ print $2 }' data_tmp_moldata.csv > ${CEP_PATH}/targets.dat

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets,
# removing all samples whose target value is 0.0 (which appear to behave as outliers in the CEP dataset)
awk -F"," 'NR!=1{ if ($2 != 0.0) print substr($1, 2, length($1)-2) }' data_tmp_moldata.csv > ${CEP_NO_OUTLIERS_PATH}/smiles.dat
awk -F"," 'NR!=1{ if ($2 != 0.0) print $2 }' data_tmp_moldata.csv > ${CEP_NO_OUTLIERS_PATH}/targets.dat

# Delete original file
rm data_tmp_moldata.csv

########################################################################################################################

# Path where data will be stored
REDUCED_CEP_PATH=${NF_HOMEDIR}/data/smiles/reduced_cep
REDUCED_CEP_NO_OUTLIERS_PATH=${NF_HOMEDIR}/data/smiles/reduced_cep_no_outliers

# Create output folders (if they do not exist)
mkdir -p ${REDUCED_CEP_PATH}
mkdir -p ${REDUCED_CEP_NO_OUTLIERS_PATH}

# Download reduced CEP dataset (link working as of 01.06.2017)
wget https://github.com/HIPS/neural-fingerprint/raw/master/data/2015-06-02-cep-pce/cep-processed.csv

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets
awk -F"," 'NR!=1{ print $1 }' cep-processed.csv > ${REDUCED_CEP_PATH}/smiles.dat
awk -F"," 'NR!=1{ print $2 }' cep-processed.csv > ${REDUCED_CEP_PATH}/targets.dat

# Split comma-separated file into two files: one containing the SMILES strings and another containing the targets,
# removing all samples whose target value is 0.0 (which appear to behave as outliers in the CEP dataset)
awk -F"," 'NR!=1{ if ($2 != 0.0) print $1 }' cep-processed.csv > ${REDUCED_CEP_NO_OUTLIERS_PATH}/smiles.dat
awk -F"," 'NR!=1{ if ($2 != 0.0) print $2 }' cep-processed.csv > ${REDUCED_CEP_NO_OUTLIERS_PATH}/targets.dat

# Delete original file
rm cep-processed.csv
