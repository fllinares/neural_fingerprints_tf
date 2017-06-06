#!/bin/bash

# Path to Python script to preprocess data
SCRIPT_PATH=${NF_HOMEDIR}/code/data_preprocessing/smiles/smiles_to_tfrecord.py

########################################################################################################################

# Dataset name
DATASET_NAME=cep

# Input arguments to Python script
SMILES_FILE=${NF_HOMEDIR}/data/smiles/${DATASET_NAME}/smiles.dat
OUTPUT_FILE=${NF_HOMEDIR}/data/tfrecords/${DATASET_NAME}/all.tfrecords
CONFIG_FILE=${NF_HOMEDIR}/code/data_preprocessing/smiles/smiles_parser_config/config_0.json
TARGETS_FILE=${NF_HOMEDIR}/data/smiles/${DATASET_NAME}/targets.dat

python ${SCRIPT_PATH} ${SMILES_FILE} ${OUTPUT_FILE} ${CONFIG_FILE} --targets_file ${TARGETS_FILE}

########################################################################################################################

# Dataset name
DATASET_NAME=cep_no_outliers

# Input arguments to Python script
SMILES_FILE=${NF_HOMEDIR}/data/smiles/${DATASET_NAME}/smiles.dat
OUTPUT_FILE=${NF_HOMEDIR}/data/tfrecords/${DATASET_NAME}/all.tfrecords
CONFIG_FILE=${NF_HOMEDIR}/code/data_preprocessing/smiles/smiles_parser_config/config_0.json
TARGETS_FILE=${NF_HOMEDIR}/data/smiles/${DATASET_NAME}/targets.dat

python ${SCRIPT_PATH} ${SMILES_FILE} ${OUTPUT_FILE} ${CONFIG_FILE} --targets_file ${TARGETS_FILE}