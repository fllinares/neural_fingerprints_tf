#!/bin/bash

# Path to Python script to preprocess data
SCRIPT_PATH=${NF_HOMEDIR}/code/data_preprocessing/batch_tfrecords.py

########################################################################################################################

# Dataset name
DATASET_NAME=malaria

# Input arguments to Python script
TFRECORDS_FILE=${NF_HOMEDIR}/data/tfrecords/${DATASET_NAME}/all.tfrecords
TFRECORDS_PATH=${NF_HOMEDIR}/data/batched_tfrecords/${DATASET_NAME}
BATCH_SIZE=32
SPLIT_PROPORTIONS="0.8 0.1 0.1"
SEED=0

python ${SCRIPT_PATH} ${TFRECORDS_FILE} ${TFRECORDS_PATH} ${BATCH_SIZE} --split_proportions ${SPLIT_PROPORTIONS} \
                                                                        --seed ${SEED}

########################################################################################################################

# Dataset name
DATASET_NAME=reduced_malaria

# Input arguments to Python script
TFRECORDS_FILE=${NF_HOMEDIR}/data/tfrecords/${DATASET_NAME}/all.tfrecords
TFRECORDS_PATH=${NF_HOMEDIR}/data/batched_tfrecords/${DATASET_NAME}
BATCH_SIZE=32
SPLIT_PROPORTIONS="0.8 0.1 0.1"
SEED=0

python ${SCRIPT_PATH} ${TFRECORDS_FILE} ${TFRECORDS_PATH} ${BATCH_SIZE} --split_proportions ${SPLIT_PROPORTIONS} \
                                                                        --seed ${SEED}