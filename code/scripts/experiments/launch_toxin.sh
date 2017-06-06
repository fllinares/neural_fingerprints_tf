#!/bin/bash

DATASET_NAME=toxin


SCRIPT_PATH=${NF_HOMEDIR}/run_neural_fingerprints.py
TRAIN_PATH=${NF_HOMEDIR}/data/batched_tfrecords/${DATASET_NAME}/batches_tr.tfrecords
OUT_DIR=${NF_HOMEDIR}/output/${DATASET_NAME}
TRAIN_LOSS=sigmoid
EVAL_LOSSES="AUC 0/1"
VAL_PATH=${NF_HOMEDIR}/data/batched_tfrecords/${DATASET_NAME}/batches_val.tfrecords
TST_PATH=${NF_HOMEDIR}/data/batched_tfrecords/${DATASET_NAME}/batches_tst.tfrecords
NUM_EPOCHS=250
CONFIG_PATH=${NF_HOMEDIR}/model_configs/${DATASET_NAME}/config_7.json
SEED=10

python ${SCRIPT_PATH} ${TRAIN_PATH} ${OUT_DIR} ${TRAIN_LOSS} ${EVAL_LOSSES} --val_path ${VAL_PATH} \
                                                                            --tst_path ${TST_PATH} \
                                                                            --num_epochs ${NUM_EPOCHS} \
                                                                            --config_path ${CONFIG_PATH} \
                                                                            --seed ${SEED}