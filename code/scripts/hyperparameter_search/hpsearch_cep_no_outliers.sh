#!/bin/bash

DATASET_NAME=cep_no_outliers


SCRIPT_PATH=${NF_HOMEDIR}/run_neural_fingerprints.py
TRAIN_PATH=${NF_HOMEDIR}/data/batched_tfrecords/${DATASET_NAME}/batches_tr.tfrecords
TRAIN_LOSS=MSE
EVAL_LOSSES="MSE Corr"
VAL_PATH=${NF_HOMEDIR}/data/batched_tfrecords/${DATASET_NAME}/batches_val.tfrecords
TST_PATH=${NF_HOMEDIR}/data/batched_tfrecords/${DATASET_NAME}/batches_tst.tfrecords
NUM_EPOCHS=250
CONFIG_PATH=${NF_HOMEDIR}/model_configs/${DATASET_NAME}
SEED=10

for f in ${CONFIG_PATH}/*.json;
do
    # Extract configuration filename without path or extension
    fname=$(basename "$f")
    fname="${fname%.*}"

    # Output directory for results using this hyperparameter configuration
    OUT_DIR=${NF_HOMEDIR}/output/hyperparameter_search/${DATASET_NAME}/${fname}

    # Train model with the configuration specified by file $f
    python ${SCRIPT_PATH} ${TRAIN_PATH} ${OUT_DIR} ${TRAIN_LOSS} ${EVAL_LOSSES} --val_path ${VAL_PATH} \
                                                                                --tst_path ${TST_PATH} \
                                                                                --num_epochs ${NUM_EPOCHS} \
                                                                                --config_path ${f} \
                                                                                --seed ${SEED}
done