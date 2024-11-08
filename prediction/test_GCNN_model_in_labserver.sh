#!/bin/bash
#$ -cwd
#$ -l h_rt=01:00:00
#$ -N TRAINING
#$ -o eo_file_pred
#$ -e eo_file_pred

echo "--------Copying dataset..--------"
rsync -avht --progress ../hdf5_graph_files/seed-1/train_graph_data.hdf5 $TMPDIR
rsync -avht --progress ../hdf5_graph_files/seed-1/test_graph_data.hdf5 $TMPDIR
echo "---------Dataset copied!---------"

NPERNODE=1
NODES=1
NP=$((NPERNODE*NODES))

TRAIN_GRAPH_PATH="${TMPDIR}../hdf5_graph_files/seed-1/train_graph_data.hdf5"
TEST_GRAPH_PATH="${TMPDIR}../hdf5_graph_files/seed-1/test_graph_data.hdf5"
LABEL_PATH="../input_label_files/pocket_rmsd_label_seed-1_sqrt.csv"

VERSION_NAME="(2)_label_sqrt" # テストするモデルをここで設定する

#CHECKPOINT_DIR="trained_model_checkpoints"
CHECKPOINT_DIR="train_scripts/trained_model_checkpoints"

RESULT_DIR="../results"

INPUT_DIM=22

BS=64
BATCH_SIZE=$((BS*NP))
NUM_LAYERS=6
HIDDEN_CHANNELS=180
DROPOUT=0.2
POOLING_TYPE="mean"


echo "======execute below command======"
echo "<<1. graph file: train | data type: train>>"
CMD="python test_GCNN_model.py"
CMD="${CMD} --graph-path ${TRAIN_GRAPH_PATH}"
CMD="${CMD} --label-path ${LABEL_PATH}"
CMD="${CMD} --result-dir ${RESULT_DIR}"
CMD="${CMD} --checkpoint-dir ${CHECKPOINT_DIR}"
CMD="${CMD} --version-name ${VERSION_NAME}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --data-type train"
CMD="${CMD} --input-dim ${INPUT_DIM}"
CMD="${CMD} --num-layers ${NUM_LAYERS}"
CMD="${CMD} --hidden-channels ${HIDDEN_CHANNELS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --use-batch-norm"
CMD="${CMD} --pooling-type ${POOLING_TYPE}"
echo -e $CMD | sed 's/ --/\n--/g'
echo "================================="
echo "↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"
$CMD

echo "======execute below command======"
echo "<<2. graph file: train | data type: validation>>"
CMD="python test_GCNN_model.py"
CMD="${CMD} --graph-path ${TRAIN_GRAPH_PATH}"
CMD="${CMD} --label-path ${LABEL_PATH}"
CMD="${CMD} --result-dir ${RESULT_DIR}"
CMD="${CMD} --checkpoint-dir ${CHECKPOINT_DIR}"
CMD="${CMD} --version-name ${VERSION_NAME}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --data-type validation"
CMD="${CMD} --input-dim ${INPUT_DIM}"
CMD="${CMD} --num-layers ${NUM_LAYERS}"
CMD="${CMD} --hidden-channels ${HIDDEN_CHANNELS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --use-batch-norm"
CMD="${CMD} --pooling-type ${POOLING_TYPE}"
echo -e $CMD | sed 's/ --/\n--/g'
echo "================================="
echo "↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"
$CMD

# test用のhdf5ファイルでテスト
echo "======execute below command======"
echo "<<3. graph file: test | data type: test>>"
CMD="python test_GCNN_model.py"
CMD="${CMD} --graph-path ${TEST_GRAPH_PATH}"
CMD="${CMD} --label-path ${LABEL_PATH}"
CMD="${CMD} --result-dir ${RESULT_DIR}"
CMD="${CMD} --checkpoint-dir ${CHECKPOINT_DIR}"
CMD="${CMD} --version-name ${VERSION_NAME}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --data-type test"
CMD="${CMD} --input-dim ${INPUT_DIM}"
CMD="${CMD} --num-layers ${NUM_LAYERS}"
CMD="${CMD} --hidden-channels ${HIDDEN_CHANNELS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --use-batch-norm"
CMD="${CMD} --pooling-type ${POOLING_TYPE}"
echo -e $CMD | sed 's/ --/\n--/g'
echo "================================="
echo "↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓"
$CMD
