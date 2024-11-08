#!/bin/sh
#$ -cwd
#$ -l h_rt=23:00:00
#$ -N TRAINING_MODEL
#$ -o eo_file_train
#$ -e eo_file_train

echo "--------Copying dataset..--------"
rsync -avht --progress ../hdf5_graph_files/seed-1/train_graph_data.hdf5 $TMPDIR
rsync -avht --progress ../input_label_files/pocket_rmsd_label_seed-1.csv $TMPDIR
echo "---------Dataset copied!---------"

# 計算資源の設定
# NPERNODE=4  # f_node
# NPERNODE=2  # h_node
NPERNODE=1  # q_node　各ノード上で1つのプロセス
NNODES=1  # ノード数:1 
NP=$((NPERNODE*NNODES)) # プロセスの総数

#変数の設定
GRAPH_PATH="${TMPDIR}../hdf5_graph_files/seed-1/train_graph_data.hdf5"
LABEL_PATH="${TMPDIR}../input_label_files/pocket_rmsd_label_seed-1.csv"
SAVE_DIR="trained_model_checkpoints"

VERSION_NAME="(3)_hc-180_l-3" # 新しいモデルの学習の際はここも変更する

RUN_ID="version_3"
PROJECT_NAME="train_GCNN_model_0704~"

EPOCHS=500
INPUT_DIM=22
PROCESSES=$((NPERNODE*2))

BS=64
TEST_BS=64
BATCH_SIZE=$((BS*NP))
TEST_BATCH_SIZE=$((TEST_BS*NP))

NUM_LAYERS=9
HIDDEN_CHANNELS=34
DROPOUT=0.63419655480525
LEARNING_RATE=0.0000383599741810468
POOLING_TYPE="mean"

echo "======execute below command======"
#ジョブの実行
CMD="python train_GCNN_model.py" # 実行するメインのソースファイル
CMD="${CMD} --graph-path ${GRAPH_PATH}"
CMD="${CMD} --label-path ${LABEL_PATH}"
CMD="${CMD} --save-dir ${SAVE_DIR}"
CMD="${CMD} --version-name ${VERSION_NAME}"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --lr ${LEARNING_RATE}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --test-batch-size ${TEST_BATCH_SIZE}"
CMD="${CMD} --project-name ${PROJECT_NAME}"
CMD="${CMD} --resume --run-id ${RUN_ID}"
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