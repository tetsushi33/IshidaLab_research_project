#!/bin/sh
#$ -cwd
#$ -l h_rt=23:00:00
#$ -N TRAINING_MODEL
#$ -o eo_file_train
#$ -e eo_file_train

<<COMMENT
酒井さんの最終的なモデルとの単純な比較
分類を回帰に変換し、データセットはそのまま
ハイパーパラメータは修士論文のものそのまま
COMMENT

echo "--------Copying dataset..--------"
rsync -avht --progress ../../hdf5_graph_files/train_graph_apo_pocket_binary_15_apo.hdf5 $TMPDIR
rsync -avht --progress ../../input_label_files/pocket_rmsd_label_2.csv $TMPDIR
echo "---------Dataset copied!---------"

# 計算資源の設定
# NPERNODE=4  # f_node
# NPERNODE=2  # h_node
NPERNODE=1  # q_node　各ノード上で1つのプロセス
NNODES=1  # ノード数:1 
NP=$((NPERNODE*NNODES)) # プロセスの総数

#変数の設定
GRAPH_PATH="${TMPDIR}../../hdf5_graph_files/train_graph_apo_pocket_binary_15_apo.hdf5"
LABEL_PATH="${TMPDIR}../../input_label_files/pocket_rmsd_label_2.csv"
SAVE_DIR="trained_model_checkpoints"

VERSION_NAME="(1)_01_just_to_regression" # 新しいモデルの学習の際はここも変更する

RUN_ID="(1)"
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
CMD="python ../train_GCNN_model.py" # 実行するメインのソースファイル
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