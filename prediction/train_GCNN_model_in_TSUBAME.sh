#!/bin/bash
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=23:00:00
#$ -N TRAINING_MODEL
#$ -o eo_file_train
#$ -e eo_file_train

. /etc/profile.d/modules.sh
echo "this host is" `hostname` "."
module load gaussian16/B01

module load cuda/12.1.0
module load openmpi/3.1.4-opa10.10
module load nccl/2.12.7
module load cudnn/8.3
# module load gcc/8.3.0
module load gcc
module load cmake/3.21.3
module list

# export NCCL_BUFFSIZE=1048576
export NCCL_IB_DISABLE=1
# export NCCL_IB_TIMEOUT=14

echo "Copying dataset..."
# rsyncコマンドを使用してデータセットを一時ディレクトリ ($TMPDIR) にコピー(実行の効率化のため)
# rsync -avht --progress ../../data/pdbbind/graph/train_graph_apo_pocket_family.hdf5 $TMPDIR
rsync -avht --progress ../fdh5_graph_files/train_graph_apo_pocket_binary_15_apo.hdf5 $TMPDIR
rsync -avht --progress ../input_csv_files/pocket_rmsd_label_2.csv $TMPDIR
echo "Dataset copied!"

# 計算資源の設定
# NPERNODE=4  # f_node
# NPERNODE=2  # h_node
NPERNODE=1  # q_node　各ノード上で1つのプロセス
NNODES=1  # ノード数:1 
NP=$((NPERNODE*NNODES)) # プロセスの総数

#変数の設定
GRAPH_PATH="${TMPDIR}/train_graph_apo_pocket_binary_15_apo.hdf5"
LABEL_PATH="${TMPDIR}/pocket_rmsd_label.csv"
SAVE_DIR="output/graph_apo"
EPOCHS=300
PROCESSES=$((NPERNODE*2))
LEARNING_RATE=0.0001 #元：0.0001
BS=256
TEST_BS=256
BATCH_SIZE=$((BS*NP))
TEST_BATCH_SIZE=$((TEST_BS*NP))
RUN_ID="predict_rmsd_from_graph_gcnn_0"
PROJECT_NAME="Softplus_Adam"
INPUT_DIM=22
NUM_LAYERS=5
HIDDEN_CHANNELS=64
DROPOUT=0.2
POOLING_TYPE="mean"

#ジョブの実行
CMD="python train_GCNN_model.py" # 実行するメインのソースファイル
CMD="${CMD} --graph-path ${GRAPH_PATH}"
CMD="${CMD} --label-path ${LABEL_PATH}"
CMD="${CMD} --save-dir ${SAVE_DIR}"
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

echo $CMD
$CMD
