#!/bin/bash
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=01:00:00
#$ -N TRAINING
#$ -o eo_file_pred
#$ -e eo_file_pred
#$ -v GPU_COMPUTE_MODE=1

. /etc/profile.d/modules.sh
#source /home/7/22M30942/.bashrc

module load cuda openmpi
module load cuda/12.1.0
module load openmpi
module load nccl/2.12.7
module load cudnn/8.3
module load gcc
module load cmake/3.21.3
module list

# export NCCL_BUFFSIZE=1048576
export NCCL_IB_DISABLE=1
# export NCCL_IB_TIMEOUT=14

echo "Copying dataset..."
rsync -avht --progress ../hdf5_graph_files/train_graph_apo_pocket_binary_15_apo.hdf5 $TMPDIR
rsync -avht --progress ../hdf5_graph_files/test_graph_apo_pocket_binary_15_apo.hdf5 $TMPDIR
echo "Dataset copied!"

NPERNODE=1
NODES=1
NP=$((NPERNODE*NODES))

TRAIN_GRAPH_PATH="${TMPDIR}../hdf5_graph_files/train_graph_apo_pocket_binary_15_apo.hdf5"
TEST_GRAPH_PATH="${TMPDIR}../hdf5_graph_files/test_graph_apo_pocket_binary_15_apo.hdf5"
LABEL_PATH="../input_csv_files/pocket_rmsd_label_2.csv"

CHECKPOINT_DIR="trained_model_checkpoints"

RESULT_DIR="../results"

BS=32
BATCH_SIZE=$((BS*NP))
INPUT_DIM=22
NUM_LAYERS=7
HIDDEN_CHANNELS=298
DROPOUT=0.11956168977992454
POOLING_TYPE="mean"

CMD="python predict_model_graph_GCNN.py"
CMD="${CMD} --graph-path ${TRAIN_GRAPH_PATH}"
CMD="${CMD} --label-path ${LABEL_PATH}"
CMD="${CMD} --result-dir ${RESULT_DIR}"
CMD="${CMD} --checkpoint-dir ${CHECKPOINT_DIR}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --data-type train"
CMD="${CMD} --input-dim ${INPUT_DIM}"
CMD="${CMD} --num-layers ${NUM_LAYERS}"
CMD="${CMD} --hidden-channels ${HIDDEN_CHANNELS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --use-batch-norm"
CMD="${CMD} --pooling-type ${POOLING_TYPE}"

echo $CMD
$CMD
CMD="python test_GCNN_model.py"
CMD="${CMD} --graph-path ${TRAIN_GRAPH_PATH}"
CMD="${CMD} --label-path ${LABEL_PATH}"
CMD="${CMD} --result-dir ${RESULT_DIR}"
CMD="${CMD} --checkpoint-path ${CHECKPOINT_PATH}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --data-type validate"
CMD="${CMD} --input-dim ${INPUT_DIM}"
CMD="${CMD} --num-layers ${NUM_LAYERS}"
CMD="${CMD} --hidden-channels ${HIDDEN_CHANNELS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --use-batch-norm"
CMD="${CMD} --pooling-type ${POOLING_TYPE}"

echo $CMD
$CMD

CMD="python predict_model_graph_GCNN.py"
CMD="${CMD} --graph-path ${TEST_GRAPH_PATH}"
CMD="${CMD} --label-path ${LABEL_PATH}"
CMD="${CMD} --result-dir ${RESULT_DIR}"
CMD="${CMD} --checkpoint-path ${CHECKPOINT_PATH}"
CMD="${CMD} --batch-size ${BATCH_SIZE}"
CMD="${CMD} --data-type test"
CMD="${CMD} --input-dim ${INPUT_DIM}"
CMD="${CMD} --num-layers ${NUM_LAYERS}"
CMD="${CMD} --hidden-channels ${HIDDEN_CHANNELS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --use-batch-norm"
CMD="${CMD} --pooling-type ${POOLING_TYPE}"

echo $CMD
$CMD
