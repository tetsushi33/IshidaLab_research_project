#!/bin/bash

# Pythonスクリプトのパスを指定
PYTHON_SCRIPT="06_pocket_detection.py"

# 範囲の設定
START=101
END=2068
STEP=100

# 範囲を分割して実行
for ((i=$START; i<=$END; i+=$STEP)); do
    START_ID=$i
    END_ID=$((i+STEP-1))

    # 最終セットの調整 (END_IDがENDを超えないようにする)
    if ((END_ID > END)); then
        END_ID=$END
    fi

    # Pythonスクリプトの実行
    echo "Running: python $PYTHON_SCRIPT $START_ID $END_ID"
    python $PYTHON_SCRIPT $START_ID $END_ID
done
