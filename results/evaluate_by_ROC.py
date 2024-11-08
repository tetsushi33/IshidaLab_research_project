import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# CSVファイルの読み込み
output_dir = "(1)_01_just_to_regression"
df = pd.read_csv(f'{output_dir}/predicted_values_test.csv')  # 適切なファイルパスに置き換えてください。


# 'predicted_label' の値から余分な文字を取り除く処理
df['predicted_label'] = df['predicted_label'].str.replace('[', '').str.replace(']', '').astype(float)

# 閾値を定義
thresholds = [1.0,1.5,2.0,2.5]
steepness = 0.1  # 曲線の急峻

# パラメータ設定（中心となるRMSD値と曲率を調整）

# ラベルの付け直しの場合はここで閾値もサイズを揃える
thresholds = np.array(thresholds)
#thresholds = 2 * np.log(thresholds + 1)

for threshold in thresholds:
    midpoint = threshold  # 変化が大きいと判断する閾値
    df['score'] = 1 / (1 + np.exp(-steepness * (df['predicted_label'] - midpoint)))
    # 二値ラベルの生成
    df['binary_label'] = (df['true_label'] > threshold).astype(int)

    # ROC曲線とAUCの計算
    fpr, tpr, thresholds = roc_curve(df['binary_label'], df['score'])
    roc_auc = auc(fpr, tpr)

    # ROC曲線のプロット
    plt.figure()
    #plt.plot(fpr, tpr, label='ROC curve (Threshold = %0.1f, AUC = %0.2f)' % (threshold, roc_auc))
    plt.plot(fpr, tpr, color='red', label='ROC curve (Threshold = %0.1f, AUC = %0.2f)' % (threshold, roc_auc))
    #plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    #plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # 閾値の値を表示
    #plt.text(0.6, 0.2, f'Threshold = {threshold}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(f"{output_dir}/ROC_test_{threshold}.png")
    plt.show()

    # AUCの表示
    print('AUC:', roc_auc)

    # 新しいCSVファイルにDataFrameを保存
    output_filepath = f'{output_dir}/updated_predicted_values_test_{threshold}.csv'
    df.to_csv(output_filepath, index=False)
