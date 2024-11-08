import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルの読み込み
df = pd.read_csv('(2)_label_sqrt/predicted_values_test.csv')

# 逆関数を適用して元の値に戻す
# logの逆変換
#df['true_label'] = np.exp(df['true_label'] / 2) - 1
#df['predicted_label'] = df['predicted_label'].apply(lambda x: np.exp(np.array(eval(x)) / 2) - 1)

# √の逆変換
#df['true_label'] = df['true_label'] ** 2
#df['predicted_label'] = df['predicted_label'].apply(lambda x: np.array(eval(x)) ** 2)

# 'predicted_label' の値から余分な文字を取り除き、float型に変換
df['predicted_label'] = df['predicted_label'].str.replace('[', '').str.replace(']', '').astype(float)

# 相関係数を計算
correlation = df['predicted_label'].corr(df['true_label'])

# 散布図をプロット
plt.figure(figsize=(8, 8))
plt.scatter(df['true_label'], df['predicted_label'], alpha=0.5)

correlation_formatted = f"{correlation:.2f}"
plt.title(f'True vs Predicted Values (test) (correlation: {correlation_formatted})')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')  # x軸とy軸のスケールを同じにする
plt.xlim(0, 5)  # x軸の範囲を0から10に固定
plt.ylim(0, 5)  # y軸の範囲を0から10に固定
plt.plot([0, 5], [0, 5], linestyle='--', color='gray')  # y=xの点線を追加 
plt.savefig(f"(2)_label_sqrt/scatter_plot_test_resized.png")

