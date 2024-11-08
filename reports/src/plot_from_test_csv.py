import pandas as pd
import matplotlib.pyplot as plt
import os

# CSVファイルを読み込む
csv_path = '../version_02_randomseed-1/predicted_values_test.csv'
df = pd.read_csv(csv_path)

save_dir = "../version_02_randomseed-1"

# predicted_label列の値をリストから数値に変換
df['predicted_label'] = df['predicted_label'].apply(lambda x: eval(x)[0])

# 散布図を描画
plt.figure(figsize=(8, 6))
plt.scatter(df['true_label'], df['predicted_label'], color='blue', label='Data points')

# y=xの直線
plt.plot([df['true_label'].min(), df['true_label'].max()], 
         [df['true_label'].min(), df['true_label'].max()], 
         'k--', label='y = x')

# x=1.5 および y=1.5 の点線
plt.axhline(y=1.5, color='red', linestyle='--', label='y = 1.5')
plt.axvline(x=1.5, color='red', linestyle='--', label='x = 1.5')

plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Scatter Plot with Reference Lines')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'scatter_plot.png'))
plt.close()