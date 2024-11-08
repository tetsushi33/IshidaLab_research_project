import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
df = pd.read_csv('../pocket_rmsd_label_seed-1.csv')

# データ数を計算
data_counts = df.groupby('data_type')['label'].count()
label_means = df.groupby('data_type')['label'].mean()  # 各data_typeごとのラベル平均を計算

# グラフの設定
fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
fig.suptitle('Distribution of Labels by Data Type')

# ビンの設定
bins = [x * 0.2 for x in range(51)]  # 0, 0.2, 0.4, ..., 10.0

# 各data_typeごとにプロット
data_types = ['test', 'train', 'validation']
colors = ['red', 'blue', 'green']  # 色は自由に設定可能

for ax, data_type, color in zip(axes, data_types, colors):
    sns.histplot(df[df['data_type'] == data_type]['label'], kde=True, ax=ax, color=color, bins=bins)
    ax.set_title(f'Data Type: {data_type}')
    ax.set_xlabel('Label Value')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 10)  # X軸の範囲を0から10に設定
    ax.text(0.95, 0.85, f'Count: {data_counts[data_type]}', transform=ax.transAxes, horizontalalignment='right',
            color='black', fontsize=12)
    ax.text(0.95, 0.75, f'Mean: {label_means[data_type]:.2f}', transform=ax.transAxes, horizontalalignment='right',
            color='black', fontsize=12)  # 平均値を表示

# x軸を共有するための設定
plt.tight_layout(rect=[0, 0, 1, 0.95])  # タイトルとグラフが重ならないように調整
plt.savefig('Distribution of label equalized seed-1')
