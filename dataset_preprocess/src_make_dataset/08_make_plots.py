import os
import sys
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# input
input_dir = "../output_csv_files/phase_07/ver_4"
input_csv = os.path.join(input_dir, "max_pocket_rmsd_results1.csv")

# output
output_dir = "../output_csv_files/phase_08/ver_1"
output_file = os.path.join(output_dir, "result_max_rmsd_plot_5.png")

def main():
    df = pd.read_csv(input_csv)
    filtered_df = df[df['max_pocket_rmsd'] <= 10]

    # max_pocket_rmsd の分布をプロット
    plt.figure(figsize=(8, 6))  # 図のサイズを指定
    plt.hist(filtered_df['max_pocket_rmsd'], bins=100, edgecolor='black', alpha=0.7)  # ヒストグラムを作成
    plt.title('Distribution of Max Pocket RMSD', fontsize=16)  # グラフタイトル
    plt.xlabel('Max Pocket RMSD', fontsize=12)  # x軸ラベル
    plt.ylabel('Frequency', fontsize=12)  # y軸ラベル
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # グリッド線を表示

    # 平均値をグラフに載せる
    mean_value = filtered_df['max_pocket_rmsd'].mean()  
    plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.legend(fontsize=12)

    plt.tight_layout()  # レイアウト調整

    # ヒストグラムを保存
    plt.savefig(output_file, dpi=300)
    plt.close()

print(f"ヒストグラムがファイルに保存されました: {output_file}")

if __name__ == "__main__":
    main()