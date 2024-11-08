import pandas as pd
import numpy as np

# CSVファイルのパス
csv_file_path = 'pocket_rmsd_label_3_equalized.csv'

# CSVファイルを読み込む
df = pd.read_csv(csv_file_path)

# data_type列の値を取得し、その平方根を計算して新しい列として追加
df['label'] = np.sqrt(df['label'])

# 新しいCSVファイルとして保存
new_csv_file_path = 'pocket_rmsd_label_4_sqrt_equalized.csv'
df.to_csv(new_csv_file_path, index=False)
