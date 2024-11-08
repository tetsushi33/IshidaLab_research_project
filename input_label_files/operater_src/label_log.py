import pandas as pd
import numpy as np

# CSVファイルのパス
csv_file_path = '../pocket_rmsd_label_seed-1.csv'

# CSVファイルを読み込む
df = pd.read_csv(csv_file_path)

# label列の値を取得し、2 * log(x + 1) (定数はe)に変換して新しい列として追加
df['label'] = 2 * np.log(df['label'] + 1)

# 新しいCSVファイルとして保存
new_csv_file_path = '../pocket_rmsd_label_seed-1_log.csv'
df.to_csv(new_csv_file_path, index=False)
