import pandas as pd
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv('../protein_family_data.csv') # このファイル自体はそのままにしておく

# データをシャッフルするためのシード値を設定
np.random.seed(22)
#----note----#
#seed-1 : 42
#seed-2 : 22

# インデックスをシャッフル
shuffled_indices = np.random.permutation(len(df))

# 新しい data_type を割り当てる
train_size = int(0.8 * len(df))
test_size = int(0.1 * len(df))
validation_size = len(df) - train_size - test_size

# インデックスに基づいて新しいラベルを割り当てる
df['data_type'] = 'validation'  # 先に全てを validation に設定
df.loc[shuffled_indices[:train_size], 'data_type'] = 'train'
df.loc[shuffled_indices[train_size:train_size + test_size], 'data_type'] = 'test'

# 結果を保存
df.to_csv('../protein_family_file_seed-2.csv', index=False)
