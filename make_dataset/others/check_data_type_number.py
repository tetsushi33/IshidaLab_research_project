import pandas as pd
# CSVファイルの読み込み
df = pd.read_csv('../../input_csv_files/pocket_rmsd_label_2.csv')

# 'label_column_name'カラムの各値の出現回数をカウント
value_counts = df['data_type'].value_counts()

# 結果の表示
print(value_counts)
#出力結果↓
#data_type
#train         3106
#test           219
#validation     207
#Name: count, dtype: int64
