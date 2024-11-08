import pandas as pd

# 入力ファイルと出力ファイルのパスを設定
input_file_path = '../../input_label_files/protein_family_file_seed-2.csv'  # seed値設定必須
output_file_path = '../../input_label_files/pocket_rmsd_label_seed-2.csv'  

# 元のCSVファイルの読み込み
df = pd.read_csv(input_file_path)

# apo_name列の更新: apo_name, apo_chain, pocket_idを結合
df['apo_name'] = df['apo_name'] + "_" + df['apo_chain'] + "_" + df['pocket_id'].astype(str)

# 新しい列を作成して既存の列を削除
df['label'] = df['max_pocket_rmsd']
#df['data_type'] = df['data_type'].replace({'test': 'validation', 'train': 'train'})

# 必要な列のみを選択
df = df[['apo_name', 'label', 'data_type']]

# 新しいCSVファイルに保存
df.to_csv(output_file_path, index=False)

print(f"Processed data saved to {output_file_path}")
