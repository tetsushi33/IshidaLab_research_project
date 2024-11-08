import pandas as pd

# CSVファイルを読み込む
file_path = "../output_csv_files/phase_03/ver_1/apo_holo_pairs.csv"  # ここに実際のCSVファイルのパスを指定してください
df = pd.read_csv(file_path)

# holo_nameとapo_nameの位置を入れ替える
df = df[['apo_name', 'holo_name', 'holo_chain', 'apo_chain', 'ligand', 'ligand_atom_count', 'loop_per']]

# apo_nameで並び替え
df_sorted = df.sort_values(by='apo_name')

# 同じapo_nameに対してholo_nameの種類数を計算し、新しい列を追加
holo_count = df_sorted.groupby('apo_name')['holo_name'].nunique().reset_index(name='holo_count')
df_final = pd.merge(df_sorted, holo_count, on='apo_name')

# apo_nameの種類数を計算
apo_name_count = df_final['apo_name'].nunique()
holo_name_count = df_final['holo_name'].nunique()

apo_combination_count = df_final[['apo_name', 'apo_chain']].drop_duplicates().shape[0]
# 結果を保存
output_file_path = "../output_csv_files/phase_03/ver_1/apo_holo_pairs_apo_based.csv"  # 出力先のファイルパスを指定してください
df_final.to_csv(output_file_path, index=False)

print("処理が完了し、結果が保存されました。")
print(f"apo_nameの種類数: {apo_name_count}")
print(f"apo_nameとapo_chainの組み合わせの種類数: {apo_combination_count}")
print(f"holo_nameの種類数: {holo_name_count}")

