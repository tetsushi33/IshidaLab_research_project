import os
import sys
from tqdm import tqdm
import argparse
import pandas as pd

# input
pocket_analysis_results = "../output_csv_files/phase_06/ver_7"
# output
output_dir = "../output_csv_files/phase_07/ver_3"
output_csv = os.path.join(output_dir, "max_pocket_rmsd_results.csv")
output_csv_2 = os.path.join(output_dir, "max_pocket_rmsd_results_no_pocket_id.csv")

#def main(start_id, end_id):
#    target_file = os.path.join(pocket_analysis_results, f"pocket_analysis_results_{start_id}_to_{end_id}.csv")
#    if not os.path.exists(target_file):
#        print("no target dir - ", target_file)
#    
#    # 抽出したい列を指定
#    columns_to_extract = ['apo_name', 'apo_chain', 'holo_name', 'holo_chain', 'pocket_id', 'pocket_rmsd', 'pocket_com', 'protein_id', 'ligand', 'ligand_atom_count']
#    # csvファイルの読み込み
#    target_data = pd.read_csv(target_file, usecols=columns_to_extract, engine='python')
#    
#    # グループ化して処理
#    grouped = target_data.groupby(['apo_name', 'apo_chain', 'pocket_id'])
#    # 各グループのカウントを計算して列に追加
#    target_data['group_count'] = grouped['pocket_rmsd'].transform('size')
#    # 各グループのうち、pocket_rmsd の値が最大の行を抽出
#    result = target_data.loc[grouped['pocket_rmsd'].idxmax()].reset_index(drop=True)
#    # インデックスをリセットして元の列を復元
#    result = result.reset_index(drop=True)
#    # pocket_rmsd を max_pocket_rmsd に名前変更
#    result = result.rename(columns={'pocket_rmsd': 'max_pocket_rmsd'})
#    # 結果の確認
#    print(result)
#    result.to_csv(output_csv, mode='a', index=False, header=False)

def main(start_id, end_id):
    target_file = os.path.join(pocket_analysis_results, f"pocket_analysis_results_{start_id}_to_{end_id}.csv")
    if not os.path.exists(target_file):
        print("no target dir - ", target_file)
    
    # 抽出したい列を指定
    columns_to_extract = ['apo_name', 'apo_chain', 'holo_name', 'holo_chain', 'pocket_id', 'pocket_rmsd', 'pocket_com', 'protein_id', 'ligand', 'ligand_atom_count']
    # csvファイルの読み込み
    target_data = pd.read_csv(target_file, usecols=columns_to_extract, engine='python')
    
    # グループ化して処理
    grouped = target_data.groupby(['apo_name', 'apo_chain', 'pocket_id'])
    # 各グループのカウントを計算して列に追加
    target_data['group_count'] = grouped['pocket_rmsd'].transform('size')
    # 各グループのうち、pocket_rmsd の値が最大の行を抽出
    result = target_data.loc[grouped['pocket_rmsd'].idxmax()].reset_index(drop=True)
    # インデックスをリセットして元の列を復元
    result = result.reset_index(drop=True)
    # pocket_rmsd を max_pocket_rmsd に名前変更
    result = result.rename(columns={'pocket_rmsd': 'max_pocket_rmsd'})
    # 列の順序を指定
    #result = result[columns_to_extract[:-1] + ['group_count', 'max_pocket_rmsd']]
    #result = result[columns_to_extract + ['group_count', 'max_pocket_rmsd']]
    result = result[['apo_name', 'apo_chain', 'holo_name', 'holo_chain', 'pocket_id', 'max_pocket_rmsd', 'pocket_com', 'protein_id', 'ligand', 'ligand_atom_count', 'group_count']]
    
    # 結果の確認
    print(result)
    result.to_csv(output_csv, mode='a', index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process protein IDs.')
    parser.add_argument('start_id', type=int, help='Start protein ID')
    parser.add_argument('end_id', type=int, help='End protein ID')
    args = parser.parse_args()

    main(args.start_id, args.end_id)
