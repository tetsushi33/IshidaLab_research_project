import os
import sys
from tqdm import tqdm
import argparse
import pandas as pd

# input
pocket_analysis_results = "../output_csv_files/phase_06/ver_4"
# output
output_dir = "../output_csv_files/phase_07/ver_2"
output_csv = os.path.join(output_dir, "max_pocket_rmsd_results.csv")
output_csv_2 = os.path.join(output_dir, "max_pocket_rmsd_results_no_pocket_id.csv")

def main(start_id, end_id):
    target_file = os.path.join(pocket_analysis_results, f"pocket_analysis_results_{start_id}_to_{end_id}.csv")
    if not os.path.exists(target_file):
        print("no target dir - ", target_file)
    # csvファイルの読み込み
    target_data = pd.read_csv(target_file, header=0)
    
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
    # 結果の確認
    print(result)
    result.to_csv(output_csv, mode='a', index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process protein IDs.')
    parser.add_argument('start_id', type=int, help='Start protein ID')
    parser.add_argument('end_id', type=int, help='End protein ID')
    args = parser.parse_args()

    main(args.start_id, args.end_id)
