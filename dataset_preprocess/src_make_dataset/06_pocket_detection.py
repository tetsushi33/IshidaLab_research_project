import os
import sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
#from IshidaLab_research_project.dataset_preprocess.src_make_dataset.modules import module_determining_pocket
from modules import module_deciding_pockets
from pymol import cmd, stored

## input
# PDB
pdbbind_dir_refined = "../PDBbind_original_data/refined-set/"
pdbbind_dir_other = "../PDBbind_original_data/v2020-other-PL/"
# 前フェーズの結果
apo_holo_pairs_csv = pd.read_csv('../output_csv_files/phase_03/ver_1/apo_holo_pairs.csv')
apo_proteins_id_csv = pd.read_csv('../output_csv_files/phase_05/ver_1/apo_proteins_id.csv')
apo_holo_pairs_with_group_id_csv = pd.read_csv("../output_csv_files/phase_05/ver_1/apo_holo_pairs_with_group_id.csv")

## output
output_csv_dir = "../output_csv_files/phase_06/ver_4"
log_dir = "../rmsd_culculating_log"

colors = ["red", "green", "blue", "yellow", "orange", "purple"]

def main(start_id, end_id):
    # 保存先ファイルの設定
    output_csv_path = os.path.join(output_csv_dir, f'pocket_analysis_results_{start_id}_to_{end_id}.csv')

    for apo_group_id in range(start_id, end_id+1):
        print(f"-----------Processing apo group {apo_group_id}...")
        # ログファイルの設定
        log_file_path = os.path.join(log_dir, f"group_id_{apo_group_id}")
        module_deciding_pockets.logging_setting(log_file_path)
        results = []
        cmd.reinitialize()

        # グループ内の代表アポタンパク質の決定
        apo_A_name, apo_A_chain = module_deciding_pockets.deciding_apo_representitive(apo_group_id, apo_proteins_id_csv)
        # アポAと対応するホロの重ね合わせ
        module_deciding_pockets.prepare_apo_for_pymol(apo_A_name, apo_A_chain)
        
        merged_pocket_ids, pockets_centroid_results, rmsd_results, merged_pockets, apo_pocket_missing_percentage = module_deciding_pockets.overlap_apoA_and_holos(apo_group_id, apo_A_name, apo_A_chain, apo_holo_pairs_with_group_id_csv)
        # pymolでの処理内容を保存
        module_deciding_pockets.save_pymol_process(apo_group_id, apo_A_name, apo_A_chain, "A")

        for (holo_name, holo_chain), pocket_id in merged_pocket_ids.items():
            if holo_name in apo_holo_pairs_csv[apo_holo_pairs_csv['apo_name'].str.upper() == apo_A_name]['holo_name'].values:
                # pocket_rmsdがinfの場合はスキップ
                if rmsd_results.get(holo_name, 0) == float('inf'):
                    continue
                
                holo_info = apo_holo_pairs_with_group_id_csv[apo_holo_pairs_with_group_id_csv['holo_name'] == holo_name].iloc[0]
                result_a = {
                    'apo_name': apo_A_name,
                    'apo_chain': apo_A_chain,
                    'holo_name': holo_name,
                    'holo_chain': holo_info['holo_chain'],
                    'pocket_id': pocket_id,
                    'pocket_rmsd': rmsd_results.get(holo_name, None),
                    'pocket_com': pockets_centroid_results.get(holo_name, None),
                    'protein_id': apo_group_id,
                    #'family50_id': apo_a_info['family50_id'],
                    'ligand': holo_info['ligand'],
                    'ligand_atom_count': holo_info['ligand_atom_count'],
                    'apo_pocket_missing_percentage': apo_pocket_missing_percentage.get(holo_name, None),
                    #'apo_loop_per': apo_pocket_loop_percentage.get(holo_name, None),
                    'holo_loop_per': holo_info['loop_per']
                }
                results.append(result_a)
        
        #print("merged pockets : ", merged_pockets)
        #print("merged pocket id : ", merged_pocket_ids)

        ## 代表以外のアポタンパク質の処理
        print("--代表以外のアポンタンパク質（アポBs）の処理")
        apo_A_Bs_mapping = {}
        if len(apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == apo_group_id]) > 1: 
            print("アポBsの数 : ", len(apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == apo_group_id])-1)
            for _, apo_B in tqdm(
                apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == apo_group_id].iterrows(),
                total=len(apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == apo_group_id]),
                desc="Processing apo Bs"
            ):
                apo_B_name = apo_B['apo_name']
                apo_B_chain = apo_B['apo_chain']
                if apo_B_name == apo_A_name and apo_B_chain == apo_A_chain:
                    continue

                # apo_Bに対応するホロタンパク質を取得
                corresponding_holos = apo_holo_pairs_csv[apo_holo_pairs_csv['apo_name'].str.upper() == apo_B_name]['holo_name'].values

                module_deciding_pockets.prepare_apo_B_for_pymol(apo_B_name, apo_B_chain)
                rmsd_result_with_holo, pocket_centroids_B, apo_B_pocket_loop_percentage, apo_B_pocket_missing_percentage = module_deciding_pockets.process_for_apo_B(apo_A_name, apo_A_chain, apo_B_name, apo_B_chain, apo_holo_pairs_csv, merged_pocket_ids, merged_pockets, apo_group_id)
                # pymolでの処理内容を保存
                #module_deciding_pockets.save_pymol_process(apo_group_id, apo_B_name, apo_B_chain, "B")
                
                for holo_name in corresponding_holos:
                    if holo_name in merged_pocket_ids:
                        # pocket_rmsdがinfの場合はスキップ
                        if rmsd_result_with_holo.get(holo_name, 0) == float('inf'):
                            continue
                        holo_info = apo_holo_pairs_with_group_id_csv[apo_holo_pairs_with_group_id_csv['holo_name'] == holo_name].iloc[0]
                        result_b = {
                            'apo_name': apo_B_name,
                            'apo_chain': apo_B_chain,
                            'holo_name': holo_name,
                            'holo_chain': holo_info['holo_chain'],
                            'pocket_id': pocket_id,
                            'pocket_rmsd': rmsd_result_with_holo.get(holo_name, None),
                            'pocket_com': pocket_centroids_B.get(holo_name, None),
                            'protein_id': apo_B['protein_id'],
                            'family50_id': apo_B['family50_id'],
                            'ligand': holo_info['ligand'],
                            'ligand_atom_count': holo_info['ligand_atom_count'],
                            'apo_pocket_missing_percentage': apo_B_pocket_missing_percentage.get(holo_name, None),
                            'apo_loop_per': apo_B_pocket_loop_percentage.get(holo_name, None),
                            'holo_loop_per': holo_info['loop_per']
                                    }
                        results.append(result_b)
        
        # 結果の保存
        if results:
            results_df = pd.DataFrame(results)
            if os.path.exists(output_csv_path):
                results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
            else:
                results_df.to_csv(output_csv_path, index=False)

    print(f"処理完了. 保存先 -> {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process protein IDs.')
    parser.add_argument('start_id', type=int, help='Start protein ID')
    parser.add_argument('end_id', type=int, help='End protein ID')
    args = parser.parse_args()

    main(args.start_id, args.end_id)