import os
import sys
import pandas as pd
from modules import module_apo_grouping
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from modules import module_determining_pocket
from modules import module_apo_grouping
from pymol import cmd, stored

# input
apo_holo_pairs_csv = pd.read_csv('../output_csv_files/phase_03/ver_1/apo_holo_pairs.csv')
apo_proteins_id_csv = pd.read_csv('../output_csv_files/phase_05/ver_1/apo_proteins_id.csv')
pdbbind_dir_refined = "../PDBbind_original_data/refined-set/"
pdbbind_dir_other = "../PDBbind_original_data/v2020-other-PL/"
apo_holo_pairs_with_group_id_csv = pd.read_csv("../output_csv_files/phase_05/ver_1/apo_holo_pairs_with_group_id.csv")

# output
output_csv_dir = "../output_csv_files/phase_06/ver_1"

colors = ["red", "green", "blue", "yellow", "orange", "purple"]

def main(start_id, end_id):
    # 引数のidに応じて保存先を分けて保存
    output_csv_path = os.path.join(output_csv_dir, f'pocket_analysis_results_{start_id}_to_{end_id}.csv')
    
    for protein_id in tqdm(range(start_id, end_id+1)): # protein_idといったら、今後はグループのidのことをさす
        results = []
        
        cmd.reinitialize()

        apo_representitive = module_determining_pocket.determine_most_common_apo(protein_id, apo_holo_pairs_csv, apo_proteins_id_csv)
        if not apo_representitive: # 代表アポがない場合（指定のprotein_idに対応するたんぱく質がなかった場合）スキップ
            continue

        # アポタンパク質Aの情報を取得し、ホロタンパク質との関連情報を決定
        print("代表アポ: ", apo_representitive)
        holo_to_pocket_id, holo_to_pocket_centroids, holo_to_pocket_rmsd, merged_pockets, apo_pocket_loop_percentage, apo_pocket_missing_percentage = module_determining_pocket.determine_pocket_id_for_apo_a(
            apo_representitive, protein_id, apo_proteins_id_csv, apo_holo_pairs_with_group_id_csv, pdbbind_dir_refined, pdbbind_dir_other, colors
        )
        print("holo_to_pocket_id : ", holo_to_pocket_id)
        '''
        例：{'4f5y': 1, '4loi': 1, '4loh': 1} 
        '''
        
        # アポタンパク質AのFASTAファイルから配列を取得
        apo_a_info = apo_proteins_id_csv[apo_proteins_id_csv['apo_name'] == apo_representitive].iloc[0]
        apo_a_name = apo_a_info['apo_name']
        apo_a_chain = apo_a_info['apo_chain']
        apo_a_fasta_path = f"../data/fasta/apo/{apo_a_name}_{apo_a_chain}.fasta"
        apo_a_protein_path = f"../mmcif/apo/{apo_a_name}.cif"
        apo_a_seq = module_determining_pocket.read_fasta(apo_a_fasta_path)

        #print(f"apo protein: {apo_a_name} {apo_representitive}")
        print("holo num:", len(holo_to_pocket_id))

        for holo_name, pocket_id in holo_to_pocket_id.items():
            if holo_name in apo_holo_pairs_csv[apo_holo_pairs_csv['apo_name'].str.upper() == apo_representitive]['holo_name'].values:
                # pocket_rmsdがinfの場合はスキップ
                if holo_to_pocket_rmsd.get(holo_name, 0) == float('inf'):
                    continue
                
                holo_info = apo_holo_pairs_with_group_id_csv[apo_holo_pairs_with_group_id_csv['holo_name'] == holo_name].iloc[0]
                result_a = {
                    'apo_name': apo_a_name,
                    'apo_chain': apo_a_chain,
                    'holo_name': holo_name,
                    'holo_chain': holo_info['holo_chain'],
                    'pocket_id': pocket_id,
                    'pocket_rmsd': holo_to_pocket_rmsd.get(holo_name, None),
                    'pocket_com': holo_to_pocket_centroids.get(holo_name, None),
                    'protein_id': protein_id,
                    'family50_id': apo_a_info['family50_id'],
                    'ligand': holo_info['ligand'],
                    'ligand_atom_count': holo_info['ligand_atom_count'],
                    'apo_pocket_missing_percentage': apo_pocket_missing_percentage.get(holo_name, None),
                    'apo_loop_per': apo_pocket_loop_percentage.get(holo_name, None),
                    'holo_loop_per': holo_info['loop_per']
                }
                print(result_a)
                results.append(result_a)

        print("-----代表以外のアポタンパク質の処理-----")
        #代表以外のアポタンパク質の処理
        apo_a_to_b_mapping = {}
        apo_ab_num = len(apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == protein_id])
        print("number of apo a & b : ", apo_ab_num)
        if len(apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == protein_id]) > 1: # 同じidをのアポタンパク質が複数個あれば
            for index, apo_b_info in apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == protein_id].iterrows(): #代表でないアポを一つずつ処理
                apo_b_name = apo_b_info['apo_name']
                apo_b_chain = apo_b_info['apo_chain']
                # アポタンパク質Bがアポタンパク質Aと同じでないことを確認
                if apo_b_name == apo_representitive:
                    continue
                # アポタンパク質Bに対応するホロタンパク質をapo_holo_pairsから取得
                corresponding_holos = apo_holo_pairs_csv[apo_holo_pairs_csv['apo_name'].str.upper() == apo_b_name]['holo_name'].values
                apo_b_fasta_path = f"../data/fasta/apo/{apo_b_name}_{apo_b_chain}.fasta"
                apo_b_seq = module_determining_pocket.read_fasta(apo_b_fasta_path)
                apo_b_protein_path = f"../mmcif/apo/{apo_b_name}.cif"
                
                # アポタンパク質AとBのペアワイズアライメントを実行
                alignment = module_determining_pocket.align_sequences(apo_a_seq, apo_b_seq)
                print(f"Alignment result: \n{alignment}")
                # アポタンパク質AとBのFASTAとmmCIF間のマッピング生成
                fasta_to_pdb_mapping_a = module_determining_pocket.create_mapping_from_mmcif_to_fasta(apo_a_protein_path, apo_a_chain)
                fasta_to_pdb_mapping_b = module_determining_pocket.create_mapping_from_mmcif_to_fasta(apo_b_protein_path, apo_b_chain)
                print(f"Mapping from fasta to B: {fasta_to_pdb_mapping_b}")
                print(f"Mapping from fasta to A: {fasta_to_pdb_mapping_a}")
                apo_a_to_b_mapping = module_determining_pocket.create_mapping_from_alignment(alignment, fasta_to_pdb_mapping_a, fasta_to_pdb_mapping_b)
                print(f"Mapping from A to B: {apo_a_to_b_mapping}")

                # アポタンパク質Bのポケット位置を特定
                pocket_rmsd_b, pocket_centroids_b, apo_b_pocket_loop_percentage, apo_b_pocket_missing_percentage = module_determining_pocket.determine_pocket_id_for_apo_b(
                    apo_b_name, apo_b_chain, holo_to_pocket_id, merged_pockets, apo_a_to_b_mapping, pdbbind_dir_refined, pdbbind_dir_other,
                )
                
                # 結果の追加
                # それらのホロタンパク質に対してのみ解析を行う
                print(corresponding_holos)
                print(holo_to_pocket_id)
                print(apo_b_pocket_missing_percentage)
                for holo_name in corresponding_holos:
                    if holo_name in holo_to_pocket_id:
                        # pocket_rmsdがinfの場合はスキップ
                        if pocket_rmsd_b.get(holo_name, 0) == float('inf'):
                            continue
                        holo_info = apo_holo_pairs_with_group_id_csv[apo_holo_pairs_with_group_id_csv['holo_name'] == holo_name].iloc[0]
                        result_b = {
                            'apo_name': apo_b_name,
                            'apo_chain': apo_b_chain,
                            'holo_name': holo_name,
                            'holo_chain': holo_info['holo_chain'],
                            'pocket_id': pocket_id,
                            'pocket_rmsd': pocket_rmsd_b.get(holo_name, None),
                            'pocket_com': pocket_centroids_b.get(holo_name, None),
                            'protein_id': apo_b_info['protein_id'],
                            'family50_id': apo_b_info['family50_id'],
                            'ligand': holo_info['ligand'],
                            'ligand_atom_count': holo_info['ligand_atom_count'],
                            'apo_pocket_missing_percentage': apo_b_pocket_missing_percentage.get(holo_name, None),
                            'apo_loop_per': apo_b_pocket_loop_percentage.get(holo_name, None),
                            'holo_loop_per': holo_info['loop_per']
                                    }
                        print(result_b)
                        results.append(result_b)
        
        # 結果をCSVファイルに保存
        if results:
            results_df = pd.DataFrame(results)
            if os.path.exists(output_csv_path):
                results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
            else:
                results_df.to_csv(output_csv_path, index=False)


    print(f"Processing completed. Results saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process protein IDs.')
    parser.add_argument('start_id', type=int, help='Start protein ID')
    parser.add_argument('end_id', type=int, help='End protein ID')
    args = parser.parse_args()

    main(args.start_id, args.end_id)