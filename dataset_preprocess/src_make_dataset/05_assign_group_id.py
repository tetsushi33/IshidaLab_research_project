import os
import pandas as pd
from modules import module_apo_grouping
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# input
similarity_matrix_csv = "../output_csv_files/phase_04/ver_1/similarity_matrix.csv"
apo_holo_pairs_csv = "../output_csv_files/phase_03/ver_1/apo_holo_pairs.csv"
fasta_dir = "../data/fasta/apo"
mmcif_dir = "../mmcif/apo"

# output
output_csv_dir = "../output_csv_files/phase_05/ver_1"
apo_protein_id_csv = os.path.join(output_csv_dir, "apo_proteins_id.csv")
holo_proteins_id_csv = os.path.join(output_csv_dir, "holo_proteins_id_2.csv")
apo_protein_beta_path = os.path.join(output_csv_dir, "apo_protein_beta.csv")
combined_path = os.path.join(output_csv_dir, "combined_apo_protein.csv")
apo_holo_pairs_with_group_id_csv = os.path.join(output_csv_dir, "apo_holo_pairs_with_group_id.csv")

def main():
    # 保存先ディレクトリの確認
    if not os.path.exists(output_csv_dir):
        print("No directory - ", output_csv_dir)
        return
    
    similarity_matrix = pd.read_csv(similarity_matrix_csv, header=0, index_col=0) # 行のラベルは読み込まない
    print("similarity_matrix shape : ", similarity_matrix.shape)

    print("=============マトリックスからアポのid表作成=============")
    if not os.path.exists(apo_protein_id_csv):
        similarity_99_id_dict = module_apo_grouping.assign_group_ids(similarity_matrix, 0.99)
        similarity_50_id_dict = module_apo_grouping.assign_group_ids(similarity_matrix, 0.50)
        # アポタンパク質名とグループIDがペになった辞書形式

        # similarity_matrixからapo_nameとapo_chainを抽出
        names_chains = similarity_matrix.index.str.split("_", expand=True)
        names_chains = pd.DataFrame(names_chains.tolist(), columns=['apo_name', 'apo_chain']) 
        # 'apo_name' と 'apo_chain' を使って 'protein_id' と 'family50_id' を取得
        names_chains['protein_id'] = names_chains.apply(lambda row: module_apo_grouping.get_id_from_dict(row, similarity_99_id_dict), axis=1)
        names_chains['family50_id'] = names_chains.apply(lambda row: module_apo_grouping.get_id_from_dict(row, similarity_50_id_dict), axis=1)
        # CSVファイルとして保存
        names_chains.to_csv(apo_protein_id_csv, index=False)
        print("結果を保存 -----> ", apo_protein_id_csv)
    else:
        print("apo_protein_id_csv データ作成済み : ", apo_protein_id_csv)

    print("=============ホロのid表も作成=============")
    if not os.path.exists(apo_holo_pairs_with_group_id_csv):
        # apo_holo_pairsにapo_protein_id_dfを結合する
        names_chains = pd.read_csv(apo_protein_id_csv)
        apo_holo_pairs = pd.read_csv(apo_holo_pairs_csv)
        merged_df = pd.merge(apo_holo_pairs, names_chains, how='left', on=['apo_name', 'apo_chain'])
        merged_df.rename(columns={"protein_id": "apo_group_id"}, inplace=True)
        merged_df.to_csv(apo_holo_pairs_with_group_id_csv, index=False)

        #holo_protein_id_df = merged_df[['holo_name', 'holo_chain', 'ligand', 'ligand_atom_count', 'loop_per', 'apo_group_id', 'family50_id']].drop_duplicates()
        #holo_protein_id_df.to_csv(holo_proteins_id_csv, index=False)
        print("結果を保存 -----> ", apo_holo_pairs_with_group_id_csv)
    else:
        print("apo_holo_pairs_with_group_id_csv データ作成済み : ", apo_holo_pairs_with_group_id_csv)


    #print("=============アポ構造のデータの補強=============")
    #apo_protein_id_df = pd.read_csv(apo_protein_id_csv)
    #apo_holo_pairs = pd.read_csv(apo_holo_pairs_csv)
    #apo_beta_records = []
    #with ThreadPoolExecutor(max_workers=28) as executor:
    #    tasks = [executor.submit(module_apo_grouping.process_apo_beta_record, row, fasta_dir, mmcif_dir, apo_holo_pairs) for _, row in apo_protein_id_df.iterrows()]
    #    for future in tqdm(as_completed(tasks), total=len(tasks), desc="Creating apo_protein_beta"):
    #        apo_beta_records.extend(future.result())
    #apo_beta_df = pd.DataFrame(apo_beta_records)
    #apo_beta_df.to_csv(apo_protein_beta_path, index=False)
#
#
    #print("=============補強分をアポのid表に追加=============")
    ## 10. Concatenate apo_protein_id.csv and apo_protein_beta.csv
    #combined_df = pd.concat([apo_protein_id_df, apo_beta_df])
    #combined_df.to_csv(combined_path, index=False)

    #print("=============apoのグループidをapo_holo_pairsに追加=============")

    
if __name__ == "__main__":
    main()