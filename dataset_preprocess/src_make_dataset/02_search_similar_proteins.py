import os
import csv
from concurrent.futures import ProcessPoolExecutor
from modules import module_search_apo
from tqdm import tqdm

#input
pocket_data_csv = "../output_csv_files/phase_01/ver_1/pocket_data.csv"
#output
output_csv_dir = "../output_csv_files/phase_02/ver_2" # 修正があったらここのバージョンを変更 
similar_apo_proteins_csv = os.path.join(output_csv_dir, "similar_apo_proteins.csv")

def main():
    # 保存先ディレクトリの確認
    if not os.path.exists(output_csv_dir):
        print("No directory - ", output_csv_dir)
        return
    print("=============入力データの展開=============")
    pocket_data = {}
    with open(pocket_data_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダー行をスキップ
        for row in reader:
            pdb_id, dominant_chain, loop_percentage, ligand_name, ligand_size = row # csvの各行を読み込み、変数に割り当てる
            if dominant_chain:  # Check if dominant_chain is not empty
                pocket_data[pdb_id] = (dominant_chain, float(loop_percentage), ligand_name, int(ligand_size)) #行の情報を全て格納
    print("num of pocket_data : ", len(pocket_data))
    print("=============類似タンパク質検索(Blast)=============")
    # searching similar protein by Blast
    similar_apo_proteins = {} # 類似のアポタンパク質用

    if os.path.exists(similar_apo_proteins_csv):
        print("類似タンパク質データ作成済み --- ", similar_apo_proteins_csv)
        with open(similar_apo_proteins_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                pdb_id, apo_pdb_id, apo_chain = row
                if pdb_id not in similar_apo_proteins:
                    similar_apo_proteins[pdb_id] = []
                similar_apo_proteins[pdb_id].append((apo_pdb_id, apo_chain))
        print("num of similar apo proteins : ", len(similar_apo_proteins))
    else:
        with ProcessPoolExecutor() as executor:
            pdb_id_chain_data_items = list(pocket_data.items()) # pocket_data : さっきのドミナントがあるデータのリスト
            #print(pdb_id_chain_data_items)
            # tqdmをexecutor.mapに適用し、プログレスバーを表示
            results = list(tqdm(executor.map(module_search_apo.handle_blast_search, pdb_id_chain_data_items), total=len(pdb_id_chain_data_items)))
            #print(results)
    
        for pdb_id, similar_proteins in results:
            if similar_proteins:
                similar_apo_proteins[pdb_id] = similar_proteins
        print("num of holo proteins in pairs : ", len(similar_apo_proteins))
        #print(similar_apo_proteins)
        
        # 結果をCSVファイルに保存
        with open(similar_apo_proteins_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["pdb_id", "apo_pdb_id", "apo_chain"])
            for pdb_id, apo_list in similar_apo_proteins.items():
                for apo_pdb_id, apo_chain in apo_list:
                    writer.writerow([pdb_id, apo_pdb_id, apo_chain])
        print("結果を保存 -----> ", similar_apo_proteins_csv)

if __name__ == "__main__":
    main()