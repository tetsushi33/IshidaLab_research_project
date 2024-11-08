import os
import csv
from concurrent.futures import ProcessPoolExecutor
from modules import module_search_apo
from tqdm import tqdm

# input
pocket_data_csv = "../output_csv_files/phase_01/ver_1/pocket_data.csv"
similar_apo_proteins_csv = "../output_csv_files/phase_02/ver_2/similar_apo_proteins.csv"
# output
output_csv_dir = "../output_csv_files/phase_03/ver_1"
ligand_info_csv = os.path.join(output_csv_dir, "ligand_info.csv")
apo_holo_pairs_csv = os.path.join(output_csv_dir, "apo_holo_pairs.csv")


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

    similar_apo_proteins = {} # ホロに対する類似アポタンパク質の辞書
    with open(similar_apo_proteins_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                pdb_id, apo_pdb_id, apo_chain = row
                if pdb_id not in similar_apo_proteins:
                    similar_apo_proteins[pdb_id] = []
                similar_apo_proteins[pdb_id].append((apo_pdb_id, apo_chain))
    print("num of holo in similar apo proteins : ", len(similar_apo_proteins))

    print("=======類似タンパク質のリガンド情報を取得===============")
    # get ligand information
    ligand_info_dict = {}
    if os.path.exists(ligand_info_csv):
        print("リガンド情報データ作成済み --- ", ligand_info_csv)
        with open(ligand_info_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                apo_pdb_id, ligand_name, atom_count = row
                ligand_info_dict[apo_pdb_id] = (ligand_name, int(atom_count))
    else:
        apo_list = [apo for apo_candidates in similar_apo_proteins.values() for apo in apo_candidates]
        '''
        以下と同じ意味
        for apo_candidates in similar_apo_proteins.values():
            for apo in apo_candidates:
                apo_list.append(apo)
        要はsimilar_apo_proteinsにあるすべてのアポを一つずつappendしているだけ
        '''
        #print(apo_list)
        # Poolを使用せずに直接ループで処理
        ligand_info_results = module_search_apo.parallel_ligand_info_extraction(apo_list)
        #print(ligand_info_results)

        for result in ligand_info_results:
            apo_pdb_id, ligand_name, atom_count = result
            if ligand_name:
                ligand_info_dict[apo_pdb_id] = (ligand_name, atom_count)
        #print(ligand_info_dict)

        ## 結果をCSVファイルに保存
        with open(ligand_info_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["apo_pdb_id", "ligand_name", "atom_count"])
            for apo_pdb_id, (ligand_name, atom_count) in ligand_info_dict.items():
                writer.writerow([apo_pdb_id, ligand_name, atom_count])
        print("結果を保存 -----> ", ligand_info_csv)

    print("=======リガンドのないタンパク質のみ抽出===============")
    # リガンドが存在しないアポタンパク質のみを取り出す
    filtered_apo_proteins = {}
    for pdb_id, apo_candidates in similar_apo_proteins.items():
        # holo_nameを小文字に変換
        pdb_id_lower = pdb_id.lower() # ホロの方
        valid_apos = []
        for apo in apo_candidates:
            # apo_nameを小文字に変換して、holo_nameと異なる場合のみ追加
            apo_pdb_id_lower = apo[0].lower() #apo[0]で(id, chain)のうちのidのみ取得
            if apo_pdb_id_lower != pdb_id_lower: # ホロとアポが同じものだったら意味がない
                ligand_info = ligand_info_dict.get(apo[0], (None, None))
                if not ligand_info[0]:  # リガンド名の部分が""だったら
                    valid_apos.append(apo)
        if valid_apos:
            filtered_apo_proteins[pdb_id] = valid_apos

    print("num of filtered_apo_proteins: ", len(filtered_apo_proteins))

    print("=======ホロポケットとアポタンパク質のペアを記録===============")
    # アポタンパク質とホロタンパク質のペアをCSVに保存
    with open(apo_holo_pairs_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["holo_name", "apo_name", "holo_chain", "apo_chain", "ligand", "ligand_atom_count", "loop_per"])
        for holo_pdb_id, apo_proteins in tqdm(filtered_apo_proteins.items(), desc="Creating apo-holo pairs"):
            for apo in apo_proteins:
                ligand_name, ligand_size = pocket_data[holo_pdb_id][2], pocket_data[holo_pdb_id][3]  # get ligand_name and ligand_size from pocket_data
                writer.writerow([holo_pdb_id, apo[0], pocket_data[holo_pdb_id][0], apo[1], ligand_name, ligand_size, pocket_data[holo_pdb_id][1]])
    print("結果を保存 -----> ", apo_holo_pairs_csv)

if __name__ == "__main__":
    main()