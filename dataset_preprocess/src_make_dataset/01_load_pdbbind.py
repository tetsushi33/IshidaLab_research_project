import os
import csv
import concurrent.futures
from modules import module_search_apo
from tqdm import tqdm

pdbbind_dir = "../PDBbind_original_data"
subdir = ["refined-set", "v2020-other-PL"]
output_csv_dir = "../output_csv_files/"

# output
pocket_data_csv = os.path.join(output_csv_dir, "phase_01/ver_1/pocket_data.csv")
no_dominant_chains_csv = os.path.join(output_csv_dir, "phase_01/ver_1/no_dominant_chains.csv")


def main():
    # 保存先ディレクトリの確認
    if not os.path.exists(pocket_data_csv):
        print("No directory - ", pocket_data_csv)
        return
    if not os.path.exists(no_dominant_chains_csv):
        print("No directory - ", no_dominant_chains_csv)
        return
    print("=============ポケットデータパス取得=============")
    holo_pocket_paths = {}
    for dir in subdir:
        for pdb_id in os.listdir(os.path.join(pdbbind_dir, dir)):
            pocket_file_path = os.path.join(pdbbind_dir,dir, pdb_id, f"{pdb_id}_pocket.pdb")
            # ex.) pocket_file_path = "../PDBbind_original_data/refined-set/1a1e/1a1e_pocket.pdb"
            if os.path.exists(pocket_file_path):
                holo_pocket_paths[pdb_id] = pocket_file_path
    
    '''
    ex.)
    holo_pocket_paths = {'1a1e' : "../../PDBbind_original_data/refined-set/1a1e/1a1e_pocket.pdb", '1a4k' : "" , ...}
    '''
    print("PDBbind_original_data/[refined-set, v2020-other-PL] から取得した _pocket.pdbファイルの数: ", len(holo_pocket_paths))

    print("=============ポケットデータ読み込み=============")
    # ドミナントチェーンの決定
        # get pocket and chain data
    pocket_data = {}
    no_dominant_protein_id = [] # ドミナントチェーンを持たないタンパク質のid記録用

    if os.path.exists(pocket_data_csv):
        print("Already prepared! ---> ", pocket_data_csv)
        with open(pocket_data_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダー行をスキップ
            for row in reader:
                pdb_id, dominant_chain, loop_percentage, ligand_name, ligand_size = row # csvの各行を読み込み、変数に割り当てる
                if dominant_chain:  # Check if dominant_chain is not empty
                    pocket_data[pdb_id] = (dominant_chain, float(loop_percentage), ligand_name, int(ligand_size)) #行の情報を全て格納
                else:
                    no_dominant_chains_csv.append(pdb_id) # idのみ記録
            print("num of pocket_data : ", len(pocket_data))
            print("num of no dominant chain : ", len(no_dominant_chains_csv))
    else:
        with concurrent.futures.ProcessPoolExecutor() as executer:
            results = list(tqdm(executer.map(module_search_apo.get_dominant_chain_2, holo_pocket_paths.values(), holo_pocket_paths.keys()), total=len(holo_pocket_paths)))
            
            for pdb_id, result in zip(holo_pocket_paths.keys(), results):
                dominant_chain, loop_percentage, ligand_name, ligand_size = result

                if dominant_chain:
                    pocket_data[pdb_id] = (dominant_chain, loop_percentage, ligand_name, ligand_size) #行の情報を全て格納
                else:
                    no_dominant_protein_id.append(pdb_id) # idのみ記録

            print("ドミナントチェーンを決定し、ポケットデータを取得")
            print("num of pocket_data : ", len(pocket_data))
            #print(pocket_data)

            # 結果をCSVファイルを作成して保存
            with open(pocket_data_csv, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["pdb_id", "dominant_chain", "loop_percentage", "ligand_name", "ligand_size"])
                for pdb_id, (dominant_chain, loop_percentage, ligand_name, ligand_size) in pocket_data.items():
                    writer.writerow([pdb_id, dominant_chain, loop_percentage, ligand_name, ligand_size])
            print("結果を保存 -----> ", pocket_data_csv)
            # no_dominant_chainsも保存
            with open(no_dominant_chains_csv, 'w') as f:
                f.write('\n'.join(no_dominant_protein_id))
            print("結果を保存 -----> ", no_dominant_chains_csv)


if __name__ == "__main__":
    main()