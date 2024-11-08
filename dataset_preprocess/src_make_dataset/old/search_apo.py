import os
import csv
from Bio import PDB, SeqIO, Entrez
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.PDB import PDBParser
from urllib.request import urlretrieve
from Bio.PDB.MMCIFParser import MMCIFParser
import sys
import pymol
from pymol import cmd
from tqdm import tqdm
import subprocess
from multiprocessing import Pool, cpu_count
import concurrent.futures
# from chimerax.core.commands import runscript  # ChimeraX の必要な関数をインポート
# from chimerax.core import session  # Session をインポート
from concurrent.futures import ProcessPoolExecutor

from modules import module_search_apo

parser = MMCIFParser(QUIET=True)

# Directories and file settings
pdbbind_dir = "../PDBbind_original_data"
subdir = ["refined-set", "v2020-other-PL"]
mmcif_dir = "../mmcif"
output_csv_dir = "../output_csv_files/"
blast_db_path = "../blast_db/pdbaa"

# csv paths
error_log_csv = os.path.join(output_csv_dir, "phaze_1/error_log.csv")
pocket_data_csv = os.path.join(output_csv_dir, "phaze_1/pocket_data.csv")
no_dominant_chains_csv = os.path.join(output_csv_dir, "phaze_1/no_dominant_chains.csv")
similar_apo_proteins_csv = os.path.join(output_csv_dir, "phaze_1/similar_apo_proteins.csv")
ligand_info_csv = os.path.join(output_csv_dir, "phaze_1/ligand_info.csv")
apo_holo_pairs_csv = os.path.join(output_csv_dir, "phaze_1/apo_holo_pairs.csv")

def main():
    # error log setting
    if not os.path.exists(error_log_csv):
        with open(error_log_csv, "w") as error_log:
            writer = csv.writer(error_log)
            writer.writerow(["pdb_id", "error_message"])

    print("=============ポケットデータパス取得=============")
    # get holo protein 
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
    print("------Finish getting holo pocket paths!!------")
    #print(holo_pocket_paths)

    print("=============ポケットデータ読み込み=============")
    # get pocket and chain data
    pocket_data = {}
    failed_id = []

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
            #results = list(tqdm(executer.map(module_search_apo.get_dominant_chain, holo_pocket_paths.values(), holo_pocket_paths.keys())))
            results = list(executer.map(module_search_apo.get_dominant_chain, holo_pocket_paths.values(), holo_pocket_paths.keys()))
            
            for pdb_id, result in zip(holo_pocket_paths.keys(), results):
                dominant_chain, loop_percentage, ligand_name, ligand_size = result

                if dominant_chain:
                    pocket_data[pdb_id] = (dominant_chain, loop_percentage, ligand_name, ligand_size) #行の情報を全て格納
                else:
                    failed_id.append(pdb_id) # idのみ記録

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
                f.write('\n'.join(failed_id))
            print("結果を保存 -----> ", no_dominant_chains_csv)

    print("=============類似タンパク質検索(Blast)=============")
    # searching similar protein by Blast
    similar_apo_proteins = {} # 類似のアポタンパク質用

    if os.path.exists(similar_apo_proteins_csv) and 0:
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
        print("num of similar apo and holo pair : ", len(similar_apo_proteins))
        #print(similar_apo_proteins)
        
        # 結果をCSVファイルに保存
        with open(similar_apo_proteins_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["pdb_id", "apo_pdb_id", "apo_chain"])
            for pdb_id, apo_list in similar_apo_proteins.items():
                for apo_pdb_id, apo_chain in apo_list:
                    writer.writerow([pdb_id, apo_pdb_id, apo_chain])
        print("結果を保存 -----> ", similar_apo_proteins_csv)

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

    print("=======ポケットチェーンとアポタンパク質の情報===============")
    # アポタンパク質とホロタンパク質のペアをCSVに保存
    with open(apo_holo_pairs_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["holo_name", "apo_name", "holo_chain", "apo_chain", "ligand", "ligand_atom_count", "loop_per"])
        for holo_pdb_id, apo_proteins in tqdm(filtered_apo_proteins.items(), desc="Creating apo-holo pairs"):
            for apo in apo_proteins:
                ligand_name, ligand_size = pocket_data[holo_pdb_id][2], pocket_data[holo_pdb_id][3]  # get ligand_name and ligand_size from pocket_data
                writer.writerow([holo_pdb_id, apo[0], pocket_data[holo_pdb_id][0], apo[1], ligand_name, ligand_size, pocket_data[holo_pdb_id][1]])

 

if __name__ == "__main__":
    main()