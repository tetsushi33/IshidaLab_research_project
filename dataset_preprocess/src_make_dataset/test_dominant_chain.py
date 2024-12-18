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

pdbbind_dir = "../PDBbind_original_data"
subdir = ["refined-set", "v2020-other-PL"]
mmcif_dir = "../mmcif"
csv_dir = "../output_csv_files"
blast_db_path = "../Blast_database/pdbaa"

def get_dominant_chain(pocket_file_path, pdb_id):
    # ex.) pocket_file_path = "../../PDBbind_original_data/refined-set/1a1e/1a1e_pocket.pdb", pdb_id = '1a1e'
    try:
        full_structure_file_path = os.path.join(mmcif_dir, "holo", f"{pdb_id}.cif")
        # Download the full structure
        if not os.path.exists(full_structure_file_path):
            print(pocket_file_path)
            print(full_structure_file_path)
            print("ない")
            return
            #download_mmcif(pdb_id, mmcif_dir, "holo") # ない場合はインターネットから取得
    
        cmd.reinitialize()
        cmd.load(pocket_file_path, "pocket") # PDBbindからの構造データ（ポケットのみ）
        cmd.load(full_structure_file_path, "full_structure") # mmcifからの構造データ（全体構造）
        print("load finished!")
        # Get all atoms in the pocket
        pocket_atoms = [
            atom for atom in cmd.get_model("pocket").atom 
            if atom.resn != "HOH"  # 水分子を除外
        ]
        pocket_length = len(pocket_atoms)
        print("pocket length: ", pocket_length)
        
        # ポケットを含むチェーンを取得
        chains_including_pocket = list(set([atom.chain for atom in cmd.get_model("pocket").atom if atom.chain.strip()]))
        print(chains_including_pocket) 
        overlapping_chains_atoms = {}
        for chain in chains_including_pocket:
            #atom_count_on_chain = cmd.count_atoms(f"pocket and chain {chain} like full_structure")
            atom_count_on_chain = cmd.count_atoms(f"pocket and chain {chain}")
            overlapping_chains_atoms[chain] = atom_count_on_chain
        print(overlapping_chains_atoms)

        if max(overlapping_chains_atoms.values()) == 0:
            print(f"No overlapping chains found for {pocket_file_path}")
            return None, None, None, None
        
        dominant_chain = max(overlapping_chains_atoms, key=overlapping_chains_atoms.get)
        print(dominant_chain)
        # ループ領域の%を計算
        cmd.select("holo_pocket", f"full_structure like pocket")
        cmd.dss("holo_pocket")
        total_atoms = cmd.count_atoms("holo_pocket")
        loop_atoms = cmd.count_atoms("holo_pocket and not (ss h+s)")
        loop_per = loop_atoms / total_atoms * 100

        ## 空のチェーンIDを持つ原子の詳細を表示(確認用)
        #empty_chain_atoms = [atom for atom in pocket_atoms if atom.chain == '']
        #if empty_chain_atoms:
        #    print(f"Found {len(empty_chain_atoms)} atoms with empty chain ID:")
        #    for atom in empty_chain_atoms:
        #        print(f"Residue: {atom.resn} {atom.resi}, Name: {atom.name}")

        cmd.delete("all")
        for chain_id in overlapping_chains_atoms.keys():
            print(chain_id)

        if overlapping_chains_atoms[dominant_chain] / pocket_length >= 0.9:
            # ドミナント率が90%以上の場合のみ許す
            print("ok!")
            return dominant_chain, loop_per
        else:
            #print(f"No dominant chain for {pocket_file_path}")
            return None, None
        
    except Exception as e:
        print(f"Error encountered for {pdb_id}: {e}")
        return None, None, None, None
    

if __name__ == "__main__":
    # テストするファイルとPDB IDを指定
    pocket_file_path = "../PDBbind_original_data/refined-set/3k8c/3k8c_pocket.pdb"  # 適切なパスに変更
    pdb_id = "3k8c"

    # 関数を呼び出して結果を取得
    dominant_chain, loop_per = get_dominant_chain(pocket_file_path, pdb_id)

    # 結果を表示
    if dominant_chain is not None:
        print(f"Dominant Chain: {dominant_chain}")
        print(f"Loop Percentage: {loop_per:.2f}%")
    else:
        print(f"No dominant chain found for PDB ID {pdb_id}")

