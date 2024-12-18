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

parser = MMCIFParser(QUIET=True)

pdbbind_dir = "../PDBbind_original_data"
subdir = ["refined-set", "v2020-other-PL"]
mmcif_dir = "../mmcif"
csv_dir = "../output_csv_files"
blast_db_path = "../Blast_database/pdbaa"

AMINO_ACID_CODE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X"
}
    
exclude_list = ["WAT", "HOH", "NA", "CL", "MG", "K", "CA", "ZN", "SO4", "PO4", "DA", "DT", "DC", "DG", "U", "A", "G", "C"]

'''
ドミナントチェーンの決定
-----------------------------------------------------------------------------
- get_dominant_chain(pocket_file_path, pdb_id)
    (-) download_mmcif(pdb_id, mmcif_dir, "holo") 
-----------------------------------------------------------------------------
'''
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
        cmd.load(pocket_file_path, "pocket") # パスを"pocket"という名前でロード
        cmd.load(full_structure_file_path, "full_structure") # パスを"full_structure"という名前でロード

        # Get all atoms in the pocket
        #pocket_atoms = [atom for atom in cmd.get_model("pocket").atom]
        pocket_atoms = [
            atom for atom in cmd.get_model("pocket").atom 
            if atom.resn != "HOH"  # 水分子を除外
        ]
        pocket_length = len(pocket_atoms)
        
        # ポケットを含むチェーンを取得
        chains_including_pocket = list(set([atom.chain for atom in cmd.get_model("pocket").atom if atom.chain.strip()])) 
        overlapping_chains_atoms = {}
        for chain in chains_including_pocket:
            #atom_count_on_chain = cmd.count_atoms(f"pocket and chain {chain} like full_structure")
            atom_count_on_chain = cmd.count_atoms(f"pocket and chain {chain}")
            overlapping_chains_atoms[chain] = atom_count_on_chain
        #overlapping_chains = []
        #for chain in chains_including_pocket:
        #    if cmd.count_atoms(f"pocket and chain {chain} like full_structure"):
        #        '''
        #        pocketとchain{}を含むfull_structure内の原子をカウント
        #        0でない=ポケットがかかっているチェーン
        #        '''
        #        overlapping_chains.append(chain)

        # ループ領域の%を計算
        cmd.select("holo_pocket", f"full_structure like pocket")
        cmd.dss("holo_pocket")
        total_atoms = cmd.count_atoms("holo_pocket")
        loop_atoms = cmd.count_atoms("holo_pocket and not (ss h+s)")
        loop_per = loop_atoms / total_atoms * 100

        #print("overlapping_chains: ",overlapping_chains)
        #print("loop per: ",loop_per)

        if max(overlapping_chains_atoms.values()) == 0:
            print(f"No overlapping chains found for {pocket_file_path}")
            return None, None, None, None

        dominant_chain = max(overlapping_chains_atoms, key=overlapping_chains_atoms.get)
        
        ## 空のチェーンIDを持つ原子の詳細を表示(確認用)
        #empty_chain_atoms = [atom for atom in pocket_atoms if atom.chain == '']
        #if empty_chain_atoms:
        #    print(f"Found {len(empty_chain_atoms)} atoms with empty chain ID:")
        #    for atom in empty_chain_atoms:
        #        print(f"Residue: {atom.resn} {atom.resi}, Name: {atom.name}")

        cmd.delete("all")
        # Get ligand information from all chains and select the largest one
        ligands = []
        for chain_id in overlapping_chains_atoms.keys():
            ligand_name, atom_count = check_ligand_info_mmcif_inner(full_structure_file_path, pdb_id)
            if ligand_name:  # Check if ligand_name is not an empty string
                ligands.append((ligand_name, atom_count))
        ligands.sort(key=lambda x: x[1], reverse=True)
        dominant_ligand, dominant_ligand_size = ligands[0] if ligands else ("", 0)  # Default values instead of None
        #print(dominant_ligand)
        #print(dominant_ligand_size)

        if overlapping_chains_atoms[dominant_chain] / pocket_length >= 0.9:
            # ドミナント率が90%以上の場合のみ許す
            return dominant_chain, loop_per, dominant_ligand, dominant_ligand_size
        else:
            #print(f"No dominant chain for {pocket_file_path}")
            return None, None, None, None
        
    except Exception as e:
        print(f"Error encountered for {pdb_id}: {e}")
        return None, None, None, None


def get_dominant_chain_2(pocket_file_path, pdb_id):
    try:
        full_structure_file_path = os.path.join(mmcif_dir, "holo", f"{pdb_id}.cif")
        if not os.path.exists(full_structure_file_path):
            print(pocket_file_path)
            print(full_structure_file_path)
            print("ない")
            return
        
        cmd.reinitialize()
        cmd.load(pocket_file_path, "pocket")
        cmd.load(full_structure_file_path, "full_structure")

        pocket_atoms = [atom for atom in cmd.get_model("pocket").atom]
        pocket_length = len(pocket_atoms)

        chains_including_pocket = list(set([atom.chain for atom in cmd.get_model("pocket").atom if atom.chain.strip()]))
        overlapping_chains = []
        for chain in chains_including_pocket:
            if cmd.count_atoms(f"pocket and chain {chain} like full_structure"):
                overlapping_chains.append(chain)

        cmd.select("holo_pocket", f"full_structure like pocket")
        cmd.dss("holo_pocket")
        total_atoms = cmd.count_atoms("holo_pocket")
        loop_atoms = cmd.count_atoms("holo_pocket and not (ss h+s)")
        loop_per = loop_atoms / total_atoms * 100

        if not overlapping_chains:
            print(f"No overlapping chains found for {pocket_file_path}")
            return None, None, None, None

        overlapping_lengths = {}
        for chain_id in overlapping_chains:
            length = len([atom for atom in cmd.get_model(f"full_structure and chain {chain_id}").atom])
            overlapping_lengths[chain_id] = length

        dominant_chain = max(overlapping_lengths, key=overlapping_lengths.get)

        cmd.delete("all")

        ligands = []
        for chain_id in overlapping_chains:
            ligand_name, atom_count = check_ligand_info_mmcif_inner(full_structure_file_path, pdb_id)
            if ligand_name:
                ligands.append((ligand_name, atom_count))
        ligands.sort(key=lambda x: x[1], reverse=True)
        dominant_ligand, dominant_ligand_size = ligands[0] if ligands else ("", 0)

        if overlapping_lengths[dominant_chain] / pocket_length >= 0.9:
            return dominant_chain, loop_per, dominant_ligand, dominant_ligand_size
        else:
            return None, None, None, None
        
    except Exception as e:
        print(f"Error encountered for {pdb_id}: {e}")
        return None, None, None, None
     

'''
Blast検索
-----------------------------------------------------------------------------
- handle_blast_search(pdb_id_chain_data)
    - get_chain_sequence(pdb_id, mmcif_path, chain_id)
        - load_nonstandard_amino_acid_mappings(csv_file_path)
    - save_fasta_sequence(pdb_id, chain_id, sequence, base_dir="../data/fasta/holo")
    - perform_blast_search(fasta_path)
-----------------------------------------------------------------------------
'''
#----------------------------------------------------------------------------------------
def handle_blast_search(pdb_id_chain_data):
    pdb_id, chain_data = pdb_id_chain_data
    chain_id = chain_data[0]
    mmcif_path = os.path.join(mmcif_dir, "holo", f"{pdb_id}.cif")
    sequence = get_chain_sequence(pdb_id, mmcif_path, chain_id)
    fasta_path = save_fasta_sequence(pdb_id, chain_id, sequence)
    # perform_blast_search now takes the path to the fasta file as argument
    return pdb_id, perform_blast_search(fasta_path)

def get_chain_sequence(pdb_id, mmcif_path, chain_id):
    aa_mapping = load_nonstandard_amino_acid_mappings("../csv_files/non_amino_2_amino.csv")

    parser = MMCIFParser(QUIET=True)
    # Load the mmcif file
    structure = parser.get_structure(pdb_id, mmcif_path)
    
    # Get the sequence for the specified chain
    chain = structure[0][chain_id]
    #print(chain)
    sequence = []
    for residue in chain: # チェーン内の全ての残機に対して処理
        # 非ポリマーはスキップ（空白の時がポリマー（普通のアミノ酸））
        if residue.id[0] != " ":
            continue

        resname = residue.get_resname().strip() # 残基の名前を取得し、前後の空白を消去
        # 標準アミノ酸リストに含まれる残基は、アルファベット一文字（対応する文字）にしてリストに格納
        if resname in AMINO_ACID_CODE:
            sequence.append(AMINO_ACID_CODE[resname])
        # ヌクレオチド（DNA、RNA）に含まれる残機はそのチェーンごとスキップしてNoneを返す
        elif resname in ["DA", "DT", "DC", "DG", "U", "A", "G", "C"]:
            print(f"ルクレオチドを含むためNone({chain_id} in {pdb_id}, 残基名:{resname})")
            return None
        # 非標準アミノ酸の場合も標準と同様に一文字に変換してリストに追加
        elif resname in aa_mapping:  
            sequence.append(aa_mapping[resname])
        # 上記のいずれにも該当しない残基は、'X'と変換してリストに追加
        else:
            # If non-standard residue, convert to "X"
            sequence.append("X")
            print(f"リストにない非標準アミノ酸({pdb_id} chain {chain_id}: {resname}) -> 'X'に変換")
    #print("residue sequence: ",sequence)
    return "".join(sequence)

def load_nonstandard_amino_acid_mappings(csv_file_path):
    """
    CSVファイルから非標準アミノ酸のマッピングをロードします。

    引数:
    csv_file_path (str): マッピングが含まれるCSVファイルへのパス。

    戻り値:
    dict: 非標準の3文字コードから標準のものへのマッピングを行う辞書。
    """
    mapping = {}
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # ヘッダーをスキップ
        for row in reader:
            nonstandard  = row[1]
            standard     = row[3]
            mapping[nonstandard] = standard

    return mapping

def save_fasta_sequence(pdb_id, chain_id, sequence, base_dir="../fasta/holo"):
    fasta_file = f"{pdb_id}_{chain_id}.fasta"
    fasta_path = os.path.join(base_dir, fasta_file)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(fasta_path, 'w') as file:
        file.write(f">{pdb_id}_{chain_id}\n{sequence}\n")

    return fasta_path

# Execute BLAST search
def perform_blast_search(fasta_path):
    # クエリシーケンスを一時ファイルに保存

    # ローカルBLAST検索を実行
    cmd = [
        "blastp",
        "-query", fasta_path,
        "-db", blast_db_path,
        "-outfmt", "6 sseqid qseqid length qcovs pident"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    #print(result)
    ''' resultの中身
    1CAN_A\t6ugp_A\t257\t100\t100.000
    1CNH_A\t6ugp_A\t257\t100\t99.611
    .
    .
    （こんな感じ）
    意味は左から、
    sseqid:ヒットしたシーケンスの ID
    qseqid:クエリシーケンスの ID
    length:アライメントの長さ
    qcovs:クエリシーケンスのカバレッジ
    pident:アライメントの同一性の割合
    '''

    hit_chains = []
    similar_protein_chains = []
    for line in result.stdout.split("\n"):
        if line:
            sseqid, qseqid, length, qcovs, pident = line.split()
            length = int(length)
            qcovs = int(qcovs)
            pident = float(pident)

            hit_chains.append(sseqid)
            if pident >= 99.0 and qcovs >= 90:
                try:
                    parts = sseqid.split('|')
                    if len(parts) == 4:
                        pdb_id, chain_id = parts[1], parts[3]
                    elif len(parts) == 3:
                        pdb_id, chain_id = parts[1], parts[2]
                    else:
                        raise ValueError(f"Unexpected sseqid format: {sseqid}")
                    similar_protein_chains.append((pdb_id, chain_id))
                except ValueError as e:
                    print(f"Error splitting sseqid: {sseqid}, Error: {e}")
    
    #if(len(similar_protein_chains) == 0):
    #    print("類似タンパク質が見つかりませんでした")
    
    #print(hit_chains)
    return similar_protein_chains

'''
リガンド情報の取得
-----------------------------------------------------------------------------
    - check_ligand_info_mmcif(args)
       (-) download_mmcif(apo_pdb_id)
       (-) safe_remove(mmcif_file)
        - check_ligand_info_mmcif_inner(mmcif_file, apo_pdb_id)

- parallel_ligand_info_extraction(apo_list):
-----------------------------------------------------------------------------
'''
def check_ligand_info_mmcif(args):
    apo_pdb_id, _ = args
    mmcif_file = download_mmcif(apo_pdb_id) # 返り値はパス
    ligand_name, atom_count = check_ligand_info_mmcif_inner(mmcif_file, apo_pdb_id)
    return apo_pdb_id, ligand_name, atom_count

# リガンドの存在と原子数をチェック
def check_ligand_info_mmcif_inner(mmcif_file, pdb_id):
    # Initial attempt to parse the mmcif file
    try:
        structure = parser.get_structure(mmcif_file, mmcif_file) # チェインのオブジェクト
    except Exception as e1:
        # If an error occurs, remove the problematic file and try downloading again
        safe_remove(mmcif_file)
        download_mmcif(pdb_id) # エラーが発生した際の再ダウンロード
        try:
            # Try parsing the mmcif file again after re-downloading
            structure = parser.get_structure(mmcif_file, mmcif_file)
        except Exception as e2:
            # If an error occurs again, log the pdb_id and the error to the csv
            with open("modules/error_log_in_check_ligand_info_mmcif_inner.csv", "w") as error_log:
                writer = csv.writer(error_log)
                writer.writerow([pdb_id, str(e2)])
            return "", 0  # それでも無理な場合はNoneと0を返す
    
    # 小さいがリガンドとして判定するもののリスト
    allowed_small_ligands = [
        "ACT", "DMS", "EDO", "GOL", "PEG", "PG4", 
        "IOD", "OCS", "CO3", "3PE", "IMD", "FMT", 
        "TRS", "TAR", "MLI", "FLC"
    ]
    # 解析から除外する残基のリスト（水分子やイオン）
    exclude_list = ["WAT", "HOH", "NA", "CL", "MG", "K", "CA", "ZN", "SO4", "PO4", "DA", "DT", "DC", "DG", "U", "A", "G", "C"]
    
    ligands = []
    for residue in structure.get_residues(): # チェーン内の各残基に対して処理
        if residue.id[0] != " " and residue.get_resname().strip() not in exclude_list: # 非ポリマー、除外リストの残基はスキップ
            ligand_name = residue.get_resname().strip() # その残基についているリガンドの残基を取得
            atom_count = len(list(residue.get_atoms()))
            # 各リガンドの名前と原子数のセットをリストにして格納
            ligands.append((ligand_name, atom_count))
    
    # 原子数最大のリガンドを選択
    ligands.sort(key=lambda x: x[1], reverse=True)
    largest_ligand = ligands[0] if ligands else ("", 0)  # デフォルト : (空のリガンド名, 0)
    
    # 原子数5以下のかつリストに載ってないリガンドはデフォルト値にしてreturn
    if largest_ligand[1] <= 5 and largest_ligand[0] not in allowed_small_ligands:
        #print(f"Residue {largest_ligand[0]} is not considered as ligand due to atom count {largest_ligand[1]}")
        return "", 0  # Consider no ligand

    return largest_ligand # ("リガンド名", 原子数)



def parallel_ligand_info_extraction(apo_list):
    # CPUのコア数に応じてプロセスプールを作成
    pool = Pool(processes=cpu_count())
    ligand_info_results = []

    try:
        # tqdmを使ってプログレスバーを表示しながら、並列処理を実行
        for result in tqdm(pool.imap_unordered(check_ligand_info_mmcif, apo_list), total=len(apo_list), desc="Processing ligand info"):
            ligand_info_results.append(result)
    finally:
        # プールを閉じてリソースを解放
        pool.close()
        pool.join()

    return ligand_info_results



'''
その他
-----------------------------------------------------------------------------
- download_mmcif(pdb_id, mmcif_dir, "holo")
- safe_remove(file_path):
-----------------------------------------------------------------------------
'''
def download_mmcif(pdb_id, destination_folder="../mmcif", subfolder="apo"):
    # 得られるデータはタンパク質全体の構造
    #print(pdb_id)
    destination_folder = os.path.join(destination_folder, subfolder)
    mmcif_file_path = os.path.join(destination_folder, f"{pdb_id}.cif")

    # ファイルがすでに存在する場合はダウンロードをスキップ
    if os.path.exists(mmcif_file_path):
        #print(f"File for {pdb_id} already exists. Skipping download.")
        return mmcif_file_path

    print(f"Downloading: {pdb_id}")  # Debugging print statement
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    exit_status = os.system(f"wget {url} -O {mmcif_file_path}")
    if exit_status != 0:
        print(f"Error downloading: {pdb_id}")
    return mmcif_file_path

def safe_remove(file_path):
    """ファイルが存在する場合にのみファイルを削除する"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            print(f"削除対象のファイルが存在しません: {file_path}")
    except Exception as e:
        print(f"ファイルの削除中にエラーが発生しました: {e}")