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
parser = MMCIFParser(QUIET=True)


# Directories and file settings
pdbbind_directory = "../PDBbind_original_data"
subdirectories = ["refined-set", "v2020-other-PL"]
mmcif_directory = "../mmcif"
csv_directory = "../result_csv_files"
blast_db_path = "../blast_db/pdbaa"

# CSV paths
pocket_data_file = os.path.join(csv_directory, "pocket_data.csv")
no_dominant_chains_file = os.path.join(csv_directory, "no_dominant_chains.csv")
similar_apo_proteins_file = os.path.join(csv_directory, "similar_apo_proteins_check.csv")
ligand_info_file = os.path.join(csv_directory, "ligand_info_check.csv")
apo_holo_pairs_file = os.path.join(csv_directory, "apo_holo_pairs.csv")

AMINO_ACID_CODE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X"
}
    
exclude_list = ["WAT", "HOH", "NA", "CL", "MG", "K", "CA", "ZN", "SO4", "PO4", "DA", "DT", "DC", "DG", "U", "A", "G", "C"]


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


# def check_ligand_info_mmcif(args):
#     apo_pdb_id, _ = args
#     mmcif_file = download_mmcif(apo_pdb_id)
#     ligand_name, atom_count = get_ligand_info_chimerax(mmcif_file)  # ChimeraX関数を呼び出します
#     return apo_pdb_id, ligand_name, atom_count


## ---ドミナントチェーンの決定関連--- ##
### ---------------------------------------------------------------------------------------------------------------- ###
def get_dominant_chain(pocket_file, pdb_id):
    try:
        full_structure_file = os.path.join(mmcif_directory, "holo", f"{pdb_id}.cif")
    
        # Download the full structure if not present
        if not os.path.exists(full_structure_file):
            download_mmcif(pdb_id, mmcif_directory, "holo") # ない場合はインターネットから取得
        
        overlapping_chains, loop_percentage = identify_overlapping_chains(pocket_file, full_structure_file)
        
        if not overlapping_chains:
            print(f"No overlapping chains found for {pocket_file}")
            return None, None, None, None

        pocket_length = len([atom for atom in cmd.get_model("pocket").atom]) # ポケット内の原子の数
        
        overlapping_lengths = {}
        for chain_id in overlapping_chains: # 重複する各チェーンの原子数を計算
            length = len([atom for atom in cmd.get_model(f"full_structure and chain {chain_id}").atom])
            overlapping_lengths[chain_id] = length

        dominant_chain = max(overlapping_lengths, key=overlapping_lengths.get) # 原子数の最も多かったチェーンをドミナントとする

        cmd.delete("all")

        # Get ligand information from all chains and select the largest one
        ligands = []
        for chain_id in overlapping_chains:
            ligand_name, atom_count = check_ligand_info_mmcif_inner(full_structure_file, pdb_id)
            if ligand_name:  # Check if ligand_name is not an empty string
                ligands.append((ligand_name, atom_count))
        ligands.sort(key=lambda x: x[1], reverse=True)
        dominant_ligand, dominant_ligand_size = ligands[0] if ligands else ("", 0)  # Default values instead of None

        if overlapping_lengths[dominant_chain] / pocket_length >= 0.9:
            return dominant_chain, loop_percentage, dominant_ligand, dominant_ligand_size
        else:
            print(f"No dominant chain for {pocket_file}")
            return None, None, None, None
    except Exception as e:
        print(f"Error encountered for {pdb_id}: {e}")
        return None, None, None, None
    
def check_ligand_info_mmcif(args):
    apo_pdb_id, _ = args
    mmcif_file = download_mmcif(apo_pdb_id)
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
        download_mmcif(pdb_id)
        try:
            # Try parsing the mmcif file again after re-downloading
            structure = parser.get_structure(mmcif_file, mmcif_file)
        except Exception as e2:
            # If an error occurs again, log the pdb_id and the error to the csv
            with open("../result_csv_files/error_log.csv", "a") as error_log:
                writer = csv.writer(error_log)
                writer.writerow([pdb_id, str(e2)])
            return "", 0  # Default values instead of None
    
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
            ligand_name = residue.get_resname().strip()
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

# Identify overlapping chains in the pocket
def identify_overlapping_chains(pocket_file, full_structure_file):
    cmd.reinitialize()
    cmd.load(pocket_file, "pocket")
    cmd.load(full_structure_file, "full_structure")
    
    # Get chain names from the pocket pdb file
    pocket_data = list(set([atom.chain for atom in cmd.get_model("pocket").atom if atom.chain.strip()]))
    
    overlapping_chains = []
    for chain in pocket_data:
        if cmd.count_atoms(f"pocket and chain {chain} like full_structure"):
            overlapping_chains.append(chain)
    
    # Select the pocket part from the holo mmcif file
    cmd.select("holo_pocket", f"full_structure like pocket")
    
    # Calculate the percentage of loop parts in the selected pocket
    cmd.dss("holo_pocket")
    total_atoms = cmd.count_atoms("holo_pocket")
    loop_atoms = cmd.count_atoms("holo_pocket and not (ss h+s)")
    loop_percentage = loop_atoms / total_atoms * 100
        
    return overlapping_chains, loop_percentage


# Download mmcif files function
def download_mmcif(pdb_id, destination_folder="../mmcif", subfolder="apo"):
    print(pdb_id)
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

## -------Blast検索関連------- ##
### ---------------------------------------------------------------------------------------------------------------- ###
def handle_blast_search(pdb_id_chain_data):
    pdb_id, chain_data = pdb_id_chain_data
    chain_id = chain_data[0]
    mmcif_path = os.path.join(mmcif_directory, "holo", f"{pdb_id}.cif")
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
    return "".join(sequence)


def save_fasta_sequence(pdb_id, chain_id, sequence, base_dir="../data/fasta/holo"):
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

    similar_proteins = []
    for line in result.stdout.split("\n"):
        if line:
            sseqid, qseqid, length, qcovs, pident = line.split()
            length = int(length)
            qcovs = int(qcovs)
            pident = float(pident)
            
            if pident >= 99.0 and qcovs >= 90:
                try:
                    pdb_id, chain_id = sseqid.split("_")
                    similar_proteins.append((pdb_id, chain_id))
                except ValueError as e:
                    print(f"Error splitting sseqid: {sseqid}, Error: {e}")
    
    if(len(similar_proteins) == 0):
        print("類似タンパク質が見つかりませんでした")

    return similar_proteins

## -------------------------- ##
### ---------------------------------------------------------------------------------------------------------------- ###

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

def main():
    aa_mapping = load_nonstandard_amino_acid_mappings("../csv_files/non_amino_2_amino.csv")
    # Ensure "error_log.csv" exists and has a header
    # エラーログファイルの準備
    if not os.path.exists("error_log.csv"):
        with open("../result_csv_files/error_log.csv", "w") as error_log:
            writer = csv.writer(error_log)
            writer.writerow(["pdb_id", "error_message"])


    # Retrieve pocket PDB files
    # ホロタンパク質のポケットファイルの収集
    holo_pocket_files = {}
    for subdirectory in subdirectories: # subdirectories = ["refined-set", "v2020-other-PL"]
        for pdb_id in os.listdir(os.path.join(pdbbind_directory, subdirectory)): # pdbbind_directory = "../data/pdbbind_dir" (指定したディレクトリ内のすべてのエントリの名前が含まれたリストが返されます。)
            pocket_file_path = os.path.join(pdbbind_directory, subdirectory, pdb_id, f"{pdb_id}_pocket.pdb")
            if os.path.exists(pocket_file_path): # pocket_file_pathがあれば(.pdbファイルがあれば)
                holo_pocket_files[pdb_id] = pocket_file_path

    print("==========Reading or generating pocket data...==========")

    # ポケットのチェインデータを取得、または保存されている結果を読み込む
    pocket_data = {} # ドミナントチェーンがわかってるデータのためのリスト
    no_dominant_chains = [] # ドミナントチェーンを識別できなかったPDB IDを追跡するためのリスト

    if os.path.exists(pocket_data_file): # pocket_data_file = ../csv_file/pocket_data.csv
        with open(pocket_data_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダー行をスキップ
            for row in reader:
                pdb_id, dominant_chain, loop_percentage, ligand_name, ligand_size = row # csvの各行を読み込み、変数に割り当てる
                if dominant_chain:  # Check if dominant_chain is not empty
                    pocket_data[pdb_id] = (dominant_chain, float(loop_percentage), ligand_name, int(ligand_size)) #行の情報を全て格納
                else:
                    no_dominant_chains.append(pdb_id) # idのみ記録
    else: # ポケットデータファイルが存在しない場合
        with concurrent.futures.ProcessPoolExecutor() as executor: # ドミナントチェーンの識別処理を並列に実行
            results = list(tqdm(executor.map(get_dominant_chain, holo_pocket_files.values(), holo_pocket_files.keys()), total=len(holo_pocket_files)))

        for pdb_id, result in zip(holo_pocket_files.keys(), results):
            dominant_chain, loop_percentage, ligand_name, ligand_size = result
            
            if dominant_chain:
                pocket_data[pdb_id] = (dominant_chain, loop_percentage, ligand_name, ligand_size) #行の情報を全て格納
            else:
                no_dominant_chains.append(pdb_id) # idのみ記録

        # ついでに結果をCSVファイルを作成して保存
        with open(pocket_data_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["pdb_id", "dominant_chain", "loop_percentage", "ligand_name", "ligand_size"])
            for pdb_id, (dominant_chain, loop_percentage, ligand_name, ligand_size) in pocket_data.items():
                writer.writerow([pdb_id, dominant_chain, loop_percentage, ligand_name, ligand_size])

        # no_dominant_chainsも保存
        with open(no_dominant_chains_file, 'w') as f:
            f.write('\n'.join(no_dominant_chains))


    print(f"No dominant chain for {len(no_dominant_chains)} PDB IDs. Examples: {no_dominant_chains[:5]}...")


    print("==========Reading or generating ligand info...==========")
    print(len(pocket_data))
    print(list(pocket_data.items())[:1])
    # BLAST検索を行い、または保存されている結果を読み込む
    similar_apo_proteins = {} # 類似のアポタンパク質用
    filtered_apo_proteins = {} # フィルタリングされたアポタンパク質用

    with ProcessPoolExecutor() as executor:
        pdb_id_chain_data_items = list(pocket_data.items())[:1]  # pocket_data : さっきのドミナントがあるデータのリスト
        # tqdmをexecutor.mapに適用し、プログレスバーを表示
        results = list(tqdm(executor.map(handle_blast_search, pdb_id_chain_data_items), total=len(pdb_id_chain_data_items)))
    for pdb_id, similar_proteins in results:
        if similar_proteins:
            similar_apo_proteins[pdb_id] = similar_proteins
    
    #print("similar_apo_proteins : ", similar_apo_proteins)

    # 結果をCSVファイルに保存
    with open(similar_apo_proteins_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", "apo_pdb_id", "apo_chain"])
        for pdb_id, apo_list in similar_apo_proteins.items():
            for apo_pdb_id, apo_chain in apo_list:
                writer.writerow([pdb_id, apo_pdb_id, apo_chain])

    print("===============リガンド情報を取得===============")
    ligand_info_dict = {}


    apo_list = [apo for apo_candidates in similar_apo_proteins.values() for apo in apo_candidates]
    # 例 : apo_list = [("3C4D", "A"), ("5E6F", "B"), ("9I0J", "A")]
    
    # Poolを使用せずに直接ループで処理
    ligand_info_results = parallel_ligand_info_extraction(apo_list)
    
    for result in ligand_info_results:
        apo_pdb_id, ligand_name, atom_count = result
        if ligand_name:
            ligand_info_dict[apo_pdb_id] = (ligand_name, atom_count)
    # 結果をCSVファイルに保存
    with open(ligand_info_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["apo_pdb_id", "ligand_name", "atom_count"])
        for apo_pdb_id, (ligand_name, atom_count) in ligand_info_dict.items():
            writer.writerow([apo_pdb_id, ligand_name, atom_count])

if __name__ == "__main__":
    main()