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

AMINO_ACID_CODE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X"
}
    
exclude_list = ["WAT", "HOH", "NA", "CL", "MG", "K", "CA", "ZN", "SO4", "PO4", "DA", "DT", "DC", "DG", "U", "A", "G", "C"]


def handle_blast_search(pdb_id_chain_data):
    pdb_id, chain_data = pdb_id_chain_data
    chain_id = chain_data[0]
    #mmcif_path = os.path.join(mmcif_dir, "holo", f"{pdb_id}.cif")
    #sequence = get_chain_sequence(pdb_id, mmcif_path, chain_id)
    #fasta_path = save_fasta_sequence(pdb_id, chain_id, sequence)
    #print(fasta_path)

    # perform_blast_search now takes the path to the fasta file as argument
    #return pdb_id, perform_blast_search(fasta_path)

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
    blast_db_path = "../../Blast_database/pdbaa"
    # ローカルBLAST検索を実行
    cmd = [
        "blastp",
        "-query", fasta_path,
        "-db", blast_db_path,
        "-outfmt", "6 sseqid qseqid length qcovs pident"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    
    #print(result.stdout)
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

    if(len(similar_protein_chains) == 0):
        print("類似タンパク質が見つかりませんでした")
    else:
        print("類似タンパク質が見つかりました")

    #print(hit_chains)
    return similar_protein_chains

    #target_results = {}
    #for pair in target_pairs:
    #    target_results[pair] = {'pident': None, 'qcovs': None}
#
    #for line in result.stdout.split("\n"):
    #    if line:
    #        sseqid, qseqid, length, qcovs, pident = line.split()
    #        length = int(length)
    #        qcovs = int(qcovs)
    #        pident = float(pident)
    #        
    #        for target_pair in target_pairs:
    #            target_sseqid, target_qseqid = target_pair
    #            if sseqid == target_sseqid and qseqid == target_qseqid:
    #                target_results[target_pair]['pident'] = pident
    #                target_results[target_pair]['qcovs'] = qcovs
    #
    #return target_results


def main():
    #tructure = parser.get_structure("../PDBbind_original_data/refined-set/4msa/4msa_pocket.pdb", "../mmcif/holo/4msa.cif")
    #tructure_all = parser.get_structure("../PDBbind_original_data/refined-set/4msa/4msa_protein.pdb", "../mmcif/holo/4msa.cif")
    #ull_structure_file_path = "../mmcif/holo/4msa.cif"
    #ocket_file_path = "../PDBbind_original_data/refined-set/4msa/4msa_pocket.pdb"
    #db_id = "4msa"
    fasta_path = "../../data/fasta/holo/5cil_A.fasta"

    #target_pairs = [("2OUV_A", "5nwe_A"), ("4LKQ_A", "5nwe_A"), ("5UWF_C", "5nwe_A")]

    result = perform_blast_search(fasta_path)
    print(result)
    print(len(result))
    
        

if __name__ == "__main__":
    main()

