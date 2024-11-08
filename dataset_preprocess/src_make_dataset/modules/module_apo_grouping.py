import os
import subprocess
import pandas as pd
from Bio.PDB import MMCIFParser
import numpy as np
from Bio.PDB import is_aa
from modules import module_search_apo
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import urllib.request

mmcif_dir = "../mmcif/apo"
fasta_dir = "../fasta/apo"
blast_db_path = "../Blast_database/pdbaa"

'''
apoタンパク質の配列をまとめたfastaファイルの作成
-----------------------------------------------------------------------------
- create_combined_fasta_file(unique_apo_combinations):
    (-) get_standardized_sequence(pdb_id, mmcif_file, chain_id):
-----------------------------------------------------------------------------
'''
def create_combined_fasta_file(unique_apo_combinations, all_apo_fasta_file_path):
    with open(all_apo_fasta_file_path, 'w') as combined_fasta:
        for row in tqdm(unique_apo_combinations.itertuples(), total=len(unique_apo_combinations), desc="Creating FASTA file"): # intertuplesは各行を名前付きタプルとして返します。これにより、行の各列に名前を使ってアクセスできます。
            pdb_id = row.apo_name
            chain_id = row.apo_chain
            mmcif_file = os.path.join(mmcif_dir, f"{pdb_id}.cif")
            sequence = get_standardized_sequence(pdb_id, mmcif_file, chain_id)
            if sequence:
                combined_fasta.write(f">{pdb_id}_{chain_id}\n{sequence}\n")


def get_standardized_sequence(pdb_id, mmcif_file, chain_id):
    """
    指定された PDB ID、mmCIF ファイル、チェーン ID に基づいてシーケンスを取得し、非標準アミノ酸を標準アミノ酸に置き換える
    """
    sequence = module_search_apo.get_chain_sequence(pdb_id, mmcif_file, chain_id)
    if sequence:
        # Replace non-standard amino acids
        sequence = sequence.replace('O', 'K').replace('U', 'C')
    return sequence

'''
類似マトリックスの作成
-----------------------------------------------------------------------------
- calculate_sequence_similarity(blast_db_path, all_fasta_file_path, output_file):
- create_similarity_matrix(blast_output_file):
-----------------------------------------------------------------------------
'''

def calculate_sequence_similarity(blast_db_path, all_fasta_file_path, output_file):
    """
    BLAST+を使用して全配列間の相同性スコアを計算する関数。
    """
    cmd = [
        "blastp",
        "-query", all_fasta_file_path,
        "-db", blast_db_path,
        "-outfmt", "7 qseqid sseqid pident",
        "-out", output_file,
        "-num_threads", "28"  # スレッド数は環境に応じて調整してください
    ]
    print(f"Running BLAST command: {' '.join(cmd)}")  # 実行するコマンドを表示
    result = subprocess.run(cmd, capture_output=True, text=True)  # stdoutとstderrをキャプチャ

    # 出力とエラーを表示
    print("BLAST command stdout:", result.stdout)
    print("BLAST command stderr:", result.stderr)

    # プロセスの終了コードをチェック
    if result.returncode != 0:
        print(f"Error: BLAST command failed with exit code {result.returncode}.")
    else:
        print("BLAST command completed successfully.")

def create_similarity_matrix(blast_output_file):
    blast_results = pd.read_csv(blast_output_file, sep='\t', comment='#', names=["qseqid", "sseqid", "pident"])
    print(f"Loaded BLAST results from {blast_output_file}. Number of records: {len(blast_results)}.")

    # 空の相同性マトリクスを作成
    all_sequences = pd.concat([blast_results['qseqid'], blast_results['sseqid']]).unique()
    similarity_matrix = pd.DataFrame(0.0, index=all_sequences, columns=all_sequences)
    print(f"Created empty similarity matrix with shape {similarity_matrix.shape}.")

    # 相同性マトリクスを埋める
    for _, row in tqdm(blast_results.iterrows(), total=len(blast_results), desc="Filling similarity matrix"):
        similarity_matrix.at[row['qseqid'], row['sseqid']] = row['pident'] / 100.0  # パーセントを小数に変換
        similarity_matrix.at[row['sseqid'], row['qseqid']] = row['pident'] / 100.0  # 対称性を保つ

    print(f"Filled similarity matrix with BLAST results. Final matrix shape: {similarity_matrix.shape}.")
    return similarity_matrix

'''
idの割り当て
-----------------------------------------------------------------------------
- assign_group_ids(similarity_matrix, threshold):
- get_id_from_dict(row, id_dict):
-----------------------------------------------------------------------------
'''

def assign_group_ids(similarity_matrix, threshold):
    # 数値型に変換
    similarity_matrix = similarity_matrix.apply(pd.to_numeric, errors='coerce')

    # マトリクスを二値化する
    binary_matrix = (similarity_matrix >= threshold).astype(int)

    # CSR形式のスパースマトリクスに変換
    '''
    CSR形式（Compressed Sparse Row）は、主にゼロが多く含まれる行列をメモリ効率よく保存し、処理速度を向上させるための形式
    '''
    sparse_matrix = csr_matrix(binary_matrix)

    # 連結成分を抽出
    '''
    connected_components 関数は、スパースマトリクスを無向グラフとして解釈し、グラフ内の連結成分（相互に接続されたグループ）を抽出する
    '''
    n_components, labels = connected_components(csgraph=sparse_matrix, directed=False)

    # タンパク質名とラベルをマッピング
    protein_to_group = {name: group_id for name, group_id in zip(similarity_matrix.index, labels)}

    return protein_to_group


def get_id_from_dict(row, id_dict):
    # 'apo_name' と 'apo_chain' を結合してキーを作成
    key = f"{row['apo_name']}_{row['apo_chain']}"
    # 辞書から対応するIDを取得
    return id_dict.get(key)

'''
idの割り当て
-----------------------------------------------------------------------------
- assign_group_ids(similarity_matrix, threshold):
- get_id_from_dict(row, id_dict):
-----------------------------------------------------------------------------
'''

def process_apo_beta_record(row, fasta_directory, mmcif_directory, apo_holo_pairs):
    results = []
    apo_name = row["apo_name"]
    apo_chain = row["apo_chain"]
    fasta_file = ensure_fasta_file(apo_name, apo_chain, fasta_directory, mmcif_directory)
    
    similar_sequences = search_similar_sequences(fasta_file, blast_db_path)
    for seq in similar_sequences:
        seq_apo_name = seq.split("_")[0].lower()  # 小文字に変換
        if not apo_holo_pairs["apo_name"].str.lower().isin([seq_apo_name]).any() and not apo_holo_pairs["holo_name"].str.lower().isin([seq_apo_name]).any():
            if is_ligand_free(seq.split("_")[0].upper()):
                print("I found!!", seq)
                results.append({
                    "apo_name": seq_apo_name.upper(),  # PDB IDを想定、必要に応じて修正
                    "apo_chain": seq.split("_")[1],
                    "protein_id": row["protein_id"],
                    "family50_id": row["family50_id"],
                    "data_augmentation": 1
                })
    return results

def ensure_fasta_file(pdb_id, chain_id, fasta_directory, mmcif_directory):
    fasta_file = os.path.join(fasta_directory, f"{pdb_id}_{chain_id}.fasta")
    if not os.path.exists(fasta_file):
        mmcif_file = os.path.join(mmcif_directory, f"{pdb_id}.cif")
        sequence = module_search_apo.get_chain_sequence(pdb_id, mmcif_file, chain_id)
        if sequence:
            with open(fasta_file, "w") as f:
                f.write(f">{pdb_id}_{chain_id}\n{sequence}\n")
        else:
            print(f"Failed to retrieve sequence for {pdb_id}_{chain_id}")
    return fasta_file

def search_similar_sequences(fasta_path, blast_db_path):
    """
    指定されたFASTAファイルに対してBLAST検索を実行し、
    特定の条件を満たす類似のタンパク質のリストを返します。
    """
    # ローカルBLAST検索を実行
    cmd = [
        "blastp",
        "-query", fasta_path,
        "-db", blast_db_path,
        "-outfmt", "6 sseqid qseqid length qcovs pident"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)

    similar_proteins = []
    for line in result.stdout.split("\n"):
        if line:
            sseqid, qseqid, length, qcovs, pident = line.split()
            length = int(length)
            qcovs = int(qcovs)
            pident = float(pident)

            # 特定の閾値を満たす場合にのみ、リストに追加します
            if pident >= 90 and qcovs >= 80:
                if pident<=90:
                    print(pident, qcovs, sseqid)
                pdb_id, chain_id = sseqid.split("_")
                similar_proteins.append(sseqid)

    return similar_proteins

def is_ligand_free(pdb_id):
    """
    指定されたPDB IDのタンパク質がリガンドフリーかどうかを判断します。
    """
    mmcif_file = os.path.join(mmcif_dir, f"{pdb_id}.cif")

    # mmCIFファイルが存在するかチェック
    if not os.path.exists(mmcif_file):
        print(f"mmCIF file for {pdb_id} not found, downloading...")
        mmcif_file = module_search_apo.download_mmcif(pdb_id, mmcif_dir)

        if mmcif_file is None:
            return False  # ダウンロードに失敗した場合

    ligand_name, atom_count = module_search_apo.check_ligand_info_mmcif_inner(mmcif_file, pdb_id)

    return ligand_name == "" and atom_count == 0