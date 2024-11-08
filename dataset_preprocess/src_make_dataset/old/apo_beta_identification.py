import os
import subprocess
import pandas as pd
from Bio import SeqIO, SearchIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices
from Bio.PDB import MMCIFParser
from itertools import combinations
from scipy.spatial import distance
import numpy as np
from Bio.PDB import is_aa
from ready_pdbbind import check_ligand_info_mmcif_inner, get_chain_sequence
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio.Blast.Applications import NcbimakeblastdbCommandline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import urllib.request


# Directories and file settings
pdbbind_directory = "../data/pdbbind_dir"
subdirectories = ["refined-set", "v2020-other-PL"]
mmcif_directory = "../data/mmcif/apo"
csv_directory = "../csv_file"
fasta_directory = "../data/fasta/apo"
blast_output_directory = "../data/fasta/blast_out"


blast_db_path = "../blast_db/pdbaa"
ll_fasta_file_path = os.path.join(fasta_directory, "all_apo_sequences.fasta")
tmalign_path = "/gs/hs0/tga-ishidalab/share/TMtools-20190822/TMalign"
apo_holo_pairs_path = os.path.join(csv_directory, "apo_holo_pairs.csv")
similarity_matrix_path = os.path.join(csv_directory, "similarity_matrix.csv")
apo_protein_id_path = os.path.join(csv_directory, "apo_protein_id.csv")
holo_protein_id_path = os.path.join(csv_directory, "holo_protein_id.csv")
apo_protein_beta_path = os.path.join(csv_directory, "apo_protein_beta.csv")
combined_path = os.path.join(csv_directory, "combined_apo_protein.csv")
tmalign_scores_path = os.path.join(csv_directory, "tmalign_scores.csv")
apo_holo_protein_id_path = os.path.join(csv_directory, "apo_holo_protein_id.csv")
# 1. 必要なライブラリと関数をインポート

def run_blast(query_file, db_path, out_file):
    """Run BLAST+ on a given FASTA file."""
    cmd = [
        "blastp",
        "-query", query_file,
        "-db", db_path,
        "-out", out_file,
        "-outfmt", "6 sseqid qseqid length qcovs pident"
    ]
    subprocess.run(cmd)


def run_tmalign(structure1, structure2):
    """Run TMalign on two given structures."""
    cmd = ["/gs/hs0/tga-ishidalab/share/TMtools-20190822/TMalign", structure1, structure2]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "TM-score=" in line:
            return float(line.split()[1])

def download_mmcif(pdb_id, directory):
    """
    指定されたPDB IDのmmCIFファイルをダウンロードします。
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    file_path = os.path.join(directory, f"{pdb_id}.cif")

    try:
        urllib.request.urlretrieve(url, file_path)
    except urllib.error.URLError as e:
        print(f"Error downloading mmCIF file for {pdb_id}: {e}")
        return None

    return file_path

def assign_group_ids(similarity_matrix, threshold):
    # マトリクスを二値化する
    binary_matrix = (similarity_matrix >= threshold).astype(int)

    # CSR形式のスパースマトリクスに変換
    sparse_matrix = csr_matrix(binary_matrix)

    # 連結成分を抽出
    n_components, labels = connected_components(csgraph=sparse_matrix, directed=False)

    # タンパク質名とラベルをマッピング
    protein_to_group = {name: group_id for name, group_id in zip(similarity_matrix.index, labels)}

    return protein_to_group

def extract_chain_from_cif(cif_path, chain_id):
    """Extracts a specific chain from a CIF file and returns its FASTA sequence."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_path)
    chain = structure[0][chain_id]
    return "".join([residue.resname for residue in chain if is_aa(residue)])

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
    mmcif_file = os.path.join(mmcif_directory, f"{pdb_id}.cif")

    # mmCIFファイルが存在するかチェック
    if not os.path.exists(mmcif_file):
        print(f"mmCIF file for {pdb_id} not found, downloading...")
        mmcif_file = download_mmcif(pdb_id, mmcif_directory)

        if mmcif_file is None:
            return False  # ダウンロードに失敗した場合

    ligand_name, atom_count = check_ligand_info_mmcif_inner(mmcif_file, pdb_id)

    return ligand_name == "" and atom_count == 0



def calculate_tmalign(structure1, structure2, tmalign_path):
    """Calculate TM-align score between two structures."""
    result = subprocess.run([tmalign_path, structure1, structure2], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "TM-score" in line and "Chain_1" in line:
            return float(line.split()[1])
    return None


def ensure_fasta_file(pdb_id, chain_id, fasta_directory, mmcif_directory):
    fasta_file = os.path.join(fasta_directory, f"{pdb_id}_{chain_id}.fasta")
    if not os.path.exists(fasta_file):
        mmcif_file = os.path.join(mmcif_directory, f"{pdb_id}.cif")
        sequence = get_chain_sequence(pdb_id, mmcif_file, chain_id)
        if sequence:
            with open(fasta_file, "w") as f:
                f.write(f">{pdb_id}_{chain_id}\n{sequence}\n")
        else:
            print(f"Failed to retrieve sequence for {pdb_id}_{chain_id}")
    return fasta_file


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

def get_standardized_sequence(pdb_id, mmcif_file, chain_id):
    """Retrieve and standardize the sequence for the specified chain."""
    sequence = get_chain_sequence(pdb_id, mmcif_file, chain_id)
    if sequence:
        # Replace non-standard amino acids
        sequence = sequence.replace('O', 'K').replace('U', 'C')
    return sequence

def create_fasta_file(row):
    pdb_id = row.apo_name
    chain_id = row.apo_chain
    mmcif_file = os.path.join(mmcif_directory, f"{pdb_id}.cif")
    fasta_file = os.path.join(fasta_directory, f"{pdb_id}_{chain_id}.fasta")

    if not os.path.exists(fasta_file):
        sequence = get_standardized_sequence(pdb_id, mmcif_file, chain_id)
        if sequence:
            with open(fasta_file, "w") as f:
                f.write(f">{pdb_id}_{chain_id}\n{sequence}\n")
        else:
            print(f"Failed to retrieve sequence for {pdb_id}_{chain_id}")
    else:
        print(f"FASTA file for {pdb_id}_{chain_id} already exists.")


    return fasta_file

def create_combined_fasta_file(unique_apo_combinations):
    """
    すべてのアポタンパク質の組み合わせから一つのFASTAファイルを作成する関数。
    """
    with open(all_fasta_file_path, 'w') as combined_fasta:
        for row in unique_apo_combinations.itertuples():
            pdb_id = row.apo_name
            chain_id = row.apo_chain
            mmcif_file = os.path.join(mmcif_directory, f"{pdb_id}.cif")
            sequence = get_standardized_sequence(pdb_id, mmcif_file, chain_id)
            if sequence:
                combined_fasta.write(f">{pdb_id}_{chain_id}\n{sequence}\n")

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


def build_similarity_matrix(blast_output_file):
    """
    BLASTの出力から相同性マトリックスを構築する関数。
    """
    # BLASTの結果を読み込む
    blast_results = pd.read_csv(blast_output_file, sep='\t', comment='#', names=["qseqid", "sseqid", "pident"])
    print(f"Loaded BLAST results from {blast_output_file}. Number of records: {len(blast_results)}.")

    # 空の相同性マトリクスを作成
    all_sequences = pd.concat([blast_results['qseqid'], blast_results['sseqid']]).unique()
    similarity_matrix = pd.DataFrame(0, index=all_sequences, columns=all_sequences)
    print(f"Created empty similarity matrix with shape {similarity_matrix.shape}.")

    # 相同性マトリクスを埋める
    for _, row in blast_results.iterrows():
        similarity_matrix.at[row['qseqid'], row['sseqid']] = row['pident'] / 100.0  # パーセントを小数に変換
        similarity_matrix.at[row['sseqid'], row['qseqid']] = row['pident'] / 100.0  # 対称性を保つ

    print(f"Filled similarity matrix with BLAST results. Final matrix shape: {similarity_matrix.shape}.")
    return similarity_matrix

def create_blast_database(fasta_file_path, db_path):
    """Create a BLAST database from a FASTA file."""
    makeblastdb_cline = NcbimakeblastdbCommandline(
        dbtype="prot",  # for protein sequences
        input_file=fasta_file_path,
        out=db_path
    )
    stdout, stderr = makeblastdb_cline()


def get_id_from_dict(row, id_dict):
    # 'apo_name' と 'apo_chain' を結合してキーを作成
    key = f"{row['apo_name']}_{row['apo_chain']}"
    # 辞書から対応するIDを取得
    return id_dict.get(key)

def main():
    # 3. apo_holo_pairs.csvを読み込む
    apo_holo_pairs = pd.read_csv(apo_holo_pairs_path)

    # 4. 全てのアポタンパク質の組み合わせから一つのFASTAファイルを作成
    unique_apo_combinations = apo_holo_pairs[['apo_name', 'apo_chain']].drop_duplicates()
    
    create_combined_fasta_file(unique_apo_combinations)

    # 5. BLASTデータベースの作成
    create_blast_database(all_fasta_file_path, blast_db_path)    
    # 6. BLAST+を使用して全配列間の相同性を計算
    blast_output_file = os.path.join(blast_output_directory, "blast_output.txt")
    calculate_sequence_similarity(blast_db_path, all_fasta_file_path, blast_output_file)    # 7. 相同性マトリクスの構築
    similarity_matrix = build_similarity_matrix(blast_output_file)    
    similarity_matrix.to_csv(similarity_matrix_path)
    
    #6. グループIDの割り当て
    similarity_matrix = pd.read_csv(similarity_matrix_path, index_col=0)

    protein_ids_dict = assign_group_ids(similarity_matrix, 0.99)  # または適切な閾値
    print(protein_ids_dict)
    family50_ids_dict = assign_group_ids(similarity_matrix, 0.50)  # または適切な閾値

    # similarity_matrixからapo_nameとapo_chainを抽出
    names_chains = similarity_matrix.index.str.split("_", expand=True)
    names_chains = pd.DataFrame(names_chains.tolist(), columns=['apo_name', 'apo_chain'])  # ここでDataFrameに変換します。
    # 'apo_name' と 'apo_chain' を使って 'protein_id' と 'family50_id' を取得
    names_chains['protein_id'] = names_chains.apply(lambda row: get_id_from_dict(row, protein_ids_dict), axis=1)
    names_chains['family50_id'] = names_chains.apply(lambda row: get_id_from_dict(row, family50_ids_dict), axis=1)
    # CSVファイルとして保存
    names_chains.to_csv(apo_protein_id_path, index=False)
    
    # apo_holo_pairsにapo_protein_id_dfを結合する
    merged_df = pd.merge(apo_holo_pairs, names_chains, how='left', on=['apo_name', 'apo_chain'])
    holo_protein_id_df = merged_df[['holo_name', 'holo_chain', 'ligand', 'ligand_atom_count', 'loop_per']].drop_duplicates()
    holo_protein_id_df.to_csv(holo_protein_id_path, index=False)

    apo_protein_id_df = pd.read_csv(apo_protein_id_path)

    # 9. Create apo_protein_beta.csv
    apo_beta_records = []
    with ThreadPoolExecutor(max_workers=28) as executor:
        tasks = [executor.submit(process_apo_beta_record, row, fasta_directory, mmcif_directory, apo_holo_pairs) for _, row in apo_protein_id_df.iterrows()]
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Creating apo_protein_beta"):
            apo_beta_records.extend(future.result())
    apo_beta_df = pd.DataFrame(apo_beta_records)
    apo_beta_df.to_csv(apo_protein_beta_path, index=False)


    print("start combined")
    
    # 10. Concatenate apo_protein_id.csv and apo_protein_beta.csv
    combined_df = pd.concat([apo_protein_id_df, apo_beta_df])
    combined_df.to_csv(combined_path, index=False)


    # 11. Create a TM-align score matrix
    pdb_ids = combined_df["apo_name"].unique()
    tmalign_scores = pd.DataFrame(index=pdb_ids, columns=pdb_ids)


    with ThreadPoolExecutor(max_workers=28) as executor:
        future_to_score = {executor.submit(calculate_tmalign, f"../data/mmcif/apo/{pdb_ids[i]}.cif", f"../data/mmcif/apo/{pdb_ids[j]}.cif", tmalign_path): (pdb_ids[i], pdb_ids[j]) for i in range(len(pdb_ids)) for j in range(i, len(pdb_ids))}
        for future in tqdm(as_completed(future_to_score), total=len(future_to_score), desc="TM-align scoring"):
            pair_i, pair_j = future_to_score[future]
            score = future.result()
            tmalign_scores.at[pair_i, pair_j] = score
            tmalign_scores.at[pair_j, pair_i] = score  # Symmetric

    # Save the matrix for future reference
    tmalign_scores.to_csv(tmalign_scores_path)

    # 12. Assign TMalign05_id and TMalign07_id based on the TM-align score matrix
    thresholds = {"TMalign05_id": 0.5, "TMalign07_id": 0.7}

    for column, threshold in thresholds.items():
        # Clustering
        clusters = {}
        cluster_id = 0
        for pdb_id in pdb_ids:
            if pdb_id not in clusters:
                clusters[pdb_id] = cluster_id
                cluster_id += 1
            for other_pdb in pdb_ids:
                if tmalign_scores.at[pdb_id, other_pdb] > threshold:
                    clusters[other_pdb] = clusters[pdb_id]
  
        # Assigning IDs
        combined_df[column] = combined_df["apo_name"].map(clusters)

    # 13. Save the final dataframe
    combined_df.to_csv(apo_holo_protein_id_path, index=False)

if __name__ == "__main__":
    main()