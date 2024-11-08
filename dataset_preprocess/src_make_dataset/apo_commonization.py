import os
import sys
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
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
#from Bio.Blast.Applications import NcbimakeblastdbCommandline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import urllib.request

from IshidaLab_research_project.dataset_preprocess.src_make_dataset.modules import module_apo_grouping
from modules import module_search_apo

# Directories and file settings
pdbbind_dir = "../PDBbind_original_data"
subdir = ["refined-set", "v2020-other-PL"]
mmcif_dir = "../mmcif/apo"
csv_dir = "../output_csv_files"
blast_db_path = "../blast_db/pdbaa"
fasta_dir = "../data/fasta/apo"
blast_output_dir = "../data/fasta/blast_out"

all_fasta_file_path = os.path.join(fasta_dir, "all_apo_sequences.fasta")

# csv paths
similarity_matrix_csv = os.path.join(csv_dir, "intermediate_datas/similarity_matrix.csv")
pocket_data_csv = os.path.join(csv_dir, "intermediate_datas/pocket_data.csv")
apo_protein_id_csv = os.path.join(csv_dir, "intermediate_datas/apo_protein_id.csv")
no_dominant_chains_csv = os.path.join(csv_dir, "intermediate_datas/no_dominant_chains.csv")
similar_apo_proteins_csv = os.path.join(csv_dir, "intermediate_datas/similar_apo_proteins.csv")
ligand_info_csv = os.path.join(csv_dir, "intermediate_datas/ligand_info.csv")

apo_holo_pairs_csv = os.path.join(csv_dir, "phaze_1/apo_holo_pairs_sakai.csv")
apo_holo_protein_id_csv = os.path.join(csv_dir, "intermediate_datas/apo_holo_protein_id.csv")

def main():
    apo_holo_pairs = pd.read_csv(apo_holo_pairs_csv)
    
    # アポタンパク質のみ取り出し、重複を削除
    unique_apo_combinations = apo_holo_pairs[['apo_name', 'apo_chain']].drop_duplicates()
    print(unique_apo_combinations.shape[0])
    module_apo_grouping.create_combined_fasta_file(unique_apo_combinations)

    #modules_2.create_blast_database(all_fasta_file_path, blast_db_path)

    # Blast+による全配列間の相同性を計算
    #blast_output = os.path.join(blast_output_dir, "blast_output_sakai.txt")
    #modules_2.calculate_sequence_similarity(blast_db_path, all_fasta_file_path, blast_output)
    """
    !!酒井さんのデータとデータ数が全然違う
    !!ただ、apo_holo
    """

    #similarity_matrix = modules_2.create_similarity_matrix(blast_output)
    #similarity_matrix.to_csv(similarity_matrix_csv)

    ####

    #similarity_matrix = pd.read_csv(similarity_matrix_csv, index_col=0)
#
    #protein_ids_dict = modules_2.assign_group_ids(similarity_matrix, 0.99)  # または適切な閾値
    #print(protein_ids_dict)
    #
    #family50_ids_dict = modules_2.assign_group_ids(similarity_matrix, 0.50)  # または適切な閾値    # similarity_matrixからapo_nameとapo_chainを抽出
    #
    #names_chains = similarity_matrix.index.str.split("_", expand=True)
    #names_chains = pd.DataFrame(names_chains.tolist(), columns=['apo_name', 'apo_chain'])  # ここでDataFrameに変換します。    # 'apo_name' と 'apo_chain' を使って 'protein_id' と 'family50_id' を取得
    #names_chains['protein_id'] = names_chains.apply(lambda row: modules_2.get_id_from_dict(row, protein_ids_dict), axis=1)
    #names_chains['family50_id'] = names_chains.apply(lambda row: modules_2.get_id_from_dict(row, family50_ids_dict), axis=1)    # CSVファイルとして保存
    #
    #names_chains.to_csv(apo_protein_id_csv, index=False)    # apo_holo_pairsにapo_protein_id_dfを結合する
    #
    #merged_df = pd.merge(apo_holo_pairs, names_chains, how='left', on=['apo_name', 'apo_chain'])
    #holo_protein_id_df = merged_df[['holo_name', 'holo_chain', 'ligand', 'ligand_atom_count', 'loop_per']].drop_duplicates()
    #holo_protein_id_df.to_csv(apo_holo_protein_id_csv, index=False)

if __name__ == "__main__":
    main()