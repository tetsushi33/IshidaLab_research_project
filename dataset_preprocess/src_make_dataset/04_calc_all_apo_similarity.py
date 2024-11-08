import os
import pandas as pd
from modules import module_apo_grouping
from Bio.Blast.Applications import NcbimakeblastdbCommandline

blast_db_path = "../Blast_database/pdbaa"

# input
apo_holo_pairs_csv = "../output_csv_files/phase_03/ver_1/apo_holo_pairs.csv"

# output
all_apo_fasta_file_path = "../fasta/apo/all_apo_sequences.fasta"

output_csv_dir = "../output_csv_files/phase_04/ver_1"
blast_output_path = os.path.join(output_csv_dir, "blast_output.txt")
similarity_matrix_csv = os.path.join(output_csv_dir, "similarity_matrix.csv")

def main():
    # 保存先ディレクトリの確認
    if not os.path.exists(output_csv_dir):
        print("No directory - ", output_csv_dir)
        return

    apo_holo_pairs = pd.read_csv(apo_holo_pairs_csv)
    print("num of apo_holo_pairs : ", apo_holo_pairs.shape[0])
    target_apo_list = apo_holo_pairs[['apo_name', 'apo_chain']].drop_duplicates() # apo_holo_pairs内のapoを重複なく抽出
    print("num of kind of apo in pairs : ", len(target_apo_list))

    print("=============アポをまとめたfastaファイル作成=============")
    if not os.path.exists(all_apo_fasta_file_path):
        module_apo_grouping.create_combined_fasta_file(target_apo_list, all_apo_fasta_file_path)
        print("結果を保存 -----> ", all_apo_fasta_file_path)
    else:
        print("all_apo_fasta データ作成済み : ", all_apo_fasta_file_path)

    print("=============Blastデータベース（マトリックス）作成=============")
    makeblastdb_cline = NcbimakeblastdbCommandline(
        dbtype="prot",  # for protein sequences
        input_file=all_apo_fasta_file_path,
        out=blast_db_path
    )
    stdout, stderr = makeblastdb_cline()

    if not os.path.exists(blast_output_path):
        with open(blast_output_path, 'w') as output_file:
            pass
        print(f"File created: {blast_output_path}")
    #module_apo_grouping.calculate_sequence_similarity(blast_db_path, all_apo_fasta_file_path, blast_output_path)
    similarity_matrix = module_apo_grouping.create_similarity_matrix(blast_output_path)    
    similarity_matrix.to_csv(similarity_matrix_csv)
    print("結果を保存 -----> ", similarity_matrix_csv)

if __name__ == "__main__":
    main()