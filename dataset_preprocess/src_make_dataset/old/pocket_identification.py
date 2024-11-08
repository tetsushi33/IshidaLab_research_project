import pandas as pd
import os
from pymol import cmd, stored
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Align import PairwiseAligner
from tqdm import tqdm
import sys
import argparse


# CSVファイルの読み込み
apo_data = pd.read_csv('../csv_file/filtered_apo_protein_info.csv')
holo_data = pd.read_csv('../csv_file/holo_protein_info.csv')
conversion_dict = pd.read_csv('../csv_file/non_amino_2_amino.csv').set_index('Non-standard AA Code')['Standard AA Code'].to_dict()
apo_holo_pairs = pd.read_csv('../csv_file/apo_holo_pairs.csv')

colors = ["red", "green", "blue", "yellow", "orange", "purple"]
# ディレクトリパスを定義
holo_pocket_dir_refined = "../data/pdbbind_dir/refined-set/"
holo_pocket_dir_other = "../data/pdbbind_dir/v2020-other-PL/"

results_df = pd.DataFrame(columns=['apo_name', 'apo_chain', 'holo_name', 'holo_chain', 'pocket_id', 'pocket_rmsd', 'protein_id', 'family50_id', 'ligand', 'ligand_atom', 'loop_per', 'pocket_com'])

AMINO_ACID_CODE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X"
}



def parse_cif_for_poly_seq_scheme(cif_file_path):
    poly_seq_scheme = []
    with open(cif_file_path, 'r') as file:
        lines = file.readlines()
        capture = False
        for line in lines:
            if line.startswith('_pdbx_poly_seq_scheme.'):
                capture = True
            elif line.startswith('#'):
                capture = False
            elif capture:
                poly_seq_scheme.append(line.strip().split())
    return poly_seq_scheme

def parse_cif_for_atom_site(cif_file_path):
    atom_site_data = []
    headers = []

    with open(cif_file_path, 'r') as file:
        lines = file.readlines()
        atom_site_section = False

        for line in lines:
            # _atom_site. で始まる行が見つかったらヘッダーに追加
            if line.startswith('_atom_site.'):
                atom_site_section = True
                headers.append(line.strip())
                continue

            # ヘッダーの読み込みが終了した後のデータ行を処理
            if atom_site_section and not line.startswith('_atom_site.'):
                if line.startswith('#'):
                    break  # セクションの終了

                # 空白文字で分割して辞書に格納
                values = line.strip().split()
                atom_site_data.append(dict(zip(headers, values)))

    return atom_site_data

def get_residue_counts(poly_seq_lines, residue_index, seq_id_index):
    residue_counts = {}
    for line in poly_seq_lines:
        if len(line) > max(residue_index, seq_id_index):
            residue = line[residue_index].strip()
            seq_id_str = line[seq_id_index].strip()

            # seq_idが数値でない場合は欠損として扱う
            if seq_id_str.isdigit():
                seq_id = int(seq_id_str)
            else:
                seq_id = None  # 欠損を示す

            residue_id = (residue, seq_id)
            residue_counts[residue_id] = residue_counts.get(residue_id, 0) + 1

    return residue_counts

def calculate_missing_coordinates_percentage(cif_file_path, chain, aligned_residue_ids):
    poly_seq_lines = parse_cif_for_poly_seq_scheme(cif_file_path)
    atom_site_lines = parse_cif_for_atom_site(cif_file_path)

    # 座標情報を持つ残基のセットを作成
    atom_site_residues = set()
    # print("chain", chain)
    for line in atom_site_lines:
        if ('_atom_site.auth_asym_id' in line and line['_atom_site.auth_asym_id'].strip() == chain) or ('_atom_site.label_asym_id' in line and line['_atom_site.label_asym_id'].strip() == chain):
            seq_id_key = '_atom_site.auth_seq_id' if '_atom_site.auth_seq_id' in line else '_atom_site.label_seq_id'
            seq_id = int(line[seq_id_key])
            atom_site_residues.add(seq_id)

    # print(atom_site_residues)
    # 残基数と欠損残基数の初期化
    total_residues = 0
    missing_residues = 0

    # aligned_residue_idsをセットに変換
    aligned_residue_ids_set = set(aligned_residue_ids)

    for residue_id in aligned_residue_ids_set:
        total_residues += 1
        # print(f"Residue {residue_id} in chain {chain}")

        if residue_id[1] not in atom_site_residues:  # 残基に座標がない場合
            missing_residues += 1
            print(f"Missing residue at {residue_id}")

    print(f"Total aligned residues: {total_residues}, Missing aligned residues: {missing_residues}")

    # 欠損割合の計算
    if total_residues == 0:
        return 0

    missing_percentage = (missing_residues / total_residues) * 100
    print(f"Missing coordinates percentage: {missing_percentage}%")
    return missing_percentage


# def calculate_missing_coordinates_percentage(structure, aligned_residue_ids):
#     missing_count = 0
#     for chain_id, res_num in aligned_residue_ids:
#         try:
#             residue = structure[0][chain_id][(' ', res_num, ' ')]  # 正しい残基を取得
#             if not residue.child_list:  # 子リスト（原子リスト）が空の場合、座標情報が欠損している
#                 missing_count += 1
#         except KeyError:
#             # 指定されたチェインIDまたは残基番号が存在しない場合
#             continue

#     total_aligned_residues = len(aligned_residue_ids)
#     if total_aligned_residues == 0:
#         return 0

#     return missing_count / total_aligned_residues * 100


def determine_most_common_apo(protein_id, apo_holo_pairs, apo_data):
    apo_names = apo_data[apo_data['protein_id'] == protein_id]['apo_name'].str.upper()
    filtered_apo_holo_pairs = apo_holo_pairs[apo_holo_pairs['apo_name'].str.upper().isin(apo_names)]

    if filtered_apo_holo_pairs.empty:
        return None

    common_apo = filtered_apo_holo_pairs['apo_name'].value_counts().idxmax().upper()
    return common_apo




def get_sequence_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    sequences = []
    for model in structure:
        for chain in model:
            sequence = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue

                resname = residue.get_resname().strip()
                if resname in AMINO_ACID_CODE:
                    sequence.append(AMINO_ACID_CODE[resname])
                elif resname in conversion_dict:
                    sequence.append(conversion_dict[resname])
                else:
                    sequence.append("X")
                    print(f"Non-standard amino acid found: {resname}. Converted to 'X'.")

            sequences.append("".join(sequence))

    combined_sequence = "".join(sequences)
    return combined_sequence

def store_residue_indices(selection_name):
    stored.residue_indices = []
    cmd.iterate(f"{selection_name} and name CA", "stored.residue_indices.append(resi)")

def calculate_rmsd_if_aligned_enough(align_result, holo_pocket_atom_count, threshold_ratio=0.3):
    """
    align_result と holo_pocket_atom_count に基づいて RMSD を計算します。
    重ね合わせに成功した分子数が指定された閾値以上の場合のみ RMSD を返します。
    
    :param align_result: cmd.align 関数の結果
    :param holo_pocket_atom_count: ホロタンパク質のポケットの分子数
    :param threshold_ratio: 重ね合わせに成功する必要がある分子数の割合 (例: 0.1)
    :return: 適切な場合のみ RMSD、そうでなければ float("inf")
    """
    aligned_atom_count = align_result[4]  # 重ね合わせで成功した分子数

    if aligned_atom_count >= threshold_ratio * holo_pocket_atom_count:
        return align_result[0]  # RMSD
    else:
        return float("inf")

def select_aligned_residues_and_store_numbers(apo_chain_id, alignment, apo_residue_numbers):
    selected_residue_numbers = []
    query = []

    apo_index = 0
    holo_index = 0

    for apo_res, holo_res in zip(alignment[0], alignment[1]):
        if apo_res != "-" and holo_res != "-":
            # Both residues are aligned
            query.append(f"(chain {apo_chain_id} and resi {apo_residue_numbers[apo_index]})")
            selected_residue_numbers.append(apo_residue_numbers[apo_index])

        if apo_res != "-":
            apo_index += 1
        if holo_res != "-":
            holo_index += 1
    
    return " or ".join(query), selected_residue_numbers

# FASTAファイルとPDBファイルの残基番号を対応付ける関数
def create_fasta_to_pdb_mapping(fasta_seq, pdb_residue_numbers):
    fasta_to_pdb_mapping = {}
    fasta_index, pdb_index = 0, 0

    while fasta_index < len(fasta_seq) and pdb_index < len(pdb_residue_numbers):
        if fasta_seq[fasta_index] != '-':
            fasta_to_pdb_mapping[fasta_index + 1] = pdb_residue_numbers[pdb_index]
            pdb_index += 1
        fasta_index += 1

    return fasta_to_pdb_mapping

def get_residue_numbers_from_mmcif(mmcif_path, chain_id):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", mmcif_path)
    residue_numbers = []

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[0] == " ":
                        residue_numbers.append(residue.id[1])

    return residue_numbers

# FASTAファイルから配列を読み込む関数
def read_fasta(file_path):
    record = SeqIO.read(file_path, "fasta")
    return str(record.seq)

# ペアワイズアライメントを行う関数
def align_sequences(seq1, seq2):
    aligner = PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]
    return best_alignment


# 隣接する残基の数をカウントする関数
def count_adjacent_residues(residues1, residues2):
    count = 0
    for resi1 in residues1:
        for resi2 in residues2:
            if abs(int(resi1) - int(resi2)) == 1:
                count += 1
    return count

# ポケットの重心を計算する関数
def calculate_centroid(selection):
    stored.xyz = []
    cmd.iterate_state(1, selection, 'stored.xyz.append([x,y,z])')
    x, y, z = zip(*stored.xyz)
    centroid = [sum(x)/len(x), sum(y)/len(y), sum(z)/len(z)]
    return centroid

def create_mapping_from_mmcif_to_fasta(mmcif_path, chain_id):
    """
    mmCIFファイルからFASTAシーケンスへのマッピングを生成する関数
    FastaシーケンスのID - 残基番号
    のマッピング
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", mmcif_path)
    mapping = {}
    index = 1  # FASTAシーケンスのインデックス

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if residue.id[0] == " ":
                        mapping[index] = residue.id[1]
                        index += 1
    return mapping

def create_mapping_from_alignment(alignment, fasta_to_pdb_mapping_a, fasta_to_pdb_mapping_b):
    """
    アポタンパク質AとBのペアワイズアライメントに基づいて、AからBへのマッピングを生成する関数
    """
    mapping = {}
    fasta_index_a, fasta_index_b = 1, 1

    for apo_a_res, apo_b_res in zip(alignment[0], alignment[1]):
        if apo_a_res != '-' and apo_b_res != '-':
            pdb_index_a = fasta_to_pdb_mapping_a[fasta_index_a]
            pdb_index_b = fasta_to_pdb_mapping_b[fasta_index_b]
            mapping[pdb_index_a] = pdb_index_b

        if apo_a_res != '-':
            fasta_index_a += 1
        if apo_b_res != '-':
            fasta_index_b += 1

    return mapping

def extract_residue_ids_from_query(selection_query):
    residue_ids = []
    parts = selection_query.split(" or ")
    for part in parts:
        chain_residue_pair = part.split(" and ")
        chain_id = chain_residue_pair[0].split(" ")[-1]
        residue_num = chain_residue_pair[1].split(" ")[-1].replace(")", "")  # ')' を取り除く
        residue_ids.append((chain_id, int(residue_num)))
    return residue_ids

    
def calculate_apo_pocket_loop_percentage(selection_name):
    """ 
    PyMOLの選択範囲に基づいて、アポタンパク質のポケット部分のループ割合を計算します。

    :param selection_name: ポケット部分を指定するPyMOLの選択範囲名
    :return: ループ部分の割合 (%)
    """
    cmd.dss(selection_name)  # セカンダリ構造の割り当て
    total_atoms = cmd.count_atoms(selection_name)
    loop_atoms = cmd.count_atoms(f"{selection_name} and not (ss h+s)")  # ループ以外のセカンダリ構造を除外
    loop_percentage = loop_atoms / total_atoms * 100 if total_atoms > 0 else 0

    return loop_percentage


# アポタンパク質AのポケットIDを決定する関数
def determine_pocket_id_for_apo_a(common_apo, protein_id, apo_data, holo_data, holo_pocket_dir_refined, holo_pocket_dir_other, colors):

    pocket_centroids = {}  # 各ポケットIDの重心を格納する辞書
    pocket_rmsd = {}
    holo_to_pocket_selection = {}  # ホロタンパク質とアポタンパク質Aのポケット選択を記録する辞書を初期化
    sub_pocket_rmsd = {}  # アポタンパク質のポケットとホロタンパク質全体のRMSDを格納する辞書
    print('アポプロテインA', common_apo)
    # アポタンパク質の選択
    apo_protein = apo_data[apo_data['apo_name'] == common_apo].iloc[0]
    apo_protein_name = apo_protein['apo_name']
    apo_protein_chain = apo_protein['apo_chain']
    apo_protein_path = f"../data/mmcif/apo/{apo_protein_name}.cif"
    cmd.delete("all")

    # チェーンIDが数字のみの場合、それを文字列として扱う
    if apo_protein_chain.isdigit():
        apo_protein_chain = f"'{apo_protein_chain}'"
    apo_protein_chain_clean = apo_protein_chain.replace("'", "")

    # アポタンパク質のロード
    cmd.load(apo_protein_path, "apo_protein")
    cmd.remove(f"apo_protein and not chain {apo_protein_chain}")

    # 重心の計算と原点への移動
    centroid = calculate_centroid('apo_protein')
    cmd.translate([-c for c in centroid], 'apo_protein')

    # アポタンパク質の残基番号を取得
    apo_residue_numbers= get_residue_numbers_from_mmcif(apo_protein_path, apo_protein_chain_clean)

    # アポタンパク質のFASTAファイルから配列を取得
    apo_protein_fasta_path = f"../data/fasta/apo/{apo_protein_name}_{apo_protein_chain_clean}.fasta"
    apo_seq = read_fasta(apo_protein_fasta_path)

    # 重心が原点にあるアポタンパク質の保存
    cmd.save(f"../data/mmcif/apo_center/{apo_protein_name}_{apo_protein_chain_clean}_centered.cif", 'apo_protein', format="cif")
    # ポケット候補の残基を保存するための辞書
    pocket_residues = {}

    # 各ホロタンパク質のポケットを処理
    holo_proteins = holo_data[holo_data['protein_id'] == protein_id]
    holo_to_pocket_centroids = {}
    holo_to_pocket_rmsd = {}
    apo_pocket_loop_percentage = {}
    apo_pocket_missing_percentage = {}

    for _, holo_protein in holo_proteins.iterrows():
        missing_percentage = 0
        pocket_name = holo_protein['holo_name']
        # ポケットファイルの存在確認
        pocket_path_refined = os.path.join(holo_pocket_dir_refined, pocket_name, f"{pocket_name}_pocket.pdb")
        pocket_path_other = os.path.join(holo_pocket_dir_other, pocket_name, f"{pocket_name}_pocket.pdb")

        # 正しいパスの決定
        pocket_path = pocket_path_refined if os.path.exists(pocket_path_refined) else pocket_path_other if os.path.exists(pocket_path_other) else None
        if pocket_path is None:
            continue  # ファイルが存在しない場合はスキップ
        
        # ポケットのロードとアラインメント
        cmd.load(pocket_path, pocket_name)
        holo_pocket_atom_count = cmd.count_atoms(pocket_name)

        # ポケットの配列を読み込む
        pocket_seq_converted = get_sequence_from_pdb(pocket_path)

        # アライメントの実行
        alignment = align_sequences(apo_seq, pocket_seq_converted)
        # print(f"Alignment result for {pocket_name} and apo {apo_protein_name}:")
        # print(alignment)

        rmsd_pocket = float('inf')  # デフォルト値を無限大に設定
        rmsd_all = float("inf")
        rmsd_pocket_sub = float("inf")


        # アライメントされた残基の選択
        try:
            selection_query, selected_numbers = select_aligned_residues_and_store_numbers(apo_protein_chain, alignment, apo_residue_numbers)
        except:
            selection_query, selected_numbers = select_aligned_residues_and_store_numbers(apo_protein_chain_clean, alignment, apo_residue_numbers)
        if selection_query:
            # 欠損座標割合の計算 
            missing_percentage = calculate_missing_coordinates_percentage(apo_protein_path, apo_protein_chain, extract_residue_ids_from_query(selection_query))
            pocket_selection = f"apo_pocket_{pocket_name}"
            cmd.select(pocket_selection, selection_query)
            selected_count = cmd.count_atoms(pocket_selection)
            apo_pocket_loop_percentage[pocket_name] = calculate_apo_pocket_loop_percentage(f"apo_pocket_{pocket_name}")

            print(f"Number of atoms selected for {pocket_selection}: {selected_count}")
            
            cmd.select(f"apo_pocket_sub_{pocket_name}", f"apo_protein like {pocket_name}")

            # 選択された残基番号をポケット残基として保存
            pocket_residues[pocket_selection] = set(selected_numbers)
            # ホロタンパク質とポケットの選択を記録
            holo_to_pocket_selection[holo_protein['holo_name']] = pocket_selection
        else:
            print(f"No aligned residues found for {pocket_name}")
            continue

        try:
            rmsd_pocket = calculate_rmsd_if_aligned_enough(cmd.align(pocket_name, pocket_selection, cycles=0), holo_pocket_atom_count)
        except pymol.CmdException as e:
            print(f"Alignment failed for {pocket_name}, {cmd.count_atoms(pocket_name)} and apo_protein, {cmd.count_atoms('apo_protein')} : {e}")

        try:
            rmsd_pocket_sub = calculate_rmsd_if_aligned_enough(cmd.align(pocket_name, f"apo_pocket_sub_{pocket_name}", cycles=0), holo_pocket_atom_count)
        except pymol.CmdException as e:
            print(f"Alignment failed for {pocket_name}, {cmd.count_atoms(pocket_name)} and apo_protein, {cmd.count_atoms('apo_protein')} : {e}")

        try:
            rmsd_all = calculate_rmsd_if_aligned_enough(cmd.align(pocket_name, "apo_protein", cycles=0), holo_pocket_atom_count)
        except pymol.CmdException as e:
            print(f"Alignment failed for {pocket_name}, {cmd.count_atoms(pocket_name)} and apo_protein, {cmd.count_atoms('apo_protein')} : {e}")

        print(f"rmsd_all:{rmsd_all}, rmsd_pocket:{rmsd_pocket}")
        rmsd_pocket_min = min(rmsd_pocket, rmsd_pocket_sub)
        rmsd = min(rmsd_all, rmsd_pocket_min)
        pocket_rmsd[pocket_name] = rmsd
        holo_to_pocket_rmsd[holo_protein['holo_name']] = rmsd
        apo_pocket_missing_percentage[holo_protein['holo_name']] = missing_percentage

        # 選択された残基の保存
        stored.residues = []
        cmd.iterate(pocket_selection, "stored.residues.append((resi, resn))")
        # print(set(stored.residues))
        pocket_residues[pocket_selection] = set(stored.residues)

        
    # ポケット候補のマージとIDの割り当て
    merged_pockets, original_to_merged_pocket_id = merge_pocket_candidates(pocket_residues)
    # print(f"merged_pockets: {merged_pockets}")


    # マージされたポケットごとにホロタンパク質をマッピング
    holo_to_merged_pocket_id = {}
    pocket_centroids = {}  # マージされたポケットの重心を格納するための辞書

    # マージされたポケットごとに重心を計算
    for pid, residues in merged_pockets:
        pocket_selection_name = f"pocket_{pid}"
        selection_residues = " or ".join([f"resi {resi}" for resi, resn in residues])
        cmd.select(pocket_selection_name, selection_residues)
        pocket_centroids[pid] = calculate_centroid(pocket_selection_name)

    for holo_name, selection in holo_to_pocket_selection.items():
        merged_pocket_id = original_to_merged_pocket_id[selection]
        holo_to_merged_pocket_id[holo_name] = merged_pocket_id
        holo_to_pocket_centroids[holo_name] = pocket_centroids[merged_pocket_id]

    # print(holo_to_merged_pocket_id)

    # 各ポケットの可視化と色付け
    color_index = 0
    for pid, residues in merged_pockets:
        selection_residues = " or ".join([f"resi {resi}" for resi, resn in residues])
        # print(pid)
        selection_name = f"pocket_{pid}"
        cmd.select(selection_name, selection_residues)
        cmd.color(colors[color_index], selection_name)
        color_index = (color_index + 1) % len(colors)
    
    # print(f"holo_to_pocket_selection: {holo_to_pocket_selection}")
    # セッションの保存
    cmd.save(f"../pse_file/pocket_visualization_protein_id_{protein_id}.pse")
    
    # ポケット選択とIDのマッピングの返却
    return holo_to_merged_pocket_id, holo_to_pocket_centroids, holo_to_pocket_rmsd, merged_pockets, apo_pocket_loop_percentage, apo_pocket_missing_percentage



# Helper function to merge pocket candidates
def merge_pocket_candidates(pocket_residues):
    # マージされたポケットIDを追跡するための辞書
    merged_pocket_tracking = {key: None for key in pocket_residues.keys()}
    
    while True:
        max_overlap = 0
        max_pair = None
        pocket_keys = list(pocket_residues.keys())
        for i in range(len(pocket_keys)):
            for j in range(i + 1, len(pocket_keys)):
                overlap = len(pocket_residues[pocket_keys[i]] & pocket_residues[pocket_keys[j]])
                overlap_percent_i = overlap / len(pocket_residues[pocket_keys[i]])
                overlap_percent_j = overlap / len(pocket_residues[pocket_keys[j]])
                if overlap_percent_i >= 0.5 or overlap_percent_j >= 0.5:
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_pair = (pocket_keys[i], pocket_keys[j])
        if max_pair:
            # マージされたポケットを追跡
            for key in merged_pocket_tracking:
                if merged_pocket_tracking[key] == max_pair[0]:
                    merged_pocket_tracking[key] = max_pair[1]
            pocket_residues[max_pair[0]].update(pocket_residues[max_pair[1]])
            del pocket_residues[max_pair[1]]
        else:
            break

    merged_pockets = []
    for pid, res in enumerate(pocket_residues.values(), start=1):
        merged_pockets.append((pid, res))
        # マージされたポケットIDを更新
        for key in merged_pocket_tracking:
            if merged_pocket_tracking[key] == None:
                merged_pocket_tracking[key] = pid

    return merged_pockets, merged_pocket_tracking

# アポタンパク質BのポケットIDを決定する関数
def determine_pocket_id_for_apo_b(apo_b_name, apo_b_chain, holo_to_pocket_id, merged_pockets, apo_a_to_b_mapping, holo_pocket_dir_refined, holo_pocket_dir_other):
    print('アポタンパク質B', apo_b_name)
    apo_b_protein_path = f"../data/mmcif/apo/{apo_b_name}.cif"
    cmd.load(apo_b_protein_path, "apo_b_protein")

    # チェーンIDが数字のみの場合、それを文字列として扱う
    if apo_b_chain.isdigit():
        apo_b_chain = f"'{apo_b_chain}'"
    
    apo_b_chain_clean = apo_b_chain.replace("'", "")
    cmd.remove(f"apo_b_protein and not chain {apo_b_chain}")
    cmd.translate([-c for c in calculate_centroid('apo_b_protein')], 'apo_b_protein')

    cmd.save(f"../data/mmcif/apo_center/{apo_b_name}_{apo_b_chain_clean}_centered.cif", 'apo_b_protein', format="cif")

    # アポタンパク質BのFASTAファイルから配列を取得
    apo_b_fasta_path = f"../data/fasta/apo/{apo_b_name}_{apo_b_chain_clean}.fasta"
    apo_b_seq = read_fasta(apo_b_fasta_path)

    pocket_rmsd_b = {}
    pocket_centroids_b = {}
    processed_pockets = {}
    apo_b_pocket_loop_percentage = {}
    apo_b_pocket_missing_percentage = {}

    for holo_name, pocket_id in holo_to_pocket_id.items():
        missing_percentage = 0
        if pocket_id not in processed_pockets:
            apo_a_residues = None
            for pid, residues in merged_pockets:
                if pid == pocket_id:
                    apo_a_residues = residues
                    break

            if apo_a_residues is None:
                print(f"No residues found for pocket ID {pocket_id}")
                sys.exit()
                continue

            apo_a_residues_mapped = [(apo_a_to_b_mapping[int(res[0])], res[1]) for res in apo_a_residues if int(res[0]) in apo_a_to_b_mapping]
            try:
                apo_b_selection_query = " or ".join([f"chain {apo_b_chain} and resi {resi} and resn {resn}" for resi, resn in apo_a_residues_mapped])
            except:
                apo_b_selection_query = " or ".join([f"chain {apo_b_chain_clean} and resi {resi} and resn {resn}" for resi, resn in apo_a_residues_mapped])
            processed_pockets[pocket_id] = apo_b_selection_query
            cmd.select(f"apo_b_pocket_{pocket_id}", apo_b_selection_query)


        else:
            apo_b_selection_query = processed_pockets[pocket_id]


        if apo_b_selection_query:
            rmsd_pocket_holo = float("inf")
            rmsd_pocket_pid = float("inf")
            rmsd_pocket_holo_align = float("inf")
            rmsd_all = float("inf")
            rmsd_pocket = float("inf")
            missing_percentage = 0
            align_target = apo_b_selection_query
            cmd.select(f"apo_b_pocket_{holo_name}_like", f"apo_b_protein like {holo_name}")

            # ホロタンパク質のポケットファイルの存在を確認
            pocket_path_refined = os.path.join(holo_pocket_dir_refined, holo_name, f"{holo_name}_pocket.pdb")
            pocket_path_other = os.path.join(holo_pocket_dir_other, holo_name, f"{holo_name}_pocket.pdb")
            pocket_path = pocket_path_refined if os.path.exists(pocket_path_refined) else pocket_path_other if os.path.exists(pocket_path_other) else None

            if pocket_path:
                holo_pocket_seq = get_sequence_from_pdb(pocket_path)
                alignment = align_sequences(apo_b_seq, holo_pocket_seq)
                holo_pocket_atom_count = cmd.count_atoms(holo_name)

                # アライメントに基づいて残基を選択
                try:
                    apo_b_residue_numbers = get_residue_numbers_from_mmcif(f"../data/mmcif/apo/{apo_b_name}.cif", apo_b_chain)
                    selection_query_holo, _ = select_aligned_residues_and_store_numbers(apo_b_chain, alignment, apo_b_residue_numbers)
                except:
                    apo_b_residue_numbers = get_residue_numbers_from_mmcif(f"../data/mmcif/apo/{apo_b_name}.cif", apo_b_chain_clean)
                    selection_query_holo, _ = select_aligned_residues_and_store_numbers(apo_b_chain_clean, alignment, get_residue_numbers_from_mmcif(f"../data/mmcif/apo/{apo_b_name}.cif", apo_b_chain_clean))
                cmd.select(f"apo_b_pocket_{holo_name}", selection_query_holo)
                # アポタンパク質Bのポケットごとのループ割合を計算
                apo_b_pocket_loop_percentage[holo_name] = calculate_apo_pocket_loop_percentage(f"apo_b_pocket_{holo_name}")
                if selection_query_holo:
                    missing_percentage = calculate_missing_coordinates_percentage(apo_b_protein_path, apo_b_chain_clean, extract_residue_ids_from_query(selection_query_holo))
                    # 選択した残基に基づいてRMSDを計算
                    try:
                        rmsd_pocket_holo_align = calculate_rmsd_if_aligned_enough(cmd.align(holo_name, f"apo_b_pocket_{holo_name}", cycles=0), holo_pocket_atom_count)
                    except pymol.CmdException as e:
                        print(f"Error aligning {holo_name} with apo_b_pocket_holo_align: {e}")

                    try:
                        rmsd_pocket_holo = calculate_rmsd_if_aligned_enough(cmd.align(holo_name, f"apo_b_pocket_{holo_name}_like", cycles=0), holo_pocket_atom_count)
                    except pymol.CmdException as e:
                        print(f"Error aligning {holo_name} with apo_b_pocket_holo_like_align: {e}")

                    # 最小RMSDの更新
                    rmsd_pocket_holo = min(rmsd_pocket_holo, rmsd_pocket_holo_align)

            try:
                rmsd_pocket_pid = calculate_rmsd_if_aligned_enough(cmd.align(holo_name, apo_b_selection_query, cycles=0), holo_pocket_atom_count)
            except pymol.CmdException as e:
                print(f"Error aligning {holo_name} with {apo_b_selection_query}: {e}")

            rmsd_pocket = min(rmsd_pocket_holo, rmsd_pocket_pid)

            try:
                rmsd_all = calculate_rmsd_if_aligned_enough(cmd.align(holo_name, "apo_b_protein", cycles=0), holo_pocket_atom_count)
            except pymol.CmdException as e:
                print(f"Error aligning {holo_name} with apo_b_protein: {e}")

            pocket_rmsd_b[holo_name] = min(rmsd_pocket, rmsd_all)
            pocket_centroid_b = calculate_centroid(apo_b_selection_query)
            pocket_centroids_b[holo_name] = pocket_centroid_b
            apo_b_pocket_missing_percentage[holo_name] = missing_percentage

        else:
            print(f"No corresponding selection found for pocket ID {pocket_id} in apo B")

    pse_filename = f"../pse_file/{apo_b_name}_{apo_b_chain_clean}_pocket.pse"
    cmd.save(pse_filename)
    # print(f"Saved PyMOL session for {apo_b_name}, chain {apo_b_chain}: {pse_filename}")

    cmd.delete("apo_b_protein")
    return pocket_rmsd_b, pocket_centroids_b, apo_b_pocket_loop_percentage, apo_b_pocket_missing_percentage




def convert_residues_apo_a_to_b(apo_a_residues, apo_a_to_b_mapping, fasta_to_pdb_mapping_b):
    apo_b_residues = []
    for resi in apo_a_residues:
        fasta_index = apo_a_to_b_mapping.get(resi)
        if fasta_index:
            pdb_resi = fasta_to_pdb_mapping_b.get(fasta_index)
            if pdb_resi:
                apo_b_residues.append(pdb_resi)
    return apo_b_residues

def convert_selection_apo_a_to_b(apo_a_selection, apo_a_to_b_mapping, apo_b_chain):
    # アポタンパク質Aの選択範囲の残基番号を取得
    apo_a_residues = get_residues_from_selection(apo_a_selection)
    apo_b_residues = []

    for resi in apo_a_residues:
        # アポタンパク質Aの残基番号をアポタンパク質Bの残基番号に変換
        if resi in apo_a_to_b_mapping:
            apo_b_residues.append(apo_a_to_b_mapping[resi])

    if not apo_b_residues:
        return None

    # アポタンパク質Bの選択範囲を構築
    apo_b_selection_query = " or ".join([f"chain {apo_b_chain} and resi {resi}" for resi in apo_b_residues])
    # print(f"Generated selection query for apo B: {apo_b_selection_query}")  # デバッグ出力

    return apo_b_selection_query


def get_residues_from_selection(selection):
    stored.residues = []
    cmd.iterate(selection, "stored.residues.append(resi)")
    return stored.residues



def main(start_id, end_id):
    # output_csv_path = '../csv_file/pocket_analysis_results.csv'
    output_csv_path = f'../csv_file/pocket_analysis_results_{start_id}_to_{end_id}.csv'



    # 既に処理済みのprotein_idを確認
    processed_ids = set()
    if os.path.exists(output_csv_path):
        existing_results_df = pd.read_csv(output_csv_path)
        processed_ids = set(existing_results_df['protein_id'].unique())


    # for protein_id in tqdm(apo_data['protein_id'].unique()):
    for protein_id in tqdm(range(start_id, end_id + 1)):
        # 結果を格納するための空のリスト
        results = []
        print(f"Start {protein_id}")
        if protein_id in processed_ids:
            print(f"Skipping already processed protein_id: {protein_id}")
            continue
        elif protein_id not in apo_data['protein_id'].unique():
            print(f"apo_data don't have {protein_id}")
            continue
        # PyMOLのセッションを初期化
        cmd.reinitialize()

        # 最も一般的なアポタンパク質を決定
        common_apo_a = determine_most_common_apo(protein_id, apo_holo_pairs, apo_data)
        if not common_apo_a:
            continue

        # アポタンパク質Aの情報を取得し、ホロタンパク質との関連情報を決定
        holo_to_pocket_id, holo_to_pocket_centroids, holo_to_pocket_rmsd, merged_pockets, apo_pocket_loop_percentage, apo_pocket_missing_percentage = determine_pocket_id_for_apo_a(
            common_apo_a, protein_id, apo_data, holo_data, holo_pocket_dir_refined, holo_pocket_dir_other, colors
        )

        # アポタンパク質AのFASTAファイルから配列を取得
        apo_a_info = apo_data[apo_data['apo_name'] == common_apo_a].iloc[0]
        apo_a_name = apo_a_info['apo_name']
        apo_a_chain = apo_a_info['apo_chain']
        apo_a_fasta_path = f"../data/fasta/apo/{apo_a_name}_{apo_a_chain}.fasta"
        apo_a_protein_path = f"../data/mmcif/apo/{apo_a_name}.cif"
        apo_a_seq = read_fasta(apo_a_fasta_path)

        print(f"apo protein: {apo_a_name} {common_apo_a}")
        for holo_name, pocket_id in holo_to_pocket_id.items():
            if holo_name in apo_holo_pairs[apo_holo_pairs['apo_name'].str.upper() == common_apo_a]['holo_name'].values:
                # pocket_rmsdがinfの場合はスキップ
                if holo_to_pocket_rmsd.get(holo_name, 0) == float('inf'):
                    continue
                holo_info = holo_data[holo_data['holo_name'] == holo_name].iloc[0]
                result_a = {
                    'apo_name': apo_a_name,
                    'apo_chain': apo_a_chain,
                    'holo_name': holo_name,
                    'holo_chain': holo_info['holo_chain'],
                    'pocket_id': pocket_id,
                    'pocket_rmsd': holo_to_pocket_rmsd.get(holo_name, None),
                    'pocket_com': holo_to_pocket_centroids.get(holo_name, None),
                    'protein_id': protein_id,
                    'family50_id': apo_a_info['family50_id'],
                    'ligand': holo_info['ligand'],
                    'ligand_atom_count': holo_info['ligand_atom_count'],
                    'apo_pocket_missing_percentage': apo_pocket_missing_percentage.get(holo_name, None),
                    'apo_loop_per': apo_pocket_loop_percentage.get(holo_name, None),
                    'holo_loop_per': holo_info['loop_per']
                }
                print(result_a)
                results.append(result_a)
        # アポタンパク質Bの処理
        apo_a_to_b_mapping = {}
        if len(apo_data[apo_data['protein_id'] == protein_id]) > 1:
            for index, apo_b_info in apo_data[apo_data['protein_id'] == protein_id].iterrows(): #代表でないアポを一つづつ処理
                apo_b_name = apo_b_info['apo_name']
                apo_b_chain = apo_b_info['apo_chain']
                # アポタンパク質Bがアポタンパク質Aと同じでないことを確認
                if apo_b_name == common_apo_a:
                    continue
                # アポタンパク質Bに対応するホロタンパク質をapo_holo_pairsから取得
                corresponding_holos = apo_holo_pairs[apo_holo_pairs['apo_name'].str.upper() == apo_b_name]['holo_name'].values

                apo_b_fasta_path = f"../data/fasta/apo/{apo_b_name}_{apo_b_chain}.fasta"
                apo_b_seq = read_fasta(apo_b_fasta_path)
                apo_b_protein_path = f"../data/mmcif/apo/{apo_b_name}.cif"
                
                # アポタンパク質AとBのペアワイズアライメントを実行
                alignment = align_sequences(apo_a_seq, apo_b_seq)
                # print(f"Alignment result: \n{alignment}")

                # アポタンパク質AとBのFASTAとmmCIF間のマッピング生成
                fasta_to_pdb_mapping_a = create_mapping_from_mmcif_to_fasta(apo_a_protein_path, apo_a_chain)
                fasta_to_pdb_mapping_b = create_mapping_from_mmcif_to_fasta(apo_b_protein_path, apo_b_chain)
                # print(f"Mapping from fasta to B: {fasta_to_pdb_mapping_b}")
                # print(f"Mapping from fasta to A: {fasta_to_pdb_mapping_a}")
                apo_a_to_b_mapping = create_mapping_from_alignment(alignment, fasta_to_pdb_mapping_a, fasta_to_pdb_mapping_b)
                # print(f"Mapping from A to B: {apo_a_to_b_mapping}")


                # アポタンパク質Bのポケット位置を特定
                pocket_rmsd_b, pocket_centroids_b, apo_b_pocket_loop_percentage, apo_b_pocket_missing_percentage = determine_pocket_id_for_apo_b(
                    apo_b_name, apo_b_chain, holo_to_pocket_id, merged_pockets, apo_a_to_b_mapping, holo_pocket_dir_refined, holo_pocket_dir_other,
                )

                # 結果の追加
                # それらのホロタンパク質に対してのみ解析を行う
                print(corresponding_holos)
                print(holo_to_pocket_id)
                print(apo_b_pocket_missing_percentage)
                for holo_name in corresponding_holos:
                    if holo_name in holo_to_pocket_id:
                        # pocket_rmsdがinfの場合はスキップ
                        if pocket_rmsd_b.get(holo_name, 0) == float('inf'):
                            continue
                        holo_info = holo_data[holo_data['holo_name'] == holo_name].iloc[0]
                        result_b = {
                            'apo_name': apo_b_name,
                            'apo_chain': apo_b_chain,
                            'holo_name': holo_name,
                            'holo_chain': holo_info['holo_chain'],
                            'pocket_id': pocket_id,
                            'pocket_rmsd': pocket_rmsd_b.get(holo_name, None),
                            'pocket_com': pocket_centroids_b.get(holo_name, None),
                            'protein_id': apo_b_info['protein_id'],
                            'family50_id': apo_b_info['family50_id'],
                            'ligand': holo_info['ligand'],
                            'ligand_atom_count': holo_info['ligand_atom_count'],
                            'apo_pocket_missing_percentage': apo_b_pocket_missing_percentage.get(holo_name, None),
                            'apo_loop_per': apo_b_pocket_loop_percentage.get(holo_name, None),
                            'holo_loop_per': holo_info['loop_per']
                                    }
                        print(result_b)
                        results.append(result_b)


        # 結果をCSVファイルに保存
        if results:
            results_df = pd.DataFrame(results)
            if os.path.exists(output_csv_path):
                results_df.to_csv(output_csv_path, mode='a', header=False, index=False)
            else:
                results_df.to_csv(output_csv_path, index=False)


    print(f"Processing completed. Results saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process protein IDs.')
    parser.add_argument('start_id', type=int, help='Start protein ID')
    parser.add_argument('end_id', type=int, help='End protein ID')
    args = parser.parse_args()

    main(args.start_id, args.end_id)