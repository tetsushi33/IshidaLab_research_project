import pandas as pd
import os
import pymol
from pymol import cmd, stored
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Align import PairwiseAligner
from tqdm import tqdm
import sys
import argparse

AMINO_ACID_CODE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X"
}



def determine_most_common_apo(protein_id, apo_holo_pairs_csv, apo_proteins_id_csv):
    apo_names = apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == protein_id]['apo_name'].str.upper() # 指定のidに一致するアポの配列
    # apo_holo_pairsから、指定のidのアポを要素に持つ行のみ抽出
    filtered_apo_holo_pairs = apo_holo_pairs_csv[apo_holo_pairs_csv['apo_name'].str.upper().isin(apo_names)]

    if filtered_apo_holo_pairs.empty: # 
        return None
    
    #print(filtered_apo_holo_pairs)

    # 最も出現回数の多いアポを代表とする
    apo_representitive = filtered_apo_holo_pairs['apo_name'].value_counts().idxmax().upper()
    return apo_representitive


'''
代表apoタンパク質ポケット位置を決定
-----------------------------------------------------------------------------
- determine_pocket_id_for_apo_a():
    (-) calculate_centroid(selection):
-----------------------------------------------------------------------------
'''

# アポタンパク質AのポケットIDを決定する関数
def determine_pocket_id_for_apo_a(apo_representitive, protein_id, apo_proteins_id_csv, apo_holo_pairs_with_group_id_csv, pdbbind_dir_refined, pdbbind_dir_others, colors):

    pocket_centroids = {}  # 各ポケットIDの重心を格納する辞書
    pocket_rmsd = {}
    holo_to_pocket_selection = {}  # ホロタンパク質とアポタンパク質Aのポケット選択を記録する辞書を初期化
    sub_pocket_rmsd = {}  # アポタンパク質のポケットとホロタンパク質全体のRMSDを格納する辞書
    # アポタンパク質の選択
    apo_protein = apo_proteins_id_csv[apo_proteins_id_csv['apo_name'] == apo_representitive].iloc[0] # 最初の行
    apo_protein_name = apo_protein['apo_name']
    apo_protein_chain = apo_protein['apo_chain']
    apo_protein_path = f"../mmcif/apo/{apo_protein_name}.cif"
    cmd.delete("all")

    # チェーンIDが数字のみの場合、それを文字列として扱う
    if apo_protein_chain.isdigit():
        apo_protein_chain = f"'{apo_protein_chain}'"
    apo_protein_chain_clean = apo_protein_chain.replace("'", "") # 文字列の下処理

    # アポタンパク質のロード
    cmd.load(apo_protein_path, "apo_protein")
    cmd.remove(f"apo_protein and not chain {apo_protein_chain}") # 対象のチェーン以外は削除

    # 重心の計算と原点への移動
    centroid = calculate_centroid('apo_protein') # 引数はpymol上での構造の名前
    print("重心座標: ", centroid)
    cmd.translate([-c for c in centroid], 'apo_protein')

    # アポタンパク質の残基番号を取得
    apo_residue_numbers= get_residue_numbers_from_mmcif(apo_protein_path, apo_protein_chain_clean) # アポ構造全体の残基たち（配列）
    print("apo_residue_numbers : ", apo_residue_numbers)

    # アポタンパク質のFASTAファイルから配列を取得
    apo_protein_fasta_path = f"../data/fasta/apo/{apo_protein_name}_{apo_protein_chain_clean}.fasta"
    apo_seq = read_fasta(apo_protein_fasta_path)
    print("apo_seq : ", apo_seq)

    # 重心が原点にあるアポタンパク質の保存
    cmd.save(f"../mmcif/apo_center/{apo_protein_name}_{apo_protein_chain_clean}_centered.cif", 'apo_protein', format="cif")
    
    # ポケット候補の残基を保存するための辞書
    pocket_residues = {}

    # 各ホロタンパク質のポケットを処理
    holo_proteins = apo_holo_pairs_with_group_id_csv[apo_holo_pairs_with_group_id_csv['apo_group_id'] == protein_id]
    holo_to_pocket_centroids = {}
    holo_to_pocket_rmsd = {}
    apo_pocket_loop_percentage = {}
    apo_pocket_missing_percentage = {}

    print("対応するホロの数 : ", len(holo_proteins))

    # 対象のアポに対応する（idが同じ）ホロに対して一つずつ処理
    for _, holo_protein in holo_proteins.iterrows():
        print("======================================================")
        missing_percentage = 0
        pocket_name = holo_protein['holo_name'] # ホロ構造のPDBIDをpocket_nameとしているだけ
        # ポケットファイルの存在確認
        pocket_path_refined = os.path.join(pdbbind_dir_refined, pocket_name, f"{pocket_name}_pocket.pdb") # 第一候補
        pocket_path_other = os.path.join(pdbbind_dir_others, pocket_name, f"{pocket_name}_pocket.pdb") # 第二候補（予備）

        # 正しいパスの決定
        pocket_path = pocket_path_refined if os.path.exists(pocket_path_refined) else pocket_path_other if os.path.exists(pocket_path_other) else None
        if pocket_path is None:
            continue  # ファイルが存在しない場合はスキップ

        print("pocket_path, pocket_name : ", pocket_path, pocket_name)
        print("--------")
        # ポケットのロードとアラインメント
        cmd.load(pocket_path, pocket_name) # pocket_pathがポケットのみのPDBデータなので、ポケット部分のみが対象
        holo_pocket_atom_count = cmd.count_atoms(pocket_name)
        print("holo_pocket_atom_count : ", holo_pocket_atom_count)

        print("--------")
        # ポケットの配列を読み込む
        pocket_seq_converted = get_sequence_from_pdb(pocket_path)
        print("pocket_seq_converted : ", pocket_seq_converted)

        # アライメントの実行
        alignment = align_sequences(apo_seq, pocket_seq_converted)
        print("--------")
        print(f"Alignment result for {pocket_name} and apo {apo_protein_name}:")
        print(alignment)
        print("--------")

        rmsd_pocket = float('inf')  # デフォルト値を無限大に設定
        rmsd_all = float("inf")
        rmsd_pocket_sub = float("inf")


        # アポタンパク質上のポケット位置（残基単位）をアライメント結果から取得
        # アライメントされた残基の選択
        try:
            selection_query, selected_numbers = select_aligned_residues_and_store_numbers(apo_protein_chain, alignment, apo_residue_numbers)
        except:
            # po_protein_chainの認識がうまくいかなかった時のための考慮（一応）
            selection_query, selected_numbers = select_aligned_residues_and_store_numbers(apo_protein_chain_clean, alignment, apo_residue_numbers)
        print("selection_query :", selection_query)
        print("--------")
        print("selected_numbers :", selected_numbers)

        if selection_query: # selection queryの形式： (chain A and resi 15) or (chain A and resi 23) or ...
            # 欠損座標割合の計算 
            missing_percentage = calculate_missing_coordinates_percentage(apo_protein_path, apo_protein_chain, extract_residue_ids_from_query(selection_query))
            pocket_selection = f"apo_pocket_{pocket_name}" 
            cmd.select(pocket_selection, selection_query) # section_query内の残基全てをもってpocetk_selection(apo_pocket_{pocket_name})と名付ける
            selected_count = cmd.count_atoms(pocket_selection)
            apo_pocket_loop_percentage[pocket_name] = calculate_apo_pocket_loop_percentage(f"apo_pocket_{pocket_name}")

            print(f"Number of atoms selected for {pocket_selection}: {selected_count}")
            
            cmd.select(f"apo_pocket_sub_{pocket_name}", f"apo_protein like {pocket_name}")

            # 選択された残基番号をポケット残基として保存
            pocket_residues[pocket_selection] = set(selected_numbers)
            # ホロタンパク質とポケットの選択を記録
            holo_to_pocket_selection[holo_protein['holo_name']] = pocket_selection # 例：holo_to_pocket_selection :  {'4f5y': 'apo_pocket_4f5y', '4loi': 'apo_pocket_4loi', '4loh': 'apo_pocket_4loh'}
            #print("pocket_residues : ", pocket_residues)
            print("holo_to_pocket_selection : ", holo_to_pocket_selection)
        else:
            print(f"No aligned residues found for {pocket_name}")
            continue

        # 3種類のRMSDを計算DADA
        # RMSDの計算自体はcmd.alignでやっている
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

        #　３種類のRMSDのうち最小のものを採用
        rmsd_pocket_min = min(rmsd_pocket, rmsd_pocket_sub)
        rmsd = min(rmsd_all, rmsd_pocket_min)
        pocket_rmsd[pocket_name] = rmsd # pocket_nameはホロのPDBIDなので、アポAに対応するホロ構造それぞれに対し、{ID, RMSD}として格納
        holo_to_pocket_rmsd[holo_protein['holo_name']] = rmsd
        apo_pocket_missing_percentage[holo_protein['holo_name']] = missing_percentage

        # 選択された残基の保存
        stored.residues = []
        cmd.iterate(pocket_selection, "stored.residues.append((resi, resn))")
        # print(set(stored.residues))
        pocket_residues[pocket_selection] = set(stored.residues)
        print("======================================================")
    
    print("--------")
    print("pocket_rmsd : ", pocket_rmsd)
    print("--------")
    print(pocket_residues)
     
    # ポケット候補のマージとIDの割り当て
    merged_pockets, original_to_merged_pocket_id = merge_pocket_candidates(pocket_residues)
    # merged_pockets : 結合後のポケットの残基番号と残基の種類のセットのリスト
    print("--------")
    print(f"merged_pockets: {merged_pockets}")

    # マージされたポケットごとにホロタンパク質をマッピング
    holo_to_merged_pocket_id = {} # アポA上のポケットが由来しているホロ名とそのポケットのid(アポA上でのIDなのでポケットが一つならidは1のみ)
    pocket_centroids = {}  # マージされたポケットの重心を格納するための辞書

    # マージされたポケットごとに重心を計算
    for pid, residues in merged_pockets:
        pocket_selection_name = f"pocket_{pid}"
        selection_residues = " or ".join([f"resi {resi}" for resi, resn in residues]) # resi: 残基番号、resn: 残基の種類
        cmd.select(pocket_selection_name, selection_residues)
        pocket_centroids[pid] = calculate_centroid(pocket_selection_name)

    for holo_name, selection in holo_to_pocket_selection.items(): # (例)holo_to_pocket_selection:{'4f5y': 'apo_pocket_4f5y', '4loi': 'apo_pocket_4loi', '4loh': 'apo_pocket_4loh'}
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


# ポケットの重心を計算する関数 (!! 各原子の重さは無視し、座標のみから平均を計算)
def calculate_centroid(selection):
    stored.xyz = [] # storedはpymolの組み込み変数
    cmd.iterate_state(1, selection, 'stored.xyz.append([x,y,z])') # 引数selectionに該当する各原子の座標をappend
    x, y, z = zip(*stored.xyz) # 各軸ごとに要素を取り出してx,y,zに格納
    centroid = [sum(x)/len(x), sum(y)/len(y), sum(z)/len(z)] # 各軸ごとに平均を計算
    return centroid

def get_residue_numbers_from_mmcif(mmcif_path, chain_id):
    # 欠損残基は飛ばす
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

# ペアワイズアライメントを行う関数
def align_sequences(seq1, seq2):
    '''
    引数の二つのアミノ酸配列から、それらのスコアとtarget,queryの並びを計算して返す
     - target : seq1
     - query : seq2
    '''
    aligner = PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]
    return best_alignment


def select_aligned_residues_and_store_numbers(apo_chain_id, alignment, apo_residue_numbers):
    #どういう更新方法？？わからない！！
    selected_residue_numbers = []
    query = []

    apo_index = 0
    holo_index = 0

    for apo_res, holo_res in zip(alignment[0], alignment[1]): # alignment[0]:target配列, alignment[1]:query配列
        if apo_res != "-" and holo_res != "-": # 残基が一致した場合
            # Both residues are aligned
            query.append(f"(chain {apo_chain_id} and resi {apo_residue_numbers[apo_index]})") # pymol用のクエリを作成してappend
            # 例："chain A and resi 34"
            selected_residue_numbers.append(apo_residue_numbers[apo_index])

        if apo_res != "-":
            apo_index += 1
        if holo_res != "-":
            holo_index += 1
    
    return " or ".join(query), selected_residue_numbers

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
            #print(f"Missing residue at {residue_id}")

    #print(f"Total aligned residues: {total_residues}, Missing aligned residues: {missing_residues}")

    # 欠損割合の計算
    if total_residues == 0:
        return 0

    missing_percentage = (missing_residues / total_residues) * 100
    #print(f"Missing coordinates percentage: {missing_percentage}%")
    return missing_percentage

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

def extract_residue_ids_from_query(selection_query):
    '''
    例：
    (chain A and resi 35)を (A, 35)にするだけ
    '''
    residue_ids = []
    parts = selection_query.split(" or ")
    for part in parts:
        chain_residue_pair = part.split(" and ")
        chain_id = chain_residue_pair[0].split(" ")[-1]
        residue_num = chain_residue_pair[1].split(" ")[-1].replace(")", "")  # ')' を取り除く
        residue_ids.append((chain_id, int(residue_num)))
    return residue_ids

def calculate_rmsd_if_aligned_enough(align_result, holo_pocket_atom_count, threshold_ratio=0.3):
    """
    align_result と holo_pocket_atom_count に基づいて RMSD を計算します。
    重ね合わせに成功した分子数が指定された閾値以上の場合のみ RMSD を返します。
    
    :param align_result: cmd.align 関数の結果
    :param holo_pocket_atom_count: ホロタンパク質のポケットの分子数
    :param threshold_ratio: 重ね合わせに成功する必要がある分子数の割合 (例: 0.1)
    :return: 適切な場合のみ RMSD、そうでなければ float("inf")
    """
    #print("align result : ", align_result)
    aligned_atom_count = align_result[4]  # 重ね合わせで成功した分子数

    # ホロ構造全体の原子数のうち一定の割合の原子数がちゃんとRMSD計算に使われていたらOK
    if aligned_atom_count >= threshold_ratio * holo_pocket_atom_count: 
        return align_result[0]  # RMSD
    else:
        return float("inf") # 正の無限大　計算不可
    
    
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

# Helper function to merge pocket candidates
def merge_pocket_candidates(pocket_residues):
    # マージされたポケットIDを追跡するための辞書
    merged_pocket_tracking = {key: None for key in pocket_residues.keys()}
    
    while True:
        max_overlap = 0
        max_pair = None
        pocket_keys = list(pocket_residues.keys())
        #print("AAAAAAAAAAA")
        for i in range(len(pocket_keys)):
            for j in range(i + 1, len(pocket_keys)): # pocket_keys(pocket_residues)内の要素に対して総当たりの比較
                overlap = len(pocket_residues[pocket_keys[i]] & pocket_residues[pocket_keys[j]]) # 二つに残基群のうち共通する残基の数
                overlap_percent_i = overlap / len(pocket_residues[pocket_keys[i]])
                overlap_percent_j = overlap / len(pocket_residues[pocket_keys[j]])
                #print("overlap_percent_i", overlap_percent_i)
                #print("overlap_percent_j", overlap_percent_j)
                #print("overlap : ", overlap)
                if overlap_percent_i >= 0.5 or overlap_percent_j >= 0.5: #共通部分が、どちらか一方にとって0.5以上
                    if overlap > max_overlap:
                        max_overlap = overlap # 共通残基数を更新 !!割合ではなく、残基の数で更新の基準にしてる
                        max_pair = (pocket_keys[i], pocket_keys[j])
        #print("max pair : ", max_pair)
        #print("BBBBBBBBBB")

        # 最も共通部分の原子数が多いペアを結合
        if max_pair:
            # マージされたポケットを追跡
            for key, value in merged_pocket_tracking.items():
                if value == max_pair[0]:
                    merged_pocket_tracking[key] = max_pair[1]
            for key in merged_pocket_tracking:
                print("key : ", key)
                print("merged_pocket_tracking[key] : ", merged_pocket_tracking[key])
                print("max_pair[0] : ", max_pair[0])
                if merged_pocket_tracking[key] == max_pair[0]:
                    merged_pocket_tracking[key] = max_pair[1]
                    print("merged_pocket_tracking : ", merged_pocket_tracking)
            pocket_residues[max_pair[0]].update(pocket_residues[max_pair[1]])
            del pocket_residues[max_pair[1]]
           # print("merged_pocket_tracking : ", merged_pocket_tracking)
            #print(len(pocket_residues))
            #print("CCCCCCCCCCC")
        else:
            break

    #print("pocket_residues : ", pocket_residues)
    merged_pockets = []
    for pid, res in enumerate(pocket_residues.values(), start=1):
        merged_pockets.append((pid, res))
        # マージされたポケットIDを更新
        for key in merged_pocket_tracking:
            if merged_pocket_tracking[key] == None:
                merged_pocket_tracking[key] = pid
    #print(merged_pocket_tracking)

    return merged_pockets, merged_pocket_tracking

'''
代表apo以外のタンパク質ポケット位置を決定
-----------------------------------------------------------------------------
- determine_pocket_id_for_apo_b():
    (-) calculate_centroid(selection):
-----------------------------------------------------------------------------
'''

# アポタンパク質BのポケットIDを決定する関数
def determine_pocket_id_for_apo_b(apo_b_name, apo_b_chain, holo_to_pocket_id, merged_pockets, apo_a_to_b_mapping, holo_pocket_dir_refined, holo_pocket_dir_other):
    print('アポタンパク質B', apo_b_name)
    apo_b_protein_path = f"../mmcif/apo/{apo_b_name}.cif"
    cmd.load(apo_b_protein_path, "apo_b_protein")

    # チェーンIDが数字のみの場合、それを文字列として扱う
    if apo_b_chain.isdigit():
        apo_b_chain = f"'{apo_b_chain}'"
    
    apo_b_chain_clean = apo_b_chain.replace("'", "")
    cmd.remove(f"apo_b_protein and not chain {apo_b_chain}") # 対象のチェーン以外を除去
    cmd.translate([-c for c in calculate_centroid('apo_b_protein')], 'apo_b_protein') # 重心を原点に移動

    cmd.save(f"../mmcif/apo_center/{apo_b_name}_{apo_b_chain_clean}_centered.cif", 'apo_b_protein', format="cif")

    # アポタンパク質BのFASTAファイルから配列を取得
    apo_b_fasta_path = f"../data/fasta/apo/{apo_b_name}_{apo_b_chain_clean}.fasta"
    apo_b_seq = read_fasta(apo_b_fasta_path)

    pocket_rmsd_b = {}
    pocket_centroids_b = {}
    processed_pockets = {}
    apo_b_pocket_loop_percentage = {}
    apo_b_pocket_missing_percentage = {}

    for holo_name, pocket_id in holo_to_pocket_id.items(): # 対象のアポグループidに対応するホロの各ポケットに対して一つずつ処理
        # ポケット残基の決定とそのクエリの作成
        missing_percentage = 0 # アポB上でポケットの座標が欠けている割合を記録
        if pocket_id not in processed_pockets: # processrd_pockets：すでに処理済みのアポB上のポケットを記録
            apo_a_residues = None
            # merged pocketsから対応するポケットを探し、残基情報をapo_a_residues取得
            for pid, residues in merged_pockets:
                if pid == pocket_id:
                    apo_a_residues = residues
                    break

            if apo_a_residues is None:
                print(f"No residues found for pocket ID {pocket_id}")
                sys.exit()
                continue
            
            # アポAのポケット対応残基をBにマッピング(resi（残基番号）, resn（残基名）)のタプル形式
            apo_a_residues_mapped = [(apo_a_to_b_mapping[int(res[0])], res[1]) for res in apo_a_residues if int(res[0]) in apo_a_to_b_mapping]
            try:
                apo_b_selection_query = " or ".join([f"chain {apo_b_chain} and resi {resi} and resn {resn}" for resi, resn in apo_a_residues_mapped])
            except:
                apo_b_selection_query = " or ".join([f"chain {apo_b_chain_clean} and resi {resi} and resn {resn}" for resi, resn in apo_a_residues_mapped])
            processed_pockets[pocket_id] = apo_b_selection_query
            cmd.select(f"apo_b_pocket_{pocket_id}", apo_b_selection_query)


        else: # クエリ作成済みの場合
            apo_b_selection_query = processed_pockets[pocket_id]

        # 
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
                # 対応するホロとのRMSDを計算
                
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


def create_mapping_from_mmcif_to_fasta(mmcif_path, chain_id):
    """
    mmCIFファイルからFASTAシーケンスへのマッピングを生成する関数
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