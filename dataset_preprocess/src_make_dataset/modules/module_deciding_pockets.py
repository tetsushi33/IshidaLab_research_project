import pandas as pd
import os
import pymol
from pymol import cmd, stored
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Align import PairwiseAligner
from tqdm import tqdm
import sys
import logging
from datetime import datetime


# PDB
pdbbind_dir_refined = "../PDBbind_original_data/refined-set/"
pdbbind_dir_other = "../PDBbind_original_data/v2020-other-PL/"

conversion_dict = pd.read_csv('../csv_files/non_amino_2_amino.csv').set_index('Non-standard AA Code')['Standard AA Code'].to_dict()

AMINO_ACID_CODE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "UNK": "X"
}

colors = ["red", "green", "blue", "yellow", "orange", "purple"]

def logging_setting(log_file_path):
    os.makedirs(log_file_path, exist_ok=True)  # フォルダが存在しない場合は作成
    # タイムスタンプ付きのログファイル名を作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_file_path, f"log_{timestamp}.log")
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,  # ログの出力レベル: DEBUG, INFO, WARNING, ERROR, CRITICAL
        format='%(levelname)s - %(message)s',  # ログのフォーマット
        handlers=[
            logging.FileHandler(log_file),  # ファイルにログを記録
        ]
    )

"""
===================================================
代表アポに対する操作
- グループ内の代表アポタンパク質の決定 : deciding_apo_representitive(apo_group_id, apo_proteins_id_csv)
- アポAの決定 : prepare_apo_for_pymol(apo_A_name, apo_A_chain)
- アポAとホロの重ね合わせ : overlap_apoA_and_holos(apo_group_id, apo_A_name, apo_A_chain, apo_holo_pairs_with_group_id_csv)
===================================================
"""
def deciding_apo_representitive(apo_group_id, apo_proteins_id_csv):
    # 入力のグループidの行を抽出
    apos_in_same_groups = apo_proteins_id_csv[apo_proteins_id_csv['protein_id'] == apo_group_id]
    
    if apos_in_same_groups.empty:
        print(f'グループ{apo_group_id}に属するアポタンパク質は存在しません')
        return None
    
    # 最も出現頻度の高いアポを代表とする
    most_common = apos_in_same_groups[['apo_name', 'apo_chain']].value_counts().idxmax()

    apo_A_name, apo_A_chain = most_common

    # 文字列の処理
    if apo_A_chain.isdigit():
        apo_A_chain = f"{apo_A_chain}"
    apo_A_chain = apo_A_chain.replace("'", "")

    return apo_A_name, apo_A_chain

def prepare_apo_for_pymol(apo_A_name, apo_A_chain):
    # アポ構造のロード
    apo_A_cif_path = f"../mmcif/apo/{apo_A_name}.cif"
    cmd.delete("all")
    cmd.load(apo_A_cif_path, "apo_protein")
    cmd.remove(f"apo_protein and not chain {apo_A_chain}") # 対象のチェーン以外は削除

    # 重心の計算と移動
    centroid = calculate_centroid('apo_protein')
    cmd.translate([-c for c in centroid], 'apo_protein')
    cmd.save(f"../mmcif/apo_center/{apo_A_name}_{apo_A_chain}_centered.cif", 'apo_protein', format="cif")

    return None


def overlap_apoA_and_holos(apo_group_id, apo_A_name, apo_A_chain, apo_holo_pairs_with_group_id_csv):
    apo_A_cif_path = f"../mmcif/apo/{apo_A_name}.cif"
    apo_residue_numbers= get_residue_numbers_from_mmcif(apo_A_cif_path, apo_A_chain) # アポ構造全体の残基たち（配列）
   
    # アポAのFastaファイルの読み込み
    apo_A_fasta_path = f"../data/fasta/apo/{apo_A_name}_{apo_A_chain}.fasta"
    record = SeqIO.read(apo_A_fasta_path, "fasta")
    apo_seq = str(record.seq)

    # アポAに対応するホロ群(グループ番号の一致が基準)
    corresponding_holos_row = apo_holo_pairs_with_group_id_csv[apo_holo_pairs_with_group_id_csv['apo_group_id'] == apo_group_id]
    # 'holo_name' と 'holo_chain' の組み合わせで重複を削除
    unique_corresponding_holos_row = corresponding_holos_row.drop_duplicates(subset=['holo_name', 'holo_chain'])

    pocket_residues = {}
    pocket_residues_2 = {}
    holo_to_pocket_selection = {}
    rmsd_results = {}
    apo_pocket_loop_percentage = {}
    apo_pocket_missing_percentage = {}

    logging.info(f"代表アポタンパク質 : {apo_A_name}_{apo_A_chain}")
    logging.info(f"対応するホロタンパク質の数 : {len(unique_corresponding_holos_row)}")
    print(f"代表アポタンパク質 : {apo_A_name}_{apo_A_chain}")
    print(f"対応するホロタンパク質の数 : {len(unique_corresponding_holos_row)}")
    
    for _, corresponding_holo_row in tqdm(unique_corresponding_holos_row.iterrows(), total=len(unique_corresponding_holos_row), desc="Processing holos"):
        logging.info("=======")
        holo_name = corresponding_holo_row['holo_name']
        holo_chain = corresponding_holo_row['holo_chain']
        logging.info(f"{holo_name}_{holo_chain}")
        # ポケットファイルの存在確認
        pocket_path_refined = os.path.join(pdbbind_dir_refined, holo_name, f"{holo_name}_pocket.pdb") # 第一候補
        pocket_path_other = os.path.join(pdbbind_dir_other, holo_name, f"{holo_name}_pocket.pdb") # 第二候補（予備）
        pdb_pocket_data_path = pocket_path_refined
        if not os.path.exists(pocket_path_refined):
            if not os.path.exists(pocket_path_other):
                logging.warning("pdb bindファイルが見つかりません")
                continue
            else:
                pdb_pocket_data_path = pocket_path_other
        
        ## -----アポAと対応ホロのアライメント
        # ホロのポケット部分をロード
        cmd.load(pdb_pocket_data_path, f"{holo_name}_pocket")
        
        holo_pocket_atom_count = cmd.count_atoms(f"{holo_name}_pocket and chain {holo_chain}")
        cmd.select(f"{holo_name}_pocket_on_chain_{holo_chain}", f"chain {holo_chain} and {holo_name}_pocket")

        # ポケットの配列を取得してアライメント（対象のチェーン部分のみに限定）
        pocket_seq_converted = get_sequence_from_pdb(pdb_pocket_data_path)
        pocket_seq_converted_chain_selected = get_sequence_from_pdb_chain_selected(pdb_pocket_data_path, holo_chain)

        alignment = align_sequences(apo_seq, pocket_seq_converted_chain_selected) # ローカルアライメントに変更
        #print(alignment)

        # ポケット対応残基の決定
        selection_query, selected_numbers = select_aligned_residues_and_store_numbers_for_local(apo_A_chain, alignment, apo_residue_numbers)
        #print(selection_query)

        if selection_query == "":
            logging.warning("アポ上のポケット対応箇所を決定できません")
            continue
        else:
            # 欠損座標割合の計算
            missing_percentage = calculate_missing_coordinates_percentage(apo_A_cif_path, apo_A_chain, extract_residue_ids_from_query_revised(selection_query))
            apo_pocket_missing_percentage[holo_name] = missing_percentage
            # 
            selection_query_of_apo_pocket = f"apo_pocket_from_{holo_name}_{apo_A_chain}"
            cmd.select(selection_query_of_apo_pocket, selection_query)
            selection_query_of_apo_pocket_sub = f"apo_pocket_sub_from_{holo_name}_{apo_A_chain}"
            cmd.select(selection_query_of_apo_pocket_sub, f"apo_protein like {holo_name}_pocket and chain {holo_chain}")
            
            selected_count = cmd.count_atoms(selection_query_of_apo_pocket)
            #apo_pocket_loop_percentage[pocket_name] = calculate_apo_pocket_loop_percentage(f"apo_pocket_{pocket_name}")

            # アポ上のポケットの名前とそれに対応する残基番号を格納
            #pocket_residues[selection_query_of_apo_pocket] = set(selected_numbers) 
            '''
            例： 'apo_pocket_from_4f5y_A': {('239', 'VAL'), ('171', 'ILE'), ...}
            '''
            stored.residues = []
            cmd.iterate(selection_query_of_apo_pocket, "stored.residues.append((resi, resn))")
            # `selected_numbers` を基にしてフィルタリング
            filtered_residues = [(resi, resn) for resi, resn in stored.residues if int(resi) in selected_numbers]
            # `pocket_residues` に格納
            pocket_residues[selection_query_of_apo_pocket] = set(filtered_residues)

            holo_to_pocket_selection[corresponding_holo_row['holo_name']] = selection_query_of_apo_pocket

        ## -----RMSDの計算
        try:
            logging.info(f"holo_pocket_atom_count : {holo_pocket_atom_count}")
            rmsd_pocket                 = calculate_rmsd_if_aligned_enough(cmd.align(f"{holo_name}_pocket and chain {holo_chain}", selection_query_of_apo_pocket,     cycles = 0), holo_pocket_atom_count)
            rmsd_pocket_sub             = calculate_rmsd_if_aligned_enough(cmd.align(f"{holo_name}_pocket and chain {holo_chain}", selection_query_of_apo_pocket_sub, cycles = 0), holo_pocket_atom_count)
            rmsd_all_structure_in_chain = calculate_rmsd_if_aligned_enough(cmd.align(f"{holo_name}_pocket and chain {holo_chain}", "apo_protein",                     cycles = 0), holo_pocket_atom_count)
            logging.info(f"rmsd_pocket: {rmsd_pocket}")
            logging.info(f"rmsd_pocket_sub: {rmsd_pocket_sub}")
            logging.info(f"rmsd_all_structure_in_chain: {rmsd_all_structure_in_chain}")
        except pymol.CmdException as e:
            #print(f"Alignment failed for {holo_name}, {cmd.count_atoms(holo_name)} and apo_protein, {cmd.count_atoms('apo_protein')} : {e}")
            print("aaaaa!!!")

        rmsd_min = min(rmsd_pocket, rmsd_pocket_sub)
        rmsd_min = min(rmsd_min, rmsd_all_structure_in_chain)
        rmsd_results[holo_name] = rmsd_min # ポケット一つに対し、由来のホロの名前とその部分のRMSDを格納
        
        # 選択した残基の保存(Pymol上)
        #stored.residues = []
        #cmd.iterate(selection_query_of_apo_pocket, "stored.residues.append((resi, resn))")
        #pocket_residues_2[selection_query_of_apo_pocket] = set(stored.residues) # 2回目の更新 <- 必要？？

    
    ## ポケットのマージ
    logging.info("=======")
    merged_pockets, original_to_merged_pocket_id = merge_pocket_candidates_2(pocket_residues) 
    logging.info(f"マージ後のポケット : {merged_pockets}")
    logging.info(f"各ポケットのマージ後のポケットid : {original_to_merged_pocket_id}")
    
    # マージされたポケットごとに重心を計算
    merged_pockets_centroids = {}
    color_index = 0
    for pid, residues in merged_pockets:
        selection_query_merged_pocket = f"merged_pocket_{pid}"
        # ↓ "apo_protein and" を入れないと、何もない空間上を選択していた
        selection_residues = " or ".join([f"apo_protein and chain {apo_A_chain} and resi {resi}" for resi, resn in residues]) # resi: 残基番号、resn: 残基の種類
        cmd.select(selection_query_merged_pocket, selection_residues)
        cmd.color(colors[color_index], selection_query_merged_pocket)
        merged_pockets_centroids[pid] = calculate_centroid(selection_query_merged_pocket)
        color_index = (color_index + 1) % len(colors)

    # マージ後のポケットIDとその重心をまとめる
    merged_pocket_ids = {}
    pockets_centroid_results = {}
    for holo_name, selection in holo_to_pocket_selection.items(): # (例)holo_to_pocket_selection:{'4f5y': 'apo_pocket_from_4f5y', '4loi': 'apo_pocket_from_4loi', '4loh': 'apo_pocket_from_4loh'}
        merged_pocket_id = original_to_merged_pocket_id[selection]
        merged_pocket_ids[holo_name] = merged_pocket_id
        pockets_centroid_results[holo_name] = merged_pockets_centroids[merged_pocket_id]
    logging.info(f"merged_pocket_ids : {merged_pocket_ids}")
    
    logging.info(f"--------------------代表アポの重ね合わせ完了--------------------")
    logging.info("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓代表以外のアポの処理↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")

    return merged_pocket_ids, pockets_centroid_results, rmsd_results, merged_pockets, apo_pocket_missing_percentage

def save_pymol_process(group_id, merged_pockets, colors):
    cmd.save(f"../pse_file/pocket_visualization_protein_id_{group_id}.pse")


"""
===================================================
代表以外のアポに対する操作
- 
===================================================
"""


def prepare_apo_B_for_pymol(apo_B_name, apo_B_chain,):
    # アポBのFastaファイルをロード
    apo_B_cif_path = f"../mmcif/apo/{apo_B_name}.cif"
    cmd.load(apo_B_cif_path, 'apo_B_protein')
    # 対象のチェーン以外を削除
    if apo_B_chain.isdigit():
        apo_B_chain = f"'{apo_B_chain}'"
    apo_B_chain = apo_B_chain.replace("'", "")
    cmd.remove(f"apo_B_protein and not chain {apo_B_chain}") # 対象のチェーン以外を除去
    # 重心を計算して原点に移動
    cmd.translate([-c for c in calculate_centroid('apo_B_protein')], 'apo_B_protein') # 重心を原点に移動
    cmd.save(f"../mmcif/apo_center/{apo_B_name}_{apo_B_chain}_centered.cif", 'apo_B_protein', format="cif")
    return None

def process_for_apo_B(apo_A_name, apo_A_chain, apo_B_name, apo_B_chain, apo_holo_pairs_csv, merged_pocket_ids, merged_pockets):
    '''
    引数；
        merged pockets :  [(1, {('163', 'TYR'), ('51', 'LEU'), ('267', 'THR'),...}), (), ...] 
        merged pocket ids :  {'4f5y': 1, '4loi': 1, '4loh': 1}
    '''
    logging.info("=================================================")
    logging.info(f"apo B : {apo_B_name}_{apo_B_chain}")
    # アポAの配列情報
    apo_A_fasta_path = f"../data/fasta/apo/{apo_A_name}_{apo_A_chain}.fasta"
    record_A = SeqIO.read(apo_A_fasta_path, "fasta")
    apo_A_seq = str(record_A.seq)
    apo_A_cif_path = f"../mmcif/apo/{apo_A_name}.cif"
    # アポBの配列情報
    apo_B_fasta_path = f"../data/fasta/apo/{apo_B_name}_{apo_B_chain}.fasta"
    record_B = SeqIO.read(apo_B_fasta_path, "fasta")
    apo_B_seq = str(record_B.seq)
    apo_B_cif_path = f"../mmcif/apo/{apo_B_name}.cif"
    
    # アポAとBでアライメント
    fasta_pdb_mapping_apo_A = create_mapping_from_mmcif_to_fasta(apo_A_cif_path, apo_A_chain)
    fasta_pdb_mapping_apo_B = create_mapping_from_mmcif_to_fasta(apo_B_cif_path, apo_B_chain)
    alignment = align_sequences(apo_A_seq, apo_B_seq)
    apo_A_B_mapping = create_mapping_from_alignment(alignment, fasta_pdb_mapping_apo_A, fasta_pdb_mapping_apo_B)

    # apo_Bに対応するホロタンパク質を取得 (使っていないが！？)
    corresponding_holos = apo_holo_pairs_csv[apo_holo_pairs_csv['apo_name'].str.upper() == apo_B_name]['holo_name'].values

    ## アポA上のそれぞれのポケットに対してアポB上にポケットを割り当て、同時にRMSDを計算する
    processed_pockets = {}
    pocket_rmsd_B ={}
    pocket_centroids_B = {}
    apo_B_pocket_missing_percentage = {}
    apo_B_pocket_loop_percentage = {}
    for holo_name, pocket_id in merged_pocket_ids.items():
        logging.info("=======")
        logging.info(f"merged pocket id : {pocket_id} (from {holo_name})")
        ## アポB上でのポケット残基を決定
        if pocket_id in processed_pockets:
            apo_B_selection_query = processed_pockets[pocket_id]
        else:
            # pocket_idに対応するポケットの残基を取得
            apo_A_pocket_residues = None
            for pid, residues in merged_pockets:
                if pid == pocket_id:
                    apo_A_pocket_residues = residues

            if apo_A_pocket_residues is None:
                print(f"ポケットID{pocket_id}の残基がありません")
                sys.exit()
                continue
            
            apo_A_residues_mapped = [(apo_A_B_mapping[int(res[0])], res[1]) for res in apo_A_pocket_residues if int(res[0]) in apo_A_B_mapping]
            apo_B_selection_query = " or ".join([f"chain {apo_B_chain} and resi {resi} and resn {resn}" for resi, resn in apo_A_residues_mapped])
            processed_pockets[pocket_id] = apo_B_selection_query
            cmd.select(f"apo_B_pocket_{pocket_id}", apo_B_selection_query)

        ## RMSDの計算（ホロ構造との比較）
        if apo_B_selection_query is None:
            print(f"アポB内に、ポケットID{pocket_id}に対応するセレクションが見つかりませんでした")
            continue
        
        # 初期値は無限大に設定
        rmsd_result_with_holo = float("inf")
        rmsd_pocket_pid = float("inf")
        rmsd_pocket_holo_align = float("inf")
        rmsd_all = float("inf")
        rmsd_pocket = float("inf")
        missing_percentage = 0

        cmd.select(f"apo_B_pocket_from_{holo_name}_like", f"apo_B_protein like {holo_name}_pocket")

        # ポケットが由来するホロ（ポケットデータ）ファイルの存在を確認
        pocket_path_refined = os.path.join(pdbbind_dir_refined, holo_name, f"{holo_name}_pocket.pdb")
        pocket_path_other = os.path.join(pdbbind_dir_other, holo_name, f"{holo_name}_pocket.pdb")
        holo_pocket_path = pocket_path_refined if os.path.exists(pocket_path_refined) else pocket_path_other if os.path.exists(pocket_path_other) else None
        if holo_pocket_path:
            holo_pocket_seq = get_sequence_from_pdb(holo_pocket_path)
            alignment = align_sequences(apo_B_seq, holo_pocket_seq)
            holo_pocket_atom_count = cmd.count_atoms(f"{holo_name}_pocket")

            # アポBとホロポケットを重ね合わせ位置をポケット位置を設定
            apo_B_residue_numbers = get_residue_numbers_from_mmcif(f"../mmcif/apo/{apo_B_name}.cif", apo_B_chain)
            selection_query_holo_pocket, _ = select_aligned_residues_and_store_numbers(apo_B_chain, alignment, apo_B_residue_numbers)
            cmd.select(f"apo_B_pocket_from_{holo_name}", selection_query_holo_pocket)
            apo_B_pocket_loop_percentage[holo_name] = calculate_apo_pocket_loop_percentage(f"apo_B_pocket_from_{holo_name}")

            if selection_query_holo_pocket:
                ## RMSDの計算
                try:
                    rmsd_pocket_holo_align = calculate_rmsd_if_aligned_enough(cmd.align(f"{holo_name}_pocket", f"apo_B_pocket_from_{holo_name}", cycles=0), holo_pocket_atom_count)
                    rmsd_pocket_holo = calculate_rmsd_if_aligned_enough(cmd.align(f"{holo_name}_pocket", f"apo_B_pocket_from_{holo_name}_like", cycles=0), holo_pocket_atom_count)
                except pymol.CmdException as e:
                    print(f"Error aligning {holo_name} with apo_B_pocket_holo_align: {e}")
                rmsd_result_with_holo = min(rmsd_pocket_holo, rmsd_pocket_holo_align)
                # 欠損割合も計算
                missing_percentage = calculate_missing_coordinates_percentage(apo_B_cif_path, apo_B_chain, extract_residue_ids_from_query(selection_query_holo_pocket))
                
        logging.info(f"rmsd_pocket_holo_align : {rmsd_pocket_holo_align}")
        logging.info(f"rmsd_pocket_holo : {rmsd_pocket_holo}")

        try:
                rmsd_pocket_pid = calculate_rmsd_if_aligned_enough(cmd.align(f"{holo_name}_pocket", apo_B_selection_query, cycles=0), holo_pocket_atom_count)
        except pymol.CmdException as e:
            print(f"Error aligning {holo_name} with {apo_B_selection_query}: {e}")
        logging.info(f"rmsd_pocket_pid : {rmsd_pocket_pid}")
        rmsd_result_with_holo = min(rmsd_result_with_holo, rmsd_pocket_pid)
        try:
                rmsd_all = calculate_rmsd_if_aligned_enough(cmd.align(f"{holo_name}_pocket", "apo_B_protein", cycles=0), holo_pocket_atom_count)
        except pymol.CmdException as e:
            print(f"Error aligning {holo_name} with apo_B_protein: {e}")
        logging.info(f"rmsd_all : {rmsd_all}")
        rmsd_result_with_holo = min(rmsd_result_with_holo, rmsd_all)

        pocket_rmsd_B[holo_name] = rmsd_result_with_holo
        pocket_centroid_b = calculate_centroid(apo_B_selection_query)
        pocket_centroids_B[holo_name] = pocket_centroid_b
        apo_B_pocket_missing_percentage[holo_name] = missing_percentage
    
    pse_filename = f"../pse_file/{apo_B_name}_{apo_B_chain}_pocket.pse"
    cmd.save(pse_filename)
    cmd.delete("apo_B_protein")
    return pocket_rmsd_B, pocket_centroids_B, apo_B_pocket_loop_percentage, apo_B_pocket_missing_percentage

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
            
"""
============================
その他
============================
"""

def calculate_centroid(selection):
    stored.xyz = [] # storedはpymolの組み込み変数
    cmd.iterate_state(1, selection, 'stored.xyz.append([x,y,z])') # 引数selectionに該当する各原子の座標をappend
    x, y, z = zip(*stored.xyz) # 各軸ごとに要素を取り出してx,y,zに格納
    centroid = [sum(x)/len(x), sum(y)/len(y), sum(z)/len(z)] # 各軸ごとに平均を計算
    return centroid

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

def get_sequence_from_pdb_chain_selected(pdb_path, chain_id = None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    sequences = []
    for model in structure:
        for chain in model:
            # 指定されたチェーンのみ抽出
            if chain_id and chain.id != chain_id:
                continue
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


def align_sequences(seq1, seq2):
    '''
    引数の二つのアミノ酸配列から、それらのスコアとtarget,queryの並びを計算して返す
     - target : seq1
     - query : seq2
    '''
    aligner = PairwiseAligner()
    aligner.mode = 'local'
    # スコアリングの調整
    aligner.match_score = 2         # 一致部分に対するスコア
    aligner.mismatch_score = -1     # 不一致部分に対するペナルティ
    aligner.open_gap_score = -2     # ギャップ開始のペナルティ
    aligner.extend_gap_score = -0.5 # ギャップ延長のペナルティ

    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]
    return best_alignment

def select_aligned_residues_and_store_numbers(apo_chain_id, alignment, apo_residue_numbers):
    '''
    アライメント結果の一致部分から、ポケットとする残基を決める
    返り値
    - (chain A and resi 15) or (chain A and resi 23) or (chain A and resi 28) or ...
    - [15, 23, 28, 34, ...] （<-これはアポ上の残基番号）
    '''
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

def select_aligned_residues_and_store_numbers_for_local(apo_chain_id, alignment, apo_residue_numbers):
    '''
    アライメント結果の一致部分から、ポケットとする残基を決める
    返り値
    - (chain A and resi 15) or (chain A and resi 23) or (chain A and resi 28) or ...
    - [15, 23, 28, 34, ...] （<-これはアポ上の残基番号）
    '''
    selected_residue_numbers = []
    query = []

    apo_index = alignment.aligned[0][0][0]
    holo_index = 0

    for apo_res, holo_res in zip(alignment[0], alignment[1]): # alignment[0]:target配列, alignment[1]:query配列
        if apo_res != "-" and holo_res != "-": # 残基が一致した場合
            # Both residues are aligned
            query.append(f"(apo_protein and chain {apo_chain_id} and resi {apo_residue_numbers[apo_index]})") # pymol用のクエリを作成してappend
            # 例："chain A and resi 34"
            selected_residue_numbers.append(apo_residue_numbers[apo_index])

        if apo_res != "-":
            apo_index += 1
        if holo_res != "-":
            holo_index += 1
    
    return " or ".join(query), selected_residue_numbers


def extract_residue_ids_from_query(selection_query):
    '''
    例：
    (apo_protein and chain A and resi 158) を (A, 158) にする
    '''
    residue_ids = []
    parts = selection_query.split(" or ")
    for part in parts:
        chain_residue_pair = part.split(" and ")
        chain_id = chain_residue_pair[0].split(" ")[-1]
        residue_num = chain_residue_pair[1].split(" ")[-1].replace(")", "")  # ')' を取り除く
        residue_ids.append((chain_id, int(residue_num)))
    return residue_ids

def extract_residue_ids_from_query_revised(selection_query):
    '''
    例：
    (apo_protein and chain A and resi 158) を (A, 158) にする
    '''
    residue_ids = []
    parts = selection_query.split(" or ")
    for part in parts:
        chain_residue_pair = part.split(" and ")
        chain_id = chain_residue_pair[1].split(" ")[-1]  # 'chain' の後ろのIDを取得
        residue_num = chain_residue_pair[2].split(" ")[-1].replace(")", "")  # 'resi' の後ろの番号を取得し、')'を除去
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

    align_result(cmd.alignの出力):
    例 : (11.465438842773438, 415, 0, 11.465438842773438, 415, 161.0, 49)
    上から
        RMSD after refinement
        Number of aligned atoms after refinement
        Number of refinement cycles
        RMSD before refinement
        Number of aligned atoms before refinement
        Raw alignment score
        Number of residues aligned
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

"""
============================
欠損座標割合の計算
============================
"""
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


"""
============================
ポケットのマージ（重なり部分の処理）
============================
"""
def merge_pocket_candidates(pocket_residues):
    '''
    返り値
    - (マージ後のポケットID(1から), その残基)
    - {(アポ上のポケットの名前, ポケットID（マージ先）), (), (), ...}
    '''
    # マージされたポケットIDを追跡するための辞書
    merged_pocket_tracking = {key: None for key in pocket_residues.keys()}
    
    while True:
        max_overlaping_residue_counts = 0
        max_pair = None
        pocket_keys = list(pocket_residues.keys())
        for i in range(len(pocket_keys)):
            for j in range(i + 1, len(pocket_keys)): 
                overlaping_residue_counts = len(pocket_residues[pocket_keys[i]] & pocket_residues[pocket_keys[j]]) # 二つに残基群のうち共通する残基の数
                overlap_percent_i = overlaping_residue_counts / len(pocket_residues[pocket_keys[i]])
                overlap_percent_j = overlaping_residue_counts / len(pocket_residues[pocket_keys[j]])
                if overlap_percent_i >= 0.5 or overlap_percent_j >= 0.5: #共通部分が、どちらか一方にとって0.5以上
                    if overlaping_residue_counts > max_overlaping_residue_counts:
                        max_overlaping_residue_counts = overlaping_residue_counts
                        max_pair = (pocket_keys[i], pocket_keys[j]) # 最も共通部分が大きいペアに更新（割合ではなく、残基の数で更新の基準にしてる）

        # 最も共通部分の原子数が多いペアを結合
        if max_pair:
            # マージされたポケットを追跡
            #merged_pocket_tracking[max_pair[0]] = max_pair[1]
            for key, value in merged_pocket_tracking.items():
                if value == max_pair[0]:
                    merged_pocket_tracking[key] = max_pair[1]
            for key in merged_pocket_tracking:
                if merged_pocket_tracking[key] == max_pair[0]:
                    merged_pocket_tracking[key] = max_pair[1]
            pocket_residues[max_pair[0]].update(pocket_residues[max_pair[1]]) # 残基リストを結合
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
    
def merge_pocket_candidates_2(pocket_residues):
    """
    ポケットの残基セットをマージする関数
    Args:
        pocket_residues (dict): {ポケット名: 残基セット} の辞書
    Returns:
        merged_pockets (list): [(ポケットID, 残基セット), ...]
        merged_pocket_tracking (dict): {元のポケット名: マージ後のポケットID}
    """
    # 初期化
    merged_pocket_tracking = {key: None for key in pocket_residues.keys()}
    merged_pockets = []
    pocket_keys = list(pocket_residues.keys())

    # マージ処理
    while True:
        max_overlapping_pair = None
        max_overlapping_count = 0

        for i in range(len(pocket_keys)):
            for j in range(i + 1, len(pocket_keys)):
                key_i, key_j = pocket_keys[i], pocket_keys[j]
                overlapping_residues = pocket_residues[key_i] & pocket_residues[key_j]
                overlap_percent_i = len(overlapping_residues) / len(pocket_residues[key_i])
                overlap_percent_j = len(overlapping_residues) / len(pocket_residues[key_j])

                # マージ条件に合うペアを見つける
                if overlap_percent_i >= 0.5 or overlap_percent_j >= 0.5:
                    if len(overlapping_residues) > max_overlapping_count:
                        max_overlapping_pair = (key_i, key_j)
                        max_overlapping_count = len(overlapping_residues)

        # マージ実行または終了
        if max_overlapping_pair:
            key_i, key_j = max_overlapping_pair

            # 残基セットをマージ
            pocket_residues[key_i].update(pocket_residues[key_j])
            del pocket_residues[key_j]
            pocket_keys.remove(key_j)

            # マージ追跡情報を更新
            for key, value in merged_pocket_tracking.items():
                if value == key_j or key == key_j:
                    merged_pocket_tracking[key] = key_i
            merged_pocket_tracking[key_j] = key_i
        else:
            break

    # マージ結果を整理
    for pid, (key, residues) in enumerate(pocket_residues.items(), start=1):
        merged_pockets.append((pid, residues))
        for original_key in merged_pocket_tracking:
            if merged_pocket_tracking[original_key] == key or original_key == key:
                merged_pocket_tracking[original_key] = pid

    return merged_pockets, merged_pocket_tracking
