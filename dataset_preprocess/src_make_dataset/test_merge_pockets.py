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
    print("merged_pocket_tracking: ", merged_pocket_tracking)
    print("pocket_keys: ", pocket_keys)
    print("----------")

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

        print("pocket_residues: ", pocket_residues)
        print("merged_pocket_tracking: ", merged_pocket_tracking)
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
        print("----------")

    # マージ結果を整理
    for pid, (key, residues) in enumerate(pocket_residues.items(), start=1):
        merged_pockets.append((pid, residues))
        for original_key in merged_pocket_tracking:
            if merged_pocket_tracking[original_key] == key or original_key == key:
                merged_pocket_tracking[original_key] = pid

    return merged_pockets, merged_pocket_tracking


def test_merge_pocket_candidates_2():
    # テストデータ: ポケット残基セット
    pocket_residues = {
        'apo_pocket_A': {('239', 'VAL'), ('172', 'LEU'), ('264', 'PRO'), ('224', 'LYS')},
        'apo_pocket_B': {('185', 'HIS'), ('239', 'VAL'), ('172', 'LEU'), ('264', 'PRO')},
        'apo_pocket_C': {('185', 'HIS'), ('239', 'VAL'), ('172', 'LEU'), ('264', 'PRO')},
        'apo_pocket_D': {('200', 'ILE'), ('201', 'ASP'), ('202', 'GLY'), ('203', 'ALA')}
    }
    # 関数を呼び出し
    merged_pockets, merged_pocket_tracking = merge_pocket_candidates_2(pocket_residues)

    # 結果を表示
    print("Merged Pockets:")
    for pocket_id, residues in merged_pockets:
        print(f"Pocket {pocket_id}: {sorted(residues)}")

    print("\nMerged Pocket Tracking:")
    for original_pocket, merged_pocket_id in merged_pocket_tracking.items():
        print(f"{original_pocket} -> Pocket {merged_pocket_id}")

# テスト実行
test_merge_pocket_candidates_2()
