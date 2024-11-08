import csv
import random

def assign_data_type():
    """
    data_type をランダムに割り当てる関数。
    'test', 'train', 'validation' を 1:8:1 の割合で割り当てる。
    """
    return random.choices(['test', 'train', 'validation'], weights=[1, 8, 1], k=1)[0]

def process_csv(input_csv_path, output_csv_path):
    """
    CSVファイルを処理し、指定された形式で新しいCSVファイルを生成する関数。

    Parameters:
    - input_csv_path: 入力CSVファイルのパス
    - output_csv_path: 出力CSVファイルのパス
    """
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        processed_rows = []

        for row in reader:
            # apo_name を apo_name, apo_chain, pocket_id で結合
            new_apo_name = f"{row['apo_name']}_{row['apo_chain']}_{row['pocket_id']}"
            # label は max_pocket_rmsd の値をそのまま利用
            label = row['max_pocket_rmsd']
            # data_type をランダムに割り当て
            data_type = assign_data_type()
            processed_rows.append({'apo_name': new_apo_name, 'label': label, 'data_type': data_type})
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['apo_name', 'label', 'data_type']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(processed_rows)


if __name__ == "__main__":
    # 入力CSVファイルと出力CSVファイルのパスを指定
    input_csv_path = '../input_csv_files/binary_protein_data_split_apo_base.csv'
    output_csv_path = '../input_csv_files/pocket_rmsd_label.csv'

    # 処理関数を呼び出し
    process_csv(input_csv_path, output_csv_path)