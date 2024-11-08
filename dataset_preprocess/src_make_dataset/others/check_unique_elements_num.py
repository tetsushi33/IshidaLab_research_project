import csv
""" 
csvファイルの特定の列の要素の種類を求める
"""
def get_unique_elements_in_column(csv_file_path, column_name):
    unique_elements = set()
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if column_name in row:
                unique_elements.add(row[column_name])
            else:
                print(f"Column {column_name} not found in the CSV file.")
                return None
    return unique_elements

# 使用例
csv_file_path = '../../../csv_files_from_sakai/csv_file/similar_apo_proteins.csv'  # 読み込むCSVファイルのパス
column_name = 'pdb_id'    # 調査する列の名前

unique_elements = get_unique_elements_in_column(csv_file_path, column_name)
if unique_elements is not None:
    print(f"the number of holo '{len(unique_elements)}':")
