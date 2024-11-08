import csv
""" 
二つのcsvファイルの特定の列の要素を比較し、片方にのみ存在する要素を確認する
"""

def get_unique_elements_in_column(csv_file_path, column_name):
    elements = set()
    with open(csv_file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if column_name in row:
                elements.add(row[column_name])
            else:
                print(f"Column {column_name} not found in the CSV file.")
                return None
    return elements

def get_diff_between_csv_columns(csv_file_path1, csv_file_path2, column_name):
    elements1 = get_unique_elements_in_column(csv_file_path1, column_name)
    elements2 = get_unique_elements_in_column(csv_file_path2, column_name)

    if elements1 is None or elements2 is None:
        return None

    unique_in_file1 = elements1 - elements2
    unique_in_file2 = elements2 - elements1

    return unique_in_file1, unique_in_file2

# 使用例
csv_file_path1 = '../../../csv_files_from_sakai/csv_file/apo_holo_pairs.csv'  # 最初のCSVファイルのパス
csv_file_path2 = '../../output_csv_files/phaze_03/ver_1/apo_holo_pairs.csv'  # 二つ目のCSVファイルのパス
column_name = 'holo_name'   # 調査する列の名前

unique_in_file1, unique_in_file2 = get_diff_between_csv_columns(csv_file_path1, csv_file_path2, column_name)

if unique_in_file1 is not None and unique_in_file2 is not None:
    print(f"Elements in '{column_name}' that are in {csv_file_path1} but not in {csv_file_path2}:")
    for element in unique_in_file1:
        print(element)

    print(f"\nElements in '{column_name}' that are in {csv_file_path2} but not in {csv_file_path1}:")
    for element in unique_in_file2:
        print(element)
