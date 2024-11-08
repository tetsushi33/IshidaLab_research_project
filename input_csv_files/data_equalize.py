import csv
import math

def read_csv(csv_file):
    data = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    #print(len(data))
    return data

def write_csv(data, filename, fieldnames):
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def equalize_distribution(data, label_column):
    # ラベルの値を取得し、データをラベルに基づいてソートする
    labels = [float(row[label_column]) for row in data if row['data_type'] == 'train']
    #labels.sort()

    # ラベルの範囲を計算する
    #min_label = min(labels)
    #max_label = max(labels)
    min_label = 0
    max_label = 10
    label_range = max_label - min_label
    #print(min_label)

    # 0.2の幅で分割する
    bin_size = 0.2
    num_bins = int(math.ceil(label_range / bin_size))

    # 各範囲内のデータ数を計算する
    bin_counts = [0] * num_bins
    for label in labels:
        bin_index = min(int((label - min_label) / bin_size), num_bins - 1)
        bin_counts[bin_index] += 1
    print("--bin counts and length (train only)--")
    print(bin_counts)
    print(len(bin_counts))

    # 最もデータ数が多い範囲の半分の値を取得する
    #max_bin_count = max(bin_counts)
    #target_count = max_bin_count / 2
    target_count = 150

    difference = [0] * num_bins
    for row in data:
        label = float(row[label_column])
        bin_index = min(int((label - min_label) / bin_size), num_bins - 1)
        if row['data_type'] == 'train':
            difference[bin_index] = bin_counts[bin_index] - target_count
    
    print("--difference--")
    print(difference)

    delete_data = []
    new_data = []
    while(True):
        for row in data:
            if row['data_type'] == 'train':
                label = float(row[label_column])
                bin_index = min(int((label - min_label) / bin_size), num_bins - 1)
                if difference[bin_index] < 0:
                    # その行を複製
                    new_data.append(row.copy())
                    difference[bin_index] +=1
                if difference[bin_index] > 0:
                    # その行は削除
                    delete_data.append(row)
                    difference[bin_index] -=1

        # difference配列の要素が全て0になったら終了
        finish = 0
        for difference_value in difference:
            if difference_value != 0:
                finish = 1
        if finish == 0:
            break

    updated_data = [row for row in data if row not in delete_data]
    updated_data.extend(new_data)

    print("↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
    print(difference)
    
    print("Final data volume:", len(updated_data))
    return updated_data
    

# CSVファイルの読み込み
csv_file = 'pocket_rmsd_label_3.csv'
data = read_csv(csv_file)

# ラベルの値を0~10の範囲に分割し、データを均等にする
updated_data = equalize_distribution(data, label_column='label')
field_names = data[0].keys() if data else []
write_csv(updated_data, csv_file, field_names)


