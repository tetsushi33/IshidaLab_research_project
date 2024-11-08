from dataset_graph_GCNN_distance import ProteinGraphDataset
import numpy as np
import torch
import pandas as pd
import h5py
from scipy.stats import norm
import sys
from pprint import pprint
# 配列が大きい場合でも省略せずに出力するように設定
np.set_printoptions(threshold=np.inf)

original_stdout = sys.stdout  # 元のstdoutを保持

csv_path = "../input_csv_files/pocket_rmsd_label_2.csv"
graph_path = "../hdf5_graph_files/train_graph_apo_pocket_binary_15_apo.hdf5" 

csv = pd.read_csv(csv_path)
hdf5 = h5py.File(graph_path, 'r')

row = csv.iloc[1]
data_type = "train"
protein_name = row['apo_name']

x = np.array(hdf5[protein_name]['x'], dtype=np.float32)
coords = x[:, -3:]  # 座標(xの最後の3列)
distances = np.sqrt(np.sum(coords**2, axis=1, keepdims=True)) # 各ノードの原点からの距離
x = np.concatenate([x[:, :-3], distances], axis=1) # xから座標情報を除き、代わりに距離を追加

edge_index = np.array(hdf5[protein_name]['edge_index'], dtype=np.int64).T # エッジのインデックス情報(各エッジがどのノード間のものか)を読み込む
edge_attr = np.array(hdf5[protein_name]['edge_attr'], dtype=np.float32).reshape(-1, 1) # エッジの属性情報を読み込む
edge_weight = np.exp(-edge_attr) # エッジの属性から、エッジの重みを計算


with open('aaaa.txt', 'w') as f:
    sys.stdout = f
    print(csv)
    print(protein_name)
    print("================x================")
    print(x)
    print("================edge_index================")
    print(edge_index)
    print("================edge_attr================")
    print(edge_attr)

# stdoutを元に戻す
sys.stdout = original_stdout
print("This will be printed to the standard console.")
print(np.shape(edge_index))

# Define the range of edge distances
edge_distances = np.arange(6, 10.1, 0.1) # 6から10までの数値を0.1刻みで生成
edge_distances = np.round(edge_distances, 1)
print(edge_distances)
# ドロップアウトの確率を計算
dropout_probs = 1 - norm.cdf(edge_distances, loc=8, scale=2) # 正規分布の累積分布関数(平均8,標準偏差2の正規分布) 
# Create a table (dictionary) of dropout probabilities
dropout_prob_table = dict(zip(edge_distances, dropout_probs)) # エッジの距離とそれに対応するドロップアウトの確率をペアにして、辞書型にする
pprint(dropout_prob_table)

#mask = np.ones(len(edge_attr), dtype=bool) # エッジ属性の長さと同じ長さのmask配列(bool配列)を作成 初期状態では全てTrue(Trueはエッジが存在することを意味する)
## 訓練データはエッジの距離に基づいた特定の確率によりエッジをドロップアウトする
#if data_type == 'train':
#    for distance in dropout_prob_table:
#        print(distance)
#        if distance >= 6:  # Only apply dropout for edges with distance >= 6
#            # 
#            mask[edge_attr.flatten() == distance] = np.random.rand(len(mask[edge_attr.flatten() == distance])) < dropout_prob_table[distance] # エッジ属性を1次元に変換し、それがdistanceと等しい場合、
## 検証データは距離が8以上のエッジを全てドロップアウト
#else:
#    mask[edge_attr.flatten() >= 8] = False

print(len(edge_attr))
# trainデータは以下の操作をする(testとvalidarionは8.0を閾値にエッジの有無を決める
mask2 = np.ones(len(edge_attr), dtype=bool)
for i in range(len(edge_attr)):
    if edge_attr[i][0] > 6:
        if edge_attr[i][0] > 10:
            mask2[i] = 0
        else:
            edge_prob = (10 - edge_attr[i][0]) / 4
            mask2[i] = edge_prob > np.random.rand()

# 1の数をカウント
count_ones = np.count_nonzero(mask2)
# 0の数をカウント
count_zeros = len(mask2) - count_ones

# 結果を表示
print("Number of 1s:", count_ones)
print("Number of 0s:", count_zeros)

edge_index = edge_index[:, mask2] # maskの値がtrueの列(エッジ)が残る(2行N列なのでこの表記 2行なのはエッジの両端を表す)
edge_weight = edge_weight[mask2]

with open('aaaa.txt', 'a') as f:
    sys.stdout = f
    print("--------------Result-------------------")
    print(edge_index)
    print(edge_weight)
# stdoutを元に戻す
sys.stdout = original_stdout
print("")
print(np.shape(edge_index))
print("This will be printed to the standard console.")