import pandas as pd
import h5py
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from scipy.spatial.distance import cdist
from torch_geometric.loader import DataLoader
from imblearn.over_sampling import RandomOverSampler
import torch_geometric.utils as utils
import networkx as nx
import matplotlib.pyplot as plt
import random
from scipy.stats import norm

residue_dict = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'UNK': 20}

class ProteinGraphDataset(Dataset):
    def __init__(self, hdf5_file, csv_file, random_edge_dropout=10, transform=None, is_train=True):
        self.csv = pd.read_csv(csv_file)
        self.data_type = 'train' if is_train else 'validation'
        self.csv = self.csv[self.csv['data_type'] == self.data_type]
        
        # HDF5ファイルを読み込みモードでオープン
        self.hdf5 = h5py.File(hdf5_file, 'r')
        
        # トレーニングデータか検証データかに応じてランダムエッジドロップアウトの設定を変更
        if self.data_type == 'train':
            self.random_edge_dropout = random_edge_dropout
        else:
            # 検証データにはランダムエッジドロップアウトを適用しない
            self.random_edge_dropout = 1

        self.dropout_prob_table = self._create_dropout_prob_table()

        super(ProteinGraphDataset, self).__init__(None, transform)

        '''
        # データタイプに応じて処理を行う
        if self.data_type == 'train':
            # データの増強（train data）
            oversample = RandomOverSampler(sampling_strategy='minority')
            x_over, y_over = oversample.fit_resample(self.csv.drop('label', axis=1), self.csv['label'])
            self.csv = pd.concat([x_over, y_over], axis=1)

            self.hdf5 = h5py.File(hdf5_file, 'r')
            self.random_edge_dropout = random_edge_dropout
        else:
            # データの増強はなし（validation data）
            self.hdf5 = h5py.File(hdf5_file, 'r')
            self.random_edge_dropout = 1  # No random edge dropout for validation data

        super(ProteinGraphDataset, self).__init__(None, transform)
        '''



    def _create_dropout_prob_table(self):
        # Define the range of edge distances
        edge_distances = np.arange(6, 10.1, 0.1) # 6から10までの数値を0.1刻みで生成
        # ドロップアウトの確率を計算
        dropout_probs = 1 - norm.cdf(edge_distances, loc=8, scale=2) # 正規分布の累積分布関数(平均8,標準偏差2の正規分布) 
        # Create a table (dictionary) of dropout probabilities
        dropout_prob_table = dict(zip(edge_distances, dropout_probs)) # エッジの距離とそれに対応するドロップアウトの確率をペアにして、辞書型にする
        return dropout_prob_table

    def __len__(self):
        #return len(self.csv) * self.random_edge_dropout
        return len(self.csv)


    def __getitem__(self, idx):
        random.seed(idx)  # Change the random seed for each item
        row = self.csv.iloc[idx % len(self.csv)]
        protein_name = row['apo_name']

        # hdf5ファイルから、他の悪質の情報を抽出
        x = np.array(self.hdf5[protein_name]['x'], dtype=np.float32) # ノード特徴量
        coords = x[:, -3:]  # 座標(xの最後の3列)
        distances = np.sqrt(np.sum(coords**2, axis=1, keepdims=True)) # 各ノードの原点からの距離
        x = np.concatenate([x[:, :-3], distances], axis=1) # xから座標情報を除き、代わりに距離を追加

        edge_index = np.array(self.hdf5[protein_name]['edge_index'], dtype=np.int64).T # エッジのインデックス情報(各エッジがどのノード間のものか)を読み込む
        edge_attr = np.array(self.hdf5[protein_name]['edge_attr'], dtype=np.float32).reshape(-1, 1) # エッジの属性情報を読み込む
        edge_weight = np.exp(-edge_attr) # エッジの属性から、エッジの重みを計算

        # Drop edges with a certain probability based on their distance
        # エッジをドロップアウトする処理
        mask = np.ones(len(edge_attr), dtype=bool) # エッジ属性の長さと同じ長さのmask配列(bool配列)を作成 初期状態では全てTrue(Trueはエッジが存在することを意味する)
        # 訓練データはエッジの距離に基づいた特定の確率によりエッジをドロップアウトする
        if self.data_type == 'train':
            for distance in self.dropout_prob_table:
                if distance >= 6:  # Only apply dropout for edges with distance >= 6
                    # 
                    mask[edge_attr.flatten() == distance] = np.random.rand(len(mask[edge_attr.flatten() == distance])) < self.dropout_prob_table[distance] # エッジ属性を1次元に変換し、それがdistanceと等しい場合、
        # 検証データは距離が8以上のエッジを全てドロップアウト
        else:
            mask[edge_attr.flatten() >= 8] = False # この.flattenなんか意味ある？
        
        # Apply the mask to the edge_index and edge_weight
        edge_index = edge_index[:, mask] # maskの値がtrueの列(エッジ)が残る(2行N列なのでこの表記 2行なのはエッジの両端を表す)
        edge_weight = edge_weight[mask]

        # csvラベルファイルから真の値(rmsd値)を取得
        y = row['label']

        # 取得データをDataオブジェクトに変換
        # 要素：ノード特徴量、エッジインデックス、エッジ属性（重み）、ラベルの4つ
        data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_weight), y=torch.tensor([y], dtype=torch.float32))
        data.protein_name = protein_name
        return data

    def plot_graph(self, data, save_path):
        G = utils.to_networkx(data, to_undirected=True)
        plt.figure(figsize=(8, 8))
        nx.draw(G, with_labels=True)
        plt.title(f"Protein Graph of {data.protein_name}")
        plt.savefig(save_path)
        plt.close()


class ProteinGraphDataset2(Dataset):
    def __init__(self, hdf5_file, csv_file, datatype, transform=None):#ここの引数はデフォルト値なので入力があればそちらを反映する
        self.csv = pd.read_csv(csv_file)
        self.data_type = datatype
        #self.data_type = 'train' if is_train else 'validate'
        self.csv = self.csv[self.csv['data_type'] == self.data_type]
        # HDF5ファイルを読み込みモードでオープン
        self.hdf5 = h5py.File(hdf5_file, 'r')
        super(ProteinGraphDataset2, self).__init__(None, transform)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        random.seed(idx)  # Change the random seed for each item
        row = self.csv.iloc[idx % len(self.csv)]
        protein_name = row['apo_name']

        # hdf5ファイルから、他の悪質の情報を抽出
        x = np.array(self.hdf5[protein_name]['x'], dtype=np.float32) # ノード特徴量
        coords = x[:, -3:]  # 座標(xの最後の3列)
        distances = np.sqrt(np.sum(coords**2, axis=1, keepdims=True)) # 各ノードの原点からの距離
        x = np.concatenate([x[:, :-3], distances], axis=1) # xから座標情報を除き、代わりに距離を追加

        edge_index = np.array(self.hdf5[protein_name]['edge_index'], dtype=np.int64).T # エッジのインデックス情報(各エッジがどのノード間のものか)を読み込む
        edge_attr = np.array(self.hdf5[protein_name]['edge_attr'], dtype=np.float32).reshape(-1, 1) # エッジの属性情報を読み込む
        edge_weight = 1 / (1 + edge_attr) # エッジ距離からエッジの重み(特徴量)を決定  

        # エッジのドロップアウト
        mask = np.ones(len(edge_attr), dtype=bool)
        if self.data_type == 'train':
            # trainデータは距離が6以上のものがドロップの可能性を持つ
            for i in range(len(edge_attr)):
                if edge_attr[i][0] > 6:
                    if edge_attr[i][0] > 10:
                        mask[i] = 0
                    else:
                        edge_prob = (10 - edge_attr[i][0]) / 4 # ドロップする確率(距離に応じて)
                        mask[i] = edge_prob > np.random.rand()
        else:
            # test, validationデータは8以下のものだけ残す
            for i in range(len(edge_attr)):
                if edge_attr[i][0] > 8:
                    mask[i] = 0
        
        edge_index = edge_index[:, mask] # maskの値がtrueの列(エッジ)が残る(2行N列なのでこの表記 2行なのはエッジの両端を表す)
        edge_weight = edge_weight[mask]

        # csvラベルファイルから真の値(rmsd値)を取得
        y = row['label']

        # 取得データをDataオブジェクトに変換
        # 要素：ノード特徴量、エッジインデックス、エッジ属性（重み）、ラベルの4つ
        data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_weight), y=torch.tensor([y], dtype=torch.float32))
        data.protein_name = protein_name
        return data

# class PredictionDataset(Dataset):
#     def __init__(self, hdf5_file, data_type='validate', transform=None):
#         self.hdf5 = h5py.File(hdf5_file, 'r')
#         self.keys = list(self.hdf5.keys())
#         super(PredictionDataset, self).__init__(None, transform)

#     def len(self):
#         return len(self.keys)

#     def get(self, idx):
#         protein_name = self.keys[idx]
#         x = np.array(self.hdf5[protein_name]['x'], dtype=np.float32)
#         edge_index = np.array(self.hdf5[protein_name]['edge_index'], dtype=np.int64).T
#         edge_attr = np.array(self.hdf5[protein_name]['edge_attr'], dtype=np.float32).reshape(-1, 1)
#         edge_weight = np.exp(-edge_attr)

#         # Drop edges with a certain distance
#         mask = edge_attr.flatten() < 8
#         edge_index = edge_index[:, mask]
#         edge_weight = edge_weight[mask]

#         data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_weight))
#         data.protein_name = protein_name
#         return data


class PredictionDataset(Dataset):
    def __init__(self, hdf5_file, label_csv, data_type='validate', transform=None):
        self.hdf5 = h5py.File(hdf5_file, 'r')
        self.keys = list(self.hdf5.keys())
        self.labels = pd.read_csv(label_csv)
        self.data_type = data_type
        super(PredictionDataset, self).__init__(None, transform)

    def len(self):
        return len(self.keys)

    def get(self, idx):
        protein_name = self.keys[idx]
        x = np.array(self.hdf5[protein_name]['x'], dtype=np.float32)
        coords = x[:, -3:]  # 座標を抽出
        distances = np.sqrt(np.sum(coords**2, axis=1, keepdims=True))
        x = np.concatenate([x[:, :-3], distances], axis=1)        
        edge_index = np.array(self.hdf5[protein_name]['edge_index'], dtype=np.int64).T
        edge_attr = np.array(self.hdf5[protein_name]['edge_attr'], dtype=np.float32).reshape(-1, 1)
        edge_weight = np.exp(-edge_attr)

        # Drop edges with a certain distance
        mask = edge_attr.flatten() < 8
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

        # Get the label from the CSV
        y = self.labels.loc[self.labels['apo_name'] == protein_name, 'label'].values[0]

        data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_weight), y=torch.tensor([y], dtype=torch.float32))
        data.protein_name = protein_name
        return data
