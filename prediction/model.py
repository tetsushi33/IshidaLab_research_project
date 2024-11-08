import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from torch.nn import Sequential as Seq, Linear, ReLU, Softplus
from torch_geometric.nn import GraphConv, global_max_pool, global_mean_pool, global_add_pool, GATConv

class GATModel(torch.nn.Module):
    def __init__(self, input_dim=29, hidden_dim=256, output_dim=1, num_heads=2):
        super(GATModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.0)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=num_heads, dropout=0.0)
        self.conv3 = GATConv(hidden_dim // 2 * num_heads, hidden_dim // 4, heads=num_heads, dropout=0.0)
        self.conv4 = GATConv(hidden_dim // 4 * num_heads, hidden_dim // 8, heads=num_heads, dropout=0.0)
        self.conv5 = GATConv(hidden_dim // 8 * num_heads, hidden_dim // 16, heads=num_heads, dropout=0.0)
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2 * num_heads)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4 * num_heads)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 8 * num_heads)
        self.fc = nn.Linear(hidden_dim // 16 * num_heads, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.view(-1, self.hidden_dim * self.num_heads)
        x = F.elu(x)
        # x = self.bn1(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv2(x, edge_index)
        x = x.view(-1, self.hidden_dim // 2 * self.num_heads)
        x = F.elu(x)
        # x = self.bn2(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv3(x, edge_index)
        x = x.view(-1, self.hidden_dim // 4 * self.num_heads)
        x = F.elu(x)
        # x = self.bn3(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv4(x, edge_index)
        x = x.view(-1, self.hidden_dim // 8 * self.num_heads)
        x = F.elu(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv5(x, edge_index)
        x = F.elu(x)
        x = global_add_pool(x, data.batch)  # Use global_add_pool for graph classification
        x = self.fc(x)
        return torch.sigmoid(x)  # apply sigmoid function


class GraphBinaryClassificationModel(torch.nn.Module):
    def __init__(self, input_dim=29, hidden_dim=256, output_dim=1):
        super(GraphBinaryClassificationModel, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, 128)
        self.conv3 = GraphConv(128, 64)
        self.conv4 = GraphConv(64, 32)
        self.conv5 = GraphConv(32, 16)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc = torch.nn.Linear(32, output_dim)

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.2)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        # x = F.dropout(x, training=self.training, p=0.2)
        # x = self.conv5(x, edge_index, edge_weight)
        # x = self.bn4(x)
        # x = F.relu(x)
        x = global_mean_pool(x, batch)  # Use global_mean_pool for graph classification
        x = F.dropout(x, training=self.training)
        x = self.fc(x)
        return torch.sigmoid(x)  # apply sigmoid function

# class GCN(torch.nn.Module):
#     def __init__(self, num_features):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_features, 128)
#         self.conv2 = GCNConv(128, 64)
#         self.classifier = nn.Linear(64, 1)

#     def forward(self, x, edge_index, batch):
#         x, edge_index, batch = data.x, data.edge_index, data.batch

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=0.5, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = F.relu(x)

#         # Global pooling.
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

#         x = self.classifier(x)

#         return torch.sigmoid(x)  # Use sigmoid activation for binary classification
        
#     def predict(self, x):
#         y_pred = self.forward(x)
#         return y_pred


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(2)

        self.conv1 = nn.Conv3d(38, 50, 3)
        self.batchnorm1 = nn.BatchNorm3d(50)
        self.conv2 = nn.Conv3d(50, 70, 3)
        self.batchnorm2 = nn.BatchNorm3d(70)
        self.conv3 = nn.Conv3d(70, 100, 3)
        self.barchnorm3 = nn.BatchNorm3d(100)
        self.fc1 = nn.Linear(100*3*3*3, 1000)
        self.barchnorm4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 100)
        self.barchnorm5 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.barchnorm3(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.barchnorm4(x)
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.barchnorm5(x)
        x = self.prelu(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred


class Net48(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(2)

        self.conv1 = self.make_conv3d_seq(38, 64, 5)
        self.conv2 = self.make_conv3d_seq(64, 64, 5)
        self.conv3 = self.make_conv3d_seq(64, 128)
        self.conv4 = self.make_conv3d_seq(128, 128)
        self.conv5 = self.make_conv3d_seq(128, 256)
        self.fc1 = nn.Linear(256*(3**3), 1000)
        self.barchnorm4 = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 100)
        self.barchnorm5 = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, 1)
        # self.activation = nn.Sigmoid()

    def make_conv3d_seq(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        conv_seq = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )
        return conv_seq

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.barchnorm4(x)
        x = self.prelu(x)
        x = self.fc2(x)
        x = self.barchnorm5(x)
        x = self.prelu(x)
        x = self.fc3(x)
        # x = self.activation(x)
        return torch.sigmoid(x)

    def predict(self, x):
        y_pred = self.forward(x)
        print(y_pred)
        return y_pred


if __name__ == '__main__':
    from torchinfo import summary

    summary(GraphBinaryClassificationModel())
    # summary(Net48(), (4, 38, 48, 48, 48))


class GraphBinaryClassificationModelOptuna(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_channels, dropout, use_batch_norm=True, pooling_type="mean"):
        super(GraphBinaryClassificationModelOptuna, self).__init__()
        
        self.use_batch_norm = use_batch_norm # バッチ正規化の利用有無
        self.pooling_type = pooling_type # プーリングの種類
        
        # Convolution layers（畳み込み層の定義）
        self.conv_layers = nn.ModuleList() # 初期化
        self.conv_layers.append(GraphConv(input_dim, hidden_channels)) # 各層の設定
        for _ in range(num_layers - 1):
            self.conv_layers.append(GraphConv(hidden_channels, hidden_channels))
        
        # BatchNorm layers（バッチ正規化層の設定）
        if self.use_batch_norm:
            self.bn_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.bn_layers.append(nn.BatchNorm1d(hidden_channels))
        
        # Final fully connected layer（最終的な全結合層の定義）
        self.fc = nn.Linear(hidden_channels, 1) # この層の出力次元は１
        
        # Dropout layer（ドロップアウト層の定義、この層は畳み込みの後に適用され、過学習を防ぐためランダムにノードを無効化する）
        self.dropout = nn.Dropout(dropout)

    def forward(self, data): # モデルの入力から出力までの処理を記述
        # 入力データから必要な情報を取り出す
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        # 畳み込み層に順番を適用し、順にバッチ正規化、ReLU活性化関数、ドロップアウトを行う
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, edge_weight)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # プーリング層の適用
        if self.pooling_type == "mean":
            x = global_mean_pool(x, data.batch) # 平均プーリング
        elif self.pooling_type == "max":
            x = global_max_pool(x, data.batch) # 最大プーリング
        elif self.pooling_type == "add":
            x = global_add_pool(x, data.batch) # 加算プーリング
        
        x = self.fc(x)
        return torch.sigmoid(x)

class DynamicGraphBinaryClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, dropout, pooling_type, use_batch_norm):
        super(DynamicGraphBinaryClassificationModel, self).__init__()
        self.layers = nn.ModuleList()

        self.use_batch_norm = use_batch_norm
        self.pooling_type = pooling_type

        # 入力層
        self.layers.append(GCNConv(input_dim, hidden_channels[0]))
        
        # 隠れ層
        for i in range(len(hidden_channels)-1):
            self.layers.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
        
        # BatchNorm layers
        if self.use_batch_norm:
            self.bn_layers = nn.ModuleList()
            for channels in hidden_channels:
                self.bn_layers.append(nn.BatchNorm1d(channels))

        # 出力層
        self.out = nn.Linear(hidden_channels[-1], 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        if self.pooling_type == "mean":
            x = global_mean_pool(x, data.batch)
        elif self.pooling_type == "max":
            x = global_max_pool(x, data.batch)
        elif self.pooling_type == "add":
            x = global_add_pool(x, data.batch)
        
        x = self.out(x)
        return torch.sigmoid(x)


import torch
from torch.nn import Module, Linear, Dropout, ModuleList, BatchNorm1d
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool, global_add_pool

class GraphRegressionModel(Module):
    def __init__(self, input_dim, num_layers, hidden_channels, dropout, use_batch_norm=True, pooling_type="mean"):
        super(GraphRegressionModel, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.pooling_type = pooling_type
         
        # Convolution layers 0
        self.conv_layers = ModuleList() # 層をリストにして入れるための箱みたいなもの
        self.conv_layers.append(GraphConv(input_dim, hidden_channels)) # 畳み込み層を追加
        for _ in range(num_layers - 1):
            self.conv_layers.append(GraphConv(hidden_channels, hidden_channels)) # 中間層ではノード特徴量の次元は変わらない

        # BatchNorm layers
        if self.use_batch_norm:
            self.bn_layers = ModuleList()
            for _ in range(num_layers):
                self.bn_layers.append(BatchNorm1d(hidden_channels))
        
        # Softplus activation function
        self.softplus = ReLU() # 負の値を取らないよう追加
        # Dropout layer
        self.dropout = Dropout(dropout) # 全チャンネル共通ドロップアウト
        # Final fully connected layer for regression
        self.fc = Linear(hidden_channels, 1)  # Output dimension 1 for regression
        

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, edge_weight)
            if self.use_batch_norm:
                x = self.bn_layers[i](x) # 畳み込み層とセットで、層の数も等しいので一緒のループ内に入れちゃっている
            x = torch.relu(x)
            x = self.dropout(x) # ドロップアウト層の適用
        
        # Pooling
        if self.pooling_type == "mean":
            x = global_mean_pool(x, data.batch) # data_batchは長さがミニバッチないの全ノード数に等しく、それぞれのノードがどのグラフ（バッチ）に属するかを指す番号を要素に持つ
        elif self.pooling_type == "max":
            x = global_max_pool(x, data.batch)
        elif self.pooling_type == "add":
            x = global_add_pool(x, data.batch)
        
        x = self.fc(x)
        # Apply Softplus activation function
        #x = self.softplus(x) # 負の値を取らないよう追加
        return x  

class DynamicGraphRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_channels, dropout, pooling_type, use_batch_norm):
        super(DynamicGraphRegressionModel, self).__init__()
        self.layers = nn.ModuleList()

        self.use_batch_norm = use_batch_norm
        self.pooling_type = pooling_type

        # 入力層
        self.layers.append(GCNConv(input_dim, hidden_channels[0]))
        
        # 隠れ層
        for i in range(len(hidden_channels)-1):
            self.layers.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
        
        # BatchNorm layers
        if self.use_batch_norm:
            self.bn_layers = nn.ModuleList()
            for channels in hidden_channels:
                self.bn_layers.append(nn.BatchNorm1d(channels))

        # 出力層
        self.out = nn.Linear(hidden_channels[-1], 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.use_batch_norm:
                x = self.bn_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling
        if self.pooling_type == "mean":
            x = global_mean_pool(x, data.batch)
        elif self.pooling_type == "max":
            x = global_max_pool(x, data.batch)
        elif self.pooling_type == "add":
            x = global_add_pool(x, data.batch)
        
        x = self.out(x)
        return x  # sigmoid関数を適用しない
