import optuna
import torch
import os
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error
import numpy as np
from dataset_graph_GCNN_distance import ProteinGraphDataset2
from model import GraphRegressionModel
import argparse
from optuna.visualization import plot_optimization_history, plot_param_importances

def objective(trial):
    # ハイパーパラメータのサジェスト
    num_layers = trial.suggest_int('num_layers', 2, 10)
    hidden_channels = trial.suggest_int('hidden_channels', 1, 512)
    dropout = trial.suggest_float('dropout', 0.0, 1.0)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    # データセットとデータローダーの設定
    train_set = ProteinGraphDataset2(args.graph_path, args.label_path, datatype='train')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # モデルの初期化
    model = GraphRegressionModel(input_dim=22, num_layers=num_layers, hidden_channels=hidden_channels, dropout=dropout, use_batch_norm=True, pooling_type="mean")
    optimizer = Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 訓練ループ
    for epoch in range(10):  # 短めのエポック数でテスト
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, data.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()

    # 検証ループ（ここでは簡単のため訓練データで検証）
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            output = model(data)
            predictions.extend(output.view(-1).cpu().numpy())
            labels.extend(data.y.view(-1).cpu().numpy())

    rmse = np.sqrt(mean_squared_error(labels, predictions))
    return rmse

def run_optimization():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    print(f"Best trial: RMSE = {study.best_value}")
    print(f"Best params: {study.best_params}")
    
    # 結果の可視化（オプション）
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model for predicting binding affinity using 3D convolutional neural network')
    parser.add_argument('--graph-path', type=str, default='../hdf5_graph_files/train_graph_apo_pocket_binary_15_apo.hdf5', help='Path to graph HDF5 file')
    parser.add_argument('--label-path', type=str, default='../input_csv_files/pocket_rmsd_label_4_sqrt.csv', help='Path to label HDF5 file')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    print("[Device]")
    print('Using ', device)
    print("device count : ", torch.cuda.device_count())
    print(" ")

    run_optimization()
