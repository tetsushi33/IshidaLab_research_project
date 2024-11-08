import os
import argparse
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from dataset_graph_GCNN_distance import ProteinGraphDataset2, PredictionDataset  # この行はdatasetの場所によって変更する必要があります
from model import GraphRegressionModel, DynamicGraphRegressionModel  # この行もmodelの場所によって変更する必要があります
import numpy as np
from sklearn.metrics import mean_squared_error

# 最後のモデルの状態を取得
def get_latest_checkpoint_epoch(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return 0
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    epochs = [int(f.split('_')[1].split('.pth')[0]) for f in checkpoint_files]
    return max(epochs, default=0) # モデルがなければ0からスタート

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Predict binding affinity using trained model')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--version-name', type=str, help='Path to checkpoint with model params' , default="trained_model/checkpoint_temp/01/model_epoch_300.pth")
    parser.add_argument('--checkpoint-dir', type=str, help='Path to checkpoint with model params' , default="trained_model/checkpoint_temp/01/model_epoch_300.pth")
    parser.add_argument('--graph-path', type=str, default='../hdf5_graph_files/test_graph_apo_pocket_binary_15_apo.hdf5', help='Path to graph HDF5 file')
    parser.add_argument('--label-path', type=str, help='Path to label HDF5 file', default="../input_csv_files/pocket_rmsd_label_2.csv")
    parser.add_argument('--result-dir', type=str, help='Path to result directory', default="../results")
    parser.add_argument('--data-type', choices=['train', 'validation', 'test'], type=str, default="test")
    parser.add_argument('--input-dim', type=int, default=22, help='input dim of model')
    parser.add_argument('--num-layers', type=int, default=9, help='input dim of model')
    parser.add_argument('--hidden-channels', type=int, default=34, help='input dim of model')
    #--hidden-channelsに nargs="*" という指定があったが、これを削除すると、実行が進んだ。この指定はなんのためだったのか？
    parser.add_argument('--dropout', type=float, default=0.63419655480525, help='input dim of model')
    parser.add_argument('--use-batch-norm', action='store_true', default=True, help='input dim of model')
    parser.add_argument('--pooling-type', type=str, default="mean", help='input dim of model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #print("PyTorch Version:", torch.__version__)
    #print("CUDA Version PyTorch is using:", torch.version.cuda)

    print('------load model------')
    model = GraphRegressionModel(input_dim=args.input_dim, num_layers=args.num_layers, hidden_channels=args.hidden_channels, dropout=args.dropout, use_batch_norm=args.use_batch_norm, pooling_type=args.pooling_type)
    device = 'cuda' if use_cuda else 'cpu'
    print("[Device]")
    print('Using ', device)
    print("device count : ", torch.cuda.device_count())
    print(" ")
    if use_cuda:
        model.to(device)

    # モデルのチェックポイントをロード
    checkpoint_dir = args.checkpoint_dir
    checkpoint_folder = os.path.join(checkpoint_dir, args.version_name)
    #latest_epoch = get_latest_checkpoint_epoch(checkpoint_folder) # ディレクトリ内の最新のエポック番号（次の番号）を取得
    latest_epoch = 475 # 手動で設定する場合はこっち
    checkpoint_path = os.path.join(checkpoint_folder, f'epoch_{latest_epoch}.pth') # ディレクトリ内にファイルパスを作成　例えば、model_epoch_10.pthのような形式になる
    checkpoint_model = torch.load(checkpoint_path) 
    model.load_state_dict(checkpoint_model['model_state_dict'])
    print("[Model checkpoint]")
    print(checkpoint_path)
    print(" ")
    print(model)
    print(" ")

    # 追加されたディレクトリチェックと作成のコード
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    result_folder = os.path.join(args.result_dir, args.version_name)

    print('-----load dataset-----')
    if args.data_type != "test":
        print(args.data_type, args.graph_path, args.label_path)
        dataset = ProteinGraphDataset2(args.graph_path, args.label_path, datatype=args.data_type) 
    else:
        #print(args.data_type, args.graph_path)
        dataset = ProteinGraphDataset2(args.graph_path, args.label_path, datatype=args.data_type)  # Updated this line to provide label path
        #dataset = ProteinGraphDataset(args.graph_path, args.label_path, is_train=(args.data_type=="train"))

    dataloader = DataLoader(dataset=dataset, num_workers=14, batch_size=args.batch_size, shuffle=False)

    # 保存先フォルダの設定
    result_folder = os.path.join(args.result_dir, args.version_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    print("[Result dir]")
    print(result_folder)
    print(" ")
    
    print('-----Start predict----') 
    all_preds = [] #予測結果
    all_labels = [] #教師ラベル
    all_apo_names = []  # タンパク質名
    bar = tqdm(total=len(dataloader))
    model.eval() # モデルを評価モードに設定
    
    with torch.no_grad():
        for data in dataloader: # バッチ単位でデータを取得
            labels = data.y.to(device)
            y_pred = model(data.to(device)) # モデルを使用してデータの予測を実行、結果がy_predに格納
            all_preds.extend(y_pred.cpu().numpy()) # CPU上に戻し、リストに追加
            all_labels.extend(labels.cpu().numpy())
            all_apo_names.extend(data.protein_name) 
            bar.update() # 進捗バーの更新

    # NumPy配列に変換
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    all_preds_np = all_preds_np.squeeze() # 1次元に変換

    # 相関係数を計算
    correlation = np.corrcoef(all_preds_np, all_labels_np)[0, 1]
    print("Correlation:", correlation)

    # 散布図をプロット
    plt.figure(figsize=(8, 8))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    correlation_formatted = f"{correlation:.2f}"
    plt.title(f'True vs Predicted Values({args.data_type}) (correlation: {correlation_formatted})')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')  # x軸とy軸のスケールを同じにする
    plt.xlim(0, 10)  # x軸の範囲を0から10に固定
    plt.ylim(0, 10)  # y軸の範囲を0から10に固定
    plt.plot([0, 10], [0, 10], linestyle='--', color='gray')  # y=xの点線を追加 
    plt.savefig(f"{result_folder}/scatter_plot_{args.data_type}.png")

    # 平均二乗誤差を計算
    print(len(all_labels_np))
    print(len(all_preds_np))
    mse = mean_squared_error(all_labels_np, all_preds_np)
    print("MSE: ", mse)


    df = pd.DataFrame({
        'apo_name': all_apo_names,
        'true_label': all_labels,
        'predicted_label': all_preds
    })

    
    df.to_csv(f"{result_folder}/predicted_values_{args.data_type}.csv", index=False) # 予測結果をCSVファイルとして保存

if __name__ == '__main__':
    main()
