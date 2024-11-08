import torch
import numpy as np
from torch.optim import Adam, SGD
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
import tqdm
import wandb
import argparse
from dataset_graph_GCNN_distance import ProteinGraphDataset2
from model import GraphRegressionModel
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_rmse(predictions, labels):
    return np.sqrt(((predictions - labels) ** 2).mean())

def calculate_metrics(output, labels):
    preds = (output > 0.5).int()
    labels_cpu = labels.cpu()
    
    return labels_cpu, preds.cpu()

# モデルとオプティマイザーのチェックポイントを保存
def save_checkpoint(model, optimizer, epoch, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, filename)

def save_model_params(filename, args):
    with open(filename, 'w') as f:
        f.write(f'Learning Rate: {args.lr}\n')
        f.write(f'Batch Size: {args.batch_size}\n')
        f.write(f'Test Batch Size: {args.test_batch_size}\n')
        f.write(f'Optimizer: {args.optim}\n')
        f.write(f'Input Dimension: {args.input_dim}\n')
        f.write(f'Number of Layers: {args.num_layers}\n')
        f.write(f'Hidden Channels: {args.hidden_channels}\n')
        f.write(f'Dropout: {args.dropout}\n')
        f.write(f'Use Batch Norm: {args.use_batch_norm}\n')
        f.write(f'Pooling Type: {args.pooling_type}\n')


# チェックポイントからモデルとオプティマイザーをロード
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

# 最後のモデルの状態を取得
def get_latest_checkpoint_epoch(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return 0
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    epochs = [int(f.split('_')[1].split('.pth')[0]) for f in checkpoint_files]
    return max(epochs, default=0) # モデルがなければ0からスタート

# ディレクトリの存在を確認し、存在しない場合は作成する関数
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description='Model for predicting binding affinity using 3D convolutional neural network')
    # GCNNの設定
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--optim', type=str, choices=['Adam', 'SGD'], default='Adam', help='optimizer (default: Adam)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='how many iterations to wait before logging training status')
    parser.add_argument('--checkpoint-interval', type=int, default=100, metavar='N', help='how many iterations to wait before saving training status')
    parser.add_argument('--resume', action='store_true', default=False, help='For resuming from checkpoint')
    parser.add_argument('--save-dir', type=str, default='./trained_model_checkpoints', help='Directory to save the results')
    parser.add_argument('--version-name', type=str, default='/version_temp', help='Directory to save the results')


    parser.add_argument('--graph-path', type=str, default='../hdf5_graph_files/train_graph_apo_pocket_binary_15_apo.hdf5', help='Path to graph HDF5 file')
    parser.add_argument('--label-path', type=str, default='../input_csv_files/pocket_rmsd_label_3_equalized.csv', help='Path to label HDF5 file')

    parser.add_argument('--run-id', type=str, default='never', help='Run id of wandb to resume')
    parser.add_argument('--project-name', type=str, default='train_GCNN_models_for_pocket_rmsd', help='Project Name of WandB')
    parser.add_argument('--project-notes', type=str, default='', help='Project Notes of WandB')
    
    parser.add_argument('--input-dim', type=int, default=22, help='input dim of model')
    parser.add_argument('--num-layers', type=int, default=4, help='input dim of model')
    parser.add_argument('--hidden-channels', type=int, default=91, help='input dim of model')
    parser.add_argument('--dropout', type=float, default=0.4, help='input dim of model')
    parser.add_argument('--use-batch-norm', action='store_true', default=True, help='input dim of model')
    parser.add_argument('--pooling-type', type=str, default="mean", help='input dim of model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available() # CUDA(NVIDIAのGPUを利用して計算を高速化するっプラットフォーム)

    torch.manual_seed(args.seed)
    # CuDNNはDeep Neural Networkの高速化のためのライブラリ
    torch.backends.cudnn.deterministic = True # deterministicモードでは同じ入力に対して同じ結果が得られるようになる
    torch.backends.cudnn.benchmark = False # benchmarkモードでは、ワークロードごとに最適なアルゴリズムが自動的に選択されるため、一般的に高速。ただし、再現性を確保したい場合は無効にすることがある。

    #print("PyTorch Version:", torch.__version__)
    #print("CUDA Version PyTorch is using:", torch.version.cuda)


    ''' モデル、optimizer、 data loadersの初期化'''
    model = GraphRegressionModel(input_dim=args.input_dim, num_layers=args.num_layers, hidden_channels=args.hidden_channels, dropout=args.dropout, use_batch_norm=args.use_batch_norm, pooling_type=args.pooling_type)
    print(model)
    device = 'cuda' if use_cuda else 'cpu'
    print("[Device]")
    print('Using', device)
    print("device count : ", torch.cuda.device_count())
    print(" ")
    if use_cuda:
        model = model.to('cuda') # モデルをGPUに転送
    weight_decay= 0.01
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay) 
    #optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)


    ''' データセットの作成'''
    train_set = ProteinGraphDataset2(args.graph_path, args.label_path, is_train=True)
    evaluate_train_set = ProteinGraphDataset2(args.graph_path, args.label_path, is_train=True)
    evaluate_test_set = ProteinGraphDataset2(args.graph_path, args.label_path, is_train=False)
    print("[Dataset for Dataloader]")
    print("train Set Size:", len(train_set))
    print("Evaluate train Set Size:", len(evaluate_train_set))
    print("Evaluate Test Set Size:", len(evaluate_test_set))
    print(" ")

    # Dataloderを使用して、バッチ単位でデータをロードできる
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    #evaluate_train_loader = DataLoader(evaluate_train_set, batch_size=args.test_batch_size)
    evaluate_test_loader = DataLoader(evaluate_test_set, batch_size=args.test_batch_size)

    # 出力ディレクトリのパス
    output_directory = './check_distribution_in_batch/size256_equalized/'
    
    # ディレクトリが存在するか確認し、存在しなければ作成
    ensure_dir(output_directory)

    ''' 学習開始'''
    print("-----Start training-----")
    for epoch in tqdm.tqdm(range(1, 2), desc="Epoch"): # tqdmは進捗を可視化バーのためのツール
        ''' 学習フェーズ'''
        model.train() # モデルをトレーニングモードに設定
        train_labels = []
        batch_means = []
        # train_loader からミニバッチごとにデータを取得し、モデルに入力して勾配を計算し、重みを更新
        for batch_idx, data in enumerate(train_loader): 
            if use_cuda:
                data = data.to('cuda')
            train_labels_batch = []
            # 真の値を収集
            train_labels.extend(data.y.cpu().numpy()) 
            train_labels_batch.extend(data.y.cpu().numpy())

            label_mean_batch = np.mean(train_labels_batch) # バッジ内の平均値

            print(f"data_num: {len(train_labels)}") # バッジ内のデータ数（累積）
            print(f"Batch {batch_idx+1} Label mean: {label_mean_batch}")

            batch_means.append(label_mean_batch)

            # バッジ内の分布を保存
            # 各バッチの分布図を描画して保存
            plt.figure(figsize=(8, 4))
            sns.histplot(train_labels_batch, kde=True, color='blue', binwidth=0.2)
            plt.title(f'Distribution of Label Values in Batch {batch_idx+1}')
            plt.xlabel('Label Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            # 保存するファイルのパスを設定
            file_path = os.path.join(output_directory, f'batch_{batch_idx+1}_distribution.png')
            plt.savefig(file_path)  # 各バッチの分布図を個別に保存
            plt.close()  # Close the figure to free memory
                
        # データフレームにラベル値を追加
        results_df = pd.DataFrame({
            'label': train_labels
        })
        # ラベルの総数と平均値を計算
        total_count = len(train_labels)
        label_mean = np.mean(train_labels)
        
        print(f"Epoch {epoch} Average Label Mean: {label_mean}")

        # ヒストグラムの描画
        plt.figure(figsize=(10, 6))
        sns.histplot(results_df['label'], kde=True, color='blue', binwidth=0.2)
        plt.title('Distribution of Label Values in One Epoch')
        plt.xlabel('Label Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.text(0.95, 0.85, f'Count: {total_count}', transform=plt.gca().transAxes, horizontalalignment='right',
                 color='black', fontsize=12)
        plt.text(0.95, 0.75, f'Mean: {label_mean:.2f}', transform=plt.gca().transAxes, horizontalalignment='right',
                 color='black', fontsize=12)
        # 保存するファイルのパスを設定
        file_path = os.path.join(output_directory, 'label_distribution.png')
        plt.savefig(file_path)
        plt.close()

        #''' 検証フェーズ'''
        #model.eval()
        #val_loss = 0
        #val_labels, val_preds = [], []
        #with torch.no_grad():
        #    for data in evaluate_test_loader:
        #        if use_cuda:
        #            data = data.to('cuda')
        #        output = model(data)
        #        val_loss += F.mse_loss(output, data.y.float().unsqueeze((1)), reduction='sum').item()
#
        #        # 真の値と、予測値を収集
        #        val_labels.extend(data.y.cpu().tolist())
        #        val_preds.extend(output.cpu().tolist())
#
        #avg_loss_val = val_loss / len(evaluate_test_loader.dataset)
        #val_preds_np = np.array(val_preds).flatten()  # モデル出力をnumpy配列に変換
        #val_labels_np = np.array(val_labels).flatten()  # 実際のラベルをnumpy配列に変換
        #val_rmse = calculate_rmse(val_preds_np, val_labels_np)
        #print('validation result -> epoch_{} | avg_loss_val: {:.2f}'.format(epoch, avg_loss_val))
        ## wandbを更新
        #wandb.log({ "validation/avg_loss": avg_loss_val, 
        #            "validation/rmse": val_rmse}, step=epoch)
#
        #''' モデルを保存'''
        #checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        #save_checkpoint(model, optimizer, epoch, checkpoint_path)
#
if __name__ == "__main__":
    main()