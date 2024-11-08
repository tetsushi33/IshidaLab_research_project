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
    parser.add_argument('--label-path', type=str, default='../input_csv_files/pocket_rmsd_label_2.csv', help='Path to label HDF5 file')

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
    device = 'cuda' if use_cuda else 'cpu'
    print("[Device]")
    print('Using', device)
    print("device count : ", torch.cuda.device_count())
    print(" ")
    if use_cuda:
        model = model.to('cuda') # モデルをGPUに転送
    weight_decay= 0.01
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay) # weght_decay -> L2ノルム正則化
    #optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)

    # チェックポイントのパスを取得
    checkpoint_dir = os.path.join(args.save_dir, args.version_name) # output/graph_apo/checkpointに設定
    # checkpoint_dirが存在しない場合、作成
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    latest_epoch = get_latest_checkpoint_epoch(checkpoint_dir) # ディレクトリ内の最新のエポック番号（次の番号）を取得
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{latest_epoch}.pth') # ディレクトリ内にファイルパスを作成　例えば、model_epoch_10.pthのような形式になる
    print("[Latest checkpoint path]")
    print(checkpoint_path)
    print(" ")

    params_filename = os.path.join(checkpoint_dir, 'params_information.txt')
    save_model_params(params_filename, args)

    # チェックポイントからモデルをロード
    if args.resume and os.path.isfile(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        start_epoch = 0
    

    ''' データセットの作成'''
    train_set = ProteinGraphDataset2(args.graph_path, args.label_path, datatype='train')
    #evaluate_train_set = ProteinGraphDataset2(args.graph_path, args.label_path, is_train=True)
    evaluate_set = ProteinGraphDataset2(args.graph_path, args.label_path, datatype='validation')
    print("[Dataset for Dataloader]")
    print("train Set Size:", len(train_set))
    #print("Evaluate train Set Size:", len(evaluate_train_set))
    print("Evaluate Test Set Size:", len(evaluate_set))
    print(" ")

    # Dataloderを使用して、バッチ単位でデータをロードできる
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    #evaluate_train_loader = DataLoader(evaluate_train_set, batch_size=args.test_batch_size)
    evaluate_loader = DataLoader(evaluate_set, batch_size=args.test_batch_size)



    ''' wandbの設定'''
    wandb_config = {
        'learning_rate': args.lr,
        'max_epochs': args.epochs,
        'batch_size': args.batch_size,
        'test_batch_size': args.test_batch_size,
        'graph': args.graph_path.split('/')[-1],
        'label': args.label_path.split('/')[-1],
        'num_layers': args.num_layers,
        'hidden_channels': args.hidden_channels,
        'dropout': args.dropout,
        'use_batch_norm': args.use_batch_norm,
        'pooling_type': args.pooling_type
    }
    # wandbの初期化
    wandb.init(project=args.project_name, name=args.version_name, entity='takanishi-t-aa', config=wandb_config, id=args.version_name, resume="allow", notes=args.project_notes)
    wandb.watch(model)


    ''' 学習開始'''
    print("-----Start training-----")
    for epoch in tqdm.tqdm(range(start_epoch+1, args.epochs+1), desc="Epoch"): # tqdmは進捗を可視化バーのためのツール
        ''' 学習フェーズ'''
        model.train() # モデルをトレーニングモードに設定
        train_loss = 0
        train_labels, train_preds = [], []
        # train_loader からミニバッチごとにデータを取得し、モデルに入力して勾配を計算し、重みを更新
        for batch_idx, data in enumerate(train_loader): 
            if use_cuda:
                data = data.to('cuda')
            optimizer.zero_grad()
            output = model(data)
            
            # 損失関数 : 平均二乗誤差(バッチサイズ分で一気に計算し、その平均を値とする)
            loss = F.mse_loss(output, data.y.float().unsqueeze((1))) # output: 予測値, data.y: 真の値

            loss.backward() # バックプロパゲーション（勾配を計算）
            optimizer.step() # 上で得た勾配から、モデルのパラメータを更新（ここで上で指定した最適化オプションを使う(Adamとか)）
            train_loss += loss.item() # 現在のバッチの損失を加算(値は一つ)

            # 真の値と、予測値を収集
            train_labels.extend(data.y.cpu().tolist()) #バッジサイズ分の要素数が追加 
            train_preds.extend(output.cpu().tolist()) #バッジサイズ分の要素数が追加 
                
        avg_loss = train_loss / len(train_loader) # (全データ数÷batch_size)で割る -> バッジ単位での平均ということ
        train_preds_np = np.array(train_preds).flatten()  # モデル出力をnumpy配列に変換
        train_labels_np = np.array(train_labels).flatten()  # 実際のラベルをnumpy配列に変換
        # 相関係数を計算
        s1 = pd.Series(train_preds_np)
        s2 = pd.Series(train_labels_np)
        train_corr =s1.corr(s2) 
        #train_rmse = calculate_rmse(train_preds_np, train_labels_np) # 相関係数(1エポック終えた後の相関係数(データ全てを使っている))
        
        print('training result   -> epoch_{} | avg_loss    : {:.5f} | R_train   : {:.4f}'.format(epoch, avg_loss, train_corr))
        #print("label average: ", np.mean(train_labels_np))
        # wandbを更新
        wandb.log({"train/avg_loss": avg_loss,
                   "train/corr": train_corr,}, step=epoch)
        
        
        ''' 検証フェーズ'''
        model.eval()
        val_loss = 0
        val_labels, val_preds = [], []
        with torch.no_grad():
            for data in evaluate_loader:
                if use_cuda:
                    data = data.to('cuda')
                output = model(data)
                val_loss += F.mse_loss(output, data.y.float().unsqueeze((1)), reduction='sum').item() # reduction=sumよりバッチサイズ内の全データでの損失の和が加算される

                # 真の値と、予測値を収集
                val_labels.extend(data.y.cpu().tolist())
                val_preds.extend(output.cpu().tolist())

        avg_loss_val = val_loss / len(evaluate_loader.dataset) # データ一つあたりの平均損失
        val_preds_np = np.array(val_preds).flatten()  # モデル出力をnumpy配列に変換
        val_labels_np = np.array(val_labels).flatten()  # 実際のラベルをnumpy配列に変換
        t1 = pd.Series(val_preds_np)
        t2 = pd.Series(val_labels_np)
        val_corr =t1.corr(t2)
        #val_rmse = calculate_rmse(val_preds_np, val_labels_np)
        print('validation result -> epoch_{} | avg_loss_val: {:.5f} | R_val     : {:.4f}'.format(epoch, avg_loss_val, val_corr))
        # wandbを更新
        wandb.log({ "validation/avg_loss": avg_loss_val, 
                    "validation/corr": val_corr}, step=epoch)

        ''' モデルを保存'''
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

if __name__ == "__main__":
    main()