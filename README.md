# 回帰によるGCNNを利用したタンパク質PocketRMSDの予測
## 各フォルダの概要
- inputcsv_files : モデルの学習に用いる元のデータセット(csvファイル)
- make_dataset : データセットの作成、修正のためのコード
- model_process : 学習モデルのコード、それらを学習、テストするためのコード等

## 実行方法
## モデルの学習
```
$ source train_graph_apo_gcnn.sh
```
シェルスクリプト内の引数を操作してtrain_model_graph_GCNN.pyを実行する

## モデルのテスト
```
$ source predict_graph_GCNN.sh
```
シェルスクリプト内の引数を操作してpredict_model_graph_GCNN.pyを実行する
