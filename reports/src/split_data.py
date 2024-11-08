import pandas as pd
import os

# CSVファイルを読み込む（ファイルパスを適切なものに変更）
csv_path = '../version_02_randomseed-1/predicted_values_test.csv'
df = pd.read_csv(csv_path)

# `predicted_label` 列を数値に変換する
df['predicted_label'] = df['predicted_label'].str.strip('[]').astype(float)

# 閾値Aを指定
A = 1.5

# 差分列を計算し追加
df['diff'] = df['predicted_label'] - df['true_label']

# TP: true_label > A かつ predicted_label > A
tp = df[(df['true_label'] > A) & (df['predicted_label'] > A)]

# TN: true_label <= A かつ predicted_label <= A
tn = df[(df['true_label'] <= A) & (df['predicted_label'] <= A)]

# FP: true_label <= A かつ predicted_label > A
fp = df[(df['true_label'] <= A) & (df['predicted_label'] > A)]

# FN: true_label > A かつ predicted_label <= A
fn = df[(df['true_label'] > A) & (df['predicted_label'] <= A)]

# 各グループを `diff` 列で大きい順にソートし、元の `true_label` と `predicted_label` も含める
columns_to_include = ['apo_name', 'true_label', 'predicted_label', 'diff']

tp_sorted = tp[columns_to_include].sort_values(by='diff', ascending=False)
tn_sorted = tn[columns_to_include].sort_values(by='diff', ascending=False)
fp_sorted = fp[columns_to_include].sort_values(by='diff', ascending=False)
fn_sorted = fn[columns_to_include].sort_values(by='diff', ascending=False)

# 各ソートされた結果を新しいCSVファイルに保存
save_dir = "../version_02_randomseed-1"
tp_sorted.to_csv(os.path.join(save_dir, 'sorted_tp.csv'), index=False)
tn_sorted.to_csv(os.path.join(save_dir, 'sorted_tn.csv'), index=False)
fp_sorted.to_csv(os.path.join(save_dir, 'sorted_fp.csv'), index=False)
fn_sorted.to_csv(os.path.join(save_dir, 'sorted_fn.csv'), index=False)

print("各分類結果を `diff` 列でソートし、元の値も含めたファイルに保存しました。")

