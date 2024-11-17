import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,  # ログの出力レベル: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',  # ログのフォーマット
    handlers=[
        logging.FileHandler("log.txt"),  # ファイルにログを記録
        logging.StreamHandler()          # 標準出力（コンソール）に表示
    ]
)

# サンプルプログラム
def divide_numbers(a, b):
    """2つの数を割り算する関数"""
    logging.info(f"開始: divide_numbers({a}, {b})")
    if b == 0:
        logging.error("ゼロ除算エラー: bが0です")
        return None
    result = a / b
    logging.info(f"終了: 結果 = {result}")
    return result

# 実行部分
logging.info("プログラム開始")

try:
    divide_numbers(10, 2)  # 正常な例
    divide_numbers(10, 0)  # エラーの例
    divide_numbers(15, 3)  # 正常な例
except Exception as e:
    logging.critical("予期せぬエラーが発生しました", exc_info=True)

logging.info("プログラム終了")
