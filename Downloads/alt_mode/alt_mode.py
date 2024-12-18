import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from collections import Counter
import time

# モデルを読み込む
loaded_model = models.load_model("bogosort_model.keras")

# 入力データを蓄積するリスト（9つの特徴量に対応）
data_buffer = []

# 入力データのシミュレーション（例: センサーデータ）
def get_new_data():
    # 例: センサーデータをランダムに生成（現実ではセンサーから取得）
    return np.random.randint(1, 10, size=(9,))

# 最頻値と分類結果を格納する変数
mode_data_result = None
classification_result = None

# 10分ごとに最頻値を計算しモデルに入力
try:
    while True:
        # 新しいデータを取得してバッファに追加
        new_data = get_new_data()
        data_buffer.append(new_data)

        # 10分分のデータが蓄積された場合
        if len(data_buffer) >= 600:  # 10分 × 60秒 = 600サンプル（1秒に1データ取得の場合）
            # 最頻値を計算
            data_array = np.array(data_buffer)
            mode_data_result = [Counter(data_array[:, i]).most_common(1)[0][0] for i in range(data_array.shape[1])]

            # モデルで分類
            mode_data_np = np.array(mode_data_result).reshape(1, -1)  # モデルの入力形状に変換
            classification_result = np.argmax(loaded_model.predict(mode_data_np), axis=1)[0]

            # 結果を変数に格納
            print(f"最頻値データ: {mode_data_result}, 分類結果: {classification_result}")

            # バッファをリセット
            data_buffer = []

        # データ取得間隔（例: 1秒ごとに新しいデータを取得）
        time.sleep(1)

except KeyboardInterrupt:
    print("終了しました。")