import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import serial
from collections import Counter
import time

# シリアルポートの設定
PORT = 'COM4'  # 使用しているポート名
BAUD_RATE = 115200  # ボーレート
TIMEOUT = 1         # タイムアウト時間（秒）

# データバッファと結果の変数
data_buffer = []  # 10分間のデータを格納
class_result = None  # クラス分類の結果
goodness_result = None  # 良し悪しの分類の結果

def read_from_serial(ser):
    """シリアルポートからデータを読み取る関数"""
    try:
        line = ser.readline().decode('utf-8').strip()
        data = line.split(',')
        return [float(value) for value in data]  # 浮動小数点数に変換
    except ValueError:
        return None

def get_from_serial():
    """シリアルポートから1つのデータを取得"""
    with serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
        sensor_data = None
        while sensor_data is None:
            sensor_data = read_from_serial(ser)
        return np.array(sensor_data)

def class_predict(sample_data):
    """クラス分類を実行"""
    class_model = models.load_model("alt_bogosort_class.keras")
    return str(np.argmax(class_model.predict(sample_data), axis=1)[0])

def goodness_predict(sample_data):
    """良し悪し分類を実行"""
    goodness_model = models.load_model("alt_bogosort_goodness.keras")
    return str(np.argmax(goodness_model.predict(sample_data), axis=1)[0])

try:
    while True:
        # データ取得
        sample_data = get_from_serial()
        data_buffer.append(sample_data)

        # 10分分のデータが蓄積された場合
        if len(data_buffer) >= 600:  # 10分 × 60秒 = 600サンプル
            # 最頻値を計算
            data_array = np.array(data_buffer)
            mode_data = np.array([
                Counter(data_array[:, i]).most_common(1)[0][0]
                for i in range(data_array.shape[1])
            ]).reshape(1, -1)

            # モデルで予測
            class_result = class_predict(mode_data)
            goodness_result = goodness_predict(mode_data)

            # 結果を出力
            print(f"最頻値データ: {mode_data[0]}")
            print(f"クラス分類結果: {class_result}")
            print(f"良し悪し分類結果: {goodness_result}")

            # バッファをリセット
            data_buffer = []

        # データ取得間隔（例: 1秒ごとに取得）
        time.sleep(1)

except KeyboardInterrupt:
    print("終了しました。")
