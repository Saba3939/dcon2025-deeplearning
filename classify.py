import numpy
import tensorflow as tf
from tensorflow.keras import layers, models
import serial

# シリアルポートの設定
PORT = 'COM4'  # 使用しているポート名（例: WindowsではCOMポート、Linux/Macでは/dev/ttyXXX）
BAUD_RATE = 115200  # ボーレート
TIMEOUT = 1000       # タイムアウト時間（秒）


def read_from_serial(ser):
    """シリアルポートからデータを読み取る関数"""
    try:
        # シリアルポートから1行を読み込む
        line = ser.readline().decode('utf-8').strip()
        # センサの値をカンマ区切りで取得（例: "12.3,45.6,78.9"）
        data = line.split(',')
        return [float(value) for value in data]  # 浮動小数点数に変換
    except ValueError:
        # データの形式が正しくない場合はNoneを返す
        return None

with serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
    # シリアルポートからデータを取得
    sensor_data = None
    while sensor_data == None:
        sensor_data = read_from_serial(ser)

#分類するデータ
sample_data = numpy.array([sensor_data])
print(sample_data)

#モデルを読み込み分類
class_model = models.load_model("bogosort_class.keras")
classified = str(numpy.argmax(class_model.predict(sample_data), axis=1)[0])

goodness_model = models.load_model("bogosort_goodness.keras")
goodness = str(numpy.argmax(goodness_model.predict(sample_data), axis=1)[0])

print(f"class = {classified}, goodness = {goodness}")