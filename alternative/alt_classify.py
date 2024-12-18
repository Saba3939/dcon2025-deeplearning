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

def get_from_serial() :
    with serial.Serial(PORT, BAUD_RATE, timeout=TIMEOUT) as ser:
        # シリアルポートからデータを取得
        sensor_data = None
        while sensor_data == None:
            sensor_data = read_from_serial(ser)
        return numpy.array([sensor_data])

#モデルを読み込み分類
def class_predict(sample_data) :
    class_model = models.load_model("alt_bogosort_class.keras")
    return str(numpy.argmax(class_model.predict(sample_data), axis=1)[0])

def goodness_predict(sample_data) :
    goodness_model = models.load_model("alt_bogosort_goodness.keras")
    return str(numpy.argmax(goodness_model.predict(sample_data), axis=1)[0])

sample = get_from_serial()
print(sample)
print(f"class = {class_predict(sample)}, goodness = {goodness_predict(sample)}")