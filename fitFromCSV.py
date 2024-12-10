import csv
import numpy
import tensorflow as tf
from tensorflow.keras import layers, models

# 入力：センサーの値x9
# 出力：どのような姿勢か

# 学習用入力データのサイズ
SIZE_ROW = 3
SIZE_COL = 9

x_train = numpy.ndarray(shape=[SIZE_ROW, SIZE_COL])
y_train = numpy.ndarray(shape=[SIZE_ROW])

# CSVファイル「sensor_data.csv」からセンサーデータを取得
with open("sensor_data.csv", encoding="utf8") as f:
    reader = csv.reader(f)
    idx_row = 0
    for i in reader:
        idx_col = 0
        for j in i:
            # \ufeffを除去
            if j[0] == "\ufeff":
                j = j[1:]

            x_train[idx_row, idx_col] = int(j)
            idx_col = idx_col + 1
        idx_row = idx_row + 1

# CSVファイル「class_data.csv」から分類データを取得
with open("class_data.csv", encoding="utf8") as f:
    reader = csv.reader(f)
    idx_row = 0
    for i in reader:
        # \ufeffを除去
        if i[0] == "\ufeff":
            j = j[1:]

        y_train[idx_row] = int(j)
        idx_row = idx_row + 1

# モデルの構築
model = models.Sequential([
    layers.Input(shape=(9,)),                 # 入力層（6つの特徴量）
    layers.Dense(64, activation='relu'),      # 隠れ層1
    layers.Dense(32, activation='relu'),      # 隠れ層2
    layers.Dense(6, activation='softmax')     # 出力層（6クラス分類）
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの概要を表示
model.summary()

# モデルを学習
model.fit(x_train, y_train, epochs=10, batch_size=32)

# モデルの出力
model.save("bogosort_model.keras")