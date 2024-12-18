import csv
import numpy
import tensorflow as tf
from tensorflow.keras import layers, models

# 入力：センサーの値x9
# 出力：どのような姿勢か

# 学習用入力データのサイズ
SIZE_ROW = 422
SIZE_COL = 9

x_train = numpy.ndarray(shape=[SIZE_ROW, SIZE_COL])
class_train = numpy.ndarray(shape=[SIZE_ROW])
goodness_train = numpy.ndarray(shape=[SIZE_ROW])

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

            x_train[idx_row, idx_col] = float(j)
            idx_col = idx_col + 1
        idx_row = idx_row + 1

# CSVファイル「class_data.csv」から分類データを取得
with open("class_data.csv", encoding="utf8") as f:
    reader = csv.reader(f)
    idx_row = 0
    for i in reader:
        # \ufeffを除去
        if i[0][0] == "\ufeff":
            i[0][0] = i[0][1:]

        class_train[idx_row] = int(i[0])
        idx_row = idx_row + 1

# モデルの構築
class_model = models.Sequential([
    layers.Input(shape=(9,)),                 # 入力層（9つの特徴量）
    layers.Dense(64, activation='relu'),      # 隠れ層1
    layers.Dense(32, activation='relu'),      # 隠れ層2
    layers.Dense(6, activation='softmax')     # 出力層（6クラス分類）
])

# モデルのコンパイル
class_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの概要を表示
# class_model.summary()

# モデルを学習
print("class_model:")
class_model.fit(x_train, class_train, epochs=32, batch_size=16)

# モデルの出力
class_model.save("bogosort_class.keras")

# CSVファイル「goodness_data.csv」から良さデータを取得
with open("goodness_data.csv", encoding="utf8") as f:
    reader = csv.reader(f)
    idx_row = 0
    for i in reader:
        # \ufeffを除去
        if i[0][0] == "\ufeff":
            i[0][0] = i[0][1:]

        goodness_train[idx_row] = int(i[0])
        idx_row = idx_row + 1

# モデルの構築
goodness_model = models.Sequential([
    layers.Input(shape=(9,)),                 # 入力層（9つの特徴量）
    layers.Dense(64, activation='relu'),      # 隠れ層1
    layers.Dense(32, activation='relu'),      # 隠れ層2
    layers.Dense(3, activation='softmax')     # 出力層（6クラス分類）
])

# モデルのコンパイル
goodness_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの概要を表示
# goodness_model.summary()

# モデルを学習
print("goodness_model:")
goodness_model.fit(x_train, goodness_train, epochs=32, batch_size=16)

# モデルの出力
goodness_model.save("bogosort_goodness.keras")