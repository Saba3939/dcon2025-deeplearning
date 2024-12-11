import numpy
import tensorflow as tf
from tensorflow.keras import layers, models

#分類するデータ
sample_data = numpy.array([[9,8,7,6,5,4,3,2,1]])

#モデルを読み込み分類
loaded_model = models.load_model("bogosort_model.keras")
predictions = str(numpy.argmax(loaded_model.predict(sample_data), axis=1)[0])