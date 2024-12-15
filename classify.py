import numpy
import tensorflow as tf
from tensorflow.keras import layers, models

#分類するデータ
sample_data = numpy.array([[9,8,7,6,5,4,3,2,1]])

#モデルを読み込み分類
class_model = models.load_model("bogosort_class.keras")
classified = str(numpy.argmax(class_model.predict(sample_data), axis=1)[0])

goodness_model = models.load_model("bogosort_goodness.keras")
goodness = str(numpy.argmax(goodness_model.predict(sample_data), axis=1)[0])