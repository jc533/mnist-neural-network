from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
# load data and pre
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = load_model('my_model.h5')
score = model.evaluate(x_test, y_test)

x = np.linspace(1, len(score), len(score))
plt.figure()
plt.plot(x,score)
plt.show()

