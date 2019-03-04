from keras.models import load_model
from keras.models import Sequential
import mnist_train
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist_train.load_data()


model = load_model('my_model.h5')
score = model.evaluate(x_test, y_test)

x = np.linspace(1, len(score), len(score))
plt.figure()
plt.plot(x,score)
plt.show()

