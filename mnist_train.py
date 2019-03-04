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

model = Sequential()

model.add(Dense(input_dim=28*28, units=500, activation='relu'))
model.add(Dense(units=500, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer="adam", metrics=['accuracy'])
train_history = model.fit(x_train, y_train, batch_size=800,
                          validation_split=0.2, epochs=10, verbose=2)

model.save('my_model.h5')

score = model.evaluate(x_train, y_train)
print('\nTrain Acc:', score[1])
score = model.evaluate(x_test, y_test)
print('\nTest Acc:', score[1])

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Train History')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.show()

del model








#並且用資料視覺化函式 庫繪出訓練過程中，
#在訓練集和測試集的正確率和損失函數值
