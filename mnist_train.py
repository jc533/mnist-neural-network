from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import Callback, History
import matplotlib.pyplot as plt
import numpy as np
# load data and pre


def load_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train/255
    x_test = x_test/255
    # turn y into one-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def build_model():

    model = Sequential()
    model.add(Dense(input_dim=28*28, units=500, activation='relu'))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.summary()
    return model

def train_model(model):

    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy','acc'])
    train_history = model.fit(x_train, y_train, batch_size=800, validation_split=0.2,
                              epochs=10, callbacks=[TestCallback((x_test, y_test))])
    return train_history


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        global test_history
        x, y = self.test_data
        loss, acc, val_loss = self.model.evaluate(x, y, verbose=0)
        # print("\n\n\n",self.model.evaluate(x, y, verbose=0),"\n\n\n")
        # print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
        print("test ",acc)
        test_history['loss'].append(loss)
        test_history['acc'].append(acc)
        test_history['val_loss'].append(val_loss)





(x_train, y_train), (x_test, y_test) = load_data()
model = build_model()
test_history = {"acc":[],"loss":[],"val_loss":[]}
train_history = train_model(model)

model.save('my_model.h5')

# score = model.evaluate(x_train, y_train)
# score = model.evaluate(x_test, y_test)
# test_acc = np.sum(score)/len(score)
# print(np.sum(score), " debug ", len(score))
# test_score = score[:]
# print(score[1])


# print("train score: ",train_acc)
# print("test score: ",test_acc)
# print(train_history.history," testing ")
# plt.plot(train_history.history['loss'])
plt.plot(train_history.history['acc'])
plt.plot(test_history['acc'])

# plt.plot(train_score)
# plt.plot(test_score)


plt.title('Train History')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.axis([0, 10, 0, 2])
plt.legend(['train acc', 'test acc'],
           loc='upper left')
plt.show()

plt.plot(train_history.history['val_loss'])
plt.plot(test_history['val_loss'])

plt.title('Train History')
plt.ylabel('val loss')
plt.xlabel('Epoch')
plt.axis([0, 10, 0, 2])
plt.legend(['train val loss', 'test val loss'],
           loc='upper left')
plt.show()

del model
