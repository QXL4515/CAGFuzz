import os
import cv2
from numpy import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from PIL import Image
import numpy as np
from keras.datasets import mnist
import keras
from LossHistory import LossHistory

def loadData(path):
    data = []
    labels = []
    for i in range(10):
        dir = './' + path + '/' + str(i)
        listImg = os.listdir(dir)
        for img in listImg:
            data.append([cv2.imread(dir + '/' + img, 0)])  # imread里的0是变为一通道
            labels.append(i)
    return data, labels

# 记录训练的loss值和准确率
history = LossHistory()

trainData, trainLabels = loadData('./datasets/MNIST/train+gen')

# testData, testLabels = loadData('test')
trainData = np.array(trainData).astype('float32').reshape(-1, 28, 28, 1)
trainData /= 255
trainLabels = np_utils.to_categorical(trainLabels, 10)
# testLabels = np_utils.to_categorical(testLabels, 10)


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# x_test = x_test.astype('float32')
# x_test /= 255
# y_test = keras.utils.to_categorical(y_test)
x_test, y_test = loadData('./datasets/MNIST/test+gen')
x_test = np.array(x_test).astype('float32').reshape(-1, 28, 28, 1)
x_test /= 255
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()

model.add(Conv2D(4, (5, 5), activation = 'tanh', padding = 'same', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(12, (5, 5), activation = 'tanh', padding = 'same'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="model/LeNet1/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = checkpoint

model.fit(trainData, trainLabels, batch_size = 256, epochs = 100, verbose = 1, shuffle=True,
          validation_data=(x_test, y_test), callbacks=[history, callbacks_list])


score = model.evaluate(x_test, y_test, verbose = 2)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
model.save('retrain_model_LeNet-1.h5')

history.loss_plot('epoch', 'Le1_retrain')










# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
#
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# batch_size = 256
# num_classes = 10
# epochs = 10
#
# img_rows, img_cols = 28, 28
# input_shape = (img_rows, img_cols, 1)
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # 处理 x
# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
#
# x_train /= 255
# x_test /= 255
#
# # 处理 y
# y_train = keras.utils.to_categorical(y_train)
# y_test = keras.utils.to_categorical(y_test)
#
#
# model = Sequential()
#
# model.add(Conv2D(4, (5, 5), activation = 'relu', padding = 'same', input_shape = input_shape))
# model.add(MaxPooling2D(pool_size = (2, 2)))
#
# model.add(Conv2D(12, (5, 5), activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D(pool_size = (2, 2)))
#
# model.add(Flatten())
# model.add(Dense(10, activation = 'softmax'))
#
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
#
# model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = batch_size, epochs = epochs, verbose = 1)
#
# score = model.evaluate(x_test, y_test, verbose = 2)
# print('Test loss: ', score[0])
# print('Test accuracy: ', score[1])
#
# model.save('model_LeNet-1.h5')
#



