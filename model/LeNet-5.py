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
            data.append([cv2.imread(dir + '/' + img, 0)])
            # img = Image.open(dir + '/' + img)
            # img = img.resize((28,28))
            # print(shape(img))
            # img = np.array(img).astype('float32').reshape(28, 28, -1)
            # print(shape(img))
            # data.append(img)
            labels.append(i)
    return data, labels


# 记录训练的loss值和准确率
history = LossHistory()

trainData, trainLabels = loadData('./datasets/MNIST/train')
# testData, testLabels = loadData('test')
trainData = np.array(trainData).astype('float32').reshape(-1, 28, 28, 1)
trainData /= 255

trainLabels = np_utils.to_categorical(trainLabels, 10)
# testLabels = np_utils.to_categorical(testLabels, 10)


# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # # x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# # # x_test = x_test.astype('float32')
# # # x_test /= 255
# # # y_test = keras.utils.to_categorical(y_test)

x_test, y_test = loadData('./datasets/MNIST/test')
x_test = np.array(x_test).astype('float32').reshape(-1, 28, 28, 1)
x_test /= 255
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="model/LeNet-5/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = checkpoint

model.fit(trainData, trainLabels, batch_size=256, epochs=100, verbose=1, shuffle=True,
          validation_data=(x_test, y_test), callbacks=[history,callbacks_list])


score = model.evaluate(x_test, y_test, verbose = 2)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
model.save('model_LeNet-5.h5')

history.loss_plot('epoch', 'Le5')