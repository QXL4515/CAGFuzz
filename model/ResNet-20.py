import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import os
import cv2
from LossHistory import LossHistory

def loadData(path):
    data = []
    labels = []
    for i in range(10):
        dir = './' + path + '/' + str(i)
        listImg = os.listdir(dir)
        for img in listImg:
            data.append([cv2.imread(dir + '/' + img)])
            labels.append(i)
    return data, labels

# 导入训练数据
x_train, y_train = loadData('./datasets/cifar-10/train')
x_train = np.array(x_train).astype('float32').reshape(-1, 32, 32, 3)
x_train /= 255
y_train = keras.utils.to_categorical(y_train, 10)


# 导入测试数据
x_test, y_test = loadData('./datasets/cifar-10/test')
x_test = np.array(x_test).astype('float32').reshape(-1, 32, 32, 3)
x_test /= 255
y_test = keras.utils.to_categorical(y_test, 10)

# 记录训练的loss值和准确率
history = LossHistory()

#ResNet Block
def resnet_block(inputs,num_filters=16,
                  kernel_size=3,strides=1,
                  activation='relu'):
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',
           kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if(activation):
        x = Activation('relu')(x)
    return x


# 建一个20层的ResNet网络
def resnet_v1(input_shape):
    inputs = Input(shape=input_shape)  # Input层，用来当做占位使用

    # 第一层
    x = resnet_block(inputs)
    print('layer1,xshape:', x.shape)
    # 第2~7层
    for i in range(6):
        a = resnet_block(inputs=x)
        b = resnet_block(inputs=a, activation=None)
        x = keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out：32*32*16
    # 第8~13层
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs=x, strides=2, num_filters=32)
        else:
            a = resnet_block(inputs=x, num_filters=32)
        b = resnet_block(inputs=a, activation=None, num_filters=32)
        if i == 0:
            x = Conv2D(32, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, b])
        x = Activation('relu')(x)
    # out:16*16*32
    # 第14~19层
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs=x, strides=2, num_filters=64)
        else:
            a = resnet_block(inputs=x, num_filters=64)

        b = resnet_block(inputs=a, activation=None, num_filters=64)
        if i == 0:
            x = Conv2D(64, kernel_size=3, strides=2, padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = keras.layers.add([x, b])  # 相加操作，要求x、b shape完全一致
        x = Activation('relu')(x)
    # out:8*8*64
    # 第20层
    x = AveragePooling2D(pool_size=2)(x)
    # out:4*4*64
    y = Flatten()(x)
    # out:1024
    outputs = Dense(10, activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 初始化模型
    # 之前的操作只是将多个神经网络层进行了相连，通过下面这一句的初始化操作，才算真正完成了一个模型的结构初始化
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = resnet_v1((32,32,3))

model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])

model.summary()

filepath="model/ResNet20/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')

def lr_sch(epoch):
    #200 total
    if epoch <20:
        return 1e-3
    if 20<=epoch<50:
        return 1e-4
    if epoch>=50:
        return 1e-5
lr_scheduler = LearningRateScheduler(lr_sch)
lr_reducer = ReduceLROnPlateau(monitor='val_acc',factor=0.2,patience=5,
                               mode='max',min_lr=1e-3)
callbacks = [checkpoint,lr_scheduler,lr_reducer,history]

model.fit(x_train,y_train,batch_size=64,epochs=100,validation_data=(x_test,y_test),
          verbose=1, callbacks=callbacks)

score = model.evaluate(x=x_test, y=y_test, verbose=0)
print('Accuracy: %.2f%%' % (score[1] * 100))

history.loss_plot('epoch', 'ResNet20')