import numpy as np
from PIL import Image
from keras.models import load_model
from Keras_coverage import NCoverage
import os
import cv2
import matplotlib.pyplot as plt

# 计算余弦相似性
def get_cossimi(x,y):
    # x = x.resize((28, 28), Image.BILINEAR)
    # y = y.resize((28, 28), Image.BILINEAR)
    # myx=np.array(x).astype('float32')   # array 后的类型是uint，要转换为浮点型，不然溢出
    # myy=np.array(y).astype('float32')
    myx = x
    myy = y
    cos1=np.sum(myx*myy)
    cos21=np.sqrt(np.sum(myy*myy))
    cos22=np.sqrt(np.sum(myx*myx))
    return (cos1/float(cos22*cos21))

def is_coverage_increase(img1, img2):
    # 导入待测模型
    model = load_model('my-model_cnn.h5')
    coverage = NCoverage(model, 0.1)

    img1 = img1.resize((28, 28))
    img1 = np.array(img1).astype('float32').reshape(-1, 28, 28, 1)
    coverage_original = len(coverage.get_neuron_coverage(img1))
    # print('原覆盖率', coverage_original)

    img2 = img2.resize((28, 28))
    img2 = np.array(img2).astype('float32').reshape(-1, 28, 28, 1)
    coverage_new = len(coverage.get_neuron_coverage(img2))
    # print('新覆盖率', coverage_new)
    if coverage_new > coverage_original:
        return True
    else:
        return False


# 读取数据
def loadData(path):
    data = []
    labels = []
    for i in range(10):
        dir = './' + path + '/' + str(i)
        listImg = os.listdir(dir)
        for img in listImg:
            data.append([cv2.imread(dir + '/' + img, 0)])
            labels.append(i)
    return data, labels

# 将txt文件转化为字典，和LossHistory里的把字典转换为txt是一套的
def readDict(model_name):
    f1 = open('./loss/{}-loss.txt'.format(model_name), 'r')
    a1 = f1.read()
    loss = eval(a1)
    f2 = open('./loss/{}-acc.txt'.format(model_name), 'r')
    a2 = f2.read()
    acc = eval(a2)
    f3 = open('./loss/{}-val_loss.txt'.format(model_name), 'r')
    a3 = f3.read()
    val_loss = eval(a3)
    f4 = open('./loss/{}-val_acc.txt'.format(model_name), 'r')
    a4 = f4.read()
    val_acc = eval(a4)
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    iters = range(len(loss['epoch']))
    # 创建一个图
    plt.figure()
    # acc
    plt.plot(iters, acc['epoch'], 'r', marker='o', label='train acc')  # plt.plot(x,y)，这个将数据画成曲线
    # loss
    plt.plot(iters, loss['epoch'], 'g', marker='D', label='train loss')
    # val_acc
    plt.plot(iters, val_acc['epoch'], 'b', marker='*', label='val acc')
    # val_loss
    plt.plot(iters, val_loss['epoch'], 'k', marker='x', label='val loss')
    plt.grid(True)  # 设置网格形式
    plt.xlabel('epoch')
    plt.ylabel('{}-acc-loss'.format(model_name))  # 给x，y轴加注释
    plt.legend(loc="center right")  # 设置图例显示位置
    plt.savefig('./loss/{}-acc-loss.png'.format(model_name))
    plt.show()