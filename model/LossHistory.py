import keras
import matplotlib.pyplot as plt
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type, model_name):
        # 将文件保存到TXT
        f1 = open('./loss/{}-loss.txt'.format(model_name), 'a', encoding='utf-8')
        f1.write(str(self.losses))
        f2 = open('./loss/{}-acc.txt'.format(model_name), 'a', encoding='utf-8')
        f2.write(str(self.accuracy))
        f3 = open('./loss/{}-val_loss.txt'.format(model_name), 'a', encoding='utf-8')
        f3.write(str(self.val_loss))
        f4 = open('./loss/{}-val_acc.txt'.format(model_name), 'a', encoding='utf-8')
        f4.write(str(self.val_acc))
        f1.close()
        f2.close()
        f3.close()
        f4.close()

        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', marker='o', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', marker='D', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', marker='*', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', marker='x', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('{}-acc-loss'.format(model_name))#给x，y轴加注释
        plt.legend(loc="center right")#设置图例显示位置
        plt.savefig('./loss/{}-acc-loss.png'.format(model_name))
        plt.show()




