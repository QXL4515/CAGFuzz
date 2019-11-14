from __future__ import print_function, division
import scipy
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
from keras.models import load_model

class CycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = '3to8'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

                if batch_i % 1000 == 0:
                    self.sample_images1(epoch, batch_i)
        # 保存模型
        self.combined.save('3to8.h5')

    def sample_images1(self, epoch, batch_i):
        # os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # 打印A类图片
        plt.imshow(gen_imgs[2])
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(32 / 100.0 / 3.0, 32 / 100.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.savefig("images/%s/3/%d_%d.png" % (self.dataset_name, epoch, batch_i),
                    format='png', transparent=True, dpi=300, pad_inches=0)
        plt.close()

        # 打印B类图片
        plt.imshow(gen_imgs[5])
        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(32 / 100.0 / 3.0, 32 / 100.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.savefig("images/%s/8/%d_%d.png" % (self.dataset_name, epoch, batch_i),
                    format='png', transparent=True, dpi=300, pad_inches=0)
        plt.close()


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=20, batch_size=1, sample_interval=20)





# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import os
# from PIL import Image
# import cv2
# from keras.utils import np_utils
#
# # mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
#
# def loadData(path):
#     data = []
#     labels = []
#     for i in range(10):
#         dir = './' + path + '/' + str(i)
#         listImg = os.listdir(dir)
#         for img in listImg:
#             data.append([cv2.imread(dir + '/' + img)])
#             labels.append(i)
#     return data, labels
#
# batch_size = 100  # 网络训练时用的批次大小
# Z_dim = 100  # 噪声的维度为100维
# # X_dim = mnist.train.images.shape[1]  # 图片尺寸
# # y_dim = mnist.train.labels.shape[1]  # 标签尺寸
# X_dim = 1024  # 图片尺寸，32*32*3
# y_dim = 10  # 标签尺寸
# h_dim = 128  # 隐藏层的单元数为128
#
#
#
# # 返回随机值
# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev)
#
#
# # X代表输入图片，应该是32*32，但是这里没有使用CNN，y是相应的label
# """ Discriminator Net model """
# X = tf.placeholder(tf.float32, shape=[None, X_dim])
# y = tf.placeholder(tf.float32, shape=[None, y_dim])
# # 权重，CGAN的输入是将图片输入与label concat起来，所以权重维度为784+10
# D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
# D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
# # 第二层有h_dim个节点
# D_W2 = tf.Variable(xavier_init([h_dim, 1]))
# D_b2 = tf.Variable(tf.zeros(shape=[1]))
#
# theta_D = [D_W1, D_W2, D_b1, D_b2]
#
#
# # 判别器D网络，这里是一个简单的神经网络，x是输入图片向量，y是相应的label
# def discriminator(x, y):
#     inputs = tf.concat(axis=1, values=[x, y])  # 拼接张量
#     D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  # relu激活函数 张量乘法
#     D_logit = tf.matmul(D_h1, D_W2) + D_b2
#     D_prob = tf.nn.sigmoid(D_logit)  # 输出的值压缩在0-1
#
#     return D_prob, D_logit
#
#
# # G网络参数，输入维度为Z_dim+y_dim，中间层有h_dim个节点，输出X_dim的数据
# """ Generator Net model """
# Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
# # 权重
# G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
# G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
#
# G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
# G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
#
# theta_G = [G_W1, G_W2, G_b1, G_b2]
#
#
# # 生成器G网络
# def generator(z, y):
#     inputs = tf.concat(axis=1, values=[z, y])
#     G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
#     G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
#     G_prob = tf.nn.sigmoid(G_log_prob)
#
#     return G_prob
#
#
# # 噪声产生的函数
# def sample_Z(m, n):
#     return np.random.uniform(-1., 1., size=[m, n])
#
#
# def plot(samples):
#     fig = plt.figure(figsize=(1, 1))
#     gs = gridspec.GridSpec(1, 1)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(samples):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.reshape(32, 32, 3), cmap='Greys_r')
#
#     return fig
#
#
# # 生成网络，基本和GAN一致
# G_sample = generator(Z, y)
# D_real, D_logit_real = discriminator(X, y)
# D_fake, D_logit_fake = discriminator(G_sample, y)
# # 优化式
# D_loss_real = tf.reduce_mean(
#     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(
#     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
# tf.summary.scalar('D_loss', D_loss)
# tf.summary.scalar('G_loss', G_loss)
#
# # 训练
# D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
#
# merged = tf.summary.merge_all()
# writer = tf.summary.FileWriter(r'./log_new', tf.get_default_graph())
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# # 输出图片在out文件夹
# if not os.path.exists('output/'):
#     os.makedirs('output/')
#
#
#
# saver = tf.train.Saver(max_to_keep=1)
#
# for j in range(10):
#     i = 0
#
#     for it in range(50000):
#         if it % 1000 == 0:
#             # n_sample 是G网络测试用的Batchsize，为100，所以输出的png图有100张
#             n_sample = 1
#             Z_sample = sample_Z(n_sample, Z_dim)  # 输入的噪声，尺寸为batchsize*noise维度
#             y_sample = np.zeros(shape=[n_sample, y_dim])  # 输入的label，尺寸为batchsize*label维度
#             y_sample[0, j] = 1
#
#             # y_sample[:10, 0] = 1
#             # y_sample[10:20, 1] = 1
#             # y_sample[20:30, 2] = 1
#             # y_sample[30:40, 3] = 1
#             # y_sample[40:50, 4] = 1
#             # y_sample[50:60, 5] = 1
#             # y_sample[60:70, 6] = 1
#             # y_sample[70:80, 7] = 1
#             # y_sample[80:90, 8] = 1
#             # y_sample[90:100, 9] = 1
#
#
#             samples = sess.run(G_sample, feed_dict={Z: Z_sample, y: y_sample})  # G网络的输入
#
#             fig = plot(samples)
#             plt.savefig('output/{}'.format(str(j)) + '/{}_'.format(str(j)) + '{}.png'.format(str(i).zfill(3)), bbox_inches='tight')  # 输出生成的图片
#             i += 1
#             plt.close(fig)
#
#
#         # mb_size是网络训练时用的Batchsize，为64
#         X_train, y_labels = loadData('data/cifar-10/train')
#         X_train = np.array(X_train).astype('float32').reshape(-1, 32, 32, 3)
#         X_train /= 255
#         y_labels = np_utils.to_categorical(y_labels, 10)
#
#         # num = 0
#         # while num < 50000:
#         #     X_train = trainData[num:num+batch_size]
#         #     y_labels = trainLabels[num:num+batch_size]
#         #     num = num + batch_size
#         # Z_dim是noise（噪声）的维度，为100维
#         Z_sample = sample_Z(batch_size, Z_dim)
#         # 交替最小化训练
#         _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_train, Z: Z_sample, y: y_labels})
#         _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y: y_labels})
#
#         result = sess.run(merged, feed_dict={X: X_train, Z: Z_sample, y: y_labels})
#         writer.add_summary(result, it)
#
#         # 输出训练时的参数
#         if it % 1000 == 0:
#             print('Iter: {}'.format(it))
#             print('D_loss: {:.4}'.format(D_loss_curr))
#             print('G_loss: {:.4}'.format(G_loss_curr))
#             print()
# saver.save(sess, 'model/my-model-cifar10')
