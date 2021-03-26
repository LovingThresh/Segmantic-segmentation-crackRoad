# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2021/2/19 21:08
# @Author : Ye
# @File : train.py
# @SoftWare : Pycharm

import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import get_file
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import models
import matplotlib.pyplot as plt

CLASS_NUMBERS = 2  # 分几类，这里分两类
HEIGHT = 227
WIDTH = 227
batch_size = 4


def customed_loss(y_true, y_pred):
    """自定义损失函数"""

    loss = binary_crossentropy(y_true, y_pred)
    return loss


def get_model():
    """获取模型， 并加载官方预训练的模型参数"""

    # 获取模型
    model = models.main()

    # 加载参数
    filename = r'I:/PycharmProjects/Semantic-segmentation/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(filename, by_name=True)

    # 编译
    model.compile(loss=customed_loss, optimizer=Adam(1e-3), metrics=['accuracy'])

    return model


def get_data():
    """
    获取样本和标签对应的行：获取训练集和验证集的数量
    :return: lines： 样本和标签的对应行： [num_train, num_val] 训练集和验证集数量
    """

    # 读取训练样本和样本对应关系的文件 lines -> [1.jpg;1.jpg\n', '10.jpg;10.png\n', ......]
    # .jpg:样本  ：  .jpg：标签

    with open(r'I:\PycharmProjects\Segmantic-segmentation-crackRoad\dataset\train.txt', 'r') as f:
        lines = f.readlines()

    print(lines)

    # 打乱行， 打乱数据有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 切分训练样本， 90% 训练： 10% 验证
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    return lines, num_train, num_val


def set_callbacks():
    """设置回调函数"""

    # 1. 有关回调函数的设置（callbacks）
    logdir = os.path.join('callbacks')
    print(logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    output_model_file = os.path.join(logdir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    callbacks = [
        ModelCheckpoint(output_model_file, save_best_only=True, save_freq='epoch'),
        ReduceLROnPlateau(factor=0.5, patience=3),
        EarlyStopping(min_delta=1e-3, patience=10)
    ]

    return callbacks, logdir


def generate_arrays_from_file(lines, batch_size):
    """
    生成器， 读取图片， 并对图片进行处理， 生成（样本，标签）
    :param lines: 样本和标签的对应行
    :param batch_size: 一次处理的图片数
    :return:  返回（样本， 标签）
    """

    numbers = len(lines)
    read_line = 0
    while True:

        x_train = []
        y_train = []

        # 一次获取batch——size大小的数据

        for t in range(batch_size):
            np.random.shuffle((lines))

        # 1. 获取训练文件的名字
        train_x_name = lines[read_line].split(';')[0]

        # 根据图片名字读取图片
        img = Image.open(r'I:\PycharmProjects\Segmantic-segmentation-crackRoad\dataset\img_png' + '/' + train_x_name)
        img = img.resize((WIDTH, HEIGHT))
        img_array = np.array(img)

        img_array = img_array / 255  # 标准化

        x_train.append(img_array)

        # 2. 获取训练样本标签的名字
        train_y_name = lines[read_line].split(';')[1].replace('\n', '')

        # 根据图片名字读取图片
        img = Image.open(r'I:\PycharmProjects\Segmantic-segmentation-crackRoad\dataset\label_png' + '/' + train_y_name)
        # img.show()
        # print(train_y_name)
        img = img.resize((int(WIDTH - 3), int(HEIGHT - 3)))  # 改变图片大小 -> (224, 224)
        img_array = np.array(img)
        # img_array, 三个通道数相同， 没法做交叉熵， 所以下面要进行”图像分层“

        # 生成标签， 标签的shape是（224， 224， class_numbers) = (224, 224, 2), 里面的值全是0
        labels = np.zeros((int(HEIGHT - 3), int(WIDTH - 3), CLASS_NUMBERS))

        # 下面将(224,224,3) => (224,224,2),不仅是通道数的变化，还有，
        # 原本背景和裂缝在一个通道里面，现在将斑马线和背景放在不同的通道里面。
        # 如，labels,第0通道放背景，是背景的位置，显示为1，其余位置显示为0
        # labels, 第1通道放斑马线，图上斑马线的位置，显示1，其余位置显示为0
        # 相当于合并的图层分层！！！！

        for cn in range(CLASS_NUMBERS):  # range(0, 2) -> 0, 1
            # 标签数值中， 裂缝的值为1， 其他为零
            labels[:, :, cn] = (img_array == cn).astype(int)

        labels = np.reshape(labels, (-1, CLASS_NUMBERS))
        y_train.append(labels)

        # 遍历所有数据，记录现在所处的行， 读取完所有数据后，read_line=0,打乱重新开始
        read_line = (read_line + 1) % numbers

        yield np.array(x_train), np.array(y_train)


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training_acc')
    plt.plot(epochs, val_acc, 'b', label='Validation_acc')
    plt.title('Training and Validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training_loss')
    plt.plot(epochs, val_loss, 'b', label='Validation_loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()


def main():
    # 过去已建立的的模型， 并加载官方与训练，模型编译
    model = get_model()

    # 获取样本（训练集&验证集）和标签的对应关系，train_num, val_num
    lines, train_nums, val_nums = get_data()

    # 设置回调函数， 并返回保存的路径
    callbacks, logdir = set_callbacks()

    # 生成样本和标签
    generate_arrays_from_file(lines, batch_size=4)

    # 训练
    history = model.fit_generator(generate_arrays_from_file(lines[:train_nums], batch_size),
                                  steps_per_epoch=max(1, train_nums // batch_size),
                                  epochs=50, callbacks=callbacks,
                                  validation_data=generate_arrays_from_file(lines[train_nums:], batch_size),
                                  validation_steps=max(1, val_nums // batch_size),
                                  initial_epoch=0)
    save_weight_path = os.path.join(logdir, 'last.h5')

    model.save_weights(save_weight_path)

    plot_history(history)


if __name__ == '__main__':
    main()
