# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2021/2/19 19:37
# @Author : Ye
# @File : models.py
# @SoftWare : Pycharm

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

class_number = 2  # 分类数

print('python版本为：' + sys.version)
print('tensorflow版本为：' + tf.__version__)


def encoder(input_height, input_width):
    """
    语义分割的第一部分，特征提取，主要用到VGG网络， 函数式API

    :param input_height: 输入图像的长
    :param input_width: 输入图像的宽
    :return: 返回： 输入图像， 提取到的5个特征
    """

    # 输入
    img_input = Input(shape=(input_height, input_width, 3))

    # 三行为一个结构单元，size减半
    # 227,227,3 -> 226, 226, 64
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((4, 4), strides=(1, 1))(x)
    f1 = x  # 暂存提取的特征

    # 224,224,64 -> 112, 112,128
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f2 = x

    # 112, 112, 128 -> 56, 56, 256
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f3 = x

    # 56, 56, 256 -> 28, 28, 512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f4 = x

    # 28, 28, 512 -> 14, 14, 512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))
    f5 = x

    return img_input, [f1, f2, f3, f4, f5]


def decoder(feature_map_list, class_number, input_height=227, input_width=227, encoder_level=3):
    """
    语义分割的后半部分，上采样，将图片放大

    :param feature_map_list: 特征图（多个），由encoder得到
    :param class_number: 分类数
    :param input_height: 输入图像长
    :param input_width: 输入图像宽
    :param encoder_level: 利用的特征图，这里利用f4
    :return: output ， 返回放大的特征图 （224, 224, 2）
    """

    # 获取一个特征图， 特征图来源encoder里面的f1, f2, f3, f4, f5;这里获取到f4
    feature_map = feature_map_list[encoder_level]

    # 解码过程， 以下（28, 28, 512） -> (224, 224, 2)

    # f4.shape=(28, 28, 512) -> (28, 28, 512)
    x = ZeroPadding2D((1, 1))(feature_map)
    x = Conv2D(512, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样， 图像长宽扩大2倍， (28, 28, 512) -> (56, 56, 256)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样， 图像长宽扩大2倍， (56, 56, 256) -> (112, 112, 128)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样， 图像长宽扩大2倍， (112, 112, 128) -> (224, 224, 64)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 再进行一次卷积， 将通道数变为2（要分类的数目） (224, 224, 64) -> (224, 224, 2)
    x = Conv2D(class_number, (3, 3), padding='same')(x)
    # reshape：(224, 224, 2) -> (224*224, 2)
    x = Reshape((int(input_height - 3) * int(input_width - 3), -1))(x)

    # 求概率
    output = Softmax()(x)

    return output


def main(Height=227, Width=227):
    """
    model 的主程序， 语义分割， 分两部分， 第一部分特征提取， 第二部分放大图片

    :param Height: 图像高
    :param Width:  图像宽
    :return: model
    """

    # 第一部分 编码， 提取特征， 图像size减小， 通道增加
    img_input, feature_map_list = encoder(input_height=Height, input_width=Width)

    # 第二部分 解码， 将图像上采样， size放大， 通道减小
    output = decoder(feature_map_list, class_number=class_number, input_height=Height, input_width=Width, encoder_level=3)

    # 构建模型
    model = Model(img_input, output)

    model.summary()

    return model


if __name__ == '__main__':
    main(Height=227, Width=227)
