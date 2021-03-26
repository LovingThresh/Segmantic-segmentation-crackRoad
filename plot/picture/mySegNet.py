#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/6/26 15:54
# @Author : v
# @File : mySegNet.py
# @Software: PyCharm
import sys

sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # 输入
    to_input(r'I:\PycharmProjects\Segmantic-segmentation-crackRoad\dataset\img_png\1.png', width=12, height=64),
    # to_input('./picture/xzlabel_0_2_json_img.png', width=12, height=12),
    #  """编码部分"""
    # 第一个单元
    # to_Conv("conv1", 256, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),

    to_Conv("conv1", 227, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),
    to_Conv("conv2", 227, 64, offset="(0,0,0)", to="(conv1-east)", height=64, depth=64, width=2),
    to_Pool("pool1", offset="(0,0,0)", to="(conv2-east)", width=2, height=60, depth=60),

    # 第二个单元
    to_Conv("conv3", 224, 128, offset="(2,0,0)", to="(pool1-east)", height=60, depth=60, width=4),
    to_Conv("conv4", 224, 128, offset="(0,0,0)", to="(conv3-east)", height=60, depth=60, width=4),
    to_Pool("pool2", offset="(0,0,0)", to="(conv4-east)", width=4, height=30, depth=30),

    # 第三个单元
    to_Conv("conv5", 112, 256, offset="(2,0,0)", to="(pool2-east)", height=30, depth=30, width=8),
    to_Conv("conv6", 112, 256, offset="(0,0,0)", to="(conv5-east)", height=30, depth=30, width=8),
    to_Pool("pool3", offset="(0,0,0)", to="(conv6-east)", width=8, height=15, depth=15),

    # 第四个单元
    to_Conv("conv7", 56, 512, offset="(2,0,0)", to="(pool3-east)", height=8, depth=8, width=16),
    to_Conv("conv8", 56, 512, offset="(0,0,0)", to="(conv7-east)", height=8, depth=8, width=16),
    to_Pool("pool4", offset="(0,0,0)", to="(conv8-east)", width=16, height=4, depth=4),

    # 第五个单元
    to_Conv("conv9", 28, 512, offset="(1,0,0)", to="(pool4-east)", height=4, depth=4, width=16),
    to_Conv("conv10", 28, 512, offset="(0,0,0)", to="(conv9-east)", height=4, depth=4, width=16),
    to_Pool("pool5", offset="(0,0,0)", to="conv10-east", width=16, height=2, depth=2),


    # """解码部分"""
    to_Conv("conv11", 14, 512, offset="(1,0,0)", to="(pool5-east)", width=16, height=2, depth=2),

    # 链接
    to_connection("pool5", "conv11"),

    # 第一个单元
    to_UnPool("unpool1", offset="(1,0,0)", to="(conv11-east)", height=4, depth=4, width=16),
    to_Conv("conv12", 28, 512, offset="(0,0,0)", to="(unpool1-east)", height=4, depth=4, width=16),

    # 第二个单元
    to_UnPool("unpool2", offset="(2,0,0)", to="(conv12-east)", height=8, depth=8, width=16),
    to_Conv("conv13", 56, 256, offset="(0,0,0)", to="(unpool2-east)", height=8, depth=8, width=16),

    # 第三个单元
    to_UnPool("unpool3", offset="(2,0,0)", to="(conv13-east)", height=15, depth=15, width=8),
    to_Conv("conv14", 112, 128, offset="(0,0,0)", to="(unpool3-east)", height=15, depth=15, width=8),

    # 第四个单元
    to_UnPool("unpool4", offset="(2,0,0)", to="(conv14-east)", height=30, depth=30, width=4),
    to_Conv("conv15", 224, 64, offset="(0,0,0)", to="(conv14-east)", height=30, depth=30, width=4),

    # 第五个单元
    to_UnPool("unpool5", offset="(2,0,0)", to="(conv15-east)", height=60, depth=60, width=2),
    to_Conv("conv16", 224, 2, offset="(0,0,0)", to="(unpool5I:\PycharmProjects\plotneuralnet\layers)", height=60, depth=60, width=1),

    # to_Conv("conv2", 128, 64, offset="(1_raster,0,0)", to="(pool1-east)", height=32, depth=32, width=2),
    # to_connection("pool1", "conv2"),
    # to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1_raster),
    to_SoftMax("soft1", 224, "(2,0,0)", "(conv16-east)", caption="SOFT", width=1, height=60, depth=60),
    # to_connection("pool2", "soft1"),
    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
