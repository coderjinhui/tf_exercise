# coding: utf-8
import tensorflow as tf
import numpy as np
'''
定义卷积神经网络模型
'''


def conv():
    # 这里的None表示 张量Tensor的第一个维度可以是任意长度(对应表示是图片的张数)
    # 第一个维度: 图片的序号
    # 第二个维度: 图片28*28个像素点
    # /255: 每个值的范围是0-255，因此/255能让数据范围变成0-1
    input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255
    input_y = tf.placeholder(tf.float32, [None, 10])

    input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])
    # 构建卷积神经网络
    # 第一层卷积之后 数据变成28*28*32
    conv1 = tf.layers.conv2d(
        inputs=input_x_images,
        filters=32,  # 使用32个过滤器，数据输出深度为32
        kernel_size=[5, 5],  # 过滤器一次扫描图片5*5的范围
        strides=1,  # 设置每过多少个数据就扫描一次
        padding='same',  #当扫描器在图像边缘时需要的补零方案，same是指依然保持5*5的范围
        activation=tf.nn.relu)
    # 第一次池化 pooling(亚采样: 数据量会<输入时的量)
    # 输出 14*14*32
    pool1 = tf.layers.max_pooling2d(inputs=conv1, strides=2, pool_size=[2, 2])

    # 第二次卷积
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        strides=1,
        padding='same',
        activation=tf.nn.relu)  #14*14*64
    # 第二次池化
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, strides=2, pool_size=[2, 2])  #7*7*64

    # 平坦化 将数据变成一维
    flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # 1024个神经元的全连接层
    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
    # 丢弃50%数据
    dropout = tf.layers.dropout(inputs=dense, rate=0.5)

    # 输出数据
    output = tf.layers.dense(inputs=dropout, units=10)

    # 计算误差 使用交叉熵 然后使用Softmax表示成百分比
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=input_y, logits=output)
    return input_x, output, [input_y, loss]
