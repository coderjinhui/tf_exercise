# coding: utf-8
'''
激活函数
用来处理一些非线性的东西
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-7, 7, 180)  # 生成-7~7的等间距180个点


# 激活函数的原始实现(直接套用数学公式定义)
def sigmoid(input):
    '''
    sigmoid: S型函数
    形状如y = 1/e^x
  '''
    y = [1 / float(1 + np.exp(-x)) for x in input]
    return y


def relu(input):
    '''
    当x<=0时 y=0,
    当x>0时 y=x的运算
  '''
    y = [x * (x > 0) for x in input]
    return y


def tanh(input):
    '''tan 函数'''
    y = []
    for x in input:
        y1 = np.exp(x) - np.exp(-x)
        y2 = float(np.exp(x) + np.exp(-x))
        y.append(y1 / y2)
    return y


def softPlus(input):
    y = [np.log(1 + np.exp(x)) for x in input]
    return y


# 使用tf定义
tf_sigmoid = tf.nn.sigmoid(x)
tf_relu = tf.nn.relu(x)
tf_tanh = tf.nn.tanh(x)
tf_softPlus = tf.nn.softplus(x)
# 运算每个的结果
sess = tf.Session()
y_sigmoid, y_relu, y_tanh, y_softPlus = sess.run(
    [tf_sigmoid, tf_relu, tf_tanh, tf_softPlus])
sess.close()

# 绘制图像
plt.subplot(221)
plt.plot(x, y_sigmoid, label="sigmoid")

plt.subplot(222)
plt.plot(x, y_relu, label="relu")

plt.subplot(223)
plt.plot(x, y_tanh, label="tanh")

plt.subplot(224)
plt.plot(x, y_softPlus, label="softPlus")

plt.show()