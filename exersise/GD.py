# coding: UTF-8
'''
使用tf的梯度下降优化方法快速解决线性回归
'''

import numpy as np
import matplotlib.pyplot as mpl
import tensorflow as tf

# 先定义一组数据(x, y)，他们近似符合 y = 6 * x + 5
# 数据量为100组
point_num = 100
points = []
for i in range(point_num):
    x1 = np.random.normal(0.0, 0.5)
    y1 = 6 * x1 + 5 + np.random.normal(0.0, 1)
    points.append([x1, y1])

# 将x y 坐标统一都拿出来
x_data = [p[0] for p in points]
y_data = [p[1] for p in points]

# 构建一个线性方程(相当于定义模型)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 初始化weight y=kx+b中的k
b = tf.Variable(tf.zeros([1]))  # 初始化b
y_test = W * x_data + b
# 定义损失函数：使用方差作为衡量标准 (y_test - y_data)^2/N
loss = tf.reduce_mean(tf.square(y_test - y_data))
# 使用梯度下降函数
optimizer = tf.train.GradientDescentOptimizer(0.1)  #学习速率(就是每次x递增的大小)
# 组合梯度下降和损失函数，将方差进行比较，寻找最小方差
train = optimizer.minimize(loss)
# 创建session
with tf.Session() as session:

    # 初始化所有变量
    init = tf.global_variables_initializer()
    # 运行 "图" 将初始化真正运行起来
    session.run(init)

    for step in range(100):
        session.run(train)
        print("step: %d, 损失量: %f, [w: %f, b: %f]") % (
            step, session.run(loss), session.run(W), session.run(b))

    # 将原数据的点和新生成的线绘制上去
    mpl.plot(x_data, y_data, 'g*', label="original data")
    mpl.plot(
        x_data, session.run(W) * x_data + session.run(b), label="trained line")
    mpl.xlabel("x")
    mpl.ylabel("y")
    mpl.title = "xian xing hui gui"
    mpl.legend()
    mpl.show()
