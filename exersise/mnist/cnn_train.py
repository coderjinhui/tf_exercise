# coding: utf-8
'''
使用mnist的手写数字图片训练手写数字
'''
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import mnist_model

# 训练库图片 55000*28*28*1 55000张图，每个图大小是28*28，颜色只有一种
mnist = input_data.read_data_sets('mnist_data', one_hot=True)
# one_hot=True 启用独热码模式，使用特殊编码表示
# 0: 1000000000
# 1: 0100000000 以此类推

# 选取测试数据集
test_x = mnist.test.images
test_y = mnist.test.labels

# 导入模型训练时必须的变量
input_x, output, variables = mnist_model.conv()
input_y, loss = variables

# 使用优化器，让误差变得最小
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 计算精度
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(input_y, axis=1), predictions=tf.argmax(output,
                                                             axis=1))[1]

# 初始化全局和局部变量
init = tf.group(tf.global_variables_initializer(),
                tf.local_variables_initializer())

# 保存变量
sess = tf.Session()
saver = tf.train.Saver()

summary = tf.summary.merge_all()
writer = tf.summary.FileWriter('./log/mnist2', sess.graph)
writer.add_graph(sess.graph)
sess.run(init)

# 开始训练
for i in range(200000):
    batch = mnist.train.next_batch(50)
    train_loss, train_op_ = sess.run([loss, train_op], {
        input_x: batch[0],
        input_y: batch[1]
    })
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy, {input_x: test_x, input_y: test_y})
        print("step: %d, loss: %.4f, [测试精度: %.2f]") % (i, train_loss,
                                                       test_accuracy)

path = saver.save(sess, './mnist_model2/test_model')

# 测试
test_output = sess.run(output, {input_x: test_x[:20]})
output_y = tf.argmax(test_output, axis=1)
print(sess.run(output_y), "预测的数字")
print(np.argmax(test_y[:20], axis=1), "真实数字")
sess.close()