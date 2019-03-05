# coding=utf-8
import tensorflow as tf
'''
tensorboard 简单使用
tensorboard --logdir dirPath
'''

W = tf.Variable(2.0, name="weight", dtype=tf.float64)
b = tf.Variable(1.0, name="BBB", dtype=tf.float64)
x = tf.placeholder(name="Input", dtype=tf.float64)

with tf.name_scope("Output"):
    y = W * x + b

path = "./log"
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(path, sess.graph)
    res = sess.run(y, {x: 3.0})
    print(res)
