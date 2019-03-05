# coding: utf-8

import tensorflow as tf

c1 = tf.constant([[2, 2]])
c2 = tf.constant([
[4], 
[4]])

mult = tf.matmul(c1, c2)

print(mult)

sess = tf.Session()
res = sess.run(mult)
print(res)
sess.close()

with tf.Session() as sess:
  res2 = sess.run(mult)
  print(res2)
