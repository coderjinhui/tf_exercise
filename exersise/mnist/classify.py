# coding: utf-8
'''
使用mnist训练好的模型进行模型预测
'''
import tensorflow as tf
import numpy as np
import image_util as image_util
import mnist_model as model
import cv2
import sys
import os

ImageToMatrix = image_util.ImageToMatrix
process_image = image_util.process_image

# 导入模型训练时必须的变量
input_x, output, _ = model.conv()

sess = tf.Session()
# 读取训练好的模型
saver = tf.train.Saver()
saver.restore(sess, "./mnist_model/test_model")


def classify(file_path):
    '''
    数字识别，传入手写数字图片文件路径
    返回值: 对应的数字
    '''
    img_data = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # 降低噪声，提高识别率
    img_data = drop_noise(img_data)
    # 调试 查看过滤器图像效果
    # cv2.imshow('image', img_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_data = process_image(img_data)
    input_data_op = tf.reshape(img_data, [-1, 28 * 28])
    input_data = sess.run(input_data_op)
    input_data = input_data / 255.0
    my_output = sess.run(output, {input_x: input_data})
    print(sess.run(output, {input_x: input_data}).flatten().tolist())
    num_choose = tf.argmax(my_output, axis=1)
    num = sess.run(num_choose)
    return num


def drop_noise(img_data):
    '''
    降噪
    使用高斯降噪
    '''
    # canny 边缘检测
    dst = cv2.Canny(img_data, 50, 50)
    # Sobel算子边缘检测(强化边缘)
    x = cv2.Sobel(dst, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(dst, cv2.CV_16S, 0, 1)
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    img = cv2.addWeighted(absx, 0.4, absy, 0.6, 0)
    # 双边滤波 滤波同时保留边缘信息
    img_median = cv2.bilateralFilter(img, 9, 75, 75)
    # 中值滤波 图像消除胡椒盐噪声, 图像变得平滑
    img_median = cv2.medianBlur(img_median, 5)
    # 由于图像处理是黑底白字，进行黑白转化
    # for i in range(img_median.shape[0]):
    #     for j in range(img_median.shape[1]):
    #         img_median[i, j] = 255 - img_median[i, j]
    return img_median


# 读取命令行参数
img_path = sys.argv[1]
if img_path.strip():
    # 如果给的参数不为空就执行图像检查
    if os.path.exists(img_path):
        num = classify(img_path)
        print("经过预测，该图像中写的数字最可能是: {:s}".format(num))
    else:
        print(img_path + u"文件不存在")

sess.close()
