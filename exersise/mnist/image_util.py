# coding: utf-8
from PIL import Image
import numpy as np
import cv2


# 将图片转化成矩阵
def ImageToMatrix(filename):
    im = Image.open(filename)
    # 显示图片
    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data, dtype='float')
    new_data = np.reshape(data, (height, width))
    return new_data


# 矩阵转化成图片
def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# 压缩图片到28*28
def process_image(image, mwidth=28, mheight=28):
    '''
      cv2.IMREAD_COLOR:读取一副彩色图片，图片的透明度会被忽略，默认为该值，实际取值为1；
      cv2.IMREAD_GRAYSCALE:以灰度模式读取一张图片，实际取值为0
      cv2.IMREAD_UNCHANGED:加载一副彩色图像，透明度不会被忽略。
    '''
    image = cv2.resize(image, (mwidth, mheight), interpolation=cv2.INTER_AREA)
    return image