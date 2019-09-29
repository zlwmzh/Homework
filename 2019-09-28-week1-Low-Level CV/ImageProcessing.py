#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/9/29 16:13
# @Author : XXX
# @Desc : week1 作业：image crop, color shift, rotation and perspective transform
# @File : ImageProcessing.py
# @Software: PyCharm

"""
将images文件夹下面的图片进行image crop, color shift, rotation and perspective transform等操作
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

# 图片路径
IMAGE_DIR = 'images'
# 定义图片格式列表
TYPE_LIST = ['png','jpeg','jpg']
# 定义操作列表
SPLT_TITLE = ['Original image','Crop image','Color Shift','Rotation Image','Affine Transform','Perspective Transform']


def readImage(img_dir):
    """
    读取图片
    :param img_dir: 文件夹路径
    :return: 图片集合
    """
    if not os.path.exists(img_dir):
        print('文件夹路径不存在，请输入正确的路径')
    else:
        # 定义图片集合
        images = []
        # 读取该路径下的所有文件
        fileNames = os.listdir(img_dir)
        # 遍历文件列表
        for file in fileNames:
            # 判断文件类型，如果是已经在TYPE_LIST中定义的类型，则对该文件进行读取
            name,type = file.split('.')
            if type in TYPE_LIST:
                image = cv.imread(os.path.join(IMAGE_DIR,file),cv.IMREAD_COLOR)
                # 这里将BGR通道的图片转换为RGB的图片，方便后面显示
                image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
                images.append(image)
        # print(np.shape(images))
        return images

def imageProcess(images):
    """
    对图片进行一些变化操作
    :param images: 图片列表
    """
    # 这个参数是用来调控子图显示的
    spltidx = 1
    # 总共有多少张图片
    imageCount = len(images)
    for index,image in enumerate(images):
        # Original image
        spltidx = imageSubplot(image,imageCount,spltidx)
        # image crop
        spltidx = imageSubplot(imageCrop(image), imageCount, spltidx)
        # color shift
        spltidx = imageSubplot(imageColorShift(image), imageCount, spltidx)
        # rotation
        spltidx = imageSubplot(imageRotation(image), imageCount, spltidx)
        # Affine Transform
        spltidx = imageSubplot(imageAffineTransform(image), imageCount, spltidx)
        # Perspective Transform
        spltidx = imageSubplot(imagePerspectiveTransform(image),imageCount,spltidx)
    plt.show()

def imageSubplot(image,imageCount,spltidx):
    """
    子图相关信息设置
    :param image:  显示的图像
    :param imageCount: 总的图片的个数
    :param spltidx: 已经显示的图片个数
    """
    splt = plt.subplot(imageCount, len(SPLT_TITLE), spltidx)
    splt.set_title(SPLT_TITLE[(spltidx-1)%len(SPLT_TITLE)])
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])
    spltidx += 1
    return spltidx

def imageCrop(image):
    """
    图像剪裁
    :param image: 图像
    :return: 变换后的图片
    """
    # 获取图片的高,宽
    h, w, _ = image.shape
    # 根据高度宽度产生随机区域
    n_start_h = np.random.randint(0,h)
    n_start_w = np.random.randint(0,w)
    n_end_h = np.random.randint(n_start_h,h)
    n_end_w = np.random.randint(n_start_w,w)
    # print(n_start_h,n_start_w,n_end_h,n_end_w)
    n_img = image[n_start_h:n_end_h,n_start_w:n_end_w]
    return n_img

def imageColorShift(image):
    """
    颜色变换。因为我们的图片有三个通道，rgb。所以我们首先产生3个0~1的整数，0代表颜色不变换，1代表颜色变换
    这样随机的三个数就表示我们应该对RGB那几个通道进行相应的变换。 然后我们在0-255之间产生变换的数值
    :param image: 图片数据
    :return: 变换后的图片
    """
    def channelColorChange(channel):
        """
        通道颜色变换
        :param channel:
        :return:
        """
        # 获取图片的高,宽
        h,w,_ = image.shape
        R, G, B = cv.split(image)
        changeList = np.random.randint(0, 255, size=(R.shape[0],R.shape[1])).astype(np.uint8)
        # print(np.shape(R))
        # print(np.shape(changeList))
        if channel == 0:
            # R 通道变换
            R = changeList
        elif channel == 1:
            # G通道变换
            G = changeList
        elif channel == 2:
            # B通道变换
            B = changeList
        return cv.merge((R, G, B))

    # 首先产生随机数列表，判断需要变换的通道 。index = 0 表示R通道   index = 1 表示G通道   index = 2 表示B通道
    channel_change_flags = np.random.randint(0,2,size=3)
    # 分别读取标识
    channel_r_flag = channel_change_flags[0]
    channel_g_flag = channel_change_flags[1]
    channel_b_flag = channel_change_flags[2]
    if channel_r_flag == 1:
        # 进行R通道的变换
        image = channelColorChange(0)
    if channel_g_flag == 1:
        # 进行G通道的颜色变换
        image = channelColorChange(1)
    if channel_b_flag == 1:
        # 进行B通道的颜色变换
        image = channelColorChange(0)
    return image

def imageRotation(image):
    """
    图片旋转
    :param image:  原始图片
    :return: 旋转后的图片
    """
    # 获取图片的高,宽
    h, w, _ = image.shape
    # 产生随机的旋转角度
    angle = np.random.randint(-360,360)
    # 生成旋转矩阵
    """
    center: 旋转中心，这里以图片的中心点旋转
    angle: 旋转角度
    scale: 放到或者缩小。这里只处理旋转，大小不变
    """
    M = cv.getRotationMatrix2D(center=(h/2,w/2),angle=angle,scale=1.0)
    # 生成旋转后的图像
    n_img = cv.warpAffine(image,M,(w,h))
    return n_img

def imageAffineTransform(image):
    """
    仿射变换
    :param image: 原始图像
    :return: 仿射变换后的图像
    """
    # 获取图片的高宽
    h,w,_ = image.shape
    # 随机提取原始图像上的三个point  以及他们在新图像的位置
    src_points_x = np.random.randint(0,w,size=3)
    src_points_y = np.random.randint(0,h,size=3)
    src_points = np.asarray([list(i) for i in zip(src_points_x,src_points_y)]).astype(np.float32)
    dst_points_x = np.random.randint(0, w, size=3)
    dst_points_y = np.random.randint(0, h, size=3)
    dst_points = np.asarray([list(i) for i in zip(dst_points_x, dst_points_y)]).astype(np.float32)
    # 生成仿射变换矩阵
    M = cv.getAffineTransform(src_points,dst_points)
    n_img = cv.warpAffine(image,M,(h,w))
    return n_img

def imagePerspectiveTransform(image):
    """
    投影变换(透视变换)：给定四个点进行转换操作。3维上处理
    :param image: 原始图像
    :return: 投影变换后的图像
    """
    # 获取图片的高宽
    h, w, _ = image.shape
    # 随机提取原始图像上的四个point  以及他们在新图像的位置
    src_points_x = np.random.randint(0, w, size=4)
    src_points_y = np.random.randint(0, h, size=4)
    src_points = np.asarray([list(i) for i in zip(src_points_x, src_points_y)]).astype(np.float32)
    dst_points_x = np.random.randint(0, w, size=4)
    dst_points_y = np.random.randint(0, h, size=4)
    dst_points = np.asarray([list(i) for i in zip(dst_points_x, dst_points_y)]).astype(np.float32)
    # 产生M矩阵(3*3矩阵  4个点+一个规则)
    M = cv.getPerspectiveTransform(src_points,dst_points)
    # 产生新的图像
    n_img = cv.warpPerspective(image,M,(h,w))
    return n_img

if __name__ == '__main__':
    imageProcess(readImage(IMAGE_DIR))