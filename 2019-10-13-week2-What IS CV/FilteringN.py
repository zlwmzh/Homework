#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/16 14:25
# @Author : Micky
# @Desc : week2 作业
# @File : FilteringN.py
# @Software: PyCharm

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def mediaBlue(image, kernel, stride = 1):
    """
    平滑中值滤波
    :param image: 原始图像
    :param kernel: 卷积核大小
    :param stride: 步长
    :return: 处理后的图像
    """

    def smoothMatrixPaddMedia(matrix, kernel, stride=1):
        """
        滑动并填充卷积核区域的中值
        :param matrix: 像素矩阵
        :param kernel: 卷积核大小
        :param stride: 步长
        :return:
        """
        matrix = np.asarray(matrix)
        row, column = matrix.shape

        l_matrix = matrix.copy()
        for r in range(0, row - kernel + 1, stride):
            for c in range(0, column - kernel + 1, stride):
                # 获取卷积核区域数据
                m = matrix[r:r + kernel, c: c + kernel]
                # 对区域进行排序
                m = np.sort(np.asarray(m).flatten())
                # 获取中值
                middel = m[len(m) // 2]
                # 修改所有的值维中值
                m[:] = middel
                # 修改原矩阵中的值
                l_matrix[r:r + kernel, c: c + kernel] = np.reshape(m, (kernel, kernel))
        return l_matrix

    # M = [[2, 6, 8, 7],
    #      [1, 5, 3, 4],
    #      [9, 10, 11, 12]]
    # smoothMatrixPaddMedia(M, kernel, 1)
    # return
    # 获取宽，高，通道
    if len(image.shape) == 2:
        # 灰度图
        return smoothMatrixPaddMedia(image, kernel, stride)
    if len(image.shape) == 3:
        # 获取B,G,R通道
        B, G, R = cv.split(image)
        # 对每个通道进行中值滤波操作
        B = smoothMatrixPaddMedia(B, kernel, stride)
        G = smoothMatrixPaddMedia(G, kernel, stride)
        R = smoothMatrixPaddMedia(R, kernel, stride)
        return cv.merge((B,G,R))

def buildSoltImage(image, soltNums):
    """
    产生椒盐噪音的图片
    :param image: 原始图片
    :param soltNums: 椒盐噪音的个数
    :return:
    """

    def solt(matrix, soltNums):
        """
        产生椒盐噪声
        :param matrix:  像素矩阵
        :param soltNums: 噪声点个数
        :return:
        """
        for s in range(soltNums):
            x = np.random.randint(0, matrix.shape[0])
            y = np.random.randint(0, matrix.shape[1])
            matrix[x][y] = 255
        return matrix


    if len(image.shape) == 2:
        # 灰度图
        return solt(image, soltNums)
    if len(image.shape) == 3:
        # 得到各通道
        B, G, R = cv.split(image)
        # 各通道产生椒盐点
        B = solt(B, soltNums)
        G = solt(G, soltNums)
        R = solt(R, soltNums)
    return cv.merge((B,G,R))


if __name__ == '__main__':
    # 图片路径
    IMAGE_PATH = 'images/koala.png'
    # 读取图片
    image = cv.imread(IMAGE_PATH)



    # 显示
    SHOW_R = 1
    SHOW_C = 4
    # 原图
    normalImg = plt.subplot(SHOW_R, SHOW_C, 1)
    normalImg.set_title('ORIGINAL IMAGE')
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    # 椒盐图片
    soltImage = plt.subplot(SHOW_R, SHOW_C, 2)
    soltImage.set_title('SOLT IMAGE')
    soltImageP = buildSoltImage(image, 10000)
    plt.imshow(cv.cvtColor(soltImageP, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])

    # 自己定义的平滑中值滤波
    mineMediaBlue = plt.subplot(SHOW_R, SHOW_C, 3)
    mineMediaBlue.set_title('MINE MEDIABLUE')
    mineMeidaBlueImage = mediaBlue(soltImageP, 3, 1)
    plt.imshow(cv.cvtColor(mineMeidaBlueImage, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])

    # 调用openCV自带的平滑中值滤波
    openCVMediaBlue = plt.subplot(SHOW_R, SHOW_C, 4)
    openCVMediaBlue.set_title('OPENCV MEDIABLUE')
    openCVMediaBlueImage = cv.medianBlur(soltImageP, 3)
    plt.imshow(cv.cvtColor(openCVMediaBlueImage, cv.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])


    plt.show()