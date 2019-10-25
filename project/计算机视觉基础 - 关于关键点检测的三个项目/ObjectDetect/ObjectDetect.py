#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/25 13:34
# @Author : Micky
# @Desc : 目标检测定位
# @File : ObjectDetect.py
# @Software: PyCharm

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def siftObjectDetect(img1, imgAll):
    """
    基于SIFT的目标检测位置
    :param img1: 被检测的图像
    :param img2: 检测的图像
    :return:
    """
    sift = cv.xfeatures2d_SIFT.create()
    kp1, dst1 = sift.detectAndCompute(img1, None)
    kp2, dst2 = sift.detectAndCompute(imgAll, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(dst1, dst2, k = 2)
    # 应用ratio检测，如果两个最相邻之间的距离之差足够大，那么就确认为是一个好的匹配点
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    drawmatchs = cv.drawMatchesKnn(img1, kp1, imgAll, kp2, good, None, flags= 2)
    # cv.imwrite('images/result.png',drawmatchs)

    point1 = []
    point2 = []
    for each in good:
        point1.append(kp1[each[0].queryIdx].pt)
        point2.append(kp2[each[0].trainIdx].pt)

    point1 = np.array(point1)
    point2 = np.array(point2)
    # 获取单映射矩阵
    H, mask = cv.findHomography(point1, point2, cv.RANSAC)

    # 获取小图的顶点
    xmin = 0
    ymin = 0
    xmax = img1.shape[1]
    ymax = img1.shape[0]
    print(xmin, ymin, xmax, ymax)
    a = np.array([[xmin, ymin],[xmax, ymax], [xmax, ymin], [xmin, ymax]], dtype='float32')
    points = np.array([a])
    pointsOut = cv.perspectiveTransform(points, H)
    pointsOut = np.reshape(pointsOut,(-1,2))[:2]
    # 在图二中绘制边框
    image = cv.rectangle(imgAll, pt1=(pointsOut[0][0], pointsOut[0][1]),
                         pt2=(pointsOut[1][0], pointsOut[1][1]), color=[0, 255, 0], thickness=5)
    return image

    # 显示
    # plt.subplot(1,2,1)
    # plt.imshow(cv.cvtColor(img1,cv.COLOR_BGR2RGB))
    # plt.xticks([]), plt.yticks([])
    #
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
    # plt.xticks([]), plt.yticks([])
    # plt.show()

img1 = cv.imread('images/detect/small.JPG')
img2 = cv.imread('images/detect/bigbg.JPG')
img2C = img2.copy()
image = siftObjectDetect(img1, img2)
cv.imwrite('images/results/result.png',image)

i1 = plt.subplot(2, 2, 1)
i1.set_title('Small Image')
plt.xticks([]), plt.yticks([])
plt.imshow(cv.cvtColor(img1,cv.COLOR_BGR2RGB))

i2 = plt.subplot(2, 2, 2)
i2.set_title('Big Bg Image')
plt.xticks([]), plt.yticks([])
plt.imshow(cv.cvtColor(img2C,cv.COLOR_BGR2RGB))

i3 = plt.subplot(2, 1, 2)
i3.set_title('Object Detect')
plt.xticks([]), plt.yticks([])
plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
plt.show()