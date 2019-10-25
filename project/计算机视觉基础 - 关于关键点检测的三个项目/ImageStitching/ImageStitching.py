#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/25 13:17
# @Author : Micky
# @Desc : 图像拼接
# @File : ImageStitching.py
# @Software: PyCharm

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

def merge_images(img1, img2):
    # 初始化SIFT检测子
    sift = cv.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)

    kp2, des2 = sift.detectAndCompute(img2, None)

    # 检测关键点:
    img1_sift_keypoints = img1.copy()
    img2_sift_keypoints = img2.copy()
    img1_sift_keypoints = cv.drawKeypoints(img1, kp1, img1_sift_keypoints,
                                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_sift_keypoints = cv.drawKeypoints(img2, kp2, img2_sift_keypoints,
                                           flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # BFMatcher 使用默认参数进行匹配

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用ratio检测，如果两个最相邻之间的距离之差足够大，那么就确认为是一个好的匹配点
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])

    point1 = []
    point2 = []
    for each in good:
        point1.append(kp1[each[0].queryIdx].pt)
        point2.append(kp2[each[0].trainIdx].pt)

    point1 = np.array(point1)
    point2 = np.array(point2)

    # 判断哪张图片在左边，如果位置相反，那么就交换两者
    num1 = 0
    num2 = 0
    flag = img1.shape[1] / 2
    for each in point1:
        if each[0] > flag:
            num1 += 1
        else:
            num2 += 1
    if num1 < num2:
        temp = img1
        img1 = img2
        img2 = temp
        temp = point1
        point1 = point2
        point2 = temp
    else:
        pass

    # 使用findHomography函数来求单应举证
    H, mask = cv.findHomography(point2, point1, cv.RANSAC)

    # 计算最终拼接图片的大小
    img2_leftup = [0, 0, 1]
    img2_leftdown = [0, img2.shape[0] - 1, 1]
    img2_rightup = [img2.shape[1] - 1, 0, 1]
    img2_rightdown = [img2.shape[1] - 1, img2.shape[0] - 1, 1]
    x1 = np.dot(img2_leftup, H[0])
    x2 = np.dot(img2_leftdown, H[0])
    x3 = np.dot(img2_rightup, H[0])
    x4 = np.dot(img2_rightdown, H[0])
    y1 = np.dot(img2_leftup, H[1])
    y2 = np.dot(img2_leftdown, H[1])
    y3 = np.dot(img2_rightup, H[1])
    y4 = np.dot(img2_rightdown, H[1])

    # 选择最终输出图片的尺寸
    y_out = int(max(y2, y4, img1.shape[0] - 1))
    x_out = int(max(x3, x4))

    # 对右边的图片进行变换，得到变换后的图像
    img_out = cv.warpPerspective(img2, H, (x_out, y_out))

    # 将变换后的图片和左边的图片拼接
    for i in range(img_out.shape[0]):
        x_temple = x1 + (x2 - x1) / (y2 - y1) * (i - y1)
        for j in range(img_out.shape[1]):
            if j < x_temple:
                if i < img1.shape[0] - 1 and j < img1.shape[1] - 1:
                    img_out[i, j] = img1[i, j]
                else:
                    img_out[i, j] = img_out[i, j]
            elif j > img1.shape[1] - 1:
                img_out[i, j] = img_out[i, j]
            else:
                if i < img1.shape[0] - 1:
                    img_out[i, j] = (img1.shape[1] - 1 - j) / (img1.shape[1] - 1 - x_temple) * img1[i, j] + (
                                                                                                            j - x_temple) / (
                                                                                                            img1.shape[
                                                                                                                1] - 1 - x_temple) * \
                                                                                                            img_out[
                                                                                                                i, j]
                else:
                    img_out[i, j] = img_out[i, j]

    return img_out

if __name__ == '__main__':
    # 读入图像
    img1 = cv.imread('./images/Hill/1Hill.jpg')
    img2 = cv.imread('./images/Hill/2Hill.jpg')
    img3 = cv.imread('./images/Hill/3Hill.jpg')
    # img1 = cv.imread('./images/bg/3-left.jpg')
    # img2 = cv.imread('./images/bg/3-right.jpg')

    # 调用函数，输出图片
    img = merge_images(merge_images(img1, img2),img3)
    cv.imwrite('./images/results/result.png', img)


    # plt.subplot(3, 3, 1)
    # plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    # plt.subplot(3, 3, 2)
    # plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    # plt.subplot(3, 3, 3)
    # plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    # plt.subplot(3, 1, 3)
    # plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))

    i1 = plt.subplot(2, 3, 1)
    i1.set_title('Image 1')
    plt.xticks([]), plt.yticks([])
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    i2 = plt.subplot(2, 3, 2)
    i2.set_title('Image 2')
    plt.xticks([]), plt.yticks([])
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    i3 = plt.subplot(2, 3, 3)
    i3.set_title('Image 3')
    plt.xticks([]), plt.yticks([])
    plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))

    i4 = plt.subplot(2, 1, 2)
    i4.set_title('Image Stitching')
    plt.xticks([]), plt.yticks([])
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
