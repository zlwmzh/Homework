#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/25 13:51
# @Author : Micky
# @Desc : 视频种的目标检测
# @File : ObjectTracking.py
# @Software: PyCharm

import cv2 as cv
import numpy as np


# 从摄像机获取视频
# 创建一个基于摄像头的视频读取流，给定基于第一个视频设备
capture = cv.VideoCapture(0)

# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('images/results/output.avi',fourcc, 20.0, (880,480))
# 设置摄像头相关参数（但是实际参数会进行稍微的偏移）
success = capture.set(cv.CAP_PROP_FRAME_WIDTH, 880)
if success:
    print("设置宽度成功")
success = capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
if success:
    print("设置高度成功")
# 打印属性
size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print(size)
# 遍历获取视频中的图像
# 读取当前时刻的摄像头捕获的图像, 返回为值：True/False, Image/None
success, frame = capture.read()
# 遍历以及等待任意键盘输入
idx = 0
sift = cv.xfeatures2d_SIFT.create()
bf = cv.BFMatcher()
img1 = cv.imread('images/detect/small.jpg')
kp1, dst1 = sift.detectAndCompute(img1, None)
MIN_MATCHES_NUMS = 10
while success and cv.waitKey(1) == -1:
    img = frame
    kp2, dst2 = sift.detectAndCompute(img, None)
    # 寻找匹配点
    matches = bf.knnMatch(dst1, dst2, k = 2)
    # 应用ratio检测，如果两个最相邻之间的距离之差足够大，那么就确认为是一个好的匹配点
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append([m])
    if len(good) > MIN_MATCHES_NUMS:
        point1 = []
        point2 = []
        for each in good:
            point1.append(kp1[each[0].queryIdx].pt)
            point2.append(kp2[each[0].trainIdx].pt)

        point1 = np.array(point1)
        point2 = np.array(point2)
        print(np.shape(point1))
        print(np.shape(point2))
        # 获取单映射矩阵
        H, mask = cv.findHomography(point1, point2, cv.RANSAC)
        # 获取小图的顶点
        xmin = 0
        ymin = 0
        xmax = img1.shape[1]
        ymax = img1.shape[0]
        a = np.array([[xmin, ymin], [xmax, ymax], [xmax, ymin], [xmin, ymax]], dtype='float32')
        points = np.array([a])
        pointsOut = cv.perspectiveTransform(points, H)
        pointsOut = np.reshape(pointsOut, (-1, 2))[:2]
        # 在图二中绘制边框
        img = cv.rectangle(img, pt1=(pointsOut[0][0], pointsOut[0][1]),
                             pt2=(pointsOut[1][0], pointsOut[1][1]), color=[0, 255, 0])
    else:
        print('未发现匹配点')
    cv.imshow('frame', img)

    # 读取下一帧的图像
    success, frame = capture.read()
    idx +=1

# 释放资源
capture.release()
cv.destroyAllWindows()