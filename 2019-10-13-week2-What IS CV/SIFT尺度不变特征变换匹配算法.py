#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/18 9:35
# @Author : Micky
# @Desc : SIFT算法
# @File : SIFT尺度不变特征变换匹配算法.py
# @Software: PyCharm

"""
SIFT 是在空间尺度上寻找极值点，并提取出位置、旋转、尺度不变量。
SIFT特点：
1. SIFT特征是图像的局部特征，它对旋转、尺度缩放，亮度变化保持不变性，对视角变换、衍射变化、噪声也保持一定的稳定性
2. 独特性好，信息量丰富，适用于在海量信息中进行匹配
3. 多量性，即使少量的物体也可以产生大量的SIFT特征向量

SIFT实质是在不同的尺度空间上查找关键点，计算出关键点的方向。
算法步骤：
1. 尺度空间极值检测：搜索所有尺度空间上的图像的位置，通过高斯微分函数识别出潜在对于尺度、旋转不变的兴趣点
2. 关键点定位： 在每个侯选位置上，通过拟合精细的模型来确定位置和尺度。关键点的选择依据与它们的稳定性
3. 方向确定： 基于图像局部的梯度方向，分配给每个关键点一个或多个方向。所有后面对图像数据的操作都相对于关键点方向、
尺度和位置进行变换，从而提供对于这些变换的不变性。
4. 关键点描述： 在每个关键点领域上，在选定的尺度上测量图像局部梯度，这些梯度被变换成一种表示，这种表示允许比较大的局部变形和光照变化

详细讲解：
一. 尺度空间极值检测
    1. 尺度空间理论：
       基本思想：在图像信息处理时引入尺度参数，通过变换尺度参数获得多尺度下的尺度空间表示序列，对这些序列进行尺度空间主轮阔提取，并以该主轮阔作为一种特征向量
                实现边缘、角点检测和不同分辨率上的特征提取
       尺度空间方法将传统的单尺度图像信息处理技术纳入尺度不断变化的动态分析框架中，更容易获取图像的本质特征。尺度空间中各尺度图像的模糊程度逐渐变大，能够模拟
       人在距离目标由近到远时目标在视网膜上的形成过程。
       当我们用眼睛观察物体时，一方面当物体所处背景的光照条件变化时，视网膜感知图像的亮度水平和对比度是不同的，因此要求尺度空间算子对图像的分析不受图像的灰度
       水平和对比度变化的影响，即满足灰度不变性和对比度不变性。另一方面，相对于某一固定坐标系，当观察者和物体之间的相对位置变化时，视网膜所感知的图像的位置、
       大小、角度和形状是不同的，因此要求尺度空间算子对图像的分析和图像的位置、大小、角度以及仿射变换无关，即满足平移不变性、尺度不变性、欧几里德不变性以及
       仿射不变性。
    2. 尺度空间表示：
       尺度空间定义为一个高斯卷积核和原图的卷积：
       L(x,y,σ) = G(x,y,σ) * I(x,y)    L(x,y,σ) 表示图像的尺度空间   G(x,y,σ) 表示高斯函数   I(x,y) 表示原图像
       (x,y) 表示图像像素位置 σ表示尺度空间因子，σ 越小表示图像被平滑的越少，图像越清晰，尺度越小。大尺度对应于图像的概貌特征，小尺度对应于图像的细节特征。
    3. 高斯金字塔的搭建
       尺度空间的实现用高斯金字塔表示。金子塔的构建分为两部分：
       a. 对图像做不同尺度的高斯模糊（这里的尺度表示尺度空间因子不同，模糊程度不同）
       b. 对图像进行下采样，改变图像的大小
       图像的金字塔模型是指，将原始图像不断下采样，得到一系列大小不一的图像，由大到小，从下到上构成的塔状模型。原图像为金子塔的第一层，
       每次下采样所得到的新图像为金字塔的一层(每层一张图像)，每个金字塔共n层。金字塔的层数根据图像的原始大小和塔顶图像的大小共同决定。
       其计算公式如下：
       n = log2{min(M,N) - t}, t∈[0,log2{min(M,N)}]   M,N表示图像宽高,t为塔顶图像的最小维数的对数值

       为了让尺度体现其连续性，高斯金字塔在简单下采样的基础上加上了高斯滤波。将图像金字塔每层的一张图像使用不同参数做高斯模糊，使得金字塔的
       每层含有多张高斯模糊图像，将金字塔每层多张图像合称为一组(Octave)，金字塔每层只有一组图像，组数和金字塔层数相等，每组含有多张(也叫层Interval)图像。
       另外，下采样时，高斯金字塔上一组图像的初始图像(底层图像)是由前一组图像的倒数第三张图像隔点采样得到的。
    4. 高斯差分金字塔(DOG金字塔)：
       在实际计算时，使用高斯金字塔每组中相邻上下两层图像相减，得到高斯差分图像，进行极值检测。
    5. 空间极值点检测（关键点的初步探查）
       关键点是由DOG空间的局部极值点组成的，关键点的初步探查是通过同一组内各DoG相邻两层图像之间比较完成的。为了寻找DoG函数的极值点，每一个像素点
       要和它所有的相邻点比较，看其是否比它的图像域和尺度域的相邻点大或者小。中间的检测点和它同尺度的8个相邻点和上下相邻尺度对应的9×2个点共26个点比较，
       以确保在尺度空间和二维图像空间都检测到极值点。
       由于要在相邻尺度进行比较，如果每组含4层的高斯差分金子塔，只能在中间两层中进行两个尺度的极值点检测，其它尺度则只能在不同组中进行。为了在每组中检测
       S个尺度的极值点，则DOG金字塔每组需S+2层图像，而DOG金字塔由高斯金字塔相邻两层相减得到，则高斯金字塔每组需S+3层图像，实际计算时S在3到5之间。
    6. 构建尺度空间需确定的参数
        σ表示尺度空间因子
        O表示组数(octave)
        S表示组内层数
        σ(o,o) = σ0 2 **(o + s / S) , o ∈[0, O-1], s∈[0,S+2]
"""

import cv2 as cv
import numpy as np

def convole(filter, matrix, padding, stride = 1):
    """
    卷积操作
    :param filter: 卷积核
    :param matrix: 被卷积的矩阵
    :param padding: 填充大小
    :param stride: 步长
    :return: 卷积后的结果
    """
    # 获取卷积核的shape
    filter_shape = filter.shape
    # 获取矩阵的shape
    matrix_shape = matrix.shape
    # 这里只处理二维卷积
    if len(filter_shape) != 2:
        raise Exception('只能处理二维卷积核')
    if len(matrix_shape) == 3:
        # 三维矩阵的处理
        for c in range(matrix_shape[-1]):
            # 按通道遍历填充
            pass
            # todo


def GuassianKernel(sigma, dim):
    """
    生成高斯卷积核
    :param sigma: 标准差，尺度因子
    :param dim: 维度
    :return: 高斯卷积核
    """
    temp = [t - (dim / 2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2 * sigma * sigma
    #  assistant、assistant.T表示x，y的值矩阵
    result = (1.0 / (temp * np.pi)) * np.exp(-(assistant ** 2 + (assistant.T) ** 2) / temp)
    return result

def downSample(image, step = 2):
    """
    对图像进行下采样操作，这里采用隔点采样操作
    :param image: 原始图像
    :param step: 隔点间隔
    :return: 下采样后的图像
    """
    return image[::step,::step,:]

def buildDog(image, n, sigma0, S = None, O = None):
    """
    生成高斯金字塔和Dog金字塔
    :param image: 原始图像
    :param n: 每一组需要提取多少个尺度的关键点
    :param sigma0: 第一组第一层尺度因子
    :param S: 每一组有多少层不同尺度的图像
    :param O: 总共有多少组
    :return:
    """
    # 因为我们需要每一组提取n个尺度的关键点，所有每一组需要S+3层不同尺度的图像。
    # 因为关键点是由DOG空间上的局部极值点提供，由于要在相邻尺度上比较，所以我们不能取到Dog每一组的第一层和最后一层，所以GOG每一组
    # 的层数至少为S+2
    if S == None:
        S = n + 3
    # 金字塔的组数和塔顶的图像宽高有关
    # 塔顶图像的宽高为3
    t = 3
    if O == None:
       O = int(np.log2(min(image.shape[0],image.shape[1]))) - t
    print('高斯金字塔共{}组，每组{}层'.format(O,S))
    # σ(o,s) = σ0 * 2 ** ((o+s)/n)  o为所在的组，s为所在的层，σ0为初始的尺度，S为每组的层数
    # 所以相邻尺度之间相差的比例因子k = σs+1 - σs = σ0 * 2 ** ((o+s+1)/S) - σ0 * 2 ** ((o+s)/S) = 2**(1/S)
    k = 2**(1/n)
    print('相邻尺度之间相差的比例因子：{}'.format(k))
    # 计算每组每层的尺度因子σ，以列表按顺序保存
    sigmas = [[ sigma0 * 2**((o + s) / n)for s in range(S)] for o in range(O)]
    print('金字塔对应的尺度因子：{}'.format(sigmas))
    # 也可以这么写，提高运算速度
    # sigma = [[(k ** s) * sigma0 * (1 << o) for s in range(S)] for o in range(O)]
    # 对图像进行下采样操作
    samplePyramid = [downSample(image, 2**o) for o in range(O)]
    sampleShape = [np.shape(eg) for eg in samplePyramid]
    print('下采样后各组图像的大小：{}'.format(sampleShape))

    # 生成高斯金子塔
    GuassianPyramid = []
    for i in range(O):
        for j in range(S):
            # 计算高斯卷积核的维度
            # 在计算高斯函数的离散近似时，在大概3*sigma距离之外的像素都可以看作不起作用，这些像素的计算也就可以忽略不计，
            # 通常，图像处理只需要计算（6*sigma+1）*（6*sigma+1）的矩阵就可以保证相关像素影响。
            # 因为卷积核一般为奇数
            dim = int(6 * sigmas[i][j] + 1)
            if dim % 2 == 0:
                dim += 1
            print('第{}组第{}层，高斯卷积核维度为{}'.format(i+1,j+1,dim))
            GuassianKernel(sigmas[i][j],dim)
            break
        break


def sift(image):
    """
    sift关键点检测
    :param image: 原始图像
    :return:
    """
    SIFT_SIGMA = 1.6
    # 这是照片在拍摄的时候会被镜头以σ=0.5进行模糊化
    SIFT_INIT_SIGMA = 0.5
    # 我们计算的图像尺度应该时镜头和高斯金字塔模糊的和作用，所以第一组第一层的高斯模糊σ初始值为：
    sigma0 = np.sqrt(SIFT_SIGMA ** 2 - SIFT_INIT_SIGMA ** 2)

    # 你想要提取的特征个数
    n = 3
    # 生成金字塔
    buildDog(image, n, sigma0)

if __name__ == '__main__':
    image = cv.imread('images/koala.png')
    sift(image)