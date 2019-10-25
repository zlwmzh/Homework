#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/17 15:06
# @Author : Micky
# @Desc : 随机抽样一致算法(RANSAC) 伪代码
# @File : Ransac算法.py
# @Software: PyCharm

"""
RANSAC是 Random Sample Consenus的缩写。 它是根据提供的数据集拟合模型的过程，得到最终的模型参数。RNASAC假设数据集中包含正常样本(绝大多数)和噪声样本，
并假定可以通过这些正确的数据样本可以拟合出正确的样本。通过随机取子集作为模型内的点，不断迭代，最终找到能够拟合大多数样本的模型。这个思想其实和KMeans聚类算法
类似，都是随机初始化几个点，然后不断调整调优。
****************************************
基本流程：
数据集为M，
1. 随机抽取n条样本得到数据集N，认为是正确的样本， 剩余的数据集为M-N
2. 用N去拟合得到模型Model
3. 用剩余的M-N数据集去测试2中得到的Model，如果适合这个模型的样本加入到N中，否则不加入
4. 统计N的个数
5. 重复1~4, 取N最多的那个模型就是我们需要的模型
*****************************************
*****************************************
*****************************************
伪代码：
samples = input samples data set
iterations = define number of iterations
n = 随机抽取的样本个数
matchNums = 模型拟合样本最少满足的样本个数


pError # 记录最优的损失误差
pModel # 记录最优的模型
i = 0
while i < iterations:
    randomSamples = 从samples随机抽取n条样本
    # 根据randomSamples拟合模型
    model = buildModel(randomSamples)
    # 用model取验证剩余的数据样本，得到满足模型的样本
    matchSamples = model.fit(otherSamples)
    if matchSamples > matchNums:
        # 找到了更优的模型
        # matchSamples+randomSamples去完整的拟合这个模型
        nModel = buildModel(matchSamples+randomSamples)
        # 计算误差，平方和均值误差
        error = betterModel.getError()
        if error < pError:
            pModel = nModel
            pError = error
     i += 1
return pModel
"""
