#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/24 11:39
# @Author : Micky
# @Desc : 逻辑回归代码实现
# @File : LogisticRegression.py
# @Software: PyCharm

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

class LogisticRegressionByMicky(object):

    def __init__(self, learning_rate=0.1, iterations=1000, tol=1e-8):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.loss = []

    def fit(self, X, y):
        """
        训练模型
        :param X: 样本数据
        :param y: 目标属性
        :return:
        """
        # 数据转换为numpy数组类型
        X = np.asarray(X)
        y = np.asarray(y)

        self.X = X
        self.y = np.reshape(y, -1)
        # 判断目标属性个数和样本个数是否一致
        sample_count, feature_count = self.X.shape
        label_count = self.y.shape[0]
        print('样本个数：{}, label个数：{}, 特征个数：{}'.format(sample_count, label_count, feature_count))
        if sample_count != label_count:
            raise Exception('x and y is no same')
        # 模型参数初始值，随机初始化。有多少个特征属性就初始化多少个模型参数
        self.coef_ = np.random.normal(loc=0.0, scale=1.0, size=feature_count)
        # 模型的截距项
        self.intercept_ = 0.0
        # 计算准确率
        print('开始计算损失：')
        current_loss = self._loss(X, y)
        # 记录损失
        self.loss.append(current_loss)
        print('当前损失为：{}'.format(current_loss))
        change_loss = current_loss + self.tol
        print('开始梯度下降算法')
        num_iter = 0
        # print(self.X)
        while num_iter < self.iterations and change_loss > self.tol:
            err = self.y - self._internel_predict_prb(self.X)
            for feature in range(feature_count):
                delta = 0
                for sample in range(sample_count):
                    delta += err[sample] * self.X[sample][feature]
                self.coef_[feature] = self.coef_[feature] + self.learning_rate * 1.0 * delta / sample_count
            # acc = self.accuracy(self._internel_predict(self.X, self.coef_), self.y)
            # 截距项，因为截距项对应的x为1，
            self.intercept_ = self.intercept_ + self.learning_rate * np.mean(err)
            # 计算损失变换量
            pre_loss = current_loss
            # 计算损失
            current_loss = self._loss(X, y)
            self.loss.append(current_loss)
            # 计算损失变换值
            change_loss = np.abs(pre_loss - current_loss)

            # 以下代码只是为了可视化
            #########################################
            plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
            plt.plot(range(len(self.loss)), self.loss)
            plt.title('LogisticRegression loss:{:.3f};迭代次数:{}'.format(current_loss, num_iter))
            plt.pause(0.4)  # 设置暂停时间，太快图表无法正常显示


            # line = plt.subplot(1, 2, 2)
            # plt.scatter(range(sample_count), self.y, c=self.y)
            # plt.plot(self.X, np.dot(self.X, self.coef_) + self.intercept_, color='r')
            # line.set_title('学习率: {};迭代次数: {}'.format(self.learning_rate, num_iter))
            # plt.pause(0.4)  # 设置暂停时间，太快图表无法正常显示
            ###########################################
            num_iter += 1

    def accuracy(self, predict, y):
        """
        计算准确率
        :param predict: 预测分类概率
        :param y: 实际分类
        :return:
        """
        return np.mean(predict == y)

    def _loss(self, X, y):
        """
        计算损失
        :param X: 样本
        :param y: 目标属性
        :return:
        """
        first = np.dot(-y, np.log(self._internel_predict_prb(X)))
        second = np.dot((1-y), np.log(1-self._internel_predict_prb(X)))
        total = np.sum((first - second))
        return total/len(y)

    def _internel_predict_prb(self, X):
        return self._sigmoid(np.dot(X, self.coef_)+ self.intercept_)

    def _sigmoid(self, x):
        """
        获取sigmoid的函数值
        :param x: x
        :return: sigmoid函数值
        """
        return 1 / (1 + np.e ** (-x))

if __name__ == '__main__':
    N = 400
    centers = 2
    data, y = make_blobs(n_samples=N, n_features=2, centers=centers)
    x = np.reshape(data[:, 0],(-1, 1))
    # plt.scatter(x, data[:, 1], c=y)
    # plt.show()
    print(x[1])
    print(y[1])
    algo = LogisticRegressionByMicky(iterations=100)
    algo.fit(x, y)
    # t_y = np.asarray([0,0,0,1,1,0])
    # t_y2 = np.asarray([0,0,0,1,1,1])
    # print(algo.accuracy(t_y, t_y2))

