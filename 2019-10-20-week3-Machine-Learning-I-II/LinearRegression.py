#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/10/23 13:44
# @Author : Micky
# @Desc : 线性回归代码实现
# @File : LinearRegression.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

class LinearRegressionByMicky(object):

    def __init__(self, learning_rate=0.001, iterations=1000, tol=1e-8, iterat_type='BGD'):
        """
        :param learn_rate:  学习率
        :param iterations: 迭代次数
        :param tol 每次迭代损失变化的产值阈值
        :param iterat_type 梯度迭代的方式
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.iterat_type = iterat_type
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.coef_c = None
        self.err = []

    def fit(self, X, y):
        """
        模型训练
        :param X: 样本数据
        :param y: 目标属性
        :return:
        """
        # 1.将数据转换未numpy数据类型
        X = np.asarray(X)
        y = np.asarray(y)
        # 2.目标属性转换未一维
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
        # 计算预测值
        predict = self._internel_predict(self.X, self.coef_, self.intercept_)
        # 计算损失loss
        print('开始计算损失')
        current_loss = self._getloss(predict)
        # 记录损失
        self.err.append(current_loss)
        print('当前损失为：{}'.format(current_loss))
        change_loss = current_loss + self.tol
        print('开始{}梯度下降算法'.format(self.iterat_type))
        if self.iterat_type == 'BGD':
            self._bgd(predict, current_loss, change_loss, feature_count, sample_count)
        if self.iterat_type == 'SGD':
            self._sgd(predict, current_loss, change_loss, feature_count, sample_count)
        if self.iterat_type == 'MBGD':
            self._mbgd(predict, current_loss, change_loss, feature_count, sample_count)

    def _bgd(self, predict, current_loss, change_loss, feature_count, sample_count):
        """
        BGD: 批量梯度迭代算法，使用当前样本的所有值的梯度作为当前theta的更新
        :param predict: 预测值
        :param current_loss: 当前损失
        :param change_loss: 变换损失
        :param feature_count: 特征个数
        :param sample_count: 样本个数
        :return:
        """
        num_iter = 0
        while num_iter < self.iterations and change_loss > self.tol:
            # 每次更新参数，所有样本需要参与
            err = self.y - predict
            # theta = theta - alpha * J的一阶导数
            # J的一阶导数又等于 实际值与预测值差值乘以当前样本特征属性求和
            # 计算theta 和截距项总的变换
            for feature in range(feature_count):
                delta = 0
                for sample in range(sample_count):
                    delta += err[sample] * self.X[sample][feature]
                self.coef_[feature] = self.coef_[feature] + self.learning_rate * 1.0 * delta / sample_count
            # 截距项，因为截距项对应的x为1，
            self.intercept_ = self.intercept_ + self.learning_rate * np.mean(err)
            # 计算损失变换量
            pre_loss = current_loss
            # 计算预测值
            predict = self._internel_predict(self.X, self.coef_, self.intercept_)
            # 计算损失
            current_loss = self._getloss(predict)
            # 记录损失
            self.err.append(current_loss)
            # 计算损失变换值
            change_loss = np.abs(pre_loss - current_loss)

            # 以下代码只是为了可视化
            #########################################
            plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
            loss = plt.subplot(1, 2, 1)
            plt.plot(range(len(self.err)), self.err)
            loss.set_title('BGD Loss损失:{:.2f};迭代次数:{}'.format(current_loss, num_iter))


            line = plt.subplot(1, 2, 2)
            plt.scatter(self.X, self.y, c='g', marker='o')
            # plt.plot(range(len(self.err)),self.err)
            plt.plot(self.X, predict)
            line.set_title('BGD 学习率: {};迭代次数: {}'.format(self.learning_rate, num_iter))
            plt.pause(0.4)  # 设置暂停时间，太快图表无法正常显示
            ###########################################


            num_iter += 1
        print('梯度迭代结束，迭代次数为：{}，模型的theta：{}，截距项为：{}，最终的损失为：{}'.format(num_iter, self.coef_, self.intercept_, self.err))

    def _sgd(self, predict, current_loss, change_loss, feature_count, sample_count):
        """
        每次使用单个样本跟新模型参数
        :param predict: 预测值
        :param current_loss: 当前损失
        :param change_loss: 改变的损失
        :param feature_count: 特征数量
        :param sample_count: 样本个数
        :return:
        """
        num_iter = 0
        idxs = [i for i in range(sample_count)]
        while num_iter < self.iterations and change_loss > self.tol:
            err = self.y - predict
            # 打乱顺序
            np.random.shuffle(idxs)
            for sample in range(sample_count):
                # 获取单个样本
                idx = idxs[sample]
                x = self.X[idx]
                predict_y = self.predict(x)
                print('预测值：{}'.format(predict_y))
                delta = self.y[idx] - predict_y
                print(self.y[idx], delta)
                for feature in range(feature_count):
                    self.coef_[feature] = self.coef_[feature] + self.learning_rate * delta * x[feature]
            # 截距项
            self.intercept_ = self.intercept_ + self.learning_rate * np.mean(err)
            # 计算损失变换量
            pre_loss = current_loss
            # 计算预测值
            predict = self._internel_predict(self.X, self.coef_, self.intercept_)
            # 计算损失
            current_loss = self._getloss(predict)
            # 记录损失
            self.err.append(current_loss)
            # 计算损失变换值
            change_loss = np.abs(pre_loss - current_loss)

            # 以下代码只是为了可视化
            #########################################
            plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
            loss = plt.subplot(1, 2, 1)
            plt.plot(range(len(self.err)), self.err)
            loss.set_title('SGD Loss损失:{:.2f};迭代次数:{}'.format(current_loss, num_iter))

            line = plt.subplot(1, 2, 2)
            plt.scatter(self.X, self.y, c='g', marker='o')
            # plt.plot(range(len(self.err)),self.err)
            plt.plot(self.X, predict)
            line.set_title('SGD 学习率: {};迭代次数: {}'.format(self.learning_rate, num_iter))
            plt.pause(0.4)  # 设置暂停时间，太快图表无法正常显示

            num_iter += 1
        print('梯度迭代结束，迭代次数为：{}，模型的theta：{}，截距项为：{}，最终的损失为：{}'.format(num_iter, self.coef_, self.intercept_, self.err))

    def _mbgd(self, predict, current_loss, change_loss, feature_count, sample_count):
        num_iter = 0
        while change_loss > self.tol and num_iter < self.iterations:
            err = self.y - predict
            for index in range(sample_count // 10):
                if sample_count < 10:
                    sample_gradient = self.X
                    sample_gradient_err = err
                else:
                    sample_gradient = self.X[index:index + 10]
                    sample_gradient_err = err[index:index + 10]
                for feature_index in range(feature_count):
                    delta = 0
                    for sample_gradient_index in range(sample_gradient.shape[0]):
                        delta += sample_gradient_err[sample_gradient_index] * sample_gradient[sample_gradient_index][
                            feature_index]
                    self.coef_[feature_index] = self.coef_[feature_index] + self.learning_rate * delta / sample_gradient.shape[0]
            # 截距项
            self.intercept_ = self.intercept_ + self.learning_rate * np.mean(err)
            # 计算损失变换量
            pre_loss = current_loss
            # 计算预测值
            predict = self._internel_predict(self.X, self.coef_, self.intercept_)
            # 计算损失
            current_loss = self._getloss(predict)
            # 记录损失
            self.err.append(current_loss)
            # 计算损失变换值
            change_loss = np.abs(pre_loss - current_loss)

            # 以下代码只是为了可视化
            #########################################
            plt.clf()  # 清除刷新前的图表，防止数据量过大消耗内存
            loss = plt.subplot(1, 2, 1)
            plt.plot(range(len(self.err)), self.err)
            loss.set_title('MBGD Loss损失:{:.2f};迭代次数:{}'.format(current_loss, num_iter))

            line = plt.subplot(1, 2, 2)
            plt.scatter(self.X, self.y, c='g', marker='o')
            # plt.plot(range(len(self.err)),self.err)
            plt.plot(self.X, predict)
            line.set_title('MBGD 学习率: {};迭代次数: {}'.format(self.learning_rate, num_iter))
            plt.pause(0.4)  # 设置暂停时间，太快图表无法正常显示

            num_iter += 1


    def _getloss(self, predict):
        """
        获取损失 , 分母加1防止为0
        :param predict: 预测值
        :return:
        """
        return np.sum(np.square(predict - self.y)) / (2 * len(predict) +1)

    def predict(self, X):
        """
        预测
        :param X:
        :return:
        """
        return self._internel_predict(X, self.coef_, self.intercept_)

    def _internel_predict(self, X, coef, intercept):
        """
        计算预测值
        :param X: 样本数据
        :param coef: 模型参数
        :param intercept: 模型截距项
        :return:
        """
        print('开始计算预测值')
        # coef = np.reshape(coef, (-1,1))
        predict = np.dot(X, coef) + intercept
        print('预测完成，shape：{}'.format(np.shape(predict)))
        return predict


if __name__ == '__main__':
    N = 10
    x = np.linspace(0, 7, N) + np.random.randn(N)
    y = 1.8 * x * 3 + x * 2 - 14 * x - 7 + np.random.randint(0, 20, N)
    x.shape = -1, 1
    y.shape = -1, 1

    # N = 10
    # d = 2
    # x = np.linspace(0, 6, N * d).reshape((N, d))
    # x = x + np.random.randn(N, d)
    # y = np.dot(x ** 3, [[1.0], [2.0]]) \
    #     + np.dot(x ** 2, [[-5.0], [-3.2]]) \
    #     + np.dot(x, [[7.0], [2.0]]) \
    #     + np.random.randn(N, 1)
    # x.shape = -1, d
    # y.shape = -1, 1

    # 模型训练  分为BGD、SGD、MBGD 可分别训练观看结果
    flag = 0
    if flag == 0:
        algo = LinearRegressionByMicky(learning_rate=0.1, iterations=100, iterat_type='BGD')
    elif flag == 1:
        algo = LinearRegressionByMicky(iterations=100, iterat_type='SGD')
    else:
        algo = LinearRegressionByMicky(learning_rate=0.1, iterations=100, iterat_type='MBGD')
    algo.fit(x, y)
    plt.show()