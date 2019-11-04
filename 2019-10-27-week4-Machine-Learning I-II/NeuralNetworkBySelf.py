#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/4 11:39
# @Author : Micky
# @Desc : 自己实现一个简单的神经网络：
# @File : NeuralNetworkBySelf.py
# @Software: PyCharm

import numpy as np
from sklearn.datasets import make_moons
from collections import  Counter


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, iterations=100, activation_function=None,
                 loss_function='crossentry', print_log=True):
        """
        传入一些初始化参数:
        :param input_nodes: 输入节点数量（样本特征属性的个数）
        :param hidden_nodes: 隐藏层节点个数（自己定义合适的）
        :param output_nodes: 输出层节点个数（想如果是回归问题的话可能是一个节点，分类的问题的话就是类别数量了）
        :param learning_rate: 学习率
        :param iterations: 迭代次数
        :param activation_function: 设置激活函数
        :param loss_function: 损失函数
        :param print_log: 是否打印日志，默认打印
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        self.itertions = iterations
        self.loss_function = loss_function
        self.print_log = print_log

        if activation_function is None:
            # 设置默认的激活函数，这里用sigmoid
            self.activation_function = self._sigmoid
        else:
            self.activation_function = activation_function

        # 创建权重矩阵
        # 1. 输入层到隐藏层的权重矩阵
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes * 0.5,
                                                        (self.input_nodes, self.hidden_nodes))
        # 2. 隐藏层到输出层的权重矩阵
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes * 0.5,
                                                         (self.hidden_nodes, self.output_nodes))

    def _sigmoid(self, x):
        """
        sigmoid函数
        :param x: 函数值
        :return: 转换后的函数值
        """
        return 1.0 / (1 + np.exp(-x))

    def _softmax(self, x):
        """
        softmax 函数
        :param x: 函数值
        :return: 转换后的函数值
        """
        return np.exp(x) / np.sum(np.exp(x),axis=1, keepdims=True)

    def crossentry(self, y):
        """
        计算交叉熵损失函数
        :param py: 预测的概率
        :param y:
        :return:
        """
        # 执行正向传播
        hidden_layer_input = np.dot(X, self.weights_input_to_hidden)
        hidden_layer_output = self.activation_function(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_to_output)
        output_layer_output = self._softmax(output_layer_input)
        corect_logprobs = -np.log(output_layer_output[range(len(y)), y])
        cross_entry = np.mean(corect_logprobs)
        return cross_entry

    def fit(self, features, labels):
        """
        模型训练
        :param features:
        :param labels:
        :return:
        """
        # 当前迭代次数
        num_iter = 0
        # 样本个数
        n_samples = features.shape[0]
        while num_iter < self.itertions:
            # 1. 构建执行前向过程
            # 1.1 输入层到隐藏层
            input_layer_output = features
            # 维度变化：(n_samples,input_nodes) * (input_nodes, hidden_nodes) = (n_samples, hidden_nodes)
            hidden_layer_input = np.dot(input_layer_output, self.weights_input_to_hidden)
            # 对隐藏层的值进行激活，得到隐藏层输出值
            # 维度(n_samples , hidden_nodes)
            hidden_layer_output = self.activation_function(hidden_layer_input)
            # 1.2 隐藏层到输出层
            # 维度：(n_samples,hidden_nodes) * (hidden_nodes, output_nodes) = (n_samples , output_nodes)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_to_output)
            # 维度(n_samples , output_nodes)
            output_layer_output = self._softmax(output_layer_input)

            # 2. 执行反向传播过程。这里需要判断使用的损失函数是什么类型的
            delta_w_h_o_error = output_layer_output
            delta_w_h_o_error[range(n_samples), labels] -= 1
            delta_w_h_o = np.dot(hidden_layer_output.T, delta_w_h_o_error)
            # print(np.shape(delta_w_h_o))
            delta_w_i_h = np.dot(delta_w_h_o_error, self.weights_hidden_to_output.T)
            delta_w_i_h_sigmoid = hidden_layer_output * (1 - hidden_layer_output)
            delta_w_i_h = delta_w_i_h * delta_w_i_h_sigmoid
            delta_w_i_h = np.dot(X.T, delta_w_i_h)

            self.weights_input_to_hidden -= self.lr * delta_w_i_h
            self.weights_hidden_to_output -= self.lr * delta_w_h_o

            if self.print_log and num_iter % 1000 == 0:
                print('itertion: {}/{}，currentLoss: {}'.format(num_iter, self.itertions, self.crossentry(y)))
            num_iter += 1


if __name__ == '__main__':

    # 产生数据
    N_SAMPLES = 200
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.2)
    print('X的shape：{}，y的shape：{}'.format(np.shape(X), np.shape(y)))

    # 超参数配置
    # 输入节点：特征个数
    input_nodes = X.shape[1]
    # 隐藏层的节点：设置为20个
    hidden_nodes = 20
    # 输出层节点：根据我们数据的分类个数来判断
    output_nodes = len(Counter(y))
    # 学习率
    learning_rate = 0.01
    # 最大迭代次数
    itertions = 30000
    loss_function = 'crossentry'
    print('input_nodes: {}, hidden_nodes: {}, output_nodes: {}, learning_rate: {}, itertions: {}, loss_function: {}'.format(input_nodes, hidden_nodes, output_nodes, learning_rate, itertions, loss_function))
    # 创建神经网络
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, iterations=itertions,
                       loss_function=loss_function)
    nn.fit(X, y)
