#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/11/4 11:08
# @Author : Micky
# @Desc : 根据老师的代码实现
# @File : NeuralNetWorkByTeacher.py
# @Software: PyCharm

# 导入需要的包
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
# 生成数据集，随机种子
np.random.seed(777)
# sklearn自带的数据集生成方法，得到的X为二维数据，y为标签（0或1）
# 数据集长度为200，噪音为0.2
X, y = sklearn.datasets.make_moons(200, noise=0.20)

# 画出散点图观察数据特点
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

# 参数设定，样本数为200，num_examples；
# 输入维度为2（二维数据）；
# 输出维度为2（二分类，ONE-HOT-ENCODING）
num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2

# 梯度下降参数
# 可以自己修改试试看结果如何？
lr = 0.01 # 学习率
reg_lambda = 0.01 # 正则化项，默认未添加

# 从模型参数正向传播，计算Loss
def calculate_loss(model):

    # softmax + cross entropy
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播计算预测的概率值
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 根据预测的概率值和真实标签计算Loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # 正则化项 (optional)
    # data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# 根据模型预测标签值，实际在训练中用不到，但在测试时需要调用 (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传递参数
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1) # 概率最大的那个对应的索引号就是标签

# 建立NN model，同时完成BP的loss回传，更新model参数
def build_model(nn_hdim, num_passes=30000, print_loss=False):

    # 权值初始化
    # np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # 定义的模型数据结构，这里是一个字典
    model = {}

    # 梯度下降，num_passes为迭代次数
    for i in range(0, num_passes):

        # 正向传播，计算概率
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 反向传播，计算loss对权值的梯度
        delta3 = probs
        delta3[range(num_examples), y] -= 1     # 选择哪一个维度对应的来减一，剩余的不处理,1-ai or aj，还记得吗？
                                                # 是关于(softmax + cross entropy)的导数相乘,
                                                # 可以自己重新推导一次
        # 你也可以自己在这里添加如下语句进行探索
        #print("before: ", delta3[:4])
        #delta3[range(4), y[:4]] -= 1
        #print(delta3)
        #print(" ")

        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)     # it's delta3 * [1]; 按列相加，保持维度
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))  # tanh 导数: 1- tanh^2
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 添加正则化项
        # 默认没有，你可以加上看看效果如何
        #dW2 += reg_lambda * W2
        #dW1 += reg_lambda * W1

        # 梯度下降，参数更新
        W1 += -lr * dW1
        b1 += -lr * db1
        W2 += -lr * dW2
        b2 += -lr * db2

        # 模型参数保存
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 打印损失变化
        # 这步很耗时
        if print_loss and i % 1000 == 0:
          print ("Loss after iteration %i: %f" %(i, calculate_loss(model)))

    return model

# 建立模型
# 你可以改参数玩，如hidden layer神经元个数等，enjoy!
model = build_model(20, print_loss=True)