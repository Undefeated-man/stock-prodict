#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 2018

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

from __future__ import print_function
import time
import numpy as np


class VectorAutoRegression(object):
    """
    向量自回归模型
    """

    _RANDOM_BIAS_SCALE = 1.0  # 随机因素的缩放因子

    def __init__(self, n, P):
        self.n = n
        self.P = P
        self.c = None
        self.Amatrix = None

        self._initial()

    def _initial(self):
        if self.P < 1:
            raise ValueError("log phase value p should be not lower than 1.")

        # 设置随机种子
        np.random.seed(int(time.time()))

        # 常数项
        self.c = np.random.rand(self.n, 1)
        # 参数矩阵A
        self.Amatrix = np.random.rand(self.P, self.n, self.n)

    def fit(self, x, y, alpha, epoches, threshold=1e-3):
        """
        x.shape = (P, T, n, 1)
        y.shape = (T, n, 1)
        """
        self.alpha = alpha
        self.sample_x = np.array(x)
        self.sample_y = np.array(y)

        ep = 1
        last_cost = None
        while ep <= epoches:
            # 取得当前参数下的预测值
            pred = self.predict(self.sample_x)
            # 计算损失函数值
            cost = self.calculateCost(pred, self.sample_y)
            # print("Epoch %4d: lost function value = %.4f" % (ep, cost))
            if last_cost is None:
                last_cost = cost
            else:
                if abs(last_cost - cost) < threshold:
                    break
                else:
                    last_cost = cost
            # 更新参数
            self.update(pred, self.sample_y)
            ep += 1

    def calculateCost(self, pred, gold):
        """LMS cost function"""
        diff = (pred - gold)
        return 0.5 * np.sum(diff * diff)

    def predict(self, x):
        """
        x.shape = (P, T, n, 1)
        """
        T = x.shape[1]
        result = np.zeros(shape=(T, self.P + 2, self.n, 1))
        for t in range(T):
            result[t, 0] = self.c
            result[t, 1] = self._RANDOM_BIAS_SCALE * \
                np.random.uniform(-0.5, 0.5, size=(self.n, 1))
            # result[t, 1] = np.zeros(shape=(self.n, 1))
            for p in range(self.P):
                result[t, 2 + p] = self.Amatrix[p] * x[p, t, :]
        return np.sum(result, axis=1)

    def update(self, pred, gold):
        """
        pred.shape == gold.shape = (T, n, 1)
        """
        diff = (pred - gold)
        self.c -= self.alpha * np.sum(diff, axis=0)
        for p in range(self.P):
            self.Amatrix[p] -= self.alpha * \
                np.sum(diff * self.sample_x[p, :], axis=0)
