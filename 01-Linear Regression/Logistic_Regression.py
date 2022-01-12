"""
@author:  Jiawy
@contact: jiawenyu2021@163.com
@commit: 实现逻辑斯蒂回归
"""

"""
步骤：
1、准备数据
2、用Class设计模型
3、构造loss和optimizer
4、训练：forward、backward、update
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

x_train = [[1.0],[2.0],[3.0]]
y_train = [[0],[0],[1]]
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1) # in_features指的是输入的二维张量的大小,out_features指的是输出的二维张量的大小

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
    