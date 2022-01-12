# encoding: utf-8
"""
@author:  Jiawy
@contact: jiawenyu2021@163.com
@commit: 随机梯度下降+反向传播函数的应用
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# w = 1.0 # 待训练参数 y = wx 
w = torch.Tensor([1.0])
w.requires_grad = True # 需要计算梯度

epoch_num = 100 
learn_rate = 0.0001

def forward(x):
    return x * w # w是tensor，x会自动转换成tensor
# 损失
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)**2
# 平均损失
def cost(xs,ys):
    cost = 0
    # zip 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    # 与 zip 相反，*zipped 可理解为解压
    for x, y in zip(xs,ys):
        cost += loss(x, y)
    return cost/len(xs)
# 梯度
def gradient(x, y):
    return 2 * x * (x * w - y)

print('before training, w=', w)
for epoch in range(epoch_num):
    for x, y in zip(x_train,y_train):
        l = loss(x, y) # forward
        l.backward()
        # w.grad.item()
        w.data = w.data - learn_rate * w.grad.data # grad也是一个tensor，所以取data才不会建立计算图
        w.grad.data.zero_() # 计算图清空，下次重新构建计算图，否则梯度会累加
    print('Epoch', epoch, 'w=', w.data, 'loss=', l.item())
print('after training, w=', w)

# 可视化
predict = w.data * x_train
fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict.numpy(), label='Fitting Line')
# 显示图例
plt.legend() 
plt.show()


