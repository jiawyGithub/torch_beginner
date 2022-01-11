# encoding: utf-8
"""
@author:  Jiawy
@contact: jiawenyu2021@163.com
@commit: 随机梯度下降
"""

import matplotlib.pyplot as plt
import numpy as np

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

w = 1.0 # 待训练参数 y = wx 
epoch_num = 100 
learn_rate = 0.0001

def forward(x):
    return x * w
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
        grad_val = gradient(x, y)
        w -= learn_rate * grad_val
    cost_val = cost(x_train, y_train)
    print('Epoch', epoch, 'w=', w, 'loss=', cost_val)
print('after training, w=', w)

# 可视化
predict = w * x_train
fig = plt.figure(figsize=(10, 5))
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predict, label='Fitting Line')
# 显示图例
plt.legend() 
plt.show()


