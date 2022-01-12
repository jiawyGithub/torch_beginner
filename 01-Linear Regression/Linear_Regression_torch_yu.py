"""
@author:  Jiawy
@contact: jiawenyu2021@163.com
@commit: 用pytorch实现线性回归，优化器使用随机梯度下降SDG
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
from torch.nn.modules.linear import Linear

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

epoch_num = 1000 
learn_rate = 0.0001

class LinearModel(torch.nn.Module):
    def __init__(self): # 构造函数
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1,1) # in_features指的是输入的二维张量的大小,out_features指的是输出的二维张量的大小

    def forward(self, x):
        y_pred = self.linear(self.linear(x))
        return y_pred

model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False) # MSE 均方损失函数 size_average=False得到整个batch所有像素loss和
print('model.named_parameters:',list(model.named_parameters()))
# for name, param in model.named_parameters(): #查看可优化的参数有哪些
#   if param.requires_grad:
#     print(name)
optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate) # 随机梯度下降

print('before training')
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

for epoch in range(epoch_num):
    y_pred = model(x_train)
    loss = criterion(y_pred,y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() #这个方法会更新所有的参数。一旦梯度被backward后，就可以调用这个函数。

    if (epoch+1) % 20 == 0:
        print('epoch',epoch,'loss',loss.item())

print('after training')
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

"""
model.eval() 作用等同于 self.train(False)
简而言之，就是评估模式。而非训练模式。
在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
在对模型进行评估时，应该配合使用with torch.no_grad() 
"""
model.eval()
with torch.no_grad():
    predict = model(x_train)

# 可视化
fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict.numpy(), label='Fitting Line')
# 显示图例
plt.legend() 
plt.show()


