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
import torch.nn.functional as F

x_train = torch.Tensor([[1.0],[2.0],[3.0]])
y_train = torch.Tensor([[0],[0],[1]])

epoch_num = 1000
learn_rate = 0.01

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1) # in_features指的是输入的二维张量的大小,out_features指的是输出的二维张量的大小

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegressionModel()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate)

print('model.named_parameters:',list(model.named_parameters()))

for epoch in range(epoch_num):
    y_pred = model(x_train)
    loss = criterion(y_pred,y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() #这个方法会更新所有的参数。一旦梯度被backward后，就可以调用这个函数。

    if (epoch+1) % 20 == 0:
        print('epoch',epoch+1,'loss',loss.item())
    
model.eval()
with torch.no_grad():
    x = np.linspace(1, 10, 200)
    x_t = torch.Tensor(x).view((200, 1))
    y_t = model(x_t)
    y = y_t.data.numpy()

# 可视化
fig = plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
# 显示图例
plt.grid() 
plt.show()