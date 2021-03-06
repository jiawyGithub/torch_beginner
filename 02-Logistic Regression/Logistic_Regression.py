# encoding: utf-8
"""
@author:  Jiawy
@contact: jiawenyu2021@163.com
@commit: 在FashionMNIST数据集上实现逻辑斯蒂回归
"""

import time
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
"""
torchvision，该包主要由3个子包组成，分别是：torchvision.datasets、torchvision.models、torchvision.transforms。
transform=transforms.ToTensor() Convert the PIL Image to Tensor
"""

def get_fashion_mnist_pred_labels(pred,labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    text = []
    for i in range(labels.size(0)):
        text.append(f"{text_labels[pred[i]]}/{text_labels[labels[i]]}")
    return text
#img 表示要描画的图像数据,row&cols 分别表示要画面几行几列，scale表示缩放比例
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图⽚张量
            ax.imshow(img.numpy())
        else:
            # PIL图⽚
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False) #坐标刻度
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontsize=8 )
    return axes


# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 5

# 下载训练集 FashionMNIST
train_dataset = datasets.FashionMNIST(
    root='../datasets', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.FashionMNIST(
    root='../datasets', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 Logistic Regression 模型
class Logistic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logistic_Regression, self).__init__()
        self.logistic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logistic(x)
        return out

model = Logistic_Regression(28 * 28, 10)  # 图片大小是28x28, 共10类
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(num_epochs):
    print('*' * 20)
    print(f'epoch {epoch+1}')
    since = time.time()
    running_loss = 0.0
    running_acc = 0.0
    model.train()
    for i, data in enumerate(train_loader, 1):
        img, label = data # img 是一个tensor
        img = img.view(img.size(0), -1)  # 将图片展开成 28x28=784
        # pytorch的view() 相当于numpy中resize()的功能，参数中的-1就代表这个位置由其他位置的数字来推断
        # tensor.size 返回的是当前张量的形状,返回值是元组tuple的一个子类.
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item()
        _, pred = torch.max(out, 1) # _是最大值本身，pred是最大值的下标
        # 在分类问题中，通常需要使用max()函数对softmax函数的输出值进行操作，求出预测值索引，然后与标签进行比对，计算准确率。
        # https://www.jianshu.com/p/3ed11362b54f
        running_acc += (pred==label).float().mean()
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print(f'[{epoch+1}/{num_epochs}] Loss: {running_loss/i:.6f}, Acc: {running_acc/i:.6f}')
    print(f'Finish {epoch+1} epoch, Loss: {running_loss/i:.6f}, Acc: {running_acc/i:.6f}')
    
    # Test
    # model.eval()
    # eval_loss = 0.
    # eval_acc = 0.
    # for data in test_loader:
    #     img, label = data
    #     img = img.view(img.size(0), -1)
    #     if use_gpu:
    #         img = img.cuda()
    #         label = label.cuda()
    #     with torch.no_grad():
    #         out = model(img)
    #         loss = criterion(out, label)
    #     eval_loss += loss.item()
    #     _, pred = torch.max(out, 1)
    #     eval_acc += (pred == label).float().mean()

    # print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_acc/len(test_loader):.6f}')
    # print(f'Time:{(time.time()-since):.1f} s')

# # 可视化 共157个batch
img_show_index=150
data_show = None
pred_show = None

model.eval()
eval_loss = 0.
eval_acc = 0.
for i,data in enumerate(test_loader):
    img, label = data
    img = img.view(img.size(0), -1)
    if use_gpu:
        img = img.cuda()
        label = label.cuda()
    with torch.no_grad():
        out = model(img)
        loss = criterion(out, label)
    eval_loss += loss.item()
    _, pred = torch.max(out, 1)
    eval_acc += (pred == label).float().mean()

    # 可视化
    if img_show_index == i:
        data_show = data
        pred_show = pred

print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_acc/len(test_loader):.6f}')
print(f'Time:{(time.time()-since):.1f} s')

# 可视化
img, label = data_show
show_images(img.reshape(img.size(0), 28, 28), 8, 8, titles=get_fashion_mnist_pred_labels(pred_show,label))
plt.show() 

# 保存模型
torch.save(model.state_dict(), './logstic.pth')
