__author__ = 'SherlockLiao'

from email.mime import image
from numpy import rint
import torch
from torch import igamma, nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
# from logger import Logger

# 定义超参数
batch_size = 128
learning_rate = 1e-2
num_epoches = 5
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速


def to_np(x):
    return x.cpu().data.numpy()


# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义 Convolution Network 模型
# Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
# MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2)
            )

        self.fc = nn.Sequential(
            nn.Linear(400, 120), 
            nn.Linear(120, 84), 
            nn.Linear(84, n_class)
            )

    def forward(self, x):
        # print('in-1',x.shape)
        out = self.conv(x)
        # print('out-1',x.shape)
        out = out.view(out.size(0), -1)
        # print('in-2',out.shape)
        out = self.fc(out)
        # print('out-2',out.shape)
        return out


model = Cnn(1, 10)  # 图片大小是28x28
if use_gpu:
    model = model.cuda()
# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# logger = Logger('./logs')
# 开始训练
for epoch in range(num_epoches):
    print('*' * 20)
    print('epoch {}'.format(epoch + 1))
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data # img 和 label 都是tensor, 默认requires_grad=false
        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        # ???
        # img = Variable(img)
        # label = Variable(label)
        # 改为：
        img.requires_gard = True
        label.requires_gard = True
        """
        ???
        Variable实质上就是可以变化的量，区别于int变量，它是一种可以变化的变量，正好符合了反向传播，参数更新的属性。
        在之前tensor不能反向传播，variable可以反向传播。
        Variable计算时，它会逐渐地生成计算图。
        这个图将所有的计算节点连接起来，最后进行误差反向传递的时候，一次性将所有Variable里面的梯度都计算出来，而tensor就没有这个能力。
        但现在Variable已经被放弃使用了，因为tensor自己已经支持自动求导的功能了
        https://blog.csdn.net/rambo_csdn_123/article/details/119056123
        但 为什么 img和label需要梯度？？？
        """

        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        # running_loss += loss.data[0] * label.size(0)
        running_loss += loss.data * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        # running_acc += num_correct.data[0]
        running_acc += num_correct.data
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ========================= Log ======================
        # step = epoch * len(train_loader) + i

        # # (1) Log the scalar values
        # info = {'loss': loss.data[0], 'accuracy': accuracy.data[0]}
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, step)

        # # (2) Log values and gradients of the parameters (histogram)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary(tag, to_np(value), step)
        #     logger.histo_summary(tag + '/grad', to_np(value.grad), step)

        # # (3) Log the images
        # info = {'images': to_np(img.view(-1, 28, 28)[:10])}
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, step)
        # if i % 300 == 0:
        #     print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
        #         epoch + 1, num_epoches, running_loss / (batch_size * i),
        #         running_acc / (batch_size * i)))
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
            train_dataset))))

    # model.eval()
    # eval_loss = 0
    # eval_acc = 0
    # for data in test_loader:
    #     img, label = data
    #     if use_gpu:
    #         img = Variable(img, volatile=True).cuda()
    #         label = Variable(label, volatile=True).cuda()
    #     else:
    #         img = Variable(img, volatile=True)
    #         label = Variable(label, volatile=True)
    #     out = model(img)
    #     loss = criterion(out, label)
    #     eval_loss += loss.data[0] * label.size(0)
    #     _, pred = torch.max(out, 1)
    #     num_correct = (pred == label).sum()
    #     eval_acc += num_correct.data[0]
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    #     test_dataset)), eval_acc / (len(test_dataset))))

model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    # ???
    # if use_gpu:
    #     img = Variable(img, volatile=True).cuda()
    #     label = Variable(label, volatile=True).cuda()
    # else:
    #     img = Variable(img, volatile=True)
    #     label = Variable(label, volatile=True)
    with torch.no_grad():
        out = model(img)
        loss = criterion(out, label)
    # eval_loss += loss.data[0] * label.size(0)
    eval_loss += loss.data * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    # eval_acc += num_correct.data[0]
    eval_acc += num_correct.data
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc / (len(test_dataset))))

# 保存模型
torch.save(model.state_dict(), './cnn.pth')
