import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import PIL.Image as pimg
import PIL.ImageDraw as draw
import PIL.ImageFont as Font
import matplotlib.pyplot as plt

batch_size = 100
learning_rate = 0.02
num_epoches = 1
# font_path = r"E:\pycharm_project\pytorch_test\torch_learn\mnist_Data\msyh.ttf"
# 数据预处理：
# transforms.ToTensor()将图片转换成PyTorch中处理的对象Tensor,并且进行归一化（数据在0~1之间）
# transforms.Normalize()做标准化。它进行了减均值，再除以标准差。两个参数分别是均值和标准差
# transforms.Compose()函数则是将各种预处理的操作组合到了一起
data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0], [1])])


train_dataset = datasets.MNIST( root='./', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./', train=False, transform=data_tf,download=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.GRU(
            input_size=28,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Linear(128*2,10)
    def forward(self, x):
        y,(c_s,h_s) = self.lstm(x,None)
        # print(y.shape)
        # print(c_s.shape)
        # print(h_s.shape)
        out = self.out(y[:,-1,:])
        # print(out.shape)
        return out


net = Net()

if torch.cuda.is_available():
    net = net.cuda()

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

net.train()
plt.ion()

a=[]
b=[]
c=[]
for epoch in range(num_epoches):
    for i,(img,label) in enumerate(train_loader):
        # print(img.size())
        #  img = img.view(img.size(0), -1)
        # img = img.reshape(-1, 784)#转换形状
        img = img.reshape(img.size(0),28, -1)#转换形状
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        out = net(img)
        # loss = loss_fn(net.y3, label)
        loss = loss_fn(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('epoch: {},i: {}, loss: {:.3}'.format(epoch,i, loss.data.item()))#损失值显示3位

#
## 模型评估,此模式下，会固定模型中的BN层和Drpout层。
#net.eval()
#eval_loss = 0
#eval_acc = 0
#for data in test_loader:
#    img, label = data
#    img = img.reshape(img.size(0),28, -1)
#    if torch.cuda.is_available():
#        img = img.cuda()
#        label = label.cuda()
#    out = net(img)
#    # loss = loss_fn(net.y3, label)
#    loss = loss_fn(out, label)
#    # 平均损失*批次=每批数据的损失，  每批数据的损失*循环次数（+=叠加）=测试数据集的总损失
#    eval_loss += loss.item() * label.size(0)
#    argmax = torch.argmax(out,1)#返回每行中最大值在每行中的索引
#    print(argmax,'************')
#
#    num_acc = (argmax == label).sum()#统计每批数据的精度
#    eval_acc += num_acc.item()#每批的精度*循环次数（+=叠加）=测试数据集的总精度
#'已经评估完所有测试集数据'
#
## print(torch.argmax(out,1))#返回每行中最大值在每行中的索引
## print(label)
## print(torch.max(out,1))#返回每行中的最大值和最大值在每行中的索引
#print('Test Loss: {:.3}, Acc: {:.3}'
#    .format(eval_loss / (len(test_dataset)),#计算所有测试数据集里的平均损失
#    eval_acc / (len(test_dataset))))#计算所有测试数据集里的平均精度
