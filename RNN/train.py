import torch
import torch.utils.data as data
import nets
import torch.optim as optim
from torchvision import transforms,datasets
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])]
)

train_dataset = datasets.MNIST( root='./', train=True, transform=tf, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=1000, shuffle=True)

net = nets.net1().to(device)

opt = optim.Adam(net.parameters())
loss_f=nn.CrossEntropyLoss()
#loss_f=nn.NLLLoss()
#loss_f=nn.MSELoss()

for epoch in range(10000):
    losss=0
    net.train()
    for i, (l, label) in enumerate(train_loader):
        l=l.to(device)
        label=label.to(device)
        z=net(l)
        loss=loss_f(z,label)
        opt.zero_grad()
        loss.backward()
        opt.step()
#        label=torch.zeros(label.size()[0],10).scatter_(1,label.cpu().reshape(-1,1),1).to(device)
        losss=losss+loss.item()
        if i%10 == 9:
            print('loss: ',losss/10)
            losss=0
    print(epoch)
    torch.save(net,'./net1.pth')





