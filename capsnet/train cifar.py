import torch
import torch.utils.data as data
import nets
import torch.optim as optim
from torchvision import transforms,datasets
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])]
)

train_dataset = datasets.CIFAR10( root='./', train=True, transform=tf, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=80, shuffle=True)

net = nets.CapsuleNet([3,32,32],10,3).to(device)

#opt = optim.SGD(net.parameters(),lr=0.0001)
opt = optim.Adam(net.parameters())

for epoch in range(10000):
    losss=0
    net.train()
    for i, (l, label) in enumerate(train_loader):
#        print(label)
#        print(l.size())
        l=l.to(device)
        label=torch.zeros(label.size(0),10).scatter_(1,label.cpu().reshape(-1,1),1).to(device)
        x,z=net(l)
        print('label: ',torch.argmax(label[0:5],dim=1).cpu().numpy().tolist())
        print('out:   ',torch.argmax(x[0:5],dim=1).cpu().numpy().tolist())
        print('acc:  ',torch.sum(torch.argmax(x,dim=1)==torch.argmax(label,dim=1)).item()/label.size(0))
        print()
#        print(label.size(),x.size(),l.size(),z.size())
        loss=nets.caps_loss(label,x,l,z,0.0005*32*32)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losss=losss+loss.item()
        if i%10 == 9:
            print('loss: ',losss/10)
            losss=0
    print(epoch)
    torch.save(net,'./net2.pth')





