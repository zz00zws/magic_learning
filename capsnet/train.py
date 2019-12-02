import torch
import torch.utils.data as data
import nets
import torch.optim as optim
from torchvision import transforms,datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])]
)

train_dataset = datasets.MNIST( root='./', train=True, transform=tf, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=100, shuffle=True)

net = nets.CapsuleNet([1,28,28],10,3).to(device)

#opt = optim.SGD(net.parameters(),lr=0.1)
opt = optim.Adam(net.parameters())

for epoch in range(10000):
    losss=0
    net.train()
    for i, (l, label) in enumerate(train_loader):
        l=l.to(device)
        label=torch.zeros(label.size(0),10).scatter_(1,label.cpu().reshape(-1,1),1).to(device)
        x,z=net(l)
        print('label: ',torch.argmax(label[0:10],dim=1).cpu().numpy().tolist())
        print('out:   ',torch.argmax(x[0:10],dim=1).cpu().numpy().tolist())
        print('acc:  ',torch.sum(torch.argmax(x,dim=1)==torch.argmax(label,dim=1)).item()/label.size(0))
        print()
        loss=nets.caps_loss(label,x,l,z,0.0005 * 784)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losss=losss+loss.item()
        if i%10 == 9:
            print('loss: ',losss/10)
            losss=0
    print(epoch)
    torch.save(net,'./net1.pth')





