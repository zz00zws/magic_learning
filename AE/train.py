import torch
import torch.nn as nn
import torch.utils.data as data
from nets import main
import torch.optim as optim
from torchvision import transforms,datasets
from torchvision.utils import save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])]
)

train_dataset = datasets.MNIST( root='./', train=True, transform=tf, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=400, shuffle=True)

net = main().to(device)
#net=torch.load('./net3.pth')
#opt = optim.SGD(net.parameters(),lr=0.001)
opt = optim.Adam(net.parameters())
#loss_f = nn.CrossEntropyLoss()
loss_f = nn.MSELoss()

losss=0
for epoch in range(10000):
    net.train()
    for i, (l, label) in enumerate(train_loader):
        l=l.to(device)
        label=label.to(device)
        out=net(l)
        loss=loss_f(out,l)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losss+=loss.item()
        if i%30 == 0 and i>1:
            print(losss/100)
            losss=0
            l = l.data
            save_image(out,"./img/fake-{}.png".format(epoch+i/60000),nrow=10)
            save_image(l,"./img/real-{}.png".format(epoch+i/60000),nrow=10)
    torch.save(net,'./net.pth')



