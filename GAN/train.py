import torch
import torch.nn as nn
import torch.utils.data as data
import nets
import torch.optim as optim
from torchvision import transforms,datasets
from torchvision.utils import save_image
import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.ToTensor()
    ,transforms.Normalize([0.5],[0.5])])

train_dataset = datasets.MNIST( root='./', train=True, transform=tf, download=True)
train_loader = data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

dnet = nets.d().to(device)
gnet = nets.g().to(device)
dopt = optim.Adam(dnet.parameters())
gopt = optim.Adam(gnet.parameters())
loss_f = nn.BCELoss()

losss=0
for epoch in range(10000):
    dnet.train()
    gnet.train()
    for i, (l, label) in enumerate(train_loader):
        rimg=l.to(device).float()
        z=torch.randn(cfg.batch_size,128).to(device).view(-1,128).float()
        fimg=gnet(z)
        rout=dnet(rimg)
        fout=dnet(fimg)
        rlabel=torch.ones(cfg.batch_size,1).to(device).float()
        flabel=torch.zeros(cfg.batch_size,1).to(device).float()
#        print(rlabel.size(),flabel.size(),rout.size(),fout.size())
        dloss1=loss_f(rout,rlabel)
        dloss2=loss_f(fout,flabel)
        dloss=dloss1+dloss2
        
        z=torch.randn(128*cfg.batch_size).to(device).view(-1,128).float()
        gfimg=gnet(z)
        fout=dnet(gfimg)
        gloss=loss_f(fout,rlabel)        
        
        dopt.zero_grad()
        dloss.backward()
        dopt.step()
        gopt.zero_grad()
        gloss.backward()
        gopt.step()
        print('1')
        save_image(fimg.data,"./img/fake/{}.png".format(epoch+i/600),nrow=10)
        save_image(l.data,"./img/real/{}.png".format(epoch+i/600),nrow=10)


