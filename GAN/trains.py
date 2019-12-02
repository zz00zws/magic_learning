import torch,os
import torch.nn as nn
import torch.utils.data as data
import nets
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
import PIL.Image as pimg
import cfg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.ToTensor()])



class get_sample(data.Dataset):
    def __init__(self):
        self.img_path = []
        for name in os.listdir('./imgss'):
            data_path = os.path.join('./imgss',name)
            self.img_path.append(data_path)


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img=pimg.open(img_path)
        img=img.convert('RGB')
        data_img =tf(img)
        return data_img

getsample=get_sample()
train_loader = data.DataLoader(getsample, batch_size=cfg.batch_size, shuffle=True)

#dnet = torch.load('./dnet.pth').to(device)
#gnet = torch.load('./gnet.pth').to(device)
dnet = nets.ds().to(device)
gnet = nets.gs().to(device)
dopt = optim.Adam(dnet.parameters(),lr=0.0002,betas=(0.5,0.999))
gopt = optim.Adam(gnet.parameters(),lr=0.0002,betas=(0.5,0.999))
loss_f = nn.BCELoss()
    
losss=0
for epoch in range(10000):
    dnet.train()
    gnet.train()
    for i, l in enumerate(train_loader):
        rimg=l.to(device).float()
        z=torch.randn(rimg.size(0),256).to(device).float()
        fimg=gnet(z)
        rout=dnet(rimg)
        fout=dnet(fimg)
        rlabel=torch.ones(rout.size(0),1).to(device).float()
        flabel=torch.zeros(fout.size(0),1).to(device).float()
#        print(rlabel.size(),flabel.size(),rout.size(),fout.size())
        dloss1=loss_f(rout,rlabel)
        dloss2=loss_f(fout,flabel)
        dloss=dloss1+dloss2
        
        z=torch.randn(rimg.size(0),256).to(device).float()
        gfimg=gnet(z)
        fout=dnet(gfimg)
        gloss=loss_f(fout,rlabel)        
        
        dopt.zero_grad()
        dloss.backward()
        dopt.step()
        gopt.zero_grad()
        gloss.backward()
        gopt.step()
        if i%10==0 and i>1:
            print('epoch:',epoch,' i: ',i)
            print('dloss: ',dloss.item())
            print('gloss: ',gloss.item())
        if i%70==0 and i>1:
            save_image(fimg.data,"./img/fake/{}.png".format(epoch+i/1422),nrow=6)
            save_image(l.data,"./img/real/{}.png".format(epoch+i/1422),nrow=6)
    torch.save(dnet,'./dnet.pth')
    torch.save(gnet,'./gnet.pth')



