import torch,os
import torch.nn as nn
import torch.utils.data as data
from netp import unetpp
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
import PIL.Image as pimg
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])])
tfl = transforms.Compose(
    [transforms.ToTensor()])

class get_sample(data.Dataset):
    def __init__(self):
        self.img_path = []
        self.label_path = []
        self.name = []
        for name in os.listdir('./data/2d_images'):
            data_path = os.path.join('./data/2d_images',name)
            l_path = os.path.join('./data/2d_masks',name)
            self.img_path.append(data_path)
            self.label_path.append(l_path)
            self.name.append(name)


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img=pimg.open(img_path)
        img=img.convert('L')
        img=img.resize((416,416))
        data_img =tf(img)
        label_path = self.label_path[index]
        imgl=pimg.open(label_path)
        imgl=imgl.convert('L')
        imgl=imgl.resize((416,416))
        label=tfl(imgl)
#        label=torch.tensor(np.array(imgl)).float()
        name=self.name[index]
        return data_img,label,name.split('.')[0]+'.png'

getsample=get_sample()
train_loader = data.DataLoader(getsample, batch_size=1, shuffle=True)

net = unetpp().to(device)
#net=torch.load('./net3.pth')
#opt = optim.SGD(net.parameters(),lr=0.001)
opt = optim.Adam(net.parameters())
#loss_f = nn.CrossEntropyLoss()
#loss_f = nn.MSELoss()
loss_f = nn.BCELoss()

for epoch in range(10000):
    net.train()
    for i, (l,label,name) in enumerate(train_loader):
        l=l.to(device)
        label=label.to(device)
        O1,O2,O3,out=net(l)
#        print(out,label[:,222])
#        print(out.size(),label.size())
        loss=loss_f(O1.view(-1,416,416).permute(0,2,1),label)+loss_f(O2.view(-1,416,416).permute(0,2,1),label)+loss_f(O3.view(-1,416,416).permute(0,2,1),label)+loss_f(out.view(-1,416,416).permute(0,2,1),label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
        l = l.data
        out=(out+0.5).int().float()
        O1=(O1+0.5).int().float()
        O2=(O2+0.5).int().float()
        O3=(O3+0.5).int().float()
#        print(out.size(),label.size())
#        save_image(out.view(-1,416,416).permute(0,2,1),"./img_out/fake/{}.png".format(epoch+i/267),nrow=3)
        save_image(out.view(-1,416,416).permute(0,2,1),"./img_out/fake/"+name[0],nrow=3,normalize=True,scale_each=True)
        save_image(O1.view(-1,416,416).permute(0,2,1),"./img_out/fake1/"+name[0],nrow=3,normalize=True,scale_each=True)
        save_image(O2.view(-1,416,416).permute(0,2,1),"./img_out/fake2/"+name[0],nrow=3,normalize=True,scale_each=True)
        save_image(O3.view(-1,416,416).permute(0,2,1),"./img_out/fake3/"+name[0],nrow=3,normalize=True,scale_each=True)
        save_image(l,"./img_out/real/"+name[0],nrow=3)
        save_image(label,"./img_out/label/"+name[0],nrow=3)
    torch.save(net,'./nets.pth')



