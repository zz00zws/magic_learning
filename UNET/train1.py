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
        for name in os.listdir('./data/label2'):
            data_path = os.path.join('./data/img',name)
            l_path = './data/label/'+ name
            self.img_path.append(data_path)
            self.label_path.append(l_path)


    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img=pimg.open(img_path)
        img=img.convert('RGB')
        data_img =tf(img)
        label_path = self.label_path[index]
        return data_img,label_path

def change(x):
    x=pimg.open(x)
    x=torch.tensor(np.array(x))
    x[x!=0]=1
    return x

getsample=get_sample()
train_loader = data.DataLoader(getsample, batch_size=1, shuffle=True)

net = unetpp().to(device)
#net=torch.load('./nets.pth').to(device)
opt = optim.SGD(net.parameters(),lr=0.001,momentum=0.9,weight_decay=0.00001)
#opt = optim.Adam(net.parameters())
#loss_f = nn.CrossEntropyLoss()
#loss_f = nn.MSELoss()
loss_f = nn.BCELoss()

for epoch in range(10000):
    net.train()
    try:
        for i, (l,label) in enumerate(train_loader):
            ll=label
            label=change(label[0]).float()
            x,y=label.size(0),label.size(1)
            label=label.to(device).view(-1,1)
    #        print(label.size())
            l=l.to(device)
            O1,O2,O3,out=net(l)
            loss=loss_f(O1,label)+loss_f(O2,label)+loss_f(O3,label)+loss_f(out,label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print('loss: ',loss.item())
            print('epoch: ',epoch,'i: ',i)
            l = l.data
            out=(out+0.5).int().float()
    #        print(out.size(),label.size())
    #        save_image(out.view(-1,416,416).permute(0,2,1),"./img_out/fake/{}.png".format(epoch+i/267),nrow=3)
            save_image(out.view(-1,y,x).permute(0,2,1),'./img_out3/'+ll[0].split('/')[-1],normalize=True,scale_each=True)
        torch.save(net,'./nets.pth')
    except:
        continue



