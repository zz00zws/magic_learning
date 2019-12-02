import torch,os
import torch.nn as nn
import torch.utils.data as data
from netp import unetpp
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image
import PIL.Image as pimg
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
#tf = transforms.Compose(
#    [transforms.ToTensor(),
#    transforms.Normalize([0.5],[0.5])])

tf = Compose([ToTensor()])

class get_sample(data.Dataset):
    def __init__(self):
        self.img_path = []
        self.label_path = []
        for name in os.listdir('./data/label'):
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
    x[x==255]=0
    return x

getsample=get_sample()
train_loader = data.DataLoader(getsample, batch_size=1, shuffle=True)

#net = unetpp().to(device)
net=torch.load('./nets.pth').to(device)
opt = optim.SGD(net.parameters(),lr=0.0001,momentum=0.9,weight_decay=0.00002)
#opt = optim.Adam(net.parameters())
loss_f = nn.CrossEntropyLoss()
#loss_f = nn.MSELoss()
#loss_f = nn.BCELoss()

for epoch in range(10000):
    net.train()
    try:
        for i, (l,label) in enumerate(train_loader):
            ll=label
            label=change(label[0]).long()
            x,y=label.size(0),label.size(1)
            label=label.to(device).view(-1)
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
#            print(out)
            out=torch.argmax(out,1).view(x,y)
#            print(out)
            imgout=pimg.fromarray(10*out.cpu().detach().numpy().astype('uint8')).convert('P')
            imgout.save('./img_out2/'+ll[0].split('/')[-1])
    #        print(imgout)
        torch.save(net,'./nets.pth')
    except:
        continue



