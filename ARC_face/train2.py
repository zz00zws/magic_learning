import torch,os
import torch.nn as nn
import torch.utils.data as data
import net2 as nets
import torch.optim as optim
from torchvision import transforms,models
import PIL.Image as pimg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose(
    [transforms.ToTensor()])

class get_sample(data.Dataset):
    def __init__(self):
        self.img_path = []
        self.label_path = []
        for s,i in enumerate(os.listdir('./data/cut')):
            for j in os.listdir(os.path.join('./data/cut',i)):
#                if s <=1000:
                data_path=os.path.join('./data/cut',i,j)
                self.img_path.append(data_path)
                self.label_path.append(s)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        label_path = self.label_path[index]
        img=pimg.open(img_path)
        img=img.resize((96,96))
        img=img.convert('RGB')
        data_img =tf(img)
        return data_img,label_path

getsample=get_sample()
train_loader = data.DataLoader(getsample, batch_size=170, shuffle=True)

net1 = models.resnet50().to(device)
net2 = nets.fc().to(device)
#net1=torch.load('./net1.pth').to(device)
#net2=torch.load('./net2.pth').to(device)
opt1 = optim.SGD(net1.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0005)
opt2 = optim.SGD(net2.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0005)
#opt1 = optim.Adam(net1.parameters(),lr=0.1,weight_decay=0.0005)
#opt2 = optim.Adam(net2.parameters(),lr=0.1,weight_decay=0.0005)
#loss_f = nn.CrossEntropyLoss()
loss_f = nn.NLLLoss()
#loss_f = nn.MSELoss()

print('start')
net1.train()
net2.train()
for epoch in range(1000):
    ii=0
    losss=0
    for i, (l,label) in enumerate(train_loader):
        l=l.to(device)
        label=torch.tensor(label).to(device).long()
        out=net1(l)
#        out=net2(out,label)
        out=net2(out)
#        print(torch.max(out,1))
        loss=loss_f(out,label)
        opt1.zero_grad()
        opt2.zero_grad()
        loss.backward()
        opt1.step()
        opt2.step()
        losss+=loss.item()
        ii+=1
        print(i%100,end=' ')
        if i%100 == 0 and i>1:
            acc=torch.argmax(out,dim=1)
            acc1=torch.sum(acc==label)
            print()
            print('acc: ',acc1.item())
            print('epoch: ',epoch,' i: ',i)
            print(losss/ii)
            losss=0
            ii=0
            torch.save(net1,'./net1.pth')
            torch.save(net2,'./net2.pth')









