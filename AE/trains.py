import torch,os
import torch.nn as nn
import torch.utils.data as data
from nets import mains
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
import PIL.Image as pimg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tf = transforms.Compose(
#    [transforms.ToTensor(),
#    transforms.Normalize([0.5],[0.5])])
tf = transforms.Compose(
    [transforms.ToTensor()])



class get_sample(data.Dataset):
    def __init__(self):
        self.img_path = []
        for name in os.listdir('./imgs'):
            data_path = os.path.join('./imgs',name)
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
train_loader = data.DataLoader(getsample, batch_size=250, shuffle=True)

net = mains().to(device)
#net=torch.load('./net3.pth')
#opt = optim.SGD(net.parameters(),lr=0.001)
opt = optim.Adam(net.parameters())
#loss_f = nn.CrossEntropyLoss()
loss_f = nn.MSELoss()

losss=0
for epoch in range(10000):
    net.train()
    for i, l in enumerate(train_loader):
        l=l.to(device)
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
            save_image(out,"./img_out/fake/{}.png".format(epoch+i/45000*250*30),nrow=10)
            save_image(l,"./img_out/real/{}.png".format(epoch+i/45000*250*30),nrow=10)
    torch.save(net,'./nets.pth')



