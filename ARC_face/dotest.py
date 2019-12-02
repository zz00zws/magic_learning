import torch,os
from torchvision import transforms
from torchvision.utils import save_image
import PIL.Image as pimg
import numpy as np

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tf = transforms.Compose(
#    [transforms.ToTensor(),
#    transforms.Normalize([0.5],[0.5])])
tf = transforms.Compose(
    [transforms.ToTensor()])
path='./sql/vic'

net1=torch.load('./pp2/net1.pth').to(device)

def mo(x):
    return torch.sqrt(torch.sum(x*x,1))

def cossame(x,y):
    xy=x@y.T
    x_y=mo(x)*mo(y)
    return xy.view(1)/x_y
with torch.no_grad():
    net1.eval()
    for l in os.listdir('./doimg'):
        img=pimg.open(os.path.join('./doimg',l)).convert('RGB')
        img=img.resize((96,96))
        img=tf(img).to(device).view(-1,3,96,96)
        out=net1(img).view(1,512)
        cos=0
        for i in os.listdir(path):
            sql=torch.tensor(np.load(os.path.join(path,i))).to(device)
            cos=max(cossame(sql,out).item(),cos)
        print(l,cos)
        
        
        









