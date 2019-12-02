import torch,os
from torchvision import transforms
import PIL.Image as pimg
import numpy as np

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tf = transforms.Compose(
#    [transforms.ToTensor(),
#    transforms.Normalize([0.5],[0.5])])
tf = transforms.Compose(
    [transforms.ToTensor()])

net1=torch.load('./pp2/net1.pth').to(device)

with torch.no_grad():
    net1.eval()
    for l in os.listdir('./sql/img'):
        img=pimg.open(os.path.join('./sql/img',l)).convert('RGB')
        img=img.resize((96,96))
        img=tf(img).to(device).view(-1,3,96,96)
        out=net1(img)
        vic=out.cpu().detach().numpy()
        np.save(os.path.join('./sql/vic',l+'.npy'),vic)









