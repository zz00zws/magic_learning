import torch,os
from torchvision import transforms
import PIL.Image as pimg
import PIL.ImageDraw as draw
import numpy as np
import cv2
import mtcnn as mt
import utils as ut

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


#vc = cv2.VideoCapture('./data/test.mp4')
#c=1
#if vc.isOpened():
#    rval,frame=vc.read()
#else:
#    rval=False
#    print(1)
#while rval:
#    print(2,end='')
#    rval,frame=vc.read()
#    cv2.imwrite('./img/'+str(c)+'.jpg',frame)
#    c=c+1
#    cv2.waitKey(1)

with torch.no_grad():
    net1.eval()
    for l in os.listdir('./img'):
        print(l,end=' ')
        img=pimg.open(os.path.join('./img',l)).convert('RGB')
        imgd=draw.ImageDraw(img)
        oms=mt.test_all(img)
        kks=ut.crop(img,oms)
        for j,im in enumerate(kks):
#            print('kk ',len(kks))
            im=im.resize((96,96))
            im=tf(im).to(device).view(-1,3,96,96)
            out=net1(im).view(1,512)
            cos=0
            for i in os.listdir(path):
                sql=torch.tensor(np.load(os.path.join(path,i))).to(device)
                cos=max(cossame(sql,out).item(),cos)
#            print('cos ',cos)
            if cos>=0.8:
                imgd.rectangle(oms[j,1:5].cpu().detach().numpy().tolist(),fill=None,outline="green")
            else:
                imgd.rectangle(oms[j,1:5].cpu().detach().numpy().tolist(),fill=None,outline="red")
        img.save('./img_out/'+l)
        
fps = 30
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_writer = cv2.VideoWriter(filename='./data/test.avi', fourcc=fourcc, fps=fps, frameSize=(544, 960))
for i,j in enumerate(os.listdir('./img_out')):
    print(1,end='')
    img = cv2.imread(filename='./img_out/'+str(i)+'.jpg')
    cv2.waitKey(100)
    video_writer.write(img)
        









