import torch.nn as nn
import torch
import cfg

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class conv(nn.Module):
    def __init__(self,x,y,z,p=0,s=1,b=True):
        super().__init__()
        self.block1=nn.Sequential(  
            nn.Conv2d(x,y,z,padding=p,stride=s,bias=b),
            nn.BatchNorm2d(y),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
            )
    def forward(self,y):
        return self.block1(y)
    
class conv2(nn.Module):
    def __init__(self,x,y,b=True):
        super().__init__()
        self.block1=nn.Sequential(  
            conv(x,y,3,1,1,b),
            conv(y,y,3,1,1,b)
            )
        self.dout=conv(y,y,3,1,2,b)
        self.cout=conv(y,y,3,1,1,b)
    def forward(self,y):
        y=self.block1(y)
        return self.dout(y),self.cout(y)

class up(nn.Module):
    def __init__(self,x,y,p=0,s=1,b=True):
        super().__init__()
        self.block1=nn.Sequential(
            conv(x,2*y,3,1,1,b),
            conv(2*y,2*y,3,1,1,b),
            nn.ConvTranspose2d(2*y,y,3,2,1,1)
            )
    def forward(self,y):
        return self.block1(y)
    
class unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1=conv2(1,64)
        self.d2=conv2(64,128)
        self.d3=conv2(128,256)
        self.d4=conv2(256,512)
        self.dd=nn.Sequential(  
            conv(512,1024,3,1,1),
            conv(1024,1024,3,1,1),
            nn.ConvTranspose2d(1024,512,3,2,1,1)
            )
        self.u4=up(1024,256)
        self.u3=up(512,128)
        self.u2=up(256,64)
        self.u1=nn.Sequential(  
            conv(128,64,3,1,1),
            conv(64,64,3,1,1),
            conv(64,cfg.class_num,3,1,1),
            nn.Sigmoid()
            )
    def forward(self,y):
        d1,c1=self.d1(y)
        d2,c2=self.d2(d1)
        d3,c3=self.d3(d2)
        d4,c4=self.d4(d3)
        dd=self.dd(d4)
        u4=torch.cat((c4,dd),1)
        u4=self.u4(u4)
        u3=torch.cat((c3,u4),1)
        u3=self.u3(u3)
        u2=torch.cat((c2,u3),1)
        u2=self.u2(u2)
        u1=torch.cat((c1,u2),1)
        u1=self.u1(u1)
        u1=u1.permute(0,2,3,1)
        return u1

if __name__ == '__main__':
    net=unet().to(device)
    z=torch.randn(1,1,512,512).to(device)
    z=net(z)
#    net2=fc()
#    z=torch.randn(5,256)
#    b=net2(z)
    print(z.size())    
    




























































