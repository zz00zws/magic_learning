import torch.nn as nn
import torch
import cfg

#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#class conv(nn.Module):
#    def __init__(self,x,y,z,p=0,s=1,b=True):
#        super().__init__()
#        self.block1=nn.Sequential(  
#            nn.Conv2d(x,y,z,padding=p,stride=s),
#            nn.BatchNorm2d(y),
#            nn.LeakyReLU(negative_slope=0.3, inplace=True)
#            )
#    def forward(self,y):
#        return self.block1(y)

class conv(nn.Module):
    def __init__(self,x,y,z,p=0,s=1,b=True):
        super().__init__()
        self.block1=nn.Sequential(  
            nn.Conv2d(x,y,z,padding=p+1,stride=s,dilation=2),
            nn.BatchNorm2d(y),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
            )
    def forward(self,y):
        return self.block1(y)
    
class conv3(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.block1=nn.Sequential(  
            conv(x,2*y,3,1,1),
            conv(2*y,2*y,3,1,1)
            )
        self.dout=conv(2*y,2*y,3,1,2)
        self.cout=conv(2*y,2*y,3,1,1)
        self.uout=nn.ConvTranspose2d(2*y,y,3,2,1,1)
    def forward(self,y):
        y=self.block1(y)
        return self.dout(y),self.cout(y),self.uout(y)
    
class conv2(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.block1=nn.Sequential(  
            conv(x,2*y,3,1,1),
            conv(2*y,2*y,3,1,1)
            )
        self.dout=conv(2*y,2*y,3,1,2)
        self.cout=nn.Sequential(  
            conv(2*y,y,3,1,1),
            nn.Sigmoid()
            )
#        self.cout=nn.Sequential(  
#            conv(2*y,y,3,1,1)
#            )
    def forward(self,y):
        y=self.block1(y)
        return self.dout(y),self.cout(y)
    
class unetpp(nn.Module):
    def __init__(self):
        super().__init__()
        self.x00=conv3(cfg.inp_num,cfg.deep_base)
        self.x01=conv2(cfg.deep_base*4,cfg.class_num)
        self.x02=conv2(cfg.deep_base*4+cfg.class_num,cfg.class_num)
        self.x03=conv2(cfg.deep_base*4+cfg.class_num*2,cfg.class_num)
        self.x04=conv2(cfg.deep_base*4+cfg.class_num*3,cfg.class_num)
        self.x10=conv3(cfg.deep_base*2,cfg.deep_base*2)
        self.x11=conv3(cfg.deep_base*8,cfg.deep_base*2)
        self.x12=conv3(cfg.deep_base*12,cfg.deep_base*2)
        self.x13=conv3(cfg.deep_base*16,cfg.deep_base*2)
        self.x20=conv3(cfg.deep_base*4,cfg.deep_base*4)
        self.x21=conv3(cfg.deep_base*16,cfg.deep_base*4)
        self.x22=conv3(cfg.deep_base*24,cfg.deep_base*4)
        self.x30=conv3(cfg.deep_base*8,cfg.deep_base*8)
        self.x31=conv3(cfg.deep_base*32,cfg.deep_base*8)
        self.x40=conv3(cfg.deep_base*16,cfg.deep_base*16)
    def forward(self,x):
        d00,c00,u00=self.x00(x)
        d10,c10,u10=self.x10(d00)
        d20,c20,u20=self.x20(d10)
        d30,c30,u30=self.x30(d20)
        d40,c40,u40=self.x40(d30)
        d31,c31,u31=self.x31(torch.cat((c30,u40),1))
        d21,c21,u21=self.x21(torch.cat((c20,u30),1))
        d11,c11,u11=self.x11(torch.cat((c10,u20),1))
        d01,c01=self.x01(torch.cat((c00,u10),1))
        d22,c22,u22=self.x22(torch.cat((c20,c21,u31),1))
        d12,c12,u12=self.x12(torch.cat((c10,c11,u21),1))
        d02,c02=self.x02(torch.cat((c00,c01,u11),1))
        d13,c13,u13=self.x13(torch.cat((c10,c11,c12,u21),1))
        d03,c03=self.x03(torch.cat((c00,c01,c02,u11),1))
        d04,c04=self.x04(torch.cat((c00,c01,c02,c03,u11),1))
        return c01.permute(0,3,2,1).reshape(-1,cfg.class_num),c02.permute(0,3,2,1).reshape(-1,cfg.class_num),c03.permute(0,3,2,1).reshape(-1,cfg.class_num),c04.permute(0,3,2,1).reshape(-1,cfg.class_num)


if __name__ == '__main__':
    net=unetpp().to(device)
    z=torch.randn(1,1,411,234).to(device)
    x,y,z,s=net(z)
#    net2=fc()
#    z=torch.randn(5,256)
#    b=net2(z)
    print(x.size(),y.size(),z.size(),s.size())    
    




























































