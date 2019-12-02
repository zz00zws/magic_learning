import torch
import torch.nn as nn
import math

#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Arcface(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)).to(device))

    def forward(self, x):
        x_norm = nn.functional.normalize(x, dim=1)
        w_norm = nn.functional.normalize(self.w, dim=0)
        cosa = torch.matmul(x_norm, w_norm) / 1.5
        a = torch.acos(cosa)
        m=0.5
#        arcsoftmax = torch.exp(torch.cos(a + m) * 10) / (torch.sum(torch.exp(cosa * 10), dim=1, keepdim=True) - 
#           torch.exp(cosa * 10) + torch.exp(torch.cos(a + m) * 10))
        arcsoftmax = torch.exp(torch.cos(a + m)*1.5) / (torch.sum(torch.exp(cosa*1.5), dim=1, keepdim=True) - 
           torch.exp(cosa*1.5) + torch.exp(torch.cos(a + m)*1.5))
        return arcsoftmax*64

class arcface(nn.Module):
    def __init__(self, in_features, out_features):
        super(arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        m = torch.tensor([0.5]).to(device)
        self.m = m
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, input, label):
#        _w = torch.norm(self.weight,dim=0).view(-1,512)
#        _x = torch.norm(input,dim=1).view(-1,1)
#        out = torch.matmul(input,self.weight.permute(1,0))
#        cosine = out/(_w*_x)
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.001 - cosine**2)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > 0, phi, cosine)
        one_hot = torch.zeros(cosine.size()).to(device).scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
        return output*64
        
class conv(nn.Module):
    def __init__(self,x,y,z,p=0,s=1,b=True,g=1):
        super().__init__()
        self.block1=nn.Sequential(  
            nn.Conv2d(x,y,z,padding=p,stride=s,bias=b,groups=g),
            nn.BatchNorm2d(y),
            nn.LeakyReLU(negative_slope=0.3, inplace=True)
            )
    def forward(self,y):
        return self.block1(y)
    
class pep(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.block1=nn.Sequential(           
            conv(x,y,1),
            conv(y,2*y,1),
            conv(2*y,2*y,3,1,g=y),
            conv(2*y,x,1)
            )
    def forward(self,z):
        return self.block1(z)+z
    
    
class ep(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.block1=nn.Sequential(           
            conv(x,y,1),
            conv(y,y,3,1,2,g=y),
            conv(y,y,1)
            )
    def forward(self,z):
        return self.block1(z)
    
class ep2(nn.Module):
    def __init__(self,x,y):
        super().__init__()
        self.block1=nn.Sequential(           
            conv(x,y,1),
            conv(y,y,3,1,g=y),
            conv(y,y,1)
            )
    def forward(self,z):
        return self.block1(z)
    
class fca(nn.Module):
    def __init__(self, c, r):
        super().__init__()
        self.c = c
        self.r = r

        hidden_channels = c // r
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c, hidden_channels, bias=False),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(hidden_channels, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out    
class main(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(           
            conv(3,12,3,1),#96
            conv(12,24,3,1,2),#48
            pep(24,7),
            ep(24,70),
            pep(70,24),
            pep(70,25),
            ep(70,150),
            pep(150,56),
            conv(150,150,1)
            )
        
        self.fca1=fca(150,8)
        
        self.block2=nn.Sequential(  
            pep(150,73),
            pep(150,71),
            pep(150,75),
            ep(150,325),
            pep(325,132),
            pep(325,124),
            pep(325,141),
            pep(325,140),
            pep(325,137),
            pep(325,135),
            pep(325,133),
            pep(325,140),
            ep(325,545),
            pep(545,276),
            conv(545,230,1),
            ep2(230,489),
            pep(489,213),
            conv(489,189,1),
            conv(189,512,3)
            )
        
        self.block3=nn.Sequential(           
            nn.Linear(512,512),
            nn.BatchNorm1d(512)
            )
        
    def forward(self,x):
        x=self.block1(x)
        x=self.fca1(x)
        x=self.block2(x)
        x=self.block3(x.view(-1,512))
        return x
    

class fc(nn.Module):
    def __init__(self):
        super().__init__()

        self.a_softmax=Arcface(512,2800)
        
    def forward(self,x):
        x=self.a_softmax(x)
        x=torch.log(x)
        return x       

#class fc(nn.Module):
#    def __init__(self):
#        super().__init__()
#
#        self.a_softmax=arcface(512,2800)
#        
#    def forward(self,x,l):
#        x=self.a_softmax(x,l)
#        x=torch.log(x)
#        return x       
            
if __name__ == '__main__':
    net=main()
    z=torch.randn(5,3,96,96)
    z=net(z)
#    net2=fc()
#    z=torch.randn(5,256)
#    b=net2(z)
    print(z.size())    
    
    
    
    
    
    
    
    
    
    
    
    