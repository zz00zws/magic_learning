import torch.nn as nn
import torch

        
#class d(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.block1=nn.Sequential(           
#            nn.Conv2d(1,64,2,bias=False),#28
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            
#            nn.Conv2d(64,128,4,bias=False),#27
#            nn.BatchNorm2d(128),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            
#            nn.Conv2d(128,128,2,bias=False),#23
#            nn.BatchNorm2d(128),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            
#            nn.Conv2d(128,256,4,bias=False),#20
#            nn.BatchNorm2d(256),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            
#            nn.Conv2d(256,256,4,2,bias=False),#9
#            nn.BatchNorm2d(256),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            
#            nn.Conv2d(256,512,4,bias=False),#3
#            nn.BatchNorm2d(512),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            
#            nn.Conv2d(512,512,4,2,bias=False),#1
#            nn.BatchNorm2d(512),
#            nn.LeakyReLU(negative_slope=0.2, inplace=True),
#            
#            nn.Conv2d(512,1,2,bias=False),#1
#            nn.Sigmoid()
#            )
#        
#    def forward(self,x):
#        x=self.block1(x)
#        return x.view(-1,1)
#        
#    
#class g(nn.Module):
#    def __init__(self):
#        super().__init__()
#        
#        self.fc=nn.Linear(128,2*2*128,bias=False)
#        
#        self.block1=nn.Sequential(           
#            nn.ConvTranspose2d(128,512,4,2,bias=False),#6
#            nn.BatchNorm2d(512),
#            nn.ReLU(inplace=True),
#            
#            nn.ConvTranspose2d(512,256,4,bias=False),#9
#            nn.BatchNorm2d(256),
#            nn.ReLU(inplace=True),
#            
#            nn.ConvTranspose2d(256,256,4,2,bias=False),#20
#            nn.BatchNorm2d(256),
#            nn.ReLU(inplace=True),
#            
#            nn.ConvTranspose2d(256,128,4,bias=False),#23
#            nn.BatchNorm2d(128),
#            nn.ReLU(inplace=True),
#                       
#            nn.ConvTranspose2d(128,128,2,bias=False),#24
#            nn.BatchNorm2d(128),
#            nn.ReLU(inplace=True),
#            
#            nn.ConvTranspose2d(128,64,4,bias=False),#27
#            nn.BatchNorm2d(64),
#            nn.ReLU(inplace=True),
#            
#            nn.ConvTranspose2d(64,1,2,bias=False),#28
#            nn.Sigmoid()
#            )
#        
#
#    def forward(self,x):
#        x = self.fc(x).view(-1,128,2,2)
#        x=self.block1(x)
#        return x
#    
    
class ds(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(           
            nn.Conv2d(3,64,4,1,1,bias=False),#95
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(64,128,4,2,1,bias=False),#47
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(128,128,4,2,1,bias=False),#23
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(128,256,4,2,1,bias=False),#11
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(256,256,4,bias=False),#8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(256,512,4,2,bias=False),#3
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(512,512,2,bias=False),#2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(512,1,2),#1
            nn.Sigmoid()
            )
        
    def forward(self,x):
        x=self.block1(x)
        return x#.view(-1,1)
        
    
class gs(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc=nn.Linear(256,2*2*128,bias=False)
        
        self.block1=nn.Sequential(           
            nn.ConvTranspose2d(128,512,4,1,1,bias=False),#3
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512,256,4,2,bias=False),#8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256,256,4,bias=False),#11
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256,128,4,2,1,1,bias=False),#23
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
                       
            nn.ConvTranspose2d(128,128,4,2,1,1,bias=False),#47
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128,64,4,2,1,1,bias=False),#95
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64,3,4,1,1,bias=False),#96
            nn.Sigmoid()
            )
        

    def forward(self,x):
        x = self.fc(x).view(-1,128,2,2)
        x=self.block1(x)
        return x    
if __name__ == '__main__':
    net=gs()
    net2=ds()
    z=torch.randn(256*10).view(-1,256)
    b=net(z)
    c=net2(b)
    print(b.size(),c.size())
#    print(b.size())
        
        
        
        
        
        
        
        
        
        
        
