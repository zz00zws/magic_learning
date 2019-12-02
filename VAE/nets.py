import torch.nn as nn
import torch

        
class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(           
            nn.Conv2d(1,64,3,padding=1),#28
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.Conv2d(64,128,3),#26
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            nn.MaxPool2d(2),#13
            
            nn.Conv2d(128,256,3),#11
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.MaxPool2d(3,2),#5
            
            nn.Conv2d(256,512,3),#3
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            nn.Conv2d(512,2,3),#1
            nn.BatchNorm2d(2),
            nn.PReLU()
            )
        
    def forward(self,x):
        x=self.block1(x)
        m,s=x[:,:1],x[:,1:]
        return s,m
        
    
class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(           
            nn.ConvTranspose2d(128,512,3),#3
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            nn.ConvTranspose2d(512,256,3),#5
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.ConvTranspose2d(256,256,3,2),#11
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.ConvTranspose2d(256,128,3),#13
            nn.BatchNorm2d(128),
            nn.PReLU(),
                       
            nn.ConvTranspose2d(128,128,2,2),#26
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            nn.ConvTranspose2d(128,64,3),#28
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.ConvTranspose2d(64,1,3,1,1),#28
            nn.Tanh()
            )
        

    def forward(self,s,m,z):
        x = z*torch.exp(s)+m
        x = x.view(-1,128,1,1)
        x=self.block1(x)
        return x
    
class main(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=encoder()
        self.dec=decoder()
        
    def forward(self,x,z):
        s,m=self.enc(x)
        x=self.dec(s,m,z)
        return s,m,x    
        
class encoders(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(           
            nn.Conv2d(3,64,3),#46
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.Conv2d(64,128,3,2,1),#23
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            nn.Conv2d(128,128,3),#21
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            nn.Conv2d(128,256,3,2),#10
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.Conv2d(256,256,3,1,1),#10
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.Conv2d(256,256,3),#8
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.Conv2d(256,256,3),#6
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.Conv2d(256,512,3,2,1),#3
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            nn.Conv2d(512,2,3),#1
            nn.BatchNorm2d(2),
            nn.PReLU()
            )
        
    def forward(self,x):
        x=self.block1(x)
        m,s=x[:,:1],x[:,1:]
        return s,m
        
    
class decoders(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(           
            nn.ConvTranspose2d(256,512,3),#3
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            nn.ConvTranspose2d(512,256,3,2,1,1),#6
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.ConvTranspose2d(256,256,3),#8
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.ConvTranspose2d(256,256,3),#10
            nn.BatchNorm2d(256),
            nn.PReLU(),
                       
            nn.ConvTranspose2d(256,256,3,1,1),#10
            nn.BatchNorm2d(256),
            nn.PReLU(),
                       
            nn.ConvTranspose2d(256,128,3,2),#21
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            nn.ConvTranspose2d(128,128,3),#23
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            nn.ConvTranspose2d(128,64,3,2,1,1),#46
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.ConvTranspose2d(64,3,3),#28
            nn.Tanh()
            )
        

    def forward(self,s,m,z):
        x = z*torch.exp(s)+m
        x = x.view(-1,256,1,1)
        x=self.block1(x)
        return x
    
class mains(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=encoders()
        self.dec=decoders()
        
    def forward(self,x,z):
        s,m=self.enc(x)
        x=self.dec(s,m,z)
        return s,m,x    
    
if __name__ == '__main__':
    net=mains()
#    net=encoders()
    z=torch.randn(256)
    a=torch.range(1,48*48*6).view(2,3,48,48)        
    a,b,c=net(a,z)
    print(b.size())
        
        
        
        
        
        
        
        
        
        
        
