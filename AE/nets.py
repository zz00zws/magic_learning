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
            
            nn.Conv2d(128,64,3),#11
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.MaxPool2d(3,2),#5
            
            nn.Conv2d(64,32,3),#3
            nn.BatchNorm2d(32),
            nn.PReLU()
            )
        
        
        self.layer_1=nn.Sequential(
            nn.Linear(32*3*3,128)
            )
        
    def forward(self,x):
        x=self.block1(x).view(-1,32*3*3)
        x=self.layer_1(x)
        return x
        
    
class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(           
            nn.ConvTranspose2d(32,64,3,2,1),#5
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.ConvTranspose2d(64,64,3,2),#7
            nn.BatchNorm2d(64),
            nn.PReLU(),
                       
            nn.ConvTranspose2d(64,128,3),#13
            nn.BatchNorm2d(128),
            nn.PReLU(),
            
            nn.ConvTranspose2d(128,64,2,2),#26
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.ConvTranspose2d(64,1,3)#28
            )
        
        
        self.layer_1=nn.Sequential(
            nn.Linear(128,32*3*3)
            )
        
    def forward(self,x):
        x=self.layer_1(x).view(-1,32,3,3)
        x=self.block1(x)
        return x
    
class main(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=encoder()
        self.dec=decoder()
        
    def forward(self,x):
        x=self.enc(x)
        x=self.dec(x)
        return x    
    
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
            
            nn.Conv2d(256,256,3),#8
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.Conv2d(256,512,3,2,1),#4
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            nn.Conv2d(512,512,3),#2
            nn.BatchNorm2d(512),
            nn.PReLU()
            )
        
        
        self.layer_1=nn.Sequential(
            nn.Linear(512*2*2,256)
            )
        
    def forward(self,x):
        x=self.block1(x).view(-1,512*2*2)
        x=self.layer_1(x)
        return x
        
    
class decoders(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer_1=nn.Sequential(
            nn.Linear(256,512*2*2)
            )
        
        self.block1=nn.Sequential(           
            nn.ConvTranspose2d(512,512,3),#4
            nn.BatchNorm2d(512),
            nn.PReLU(),
            
            nn.ConvTranspose2d(512,256,3,2,1,1),#7
            nn.BatchNorm2d(256),
            nn.PReLU(),
            
            nn.ConvTranspose2d(256,256,3),#10
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
            
            nn.ConvTranspose2d(64,3,3),#48
            nn.Sigmoid()
            )
                
    def forward(self,x):
        x=self.layer_1(x).view(-1,512,2,2)
        x=self.block1(x)
        return x
    
class mains(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=encoders()
        self.dec=decoders()
        
    def forward(self,x):
        x=self.enc(x)
        x=self.dec(x)
        return x    
    
if __name__ == '__main__':
    net=mains()
    a=torch.range(1,48*48*6).view(2,3,48,48)        
    b=net(a)
    print(b.size())
        
        
        
        
        
        
        
        
        
        
        
