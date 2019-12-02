import torch
import torch.nn as nn
import cfg



class conv(nn.Module):
    def __init__(self,x,y,z,p=0,s=1):
        super().__init__()
        self.block1=nn.Sequential(  
            nn.Conv2d(x,y,z,padding=p,stride=s,bias=True),
            nn.BatchNorm2d(y),
            nn.PReLU()
            )
    def forward(self,y):
        return self.block1(y)

class res(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.block1=nn.Sequential(           
            conv(x,int(x/2),1),
            conv(int(x/2),x,3,1)
            )
    def forward(self,y):
        return self.block1(y)+y
    


class darknet(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1=nn.Sequential(           
            conv(3,32,3,1),
            conv(32,64,3,1,2),
            res(64),
            conv(64,128,3,1,2),
            res(128),
            res(128),
            conv(128,256,3,1,2),
            res(256),
            res(256),
            res(256),
            res(256),
            res(256),
            res(256),
            res(256),
            res(256)
            )
        
        self.block2=nn.Sequential(           
            conv(256,512,3,1,2),
            res(512),
            res(512),
            res(512),
            res(512),
            res(512),
            res(512),
            res(512),
            res(512)
            )
        
        self.block3=nn.Sequential(           
            conv(512,1024,3,1,2),
            res(1024),
            res(1024),
            res(1024),
            res(1024)
            )
    def forward(self,x):
        x=self.block1(x)
        y=self.block2(x)
        z=self.block3(y)
        return x,y,z

class conset(nn.Module):
    def __init__(self,x,y):
        super(conset, self).__init__()
        
        self.block=nn.Sequential(
            conv(x,y,1),
            conv(y,x,3,1),
            conv(x,y,1),
            conv(y,x,3,1),
            conv(x,y,1)
            )

    def forward(self,x):
        return self.block(x)
    
class upsamp(nn.Module):

    def __init__(self):
        super(upsamp, self).__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2, mode='nearest')

class mainet(nn.Module):
    def __init__(self):
        super(mainet, self).__init__()
        
        self.dnet=darknet()
        
        self.cs13=nn.Sequential(
            conset(1024, 512)
            )
        self.cs26=nn.Sequential(
            conset(768, 256)
            )
        self.cs52=nn.Sequential(
            conset(384, 128)
            )
        
        self.ou13=nn.Sequential(
            conv(512,1024,3,1),
            nn.Conv2d(1024,3*(cfg.CLASS_NUM+5),1)
            )
        self.ou26=nn.Sequential(
            conv(256,512,3,1),
            nn.Conv2d(512,3*(cfg.CLASS_NUM+5),1)
            )
        self.ou52=nn.Sequential(
            conv(128,256,3,1),
            nn.Conv2d(256,3*(cfg.CLASS_NUM+5),1)
            )
        
        self.up_13_26=nn.Sequential(
            conv(512,256,3,1),
            upsamp()
            )
        self.up_26_52=nn.Sequential(
            conv(256,128,3,1),
            upsamp()
            )
        
    def forward(self,x):
        x_52,x_26,x_13=self.dnet(x)
        x_13=self.cs13(x_13)
#        print(self.up_13_26(x_13).size(),x_26.size(),x_13.size())
        out_26=torch.cat((self.up_13_26(x_13),x_26),dim=1)
        out_26=self.cs26(out_26)
        out_52=torch.cat((self.up_26_52(out_26),x_52),dim=1)
        out_52=self.cs52(out_52)
        
        out_13=self.ou13(x_13)
        out_26=self.ou26(out_26)
        out_52=self.ou52(out_52)
        
        return out_13,out_26,out_52
        
        
if __name__ == '__main__':
    trunk = mainet().cuda()

    x = torch.Tensor(2, 3, 416, 416).cuda()

    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)
        
        
        
    


        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    