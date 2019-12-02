import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class center_loss(nn.Module):
    def __init__(self,cls_num,feature_num):
        super().__init__()
        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num,feature_num))

    def forward(self, xs,ys):
#        xs = torch.nn.functional.normalize(xs)
        center_exp = self.center.index_select(dim=0, index=ys.long())
        count = torch.histc(ys, bins=self.cls_num, min=0, max=self.cls_num-1)
        count_dis = count.index_select(dim=0, index=ys.long())
        return torch.sum(torch.sqrt(torch.sum((xs-center_exp)**2,dim=1))/count_dis.float())


class arcloss(nn.Module):
    def __init__(self,feature_dim,cls_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim,cls_dim))

    def forward(self, x):
        _w = torch.norm(self.W,dim=0).view(-1,10)
        _x = torch.norm(x,dim=1).view(-1,1)
        out = torch.matmul(x,self.W)
        cosa = out/(_w*_x)
        a = torch.acos(cosa*0.5)
        top = torch.exp(_w*_x*torch.cos(a+0.05)/0.5)
        _top = torch.exp(_w*_x*torch.cos(a)/0.5)
        bottom = torch.sum(torch.exp(out),dim=1).view(-1,1)
        return top/(bottom - _top + top)

class net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(           
            nn.Conv2d(1,64,3,padding=1),#28
            nn.BatchNorm2d(64),
            nn.PReLU(),
            
            nn.Conv2d(64,32,3),#26
            nn.BatchNorm2d(32),
            nn.PReLU(),
            
            nn.MaxPool2d(2),#13
            
            nn.Conv2d(32,32,3),#11
            nn.BatchNorm2d(32),
            nn.PReLU(),
            
            nn.MaxPool2d(3,2),#5
            
            nn.Conv2d(32,16,3),#3
            nn.BatchNorm2d(16),
            nn.PReLU(),
            
            nn.Conv2d(16,2,3)#1
            )
        self.a_softmax=arcloss(2,10)

        self.center_loss_layer = center_loss(10,2)
        self.Loss = nn.MSELoss()

    def forward(self,x):
        x=self.block1(x).view(-1,2)
        y=self.a_softmax(x)
        return x,y
        
    def get_loss(self,features,outputs,label):
        loss_center = self.center_loss_layer(features,label)
        label=torch.zeros(label.size()[0],10).scatter_(1,label.cpu().reshape(-1,1),1).to(device)
        loss_cls = self.Loss(outputs,label)
        return loss_cls,loss_center
        
        
        
        
        
        
        
        
        
        
        
        
        
        
