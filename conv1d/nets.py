import torch
import torch.nn as nn

class net1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size=29,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
 
        self.out = nn.Linear(256,1)
        
    def forward(self,x):
        x,h=self.rnn(x,None)
        x=self.out(x[:,-1,:])
        return x
    
    
class net2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2),#14
            nn.BatchNorm1d(64),
            nn.PReLU(),  
            
            nn.Conv1d(64, 128, kernel_size=2, stride=1), #13 
            nn.BatchNorm1d(128),
            nn.PReLU(), 
            
            nn.Conv1d(128, 128, kernel_size=3, stride=2),  #6
            nn.BatchNorm1d(128),
            nn.PReLU(),
            
            nn.Conv1d(128, 256, kernel_size=2, stride=1),#5 
            nn.BatchNorm1d(256),
            nn.PReLU(), 
            
            nn.Conv1d(256, 256, kernel_size=3, stride=1),#3 
            nn.BatchNorm1d(256),
            nn.PReLU(), 
            
            nn.Conv1d(256, 512, kernel_size=3, stride=1), #1
#            nn.BatchNorm1d(512),
            nn.PReLU() 
        ) 
        self.out = nn.Linear(512,1)
        
    def forward(self,x):
        x=self.layer(x)
        x=self.out(x.view(1,-1))
        return x        
    
    
    
    
    
    
    
    
    
    
    