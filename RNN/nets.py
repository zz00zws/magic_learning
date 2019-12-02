import torch
import torch.nn as nn

class net1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.rnn = nn.RNN(
            input_size=28,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
 
        self.out = nn.Linear(128,10)
        
    def forward(self,x):
        x,h=self.rnn(x.view(-1,28,28),None)
        x=self.out(x[:,-1,:])
        return x
    
class net2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, stride=1,padding=1),  
            nn.BatchNorm1d(128),
            nn.PReLU(),  
            nn.Conv1d(128, 256, kernel_size=3, stride=1),  
            nn.BatchNorm1d(256),
            nn.PReLU(), 
            nn.Conv1d(256, 256, kernel_size=3, stride=1), 
            nn.BatchNorm1d(256),
            nn.PReLU() 
        ) 
        self.out = nn.Linear(256,10)
        
    def forward(self,x):
        x,h=self.layer(x.view(-1,28,28),None)
        x=self.out(x[:,-1,:])
        return x    
    
    
    
    
    
    
    
    
    
    
    
    