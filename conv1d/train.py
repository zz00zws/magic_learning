import torch
import nets
import torch.optim as optim
import torch.nn as nn
import csv
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

b=[]
with open('./test.csv') as f:
    a=csv.reader(f)

    for i in a:
        if i[5] != '':
            b.append(i[5])
      
#print(b[1:])
b=torch.tensor(list(map(float,b[1:]))).to(device).float()-7
#print(b.size())

#net = nets.net1().to(device)
net = torch.load('./net1.pth').to(device)

opt = optim.Adam(net.parameters())
#loss_f=nn.CrossEntropyLoss()
#loss_f=nn.NLLLoss()
loss_f=nn.MSELoss()

a=0
x=[]
y=[]
y2=[]
for epoch in range(10000):
    losss=0
    net.train()
    for i in range(b.size(0)-30):
        l=b[i:i+29].view(1,1,-1)
        z=net(l).view(1,1).float()
        loss=loss_f(z,b[i+30].view(1,1))
        opt.zero_grad()
        loss.backward()
        opt.step()
#        label=torch.zeros(label.size()[0],10).scatter_(1,label.cpu().reshape(-1,1),1).to(device)
        losss=losss+loss.item()
        
        a=a+1
        x.append(a)
        y.append(z.view(1).item())
        y2.append(b[i+30].item())
        plt.plot(x,y,color='red',linestyle = '-')
        plt.plot(x,y2,color='green',linestyle = '-')
        plt.show()
        if a>=100:
            del x[0]
            del y[0]
            del y2[0]
        
        if i%100 == 99:
            print('loss: ',losss/10)
            losss=0
    print(epoch)
    torch.save(net,'./net1.pth')





