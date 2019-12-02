import torch
import cfg
import torch.nn as nn
import nets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f1=open(cfg.word_path,'r+',encoding='UTF-8')
f2=open(cfg.num_path,'r+',encoding='UTF-8')

words=f1.read()
nums=f2.read()
num=torch.tensor(list(map(int,nums.split()))).to(device)

#for i in num:
#    print(words.split()[int(i)],end='')
#    print(int(i))
    
net=nets.CBOW(num.size(0),10).to(device)
#opt=torch.optim.SGD(net.parameters(),lr=0.001)
opt=torch.optim.Adam(net.parameters())

losss=0
i=0
while True:
    i=i+1
    x=torch.randint(2,len(num)-2,[3000]).to(device)
    x1,x2,x3,x4,x5=num[x-2].view(-1,1),num[x-1].view(-1,1),num[x].view(-1,1),num[x+1].view(-1,1),num[x+2].view(-1,1)
    y3=net(x1,x2,x4,x5)
    loss=net.getLoss(x3,y3)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losss=losss+loss.item()
    if i%100==0:
        print('loss=',losss/100)
        print(i)
        losss=0
        torch.save(net,'./net_n2v.pth')












































