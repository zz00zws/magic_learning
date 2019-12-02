import torch
import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net=torch.load(cfg.save_path).to(device)
with open(cfg.word_path,encoding='UTF-8') as f:
    a=f.read().split()
dicn2w=dict(zip(range(len(a)),a))
dicw2n=dict(zip(a,range(len(a))))

s='爸'
x=torch.tensor([[]]).long()
for i in s:
    ss=dicw2n[i]
    x=torch.cat((x,torch.tensor([[int(ss)]])),1)

x=x.to(device)
p = torch.range(0,len(s)-1).long().view(1,-1).to(device)

#a=False
a=True
for i in range(512-len(s)):
    y = net(x, p)
    y = y[:, -1:]
    v, y = torch.topk(y, 8, dim=-1)
    v, y = v.reshape(-1, 8), y.reshape(-1, 8)
    v = torch.multinomial(torch.softmax(v, dim=-1), 1)
    y = torch.gather(y, -1, v)
    x = torch.cat([x, y], dim=1)
    p = torch.tensor([range(i+len(s)+1)]).to(device)

z=x.view(-1).clone().cpu().numpy().tolist()
for i in z:
    d=dicn2w[i]
    if d == '[start]':
        a=True
    if a:
        if d== '[space]':
            d='\n'
        print(d,end='')














































