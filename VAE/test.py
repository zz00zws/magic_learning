import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

get_sample = Get_Sample(p=path)
datas = data.DataLoader(get_sample,100,shuffle=True)
NET_f = torch.load('./net_f.pth').to(device)
NET_k = torch.load('./net_k.pth').to(device)
loss_f = nn.BCELoss()
loss_k = nn.MSELoss()
NET_f.eval()
NET_k.eval()

#下面这块打开可以输出精确度
#eval_loss = 0
#acc = 0
#for dataa in datas:
#    img,label = dataa
#    img = img.to(device).float()
#    target = label[0:,4:5].to(device).int().view(-1)
#    out = NET(img).view(-1)
#    out = torch.round(out).int()
#    acc = acc + (out==target).sum().item()
#    print(acc)
# 
#mean_acc = acc/len(os.listdir(path))
#print(mean_acc)

#下面这块得到测试集的画框结果
for i,dataa in enumerate(datas):
    img,label,path_k = dataa
    img = img.to(device).float()
    target = label[0:,4:5].to(device).int().view(-1)
    out = NET_f(img).view(-1)
    out = torch.round(out).int()
    for j,(x,y,z) in enumerate(zip(img,out,path_k)):
        x=x.view(-1,224,224,3)
        img_k = pimg.open(z)
        if y == 1:
            out_k=NET_k(x)
            img_draw = draw.ImageDraw(img_k)
            img_draw.rectangle(xy=list(out_k.view(-1).cpu().clone().int().numpy()),fill=None,outline="green")
        img_k.save('{}/{}.png'.format(path_out,str(i)+str(j)))
            














