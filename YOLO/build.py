import torch
import os
import PIL.Image as pimg
import PIL.ImageDraw as draw


path_label='./data/Anno_XML'
path_img='./data/Original'

y=True
#yy={'pedestrian\n','tree\n','car\n','bicycle\n','store\n','building\n','road\n','sky\n','sidewalk\n'}
yy={'pedestrian\n','car\n','bicycle\n'}
dic=dict(zip(yy,range(3)))
f = open(r'./label.txt','w+')
for i in os.listdir(path_label):
    xx=[]
    ss=[]
    f1=open(os.path.join(path_label,i))
    a=f1.readlines()
    x=[]
    t=False
    for j,k in enumerate(a):
        if k=='<name>\n':
            if a[j+1] in yy:
                if x!=[]:
                    x=torch.tensor(list(map(int,x))).view(-1,2)
                    f.write(str(torch.max(x[:,0]).item()/2)+' ')
                    f.write(str(torch.max(x[:,1]).item()/2)+' ')
                    f.write(str(torch.min(x[:,0]).item()/2)+' ')
                    f.write(str(torch.min(x[:,1]).item()/2)+' ')
                y=True
                t=True
                x=[]
                f.write(str(dic[a[j+1]])+' ')
            else:
                y=False
        if k=='<x>\n' and y:
            x.append(a[j+1])
        if k=='<y>\n' and y:
            x.append(a[j+1])
    if x!=[]:
        x=torch.tensor(list(map(int,x))).view(-1,2)
        f.write(str(torch.max(x[:,0]).item()/2)+' ')
        f.write(str(torch.max(x[:,1]).item()/2)+' ')
        f.write(str(torch.min(x[:,0]).item()/2)+' ')
        f.write(str(torch.min(x[:,1]).item()/2)+' ')
    if t:
#        img=pimg.open(os.path.join(path_img,i.split('_')[0]+'.JPG'))
#        img=img.resize((640,480))
#        img.save(os.path.join(path_img,i.split('_')[0]+'.JPG'))
        f.write(i.split('_')[0]+'.JPG ')
        f.write('\n')
        print(i.split('_')[0]+'.JPG')
f.close()




























