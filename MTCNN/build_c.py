import os
import torch
import utils as ut
import PIL.Image as pimg
import PIL.ImageDraw as draw

path_img = r"./data/cat"
path_label = r"./data/label"
path_imgs = r"./data/img_train"
path_labels1 = r"./data/label1.txt"
path_labels2 = r"./data/label2.txt"
    
f1 = open(path_labels1,'a+')
f2 = open(path_labels2,'a+')

ss=0
for i in os.listdir(path_label):
    for j in os.listdir(os.path.join(path_label,i)):
        print(ss)
        ss=ss+1
        name=str(j.split('.')[0])+'.jpg'
        with open(os.path.join(path_label,i,j)) as f:
            data = f.readlines()[0].split()
        if data[0]!='9':
            continue
        img=pimg.open(os.path.join(path_img,i,name))
        img=img.convert('RGB')
        data=torch.tensor(list(map(int,data[1:]))).view(-1,2).float()
        x,y = img.size
        img_draw=draw.Draw(img)
        x1=torch.min(data[:,0]).view(-1)
        y1=torch.min(data[:,1]).view(-1)
        x2=torch.max(data[:,0]).view(-1)
        y2=torch.max(data[:,1]).view(-1)
        cx=torch.mean(data[:3,0]).view(-1)
        cy=torch.mean(data[:3,1]).view(-1)
        tk=torch.pow(torch.pow(x2-x1,2)+torch.pow(y2-y1,2),0.5)*0.5
        l=torch.cat((cx-tk,cy-tk,cx+tk,cy+tk),0)
#        try:
        spo=pos=0
        while spo == 0 and pos<=70:
            dcp=torch.randint((-tk*0.3).int().item(),(tk*0.3).int().item(),size=[2]).float()
            dks=torch.randint((tk*0.8).int().item(),(tk*1.25).int().item(),size=[1]).float()
            k=torch.cat((torch.tensor([0]).float(),cx+dcp[0:1]-dks,
                        cy+dcp[1:2]-dks,cx+dcp[0:1]+dks,
                        cy+dcp[1:2]+dks),0).view(-1,5)
            tk_=torch.cat((torch.tensor([0]).float(),l),0).view(-1,5)
            p=ut.iou(k,tk_)
            pos=pos+1
            if torch.min(k[:,1:]).item()<0 or k[:,3].item()>x or k[:,4].item()>y:
                continue
            if p[0]>=0.7:
                spo=spo+1
                ll=((l.view(-1,2).float()-k[:,1:3].view(-1,2).float())/(dks*2)).view(-1).numpy().tolist()
                f1.write(name+' ')
                for ii in ll:
                    f1.write(str(ii)+' ')
                f1.write('\n')
                cropimg=img.crop(k.view(-1)[1:].numpy().tolist())
                cropimg=cropimg.resize((48,48))
                cropimg.save(os.path.join(path_imgs,'48','1',name))
                cropimg=cropimg.resize((24,24))
                cropimg.save(os.path.join(path_imgs,'24','1',name))
                cropimg=cropimg.resize((12,12))
                cropimg.save(os.path.join(path_imgs,'12','1',name))
                print(1)
        spo=pos=0
        while spo == 0 and pos<=70:
            dcp=torch.randint((-tk*0.5).int().item(),(tk*0.5).int().item(),size=[2]).float()
            dks=torch.randint((tk*0.7).int().item(),(tk*1.43).int().item(),size=[1]).float()
            k=torch.cat((torch.tensor([0]).float(),cx+dcp[0:1]-dks,
                        cy+dcp[1:2]-dks,cx+dcp[0:1]+dks,
                        cy+dcp[1:2]+dks),0).view(-1,5)
            tk_=torch.cat((torch.tensor([0]).float(),l),0).view(-1,5)
            p=ut.iou(k,tk_)
            pos=pos+1
            if torch.min(k[:,1:]).item()<0 or k[:,3].item()>x or k[:,4].item()>y:
                continue
            if p[0]>=0.35 and p[0]<=0.5:
                spo=spo+1
                ll=((l.view(-1,2).float()-k[:,1:3].view(-1,2).float())/(dks*2)).view(-1).numpy().tolist()
                f2.write(name+' ')
                for ii in ll:
                    f2.write(str(ii)+' ')
                f2.write('\n')
                cropimg=img.crop(k.view(-1)[1:].numpy().tolist())
                cropimg=cropimg.resize((48,48))
                cropimg.save(os.path.join(path_imgs,'48','2',name))
                cropimg=cropimg.resize((24,24))
                cropimg.save(os.path.join(path_imgs,'24','2',name))
                cropimg=cropimg.resize((12,12))
                cropimg.save(os.path.join(path_imgs,'12','2',name))
                print(2)
        spo=pos=0
        while spo<=2 and pos<=50:
            pos=pos+1
            dcp=torch.randint(0,min(x,y),size=[2])
            k=torch.cat((torch.tensor([0]),dcp[:1]-48),0)
            k=torch.cat((k,dcp[1:]-48),0)
            k=torch.cat((k,dcp[:1]+48),0)
            k=torch.cat((k,dcp[1:]+48),0).view(-1,5).float()
            if k[:,1]<=0 or k[:,2]<=0 or k[:,3]>=x or k[:,4]>=y:
                continue
            tk=torch.cat((torch.tensor([0]).float(),l[:2].float()),0)
            tk=torch.cat((tk,l[:2].float()+l[2:4].float()),0).view(-1,5)
            p=ut.iou(k,tk)
            if p[0]<=0.17:
                spo=spo+1
                name1=str(spo)+name
                cropimg=img.crop(k.view(-1)[1:].numpy().tolist())
                cropimg=cropimg.resize((48,48))
                cropimg.save(os.path.join(path_imgs,'48','0',name1))
                cropimg=cropimg.resize((24,24))
                cropimg.save(os.path.join(path_imgs,'24','0',name1))
                cropimg=cropimg.resize((12,12))
                cropimg.save(os.path.join(path_imgs,'12','0',name1))
                print(3)
#        except:
#            continue
            
            
            
        
        
        
        
        
    
    



























