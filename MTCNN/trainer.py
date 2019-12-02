import torch
import torch.nn as nn
import torch.utils.data as data
import PIL.Image as pimg
import os
import torch.optim as optim
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_img1 = r"./data/label1.txt"
with open(path_img1,'r') as f:
    line = f.read().strip()
    a = line.split("\n")
path_img2 = r"./data/label2.txt"
with open(path_img2,'r') as f:
    line = f.read().strip()
    b = line.split("\n")   

class Get_Sample(data.Dataset):
    def __init__(self,p):
        self.img_path = []
        self.labels = []
        
        for name in os.listdir(os.path.join(p,'0')):
            data_path = os.path.join(p,'0',name)
            label = [0,0,0,0,0]
            self.img_path.append(data_path)
            self.labels.append(label)
            
        for name in a:
            data_path = os.path.join(p,'1',name.split()[0])
            label = [float(1)]+list(map(float,name.split()[1:]))
            self.img_path.append(data_path)
            self.labels.append(label)
            
        for name in b:
            data_path = os.path.join(p,'2',name.split()[0])
            label = [float(0.35)]+list(map(float,name.split()[1:]))
            self.img_path.append(data_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img=pimg.open(img_path)
        img=img.convert('RGB')
        img_tensor = torchvision.transforms.ToTensor()(img) 
        norm_t=torchvision.transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        data_img=norm_t(img_tensor)
        data_label = torch.tensor(self.labels[index],dtype=torch.float)
        return data_img,data_label

class trains():
    def __init__(self,net,save_path,datas_path):
        
        self.save_p=save_path
        if os.path.exists(save_path):
            self.NET = torch.load(save_path).to(device)
        else:
            self.NET = net().to(device)
        self.opt = optim.Adam(self.NET.parameters())
        self.get_sample = Get_Sample(p = datas_path)
        self.c_loss_f=nn.BCELoss()
        self.l_loss_f=nn.MSELoss()
        
    def train(self):
        
        datas = data.DataLoader(self.get_sample,250,shuffle=True,num_workers=3)
        print('start')
        a=0
        while True:
            losss=0
            try:
                for i,(img,label_all) in enumerate(datas):
                        img=img.to(device)
                        label_c=label_all[:,:1].to(device)
                        label_l=label_all[:,1:].to(device)
                        out_c,out_l=self.NET(img)
                        out_c=out_c.view(-1,1)
                        out_l=out_l.view(-1,4)
                        mask_0=torch.lt(label_c,0.2)
                        mask_2=torch.lt(label_c,0.5)^mask_0
                        mask_1=torch.gt(label_c,0.5)
                        lab_c=torch.masked_select(label_c,mask_0+mask_1)
                        ou_c=torch.masked_select(out_c,mask_0+mask_1)
                        lab_l=torch.masked_select(label_l,mask_2+mask_1)
                        ou_l=torch.masked_select(out_l,mask_2+mask_1)
                        c_loss=self.c_loss_f(ou_c,lab_c)
                        l_loss=self.l_loss_f(ou_l,lab_l)
                        loss=c_loss+l_loss
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                        losss=losss+loss.item()
                        if i%40 == 0 and i!=0:
                            print(c_loss.item())  
                            print(l_loss.item())  
                            print(losss/40) 
                            a=a+1                        
                            print(a)  
                            losss=0
                            torch.save(self.NET,self.save_p)
            except:
                continue
            print('ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
            
            
            
            
            
            
            
            




























